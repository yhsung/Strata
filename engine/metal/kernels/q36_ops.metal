// metal/kernels/q36_ops.metal - Qwen3.6-35B-A3B (qwen35moe) residual, RMSNorm+xprep, and token embedding kernels.
#include <metal_stdlib>
using namespace metal;

#ifndef STRATA_XH16_DECL
#define STRATA_XH16_DECL
constant bool XH16 [[function_constant(25)]];
constant bool xh16 = is_function_constant_defined(XH16) && XH16;
static inline void put_x16(device float* xp, uint plane, uint idx, uint jj, float v) {
    device half* xh = (device half*)(xp + 4 * plane + plane / 16);
    const float q2b[4] = {1.0f, 0.25f, 0.0625f, 0.015625f};
    xh[idx] = half(v);
    xh[plane + idx] = half(v * q2b[jj & 3]);
    xh[2 * plane + idx] = half((jj & 1) ? v * 0.00390625f : v);
}
#endif

struct Q36NormArgs {
    uint n;
    uint plane;
    float eps;
};

struct Q36EmbedArgs {
    uint row;
    uint n_embd;
    uint row_bytes;
    uint type;
};

static inline void write_xprep_chunk(device float* xp, uint plane, uint base, uint count, float v,
                                     ushort tid, threadgroup float* ys) {
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f,
                          0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    if (tid < count) {
        const uint idx = base + tid;
        const uint jj = idx & 15;
        xp[idx] = v;
        if (plane) {
            xp[plane + idx] = v * q2s[jj & 7];
            xp[2 * plane + idx] = v * p4s[jj & 3];
            xp[3 * plane + idx] = (jj & 1) ? v * 0.00390625f : v;
            if (xh16) put_x16(xp, plane, idx, jj, v);
        }
        ys[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (plane && (tid & 15) == 0 && tid < count) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) {
            s0 += ys[tid + k];
            s1 += ys[tid + k + 1];
        }
        xp[4 * plane + (base + tid) / 16] = s0 + s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid: (1, T, 1), tg: (256, 1, 1)
kernel void q36_rms_prep(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device float* xp [[buffer(2)]],
                         constant Q36NormArgs& a [[buffer(3)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    threadgroup float ys[256];
    const uint row = tg.y;
    device const float* xr = x + (ulong)row * a.n;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float xv = xr[i];
        ss = fma(xv, xv, ss);
    }
    const float ssum = simd_sum(ss);
    if (lane == 0) sh[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(a.n) + a.eps);

    for (uint base = 0; base < a.n; base += 256) {
        const uint count = min(256u, a.n - base);
        const float v = tid < count ? (xr[base + tid] * r * w[base + tid]) : 0.0f;
        write_xprep_chunk(xp, a.plane, row * a.n + base, count, v, tid, ys);
    }
}

// grid: (1, T, 1), tg: (256, 1, 1) — x += y; xp = rms(x) * w (+ xprep planes)
kernel void q36_add_rms_prep(device float* x [[buffer(0)]],
                             device const float* y [[buffer(1)]],
                             device const float* w [[buffer(2)]],
                             device float* xp [[buffer(3)]],
                             constant Q36NormArgs& a [[buffer(4)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    threadgroup float ys[256];
    const uint row = tg.y;
    device float* xr = x + (ulong)row * a.n;
    device const float* yr = y + (ulong)row * a.n;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        ss = fma(v, v, ss);
    }
    const float ssum = simd_sum(ss);
    if (lane == 0) sh[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(a.n) + a.eps);

    for (uint base = 0; base < a.n; base += 256) {
        const uint count = min(256u, a.n - base);
        const float v = tid < count ? (xr[base + tid] * r * w[base + tid]) : 0.0f;
        write_xprep_chunk(xp, a.plane, row * a.n + base, count, v, tid, ys);
    }
}

kernel void q36_add(device float* x [[buffer(0)]],
                    device const float* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) x[gid] += y[gid];
}

static inline ushort ld16(device const uchar* p) { return ushort(p[0]) | (ushort(p[1]) << 8); }
static inline float h2f(ushort h) { return float(as_type<half>(h)); }
static inline void scale_min_k4(int j, device const uchar* q, thread int& d, thread int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline float decode_emb_elem(device const uchar* row, uint type, uint c, ushort l) {
    float val = 0.0f;
    if (type == 0u) {  // F32
        val = ((device const float*)row)[32 * c + l];
    } else if (type == 1u) {  // F16
        val = h2f(ld16(row + 64 * c + 2 * l));
    } else if (type == 30u) {  // BF16
        val = as_type<float>(uint(ld16(row + 64 * c + 2 * l)) << 16);
    } else if (type == 8u) {  // Q8_0
        device const uchar* b = row + 34 * c;
        const float d = h2f(ld16(b));
        val = d * float(((device const char*)(b + 2))[l]);
    } else if (type == 12u) {  // Q4_K
        device const uchar* b = row + 144 * (c >> 3);
        const int j = int(c & 7);
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * float(sc), m1 = dmin * float(m);
        device const uchar* qs = b + 16 + 32 * (j >> 1);
        const int q = (j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF);
        val = d1 * float(q) - m1;
    } else if (type == 13u) {  // Q5_K
        device const uchar* b = row + 176 * (c >> 3);
        const int j = int(c & 7);
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * float(sc), m1 = dmin * float(m);
        device const uchar* qh = b + 16;
        device const uchar* qs = b + 48 + 32 * (j >> 1);
        const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + (((qh[l] >> j) & 1) ? 16 : 0);
        val = d1 * float(q) - m1;
    } else if (type == 14u) {  // Q6_K
        device const uchar* b = row + 210 * (c >> 3);
        const int j = int(c & 7), n = j >> 2, r = j & 3;
        device const uchar* ql = b + 64 * n;
        device const uchar* qh = b + 128 + 32 * n;
        device const char* sc = (device const char*)(b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        const int is = int(l) / 16;
        int q;
        if (r == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (r == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (r == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
        else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        val = d * float(sc[is + 2 * r]) * float(q - 32);
    }
    return val;
}

// grid: (ceil((n_embd / 32) / 8), 1, 1), tg: (256, 1, 1) — one simdgroup decodes one 32-element chunk.
kernel void q36_embed_row(device const uchar* emb [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant Q36EmbedArgs& a [[buffer(2)]],
                          uint tgx [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort l [[thread_index_in_simdgroup]]) {
    const uint c = tgx * 8 + sg;
    const uint nc = a.n_embd / 32;
    if (c >= nc) return;
    device const uchar* row = emb + (ulong)a.row * a.row_bytes;
    out[32 * c + l] = decode_emb_elem(row, a.type, c, l);
}

// grid: (ceil((n_embd / 32) / 8), T, 1), tg: (256, 1, 1) — batch embed or vision row copy per token row tg.y.
kernel void q36_embed_batch(device const uchar* emb [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device const int* tok_ids [[buffer(2)]],
                            device const float* img_rows [[buffer(3)]],
                            device const int* cell_row [[buffer(4)]],
                            constant Q36EmbedArgs& a [[buffer(5)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort l [[thread_index_in_simdgroup]]) {
    const uint c = tg.x * 8 + sg;
    const uint nc = a.n_embd / 32;
    if (c >= nc) return;
    const uint t = tg.y;
    const uint idx = 32 * c + l;
    const int cr = cell_row[t];
    float val = 0.0f;
    if (cr >= 0) {
        val = img_rows[(ulong)cr * a.n_embd + idx];
    } else {
        device const uchar* row = emb + (ulong)tok_ids[t] * a.row_bytes;
        val = decode_emb_elem(row, a.type, c, l);
    }
    out[(ulong)t * a.n_embd + idx] = val;
}

// grid: (1, T, 1), tg: (256, 1, 1) — prompt RMSNorm -> fp16 directly for dense_mm GEMMs.
kernel void q36_rms_f16(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device half* yh [[buffer(2)]],
                        constant Q36NormArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    const uint row = tg.y;
    device const float* xr = x + (ulong)row * a.n;
    device half* orow = yh + (ulong)row * a.n;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float xv = xr[i];
        ss = fma(xv, xv, ss);
    }
    const float ssum = simd_sum(ss);
    if (lane == 0) sh[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(a.n) + a.eps);

    for (uint i = tid; i < a.n; i += 256) {
        orow[i] = half(xr[i] * r * w[i]);
    }
}

// grid: (1, T, 1), tg: (256, 1, 1) — x += y; writes rms(x) * w to both fp32 (for mm_gather) and fp16 (for dense_mm).
kernel void q36_add_rms_f16(device float* x [[buffer(0)]],
                            device const float* y [[buffer(1)]],
                            device const float* w [[buffer(2)]],
                            device float* yf [[buffer(3)]],
                            device half* yh [[buffer(4)]],
                            constant Q36NormArgs& a [[buffer(5)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    const uint row = tg.y;
    device float* xr = x + (ulong)row * a.n;
    device const float* yr = y + (ulong)row * a.n;
    device float* of = yf + (ulong)row * a.n;
    device half* oh = yh + (ulong)row * a.n;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        ss = fma(v, v, ss);
    }
    const float ssum = simd_sum(ss);
    if (lane == 0) sh[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(a.n) + a.eps);

    for (uint i = tid; i < a.n; i += 256) {
        const float v = xr[i] * r * w[i];
        of[i] = v;
        oh[i] = half(v);
    }
}

// grid: (1, T, 1), tg: (256, 1, 1) — plain fp32 RMSNorm (for hmain / hmtp).
kernel void q36_rms_f32(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant Q36NormArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    const uint row = tg.y;
    device const float* xr = x + (ulong)row * a.n;
    device float* yr = y + (ulong)row * a.n;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float xv = xr[i];
        ss = fma(xv, xv, ss);
    }
    const float ssum = simd_sum(ss);
    if (lane == 0) sh[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(a.n) + a.eps);

    for (uint i = tid; i < a.n; i += 256) {
        yr[i] = xr[i] * r * w[i];
    }
}

// grid: (1, T, 1), tg: (256, 1, 1) — [rms(e)*w_e | rms(h)*w_h] -> xprep layout with plane = T * 2 * a.n.
kernel void q36_mtp_concat_prep(device const float* e [[buffer(0)]],
                                device const float* h [[buffer(1)]],
                                device const float* w_e [[buffer(2)]],
                                device const float* w_h [[buffer(3)]],
                                device float* xp [[buffer(4)]],
                                constant Q36NormArgs& a [[buffer(5)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    threadgroup float ys[256];
    const uint row = tg.y;
    const uint n = a.n;
    device const float* er = e + (ulong)row * n;
    device const float* hr = h + (ulong)row * n;

    float sse = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        const float ev = er[i];
        sse = fma(ev, ev, sse);
    }
    const float se_sum = simd_sum(sse);
    if (lane == 0) sh[sg] = se_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tote = 0.0f;
    for (int i = 0; i < 8; ++i) tote += sh[i];
    const float re = rsqrt(tote / float(n) + a.eps);
    for (uint base = 0; base < n; base += 256) {
        const uint count = min(256u, n - base);
        const float v = tid < count ? (er[base + tid] * re * w_e[base + tid]) : 0.0f;
        write_xprep_chunk(xp, a.plane, row * 2u * n + base, count, v, tid, ys);
    }

    float ssh = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        const float hv = hr[i];
        ssh = fma(hv, hv, ssh);
    }
    const float sh_sum = simd_sum(ssh);
    if (lane == 0) sh[sg] = sh_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float toth = 0.0f;
    for (int i = 0; i < 8; ++i) toth += sh[i];
    const float rh = rsqrt(toth / float(n) + a.eps);
    for (uint base = 0; base < n; base += 256) {
        const uint count = min(256u, n - base);
        const float v = tid < count ? (hr[base + tid] * rh * w_h[base + tid]) : 0.0f;
        write_xprep_chunk(xp, a.plane, row * 2u * n + n + base, count, v, tid, ys);
    }
}
