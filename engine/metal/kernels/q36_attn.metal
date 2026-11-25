// metal/kernels/q36_attn.metal - Qwen3.6-35B-A3B (qwen35moe) Dense GQA + output gate + per-head RMSNorm + IMROPE.
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

struct Q36AttnArgs {
    uint H;
    uint HKV;
    uint D;
    uint n_rot;
    int pos_t;
    int pos_h;
    int pos_w;
    int kv_pos;
    uint n_pos;
    uint ctx;
    uint plane;
    float base;
    float eps;
    float scale;
};

// grid: (H + HKV, 1, 1), tg: (256, 1, 1) — D == 256 (one thread per head dimension).
kernel void q36_attn_prep(device const float* qg [[buffer(0)]],
                          device const float* kx [[buffer(1)]],
                          device const float* vx [[buffer(2)]],
                          device const float* qw [[buffer(3)]],
                          device const float* kw [[buffer(4)]],
                          device float* q [[buffer(5)]],
                          device float* gate [[buffer(6)]],
                          device float* kc [[buffer(7)]],
                          device float* vc [[buffer(8)]],
                          constant Q36AttnArgs& a [[buffer(9)]],
                          uint head [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[8];
    threadgroup float buf[256];

    const uint D = a.D;
    const bool isq = head < a.H;
    device const float* src = isq ? (qg + (ulong)head * 2 * D) : (kx + (ulong)(head - a.H) * D);

    const float v0 = tid < D ? src[tid] : 0.0f;
    const float ssq = simd_sum(v0 * v0);
    if (lane == 0) sh[sg] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float r = rsqrt(tot / float(D) + a.eps);
    if (tid < D) buf[tid] = v0 * r * (isq ? qw[tid] : kw[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < D) {
        const uint half_rot = a.n_rot / 2;
        float v = buf[tid];
        if (tid < a.n_rot) {
            const uint p = tid < half_rot ? tid : (tid - half_rot);
            const uint m = p % 3;
            const int mpos = (m == 0) ? a.pos_t : ((m == 1) ? a.pos_h : a.pos_w);
            const float theta = float(mpos) * pow(a.base, -2.0f * float(p) / float(a.n_rot));
            const float cs = cos(theta), sn = sin(theta);
            v = (tid < half_rot) ? (buf[tid] * cs - buf[tid + half_rot] * sn)
                                 : (buf[tid - half_rot] * sn + buf[tid] * cs);
        }
        if (isq) {
            q[(ulong)head * D + tid] = v;
            gate[(ulong)head * D + tid] = qg[(ulong)head * 2 * D + D + tid];
        } else {
            const ulong kv_off = ((ulong)(head - a.H) * a.ctx + (ulong)a.kv_pos) * D + tid;
            kc[kv_off] = v;
            vc[kv_off] = vx[(ulong)(head - a.H) * D + tid];
        }
    }
}

// grid: (H, 1, 1), tg: (256, 1, 1) — D == 256, online softmax over arbitrary n_pos, output * sigmoid(gate) + xprep.
kernel void q36_attn_decode(device const float* q [[buffer(0)]],
                            device const float* kc [[buffer(1)]],
                            device const float* vc [[buffer(2)]],
                            device const float* gate [[buffer(3)]],
                            device float* att [[buffer(4)]],
                            constant Q36AttnArgs& a [[buffer(5)]],
                            uint h [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint TILE = 64;
    threadgroup float q_sh[256];
    threadgroup float sc[TILE];
    threadgroup float sh[8];
    threadgroup float ys[256];

    const uint D = a.D;
    const uint kvh = h / (a.H / a.HKV);
    const ulong kv_base = (ulong)kvh * a.ctx * D;
    if (tid < D) q_sh[tid] = q[(ulong)h * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -INFINITY;
    float sum = 0.0f;
    float acc = 0.0f;

    for (uint p0 = 0; p0 < a.n_pos; p0 += TILE) {
        const uint nc = min(TILE, a.n_pos - p0);
        // 8 simdgroups x 8 steps = 64 positions; each simdgroup reduces 256 dims (8 dims/lane).
        for (uint step = 0; step < 8; ++step) {
            const uint t = step * 8 + sg;
            float dot = 0.0f;
            if (t < nc) {
                device const float* kp = kc + kv_base + (ulong)(p0 + t) * D;
                for (uint d = lane; d < D; d += 32) {
                    dot = fma(q_sh[d], kp[d], dot);
                }
            }
            const float tot = simd_sum(dot);
            if (t < nc && lane == 0) sc[t] = tot * a.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tmx = -INFINITY;
        if (tid < nc) tmx = sc[tid];
        const float smx = simd_max(tmx);
        if (lane == 0) sh[sg] = smx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_mx = -INFINITY;
        for (int i = 0; i < 8; ++i) tile_mx = max(tile_mx, sh[i]);

        const float new_mx = max(mx, tile_mx);
        const float corr = (mx == -INFINITY) ? 0.0f : exp(mx - new_mx);

        float tw = 0.0f;
        if (tid < nc) {
            tw = exp(sc[tid] - new_mx);
            sc[tid] = tw;
        }
        const float ssum = simd_sum(tw);
        if (lane == 0) sh[sg] = ssum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_sum = 0.0f;
        for (int i = 0; i < 8; ++i) tile_sum += sh[i];

        acc *= corr;
        sum = sum * corr + tile_sum;
        mx = new_mx;

        if (tid < D) {
            device const float* vp = vc + kv_base + (ulong)p0 * D + tid;
            for (uint t = 0; t < nc; ++t) {
                acc = fma(sc[t], vp[(ulong)t * D], acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < D) {
        const float g = gate[(ulong)h * D + tid];
        const float v = (acc / sum) * (1.0f / (1.0f + exp(-g)));
        const uint idx = h * D + tid;
        att[idx] = v;
        if (a.plane) {
            const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f,
                                  0.00006103515625f};
            const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
            device float* raw = att + idx;
            const uint jj = tid & 15;
            raw[a.plane] = v * q2s[jj & 7];
            raw[2 * a.plane] = v * p4s[jj & 3];
            raw[3 * a.plane] = (jj & 1) ? v * 0.00390625f : v;
            if (xh16) put_x16(att, a.plane, idx, jj, v);
            ys[tid] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (a.plane && (tid & 15) == 0 && tid < D) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) {
            s0 += ys[tid + k];
            s1 += ys[tid + k + 1];
        }
        att[4 * a.plane + (h * D + tid) / 16] = s0 + s1;
    }
}
