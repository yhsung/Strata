// metal/tests/q36_ops_parity.cpp - TDD Cycle 3: Qwen3.6 residual/norm/embed ops (q36_ops.metal)
// + full Qwen3.6 MoE block parity (256 experts top-8, Q4_K/Q6_K routed + Q8_0 shared + F32 scalar gate).
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};

struct Q36NormArgs {
    uint32_t n;
    uint32_t plane;
    float eps;
};

struct Q36EmbedArgs {
    uint32_t row;
    uint32_t n_embd;
    uint32_t row_bytes;
    uint32_t type;
};

struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct GlueArgs { uint32_t n, n_tok, n_embd, topk; };
constexpr int P_INTS = 16 + 128 + 128 + 4 + 128 * 3;
}  // namespace

int main() {
    mtl::Context cx;
    const int lib_ops = cx.compile({"q36_ops.metal"}, "#define NO_GGML_COMMON 1\n");
    const int lib_moe = cx.compile(
        {"quant.metal", "units.metal", "moe.metal", "gemv.metal", "router.metal", "glue.metal"},
        "#define GEO_TOPK 8\n#define STRATA_KQ_DOWN 1\n");
    std::printf("device %s\n", cx.device_name().c_str());

    int fails = 0;
    const uint32_t N = 2048;
    const uint32_t NFF = 512;
    const uint32_t NEXP = 256;
    const uint32_t TOPK = 8;
    const float eps = 1e-6f;
    ref::Rng r(2026);

    // 1. q36_rms_prep + q36_add_rms_prep + q36_add
    {
        mtl::Pipeline p_rms = cx.pipeline(lib_ops, "q36_rms_prep");
        mtl::Pipeline p_arms = cx.pipeline(lib_ops, "q36_add_rms_prep");
        mtl::Pipeline p_add = cx.pipeline(lib_ops, "q36_add");
        mtl::require_fit(p_rms, 256, "q36_rms_prep");
        mtl::require_fit(p_arms, 256, "q36_add_rms_prep");
        mtl::require_fit(p_add, 256, "q36_add");

        const uint32_t plane = N;
        mtl::Buffer x = cx.alloc(4 * N), y = cx.alloc(4 * N), w = cx.alloc(4 * N),
                    xp = cx.alloc(4 * (4 * plane + plane / 16));
        std::vector<double> x0(N), y0(N), w0(N);
        for (uint32_t i = 0; i < N; ++i) {
            x0[i] = x.as<float>()[i] = r.sym() * 2.0f;
            y0[i] = y.as<float>()[i] = r.sym() * 1.5f;
            w0[i] = w.as<float>()[i] = 1.0f + 0.3f * r.sym();
        }

        Q36NormArgs na{N, plane, eps};
        mtl::Dispatch d1;
        d1.p = &p_rms;
        d1.args = {mtl::Arg::B(x), mtl::Arg::B(w), mtl::Arg::B(xp), mtl::Arg::V(na)};
        d1.grid[0] = 1;
        d1.tg[0] = 256;
        cx.run({d1});

        double ss = 0;
        for (uint32_t i = 0; i < N; ++i) ss += x0[i] * x0[i];
        double rms = 1.0 / std::sqrt(ss / N + eps);
        Err erms, exsum;
        for (uint32_t i = 0; i < N; ++i) erms.add(xp.as<float>()[i], x0[i] * rms * w0[i]);
        for (uint32_t g = 0; g < N / 16; ++g) {
            double s = 0;
            for (int j = 0; j < 16; ++j) s += x0[g * 16 + j] * rms * w0[g * 16 + j];
            exsum.add(xp.as<float>()[4 * plane + g], s);
        }
        bool ok1 = erms.l2() < 1e-5 && exsum.l2() < 1e-5;
        fails += !ok1;
        std::printf("q36_rms_prep: h rel L2 %.2e, sum16 %.2e  %s\n", erms.l2(), exsum.l2(), ok1 ? "PASS" : "FAIL");

        mtl::Dispatch d2;
        d2.p = &p_arms;
        d2.args = {mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::B(w), mtl::Arg::B(xp), mtl::Arg::V(na)};
        d2.grid[0] = 1;
        d2.tg[0] = 256;
        cx.run({d2});

        std::vector<double> x1(N);
        ss = 0;
        for (uint32_t i = 0; i < N; ++i) { x1[i] = x0[i] + y0[i]; ss += x1[i] * x1[i]; }
        rms = 1.0 / std::sqrt(ss / N + eps);
        Err ex, eh;
        for (uint32_t i = 0; i < N; ++i) {
            ex.add(x.as<float>()[i], x1[i]);
            eh.add(xp.as<float>()[i], x1[i] * rms * w0[i]);
        }
        bool ok2 = ex.l2() < 1e-6 && eh.l2() < 1e-5;
        fails += !ok2;
        std::printf("q36_add_rms_prep: x rel L2 %.2e, h %.2e  %s\n", ex.l2(), eh.l2(), ok2 ? "PASS" : "FAIL");

        mtl::Dispatch d3;
        d3.p = &p_add;
        d3.args = {mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(N)};
        d3.grid[0] = (N + 255) / 256;
        d3.tg[0] = 256;
        cx.run({d3});

        Err ea;
        for (uint32_t i = 0; i < N; ++i) ea.add(x.as<float>()[i], x1[i] + y0[i]);
        bool ok3 = ea.l2() < 1e-6;
        fails += !ok3;
        std::printf("q36_add: rel L2 %.2e  %s\n", ea.l2(), ok3 ? "PASS" : "FAIL");
    }

    // 2. q36_embed_row (Q8_0, Q4_K, Q6_K)
    {
        mtl::Pipeline p_emb = cx.pipeline(lib_ops, "q36_embed_row");
        mtl::require_fit(p_emb, 256, "q36_embed_row");
        const int nrows = 8, pick = 5;
        for (int tp : {ref::Q8_0, ref::Q4_K, ref::Q6_K}) {
            const size_t rb = ref::row_bytes(tp, N);
            mtl::Buffer emb = cx.alloc(rb * nrows), out = cx.alloc(4 * N);
            ref::random_blocks(tp, emb.as<uint8_t>(), nrows, N, 100 + tp, 0.5f);
            Q36EmbedArgs ea{(uint32_t)pick, N, (uint32_t)rb, (uint32_t)tp};
            mtl::Dispatch d;
            d.p = &p_emb;
            d.args = {mtl::Arg::B(emb), mtl::Arg::B(out), mtl::Arg::V(ea)};
            d.grid[0] = (N / 32 + 7) / 8;
            d.tg[0] = 256;
            cx.run({d});

            std::vector<float> ref_row(N);
            ref::dequant(tp, emb.as<uint8_t>() + pick * rb, ref_row.data(), N);
            Err ee;
            for (uint32_t i = 0; i < N; ++i) ee.add(out.as<float>()[i], ref_row[i]);
            bool ok = ee.l2() < 1e-6;
            fails += !ok;
            std::printf("q36_embed_row (%s): rel L2 %.2e  %s\n", ref::type_name(tp), ee.l2(), ok ? "PASS" : "FAIL");
        }
    }

    // 3. Full Qwen3.6 MoE block (256 experts top-8, Q4_K gate/up, Q4_K and Q6_K down + Q8_0 shared + F32 scalar gate)
    for (int dn_tp : {ref::Q4_K, ref::Q6_K}) {
        const int gu_tp = ref::Q4_K;
        const int sh_tp = ref::Q8_0;
        const size_t r_emb_f32 = ref::row_bytes(ref::F32, N);
        const size_t r_gu = ref::row_bytes(gu_tp, N);
        const size_t r_dn = ref::row_bytes(dn_tp, NFF);
        const size_t r_sh_gu = ref::row_bytes(sh_tp, N);
        const size_t r_sh_dn = ref::row_bytes(sh_tp, NFF);
        const size_t e_gu = r_gu * NFF;
        const size_t e_dn = r_dn * NFF;  // wait: down is [N x NFF] -> r_dn * N!
        (void)e_dn;

        mtl::Buffer W_rt = cx.alloc(r_emb_f32 * NEXP);
        mtl::Buffer W_g = cx.alloc(e_gu * NEXP), W_u = cx.alloc(e_gu * NEXP), W_d = cx.alloc(r_dn * N * NEXP);
        mtl::Buffer W_sg = cx.alloc(r_sh_gu * NFF), W_su = cx.alloc(r_sh_gu * NFF), W_sd = cx.alloc(r_sh_dn * N);
        mtl::Buffer W_sgi = cx.alloc(r_emb_f32);

        ref::random_blocks(ref::F32, W_rt.as<uint8_t>(), NEXP, N, 11, 0.2f);
        ref::random_blocks(gu_tp, W_g.as<uint8_t>(), NEXP * NFF, N, 12, 0.2f);
        ref::random_blocks(gu_tp, W_u.as<uint8_t>(), NEXP * NFF, N, 13, 0.2f);
        ref::random_blocks(dn_tp, W_d.as<uint8_t>(), NEXP * N, NFF, 14, 0.2f);
        ref::random_blocks(sh_tp, W_sg.as<uint8_t>(), NFF, N, 15, 0.2f);
        ref::random_blocks(sh_tp, W_su.as<uint8_t>(), NFF, N, 16, 0.2f);
        ref::random_blocks(sh_tp, W_sd.as<uint8_t>(), N, NFF, 17, 0.2f);
        ref::random_blocks(ref::F32, W_sgi.as<uint8_t>(), 1, N, 18, 0.2f);

        // Combine W_g, W_u, W_d offsets into one buffer view or separate offsets? moe_gu / moe_down take single W + gate_off/up_off/down_off!
        // Put W_g, W_u, W_d into one contiguous buffer W_exps (just like a GGUF wrapper!).
        const size_t sz_g = e_gu * NEXP, sz_u = e_gu * NEXP, sz_d = r_dn * N * NEXP;
        mtl::Buffer W_exps = cx.alloc(sz_g + sz_u + sz_d);
        std::memcpy(W_exps.ptr, W_g.ptr, sz_g);
        std::memcpy((uint8_t*)W_exps.ptr + sz_g, W_u.ptr, sz_u);
        std::memcpy((uint8_t*)W_exps.ptr + sz_g + sz_u, W_d.ptr, sz_d);
        cx.free(W_g); cx.free(W_u); cx.free(W_d);

        mtl::Buffer x_raw = cx.alloc(4 * N), w_rms = cx.alloc(4 * N);
        mtl::Buffer xp = cx.alloc(4 * (4 * N + N / 16));
        mtl::Buffer sh_xp = cx.alloc(4 * (4 * NFF + NFF / 16));
        mtl::Buffer logits = cx.alloc(4 * 512), ids = cx.alloc(4 * TOPK), rw = cx.alloc(4 * TOPK);
        mtl::Buffer plan = cx.alloc(4 * P_INTS), cnt = cx.alloc(16);
        mtl::Buffer h_moe = cx.alloc(4 * TOPK * NFF), ye = cx.alloc(4 * TOPK * N);
        mtl::Buffer sg_out = cx.alloc(4 * NFF), su_out = cx.alloc(4 * NFF), sgate_v = cx.alloc(16);
        mtl::Buffer ysh = cx.alloc(4 * N), bo = cx.alloc(4 * N);

        std::vector<double> x0(N), wn0(N);
        for (uint32_t i = 0; i < N; ++i) {
            x0[i] = x_raw.as<float>()[i] = r.sym();
            wn0[i] = w_rms.as<float>()[i] = 1.0f + 0.1f * r.sym();
        }

        mtl::Pipeline p_rms = cx.pipeline(lib_ops, "q36_rms_prep");
        mtl::Pipeline p_rr = cx.pipeline(lib_moe, "router_route_0", mtl::Constants().i(0, 1));
        mtl::Pipeline p_gu = cx.pipeline(lib_moe, "moe_gu_12_32_1_4");
        mtl::Pipeline p_dn = cx.pipeline(lib_moe, "moe_down_" + std::to_string(dn_tp) + "_8_1_4");
        mtl::Pipeline p_gv_sh = cx.pipeline(lib_moe, "gemv_8_32_1_4_1", mtl::Constants().i(0, 1));
        mtl::Pipeline p_gv_sg = cx.pipeline(lib_moe, "gemv_0_32_1_4_1", mtl::Constants().i(0, 1));
        mtl::Pipeline p_sw = cx.pipeline(lib_moe, "swiglu_prep");
        mtl::Pipeline p_comb = cx.pipeline(lib_moe, "moe_combine_sh");
        mtl::require_fit(p_rr, 512, "router_route_0");
        mtl::require_fit(p_gu, 128, "moe_gu");
        mtl::require_fit(p_dn, 128, "moe_down");

        Q36NormArgs na{N, N, eps};
        mtl::Dispatch d_rms{&p_rms, {mtl::Arg::B(x_raw), mtl::Arg::B(w_rms), mtl::Arg::B(xp), mtl::Arg::V(na)}, {1, 1, 1}, {256, 1, 1}};

        GemvArgs ga_rt{N, NEXP, 1, (uint32_t)r_emb_f32, N, 512};
        MoeArgs ma{0, sz_g, sz_g + sz_u, e_gu, r_dn * N, (uint32_t)r_gu, (uint32_t)r_dn,
                   N, NFF, 1, TOPK, NEXP, NFF / 4, N / 16, 0};
        mtl::Dispatch d_rr{&p_rr,
                           {mtl::Arg::B(W_rt), mtl::Arg::B(xp), mtl::Arg::B(xp, 4 * 4 * N), mtl::Arg::B(logits),
                            mtl::Arg::V(ga_rt), mtl::Arg::B(ids), mtl::Arg::B(rw), mtl::Arg::B(plan),
                            mtl::Arg::V(ma), mtl::Arg::B(cnt)},
                           {NEXP / 16, 1, 1},
                           {512, 1, 1}};

        mtl::Dispatch d_gu{&p_gu,
                           {mtl::Arg::B(W_exps), mtl::Arg::B(xp, 4 * 2 * N), mtl::Arg::B(h_moe), mtl::Arg::B(plan),
                            mtl::Arg::V(ma), mtl::Arg::B(xp, 4 * 4 * N), mtl::Arg::B(plan),
                            mtl::Arg::B(W_exps), mtl::Arg::B(W_exps), mtl::Arg::B(W_exps)},
                           {1, 1, 1},
                           {128, 1, 1}};
        d_gu.indirect = &plan;
        d_gu.ind_off = 4 * 4;

        mtl::Dispatch d_dn{&p_dn,
                           {mtl::Arg::B(W_exps), mtl::Arg::B(h_moe), mtl::Arg::B(ye), mtl::Arg::B(plan),
                            mtl::Arg::V(ma), mtl::Arg::B(plan), mtl::Arg::B(plan),
                            mtl::Arg::B(W_exps), mtl::Arg::B(W_exps), mtl::Arg::B(W_exps)},
                           {1, 1, 1},
                           {128, 1, 1}};
        d_dn.indirect = &plan;
        d_dn.ind_off = 4 * 8;

        GemvArgs ga_sgu{N, NFF, 1, (uint32_t)r_sh_gu, N, NFF};
        mtl::Dispatch d_sg{&p_gv_sh, {mtl::Arg::B(W_sg), mtl::Arg::B(xp), mtl::Arg::B(xp, 4 * 4 * N), mtl::Arg::B(sg_out), mtl::Arg::V(ga_sgu)}, {NFF / 4, 1, 1}, {128, 1, 1}};
        mtl::Dispatch d_su{&p_gv_sh, {mtl::Arg::B(W_su), mtl::Arg::B(xp), mtl::Arg::B(xp, 4 * 4 * N), mtl::Arg::B(su_out), mtl::Arg::V(ga_sgu)}, {NFF / 4, 1, 1}, {128, 1, 1}};

        GlueArgs gl_sw{NFF, 1, N, TOPK};
        mtl::Dispatch d_sw{&p_sw, {mtl::Arg::B(sg_out), mtl::Arg::B(su_out), mtl::Arg::B(sh_xp), mtl::Arg::V(gl_sw)}, {NFF / 16, 1, 1}, {16, 1, 1}};

        GemvArgs ga_sdn{NFF, N, 1, (uint32_t)r_sh_dn, NFF, N};
        mtl::Dispatch d_sd{&p_gv_sh, {mtl::Arg::B(W_sd), mtl::Arg::B(sh_xp), mtl::Arg::B(sh_xp, 4 * 4 * NFF), mtl::Arg::B(ysh), mtl::Arg::V(ga_sdn)}, {N / 4, 1, 1}, {128, 1, 1}};

        GemvArgs ga_sgi{N, 1, 1, (uint32_t)r_emb_f32, N, 1};
        mtl::Dispatch d_sgi{&p_gv_sg, {mtl::Arg::B(W_sgi), mtl::Arg::B(xp), mtl::Arg::B(xp, 4 * 4 * N), mtl::Arg::B(sgate_v), mtl::Arg::V(ga_sgi)}, {1, 1, 1}, {128, 1, 1}};

        GlueArgs gl_c{N, 1, N, TOPK};
        mtl::Dispatch d_comb{&p_comb, {mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::B(ysh), mtl::Arg::B(sgate_v), mtl::Arg::B(bo), mtl::Arg::V(gl_c)}, {(N + 127) / 128, 1, 1}, {128, 1, 1}};

        cx.run_graph({d_rms, d_rr, d_gu, d_dn, d_sg, d_su, d_sw, d_sd, d_sgi, d_comb});

        // CPU double reference
        double ss = 0;
        for (uint32_t i = 0; i < N; ++i) ss += x0[i] * x0[i];
        const double rms = 1.0 / std::sqrt(ss / N + eps);
        std::vector<double> h2(N);
        for (uint32_t i = 0; i < N; ++i) h2[i] = x0[i] * rms * wn0[i];

        std::vector<double> rl(NEXP);
        for (uint32_t e = 0; e < NEXP; ++e) {
            double acc = 0;
            const float* row = W_rt.as<float>() + e * N;
            for (uint32_t i = 0; i < N; ++i) acc += (double)row[i] * h2[i];
            rl[e] = acc;
        }
        double rmx = *std::max_element(rl.begin(), rl.end());
        double rsum = 0;
        for (double& v : rl) { v = std::exp(v - rmx); rsum += v; }
        for (double& v : rl) v /= rsum;
        std::vector<uint32_t> idx(NEXP);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](uint32_t a, uint32_t b) {
            return rl[a] != rl[b] ? rl[a] > rl[b] : a < b;
        });
        double wsum = 0;
        for (uint32_t k = 0; k < TOPK; ++k) wsum += rl[idx[k]];
        wsum = std::max(wsum, 6.103515625e-5);

        std::vector<double> bo_ref(N, 0.0);
        std::vector<float> tmp_row(std::max(N, NFF));
        for (uint32_t k = 0; k < TOPK; ++k) {
            const uint32_t e = idx[k];
            const double wk = rl[e] / wsum;
            std::vector<double> eh(NFF);
            for (uint32_t j = 0; j < NFF; ++j) {
                ref::dequant(gu_tp, (const uint8_t*)W_exps.ptr + e * e_gu + j * r_gu, tmp_row.data(), N);
                double gv = 0;
                for (uint32_t i = 0; i < N; ++i) gv += (double)tmp_row[i] * h2[i];
                ref::dequant(gu_tp, (const uint8_t*)W_exps.ptr + sz_g + e * e_gu + j * r_gu, tmp_row.data(), N);
                double uv = 0;
                for (uint32_t i = 0; i < N; ++i) uv += (double)tmp_row[i] * h2[i];
                eh[j] = (gv / (1.0 + std::exp(-gv))) * uv;
            }
            for (uint32_t i = 0; i < N; ++i) {
                ref::dequant(dn_tp, (const uint8_t*)W_exps.ptr + sz_g + sz_u + e * (r_dn * N) + i * r_dn, tmp_row.data(), NFF);
                double dv = 0;
                for (uint32_t j = 0; j < NFF; ++j) dv += (double)tmp_row[j] * eh[j];
                bo_ref[i] += wk * dv;
            }
        }
        // Shared expert
        double sg_dot = 0;
        for (uint32_t i = 0; i < N; ++i) sg_dot += (double)W_sgi.as<float>()[i] * h2[i];
        const double sg_s = 1.0 / (1.0 + std::exp(-sg_dot));
        std::vector<double> sh_h(NFF);
        for (uint32_t j = 0; j < NFF; ++j) {
            ref::dequant(sh_tp, W_sg.as<uint8_t>() + j * r_sh_gu, tmp_row.data(), N);
            double gv = 0;
            for (uint32_t i = 0; i < N; ++i) gv += (double)tmp_row[i] * h2[i];
            ref::dequant(sh_tp, W_su.as<uint8_t>() + j * r_sh_gu, tmp_row.data(), N);
            double uv = 0;
            for (uint32_t i = 0; i < N; ++i) uv += (double)tmp_row[i] * h2[i];
            sh_h[j] = (gv / (1.0 + std::exp(-gv))) * uv;
        }
        for (uint32_t i = 0; i < N; ++i) {
            ref::dequant(sh_tp, W_sd.as<uint8_t>() + i * r_sh_dn, tmp_row.data(), NFF);
            double dv = 0;
            for (uint32_t j = 0; j < NFF; ++j) dv += (double)tmp_row[j] * sh_h[j];
            bo_ref[i] += sg_s * dv;
        }

        Err ebo;
        for (uint32_t i = 0; i < N; ++i) ebo.add(bo.as<float>()[i], bo_ref[i]);
        const bool ok = ebo.l2() < 1e-5;
        fails += !ok;
        std::printf("q36_moe_block (gu=%s dn=%s sh=%s 256/top8): rel L2 %.2e  %s\n",
                    ref::type_name(gu_tp), ref::type_name(dn_tp), ref::type_name(sh_tp), ebo.l2(), ok ? "PASS" : "FAIL");
    }

    return fails ? 1 : 0;
}
