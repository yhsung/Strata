// metal/tests/q36_prompt_gemm_test.cpp - Milestone 2 TDD: GDN prompt SiLU FC32 + Q36MetalEngine T=64 prompt GEMM parity (Resident & Stream).
#include "mtl_rt.h"
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct GdnPArgs {
    uint32_t n_tok;
    uint32_t h_k;
    uint32_t h_v;
    uint32_t channels;
    float eps;
    uint32_t f32;
};

struct GgufWriter {
    struct Kv {
        std::string key;
        uint32_t type;
        uint32_t u = 0;
        float f = 0.0f;
        std::string s;
        std::vector<uint32_t> arr;
    };
    struct Ten {
        std::string name;
        std::vector<uint64_t> dims;
        int type;
        std::vector<uint8_t> data;
    };
    std::vector<Kv> kvs;
    std::vector<Ten> tens;

    void add_u32(const std::string& k, uint32_t v) { kvs.push_back({k, 4, v, 0.0f, {}, {}}); }
    void add_f32(const std::string& k, float v) { kvs.push_back({k, 6, 0, v, {}, {}}); }
    void add_str(const std::string& k, const std::string& v) { kvs.push_back({k, 8, 0, 0.0f, v, {}}); }
    void add_u32_arr(const std::string& k, std::vector<uint32_t> v) { kvs.push_back({k, 9, 0, 0.0f, {}, std::move(v)}); }

    void add_tensor(const std::string& name, std::vector<uint64_t> dims, int type, uint64_t seed, float scale = 0.05f) {
        uint64_t n = 1;
        for (uint64_t d : dims) n *= d;
        const int cols = (int)dims[0];
        const int rows = (int)(n / cols);
        std::vector<uint8_t> buf(ref::row_bytes(type, cols) * rows);
        ref::random_blocks(type, buf.data(), rows, cols, seed, scale);
        tens.push_back({name, std::move(dims), type, std::move(buf)});
    }
    void add_ones_f32(const std::string& name, uint64_t n, float val = 1.0f) {
        std::vector<uint8_t> buf(4 * n);
        float* p = reinterpret_cast<float*>(buf.data());
        for (uint64_t i = 0; i < n; ++i) p[i] = val;
        tens.push_back({name, {n}, ref::F32, std::move(buf)});
    }
    bool write(const std::string& path) const {
        std::vector<uint8_t> hdr;
        auto pu32 = [&](uint32_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
        auto pu64 = [&](uint64_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 8); };
        auto pf32 = [&](float v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
        auto pstr = [&](const std::string& s) { pu64(s.size()); hdr.insert(hdr.end(), s.begin(), s.end()); };
        pu32(0x46554747u); pu32(3); pu64(tens.size()); pu64(kvs.size());
        for (const auto& kv : kvs) {
            pstr(kv.key); pu32(kv.type);
            if (kv.type == 4) pu32(kv.u);
            else if (kv.type == 6) pf32(kv.f);
            else if (kv.type == 8) pstr(kv.s);
            else { pu32(4); pu64(kv.arr.size()); for (uint32_t x : kv.arr) pu32(x); }
        }
        const uint64_t align = 32;
        uint64_t off = 0;
        for (const auto& t : tens) {
            pstr(t.name); pu32((uint32_t)t.dims.size());
            for (uint64_t d : t.dims) pu64(d);
            pu32((uint32_t)t.type); pu64(off);
            off = (off + t.data.size() + align - 1) / align * align;
        }
        while (hdr.size() % align != 0) hdr.push_back(0);
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out.write((const char*)hdr.data(), hdr.size());
        std::vector<uint8_t> pad(align, 0);
        for (const auto& t : tens) {
            out.write((const char*)t.data.data(), t.data.size());
            const size_t rem = (align - (t.data.size() % align)) % align;
            if (rem) out.write((const char*)pad.data(), rem);
        }
        return (bool)out;
    }
};

struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};

int argmax(const float* p, int64_t n) {
    int best = 0;
    for (int64_t i = 1; i < n; ++i) if (p[i] > p[best]) best = (int)i;
    return best;
}

// Direct test of gdn_prompt.metal FC 32 (Qwen3.6 SiLU gate) vs default sigmoid gate.
int test_gdn_prompt_silu() {
    constexpr int T = 64, HK = 2, HV = 4, S = 128, C = (2 * HK + HV) * S, ZV = HV * S;
    constexpr float EPS = 1e-6f;
    mtl::Context cx;
    const int lib = cx.compile({"gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p_pre = cx.pipeline(lib, "gdn_pre");
    mtl::Pipeline p_scan = cx.pipeline(lib, "gdn_scan_4");
    mtl::Pipeline p_post_sig = cx.pipeline(lib, "gdn_post");
    mtl::Pipeline p_post_silu = cx.pipeline(lib, "gdn_post", mtl::Constants().b(32, true));

    mtl::Buffer qkv = cx.alloc(4 * (size_t)T * C);
    mtl::Buffer z = cx.alloc(4 * (size_t)T * ZV);
    mtl::Buffer alpha = cx.alloc(4 * (size_t)T * HV);
    mtl::Buffer beta = cx.alloc(4 * (size_t)T * HV);
    mtl::Buffer conv_w = cx.alloc(4 * (size_t)C * 4);
    mtl::Buffer dt = cx.alloc(4 * (size_t)HV);
    mtl::Buffer ssm_a = cx.alloc(4 * (size_t)HV);
    mtl::Buffer gamma = cx.alloc(4 * (size_t)S);
    mtl::Buffer hist_in = cx.alloc(4 * (size_t)C * 3);
    mtl::Buffer hist_out = cx.alloc(4 * (size_t)C * 3);
    mtl::Buffer qn = cx.alloc(4 * (size_t)T * HK * S);
    mtl::Buffer kn = cx.alloc(4 * (size_t)T * HK * S);
    mtl::Buffer vc = cx.alloc(4 * (size_t)T * HV * S);
    mtl::Buffer gb = cx.alloc(8 * (size_t)T * HV);
    mtl::Buffer kq = cx.alloc(4 * (size_t)T * HK);
    mtl::Buffer state = cx.alloc(4 * (size_t)S * HV * S);
    mtl::Buffer o_buf = cx.alloc(4 * (size_t)T * ZV);
    mtl::Buffer yh_silu = cx.alloc(2 * (size_t)T * ZV);
    mtl::Buffer y_silu = cx.alloc(4 * (size_t)T * ZV);
    mtl::Buffer yh_sig = cx.alloc(2 * (size_t)T * ZV);
    mtl::Buffer y_sig = cx.alloc(4 * (size_t)T * ZV);

    ref::Rng rng(1234);
    for (int i = 0; i < T * C; ++i) qkv.as<float>()[i] = 0.5f * rng.sym();
    for (int i = 0; i < T * ZV; ++i) z.as<float>()[i] = -1.5f + 3.0f * rng.uni();
    for (int i = 0; i < T * HV; ++i) {
        alpha.as<float>()[i] = rng.sym();
        beta.as<float>()[i] = rng.sym();
    }
    for (int i = 0; i < C * 4; ++i) conv_w.as<float>()[i] = 0.25f * rng.sym();
    for (int i = 0; i < C * 3; ++i) hist_in.as<float>()[i] = 0.2f * rng.sym();
    for (int i = 0; i < HV; ++i) {
        dt.as<float>()[i] = 0.1f * (i + 1);
        ssm_a.as<float>()[i] = -0.5f - 0.1f * i;
    }
    for (int i = 0; i < S; ++i) gamma.as<float>()[i] = 0.8f + 0.01f * (i % 20);
    std::vector<double> state_ref((size_t)S * HV * S);
    for (size_t i = 0; i < state_ref.size(); ++i) {
        float v = 0.05f * rng.sym();
        state.as<float>()[i] = v;
        state_ref[i] = v;
    }

    GdnPArgs ga{(uint32_t)T, (uint32_t)HK, (uint32_t)HV, (uint32_t)C, EPS, 1u};
    std::vector<mtl::Dispatch> ds;
    {
        mtl::Dispatch d;
        d.p = &p_pre;
        d.args = {mtl::Arg::B(qkv), mtl::Arg::B(hist_in), mtl::Arg::B(conv_w), mtl::Arg::B(alpha),
                  mtl::Arg::B(beta), mtl::Arg::B(dt), mtl::Arg::B(ssm_a), mtl::Arg::B(qn),
                  mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(kq),
                  mtl::Arg::B(hist_out), mtl::Arg::V(ga)};
        d.grid[0] = HK + HV / 2;
        d.grid[1] = T;
        d.tg[0] = 256;
        ds.push_back(d);
    }
    {
        mtl::Dispatch d;
        d.p = &p_scan;
        d.args = {mtl::Arg::B(state), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc),
                  mtl::Arg::B(gb), mtl::Arg::B(kq), mtl::Arg::B(o_buf), mtl::Arg::V(ga)};
        d.grid[0] = HV;
        d.tg[0] = 128;
        ds.push_back(d);
    }
    {
        mtl::Dispatch d;
        d.p = &p_post_silu;
        d.args = {mtl::Arg::B(o_buf), mtl::Arg::B(z), mtl::Arg::B(gamma), mtl::Arg::B(yh_silu),
                  mtl::Arg::B(y_silu), mtl::Arg::V(ga)};
        d.grid[0] = HV;
        d.grid[1] = T;
        d.tg[0] = 128;
        ds.push_back(d);
    }
    {
        mtl::Dispatch d;
        d.p = &p_post_sig;
        d.args = {mtl::Arg::B(o_buf), mtl::Arg::B(z), mtl::Arg::B(gamma), mtl::Arg::B(yh_sig),
                  mtl::Arg::B(y_sig), mtl::Arg::V(ga)};
        d.grid[0] = HV;
        d.grid[1] = T;
        d.tg[0] = 128;
        ds.push_back(d);
    }
    cx.run(ds);

    // CPU reference
    const float* qkv_p = qkv.as<float>();
    const float* hist_p = hist_in.as<float>();
    const float* cw_p = conv_w.as<float>();
    const float* al_p = alpha.as<float>();
    const float* be_p = beta.as<float>();
    const float* dt_p = dt.as<float>();
    const float* sa_p = ssm_a.as<float>();
    const float* z_p = z.as<float>();
    const float* g_p = gamma.as<float>();

    Err err_silu, err_sig, err_state;
    const double scale = 1.0 / std::sqrt((double)S);
    std::vector<double> ch_conv(C);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) {
                int src = t - 3 + k;
                double xv = src >= 0 ? qkv_p[src * C + c] : hist_p[c * 3 + (src + 3)];
                s += xv * cw_p[c * 4 + k];
            }
            ch_conv[c] = s / (1.0 + std::exp(-s));
        }
        std::vector<std::vector<double>> qn_r(HK, std::vector<double>(S)), kn_r(HK, std::vector<double>(S));
        for (int hk = 0; hk < HK; ++hk) {
            double sq = 0.0, sk = 0.0;
            for (int i = 0; i < S; ++i) {
                double qv = ch_conv[hk * S + i];
                double kv = ch_conv[HK * S + hk * S + i];
                sq += qv * qv;
                sk += kv * kv;
            }
            double rq = 1.0 / std::sqrt(sq / S + EPS / S) * scale;
            double rk = 1.0 / std::sqrt(sk / S + EPS / S) * scale;
            for (int i = 0; i < S; ++i) {
                qn_r[hk][i] = ch_conv[hk * S + i] * rq;
                kn_r[hk][i] = ch_conv[HK * S + hk * S + i] * rk;
            }
        }
        for (int hv = 0; hv < HV; ++hv) {
            int hk = hv % HK;
            double v_al = al_p[t * HV + hv] + dt_p[hv];
            double sp = std::log1p(std::exp(v_al));
            double gv = std::exp(sp * sa_p[hv]);
            double bv = 1.0 / (1.0 + std::exp(-be_p[t * HV + hv]));
            std::vector<double> o(S, 0.0);
            double oo = 0.0;
            for (int j = 0; j < S; ++j) {
                double a_j = 0.0;
                for (int i = 0; i < S; ++i) a_j += state_ref[(i * HV + hv) * S + j] * kn_r[hk][i];
                double vv = ch_conv[2 * HK * S + hv * S + j];
                double delta = (vv - gv * a_j) * bv;
                double oj = 0.0;
                for (int i = 0; i < S; ++i) {
                    double sn = gv * state_ref[(i * HV + hv) * S + j] + kn_r[hk][i] * delta;
                    state_ref[(i * HV + hv) * S + j] = sn;
                    oj += sn * qn_r[hk][i];
                }
                o[j] = oj * scale;
                oo += o[j] * o[j];
            }
            double rs = 1.0 / std::sqrt(oo / S + EPS);
            for (int j = 0; j < S; ++j) {
                int idx = (t * HV + hv) * S + j;
                double zv = z_p[idx];
                double base = rs * o[j] * g_p[j];
                double y_silu_r = base * (zv / (1.0 + std::exp(-zv)));
                double y_sig_r = base * (1.0 / (1.0 + std::exp(-zv)));
                err_silu.add(y_silu.as<float>()[idx], y_silu_r);
                err_sig.add(y_sig.as<float>()[idx], y_sig_r);
            }
        }
    }
    for (size_t i = 0; i < state_ref.size(); ++i) err_state.add(state.as<float>()[i], state_ref[i]);

    std::printf("gdn_prompt FC32 SiLU rel L2=%.3e, default sigmoid rel L2=%.3e, state rel L2=%.3e\n",
                err_silu.l2(), err_sig.l2(), err_state.l2());
    if (!(err_silu.l2() < 1e-4) || !(err_sig.l2() < 1e-4) || !(err_state.l2() < 1e-4)) {
        std::fprintf(stderr, "FAIL: gdn_prompt FC32 SiLU check\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    if (test_gdn_prompt_silu() != 0) return 1;

    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256, TOPK = 8, NVOCAB = 256;
    constexpr int HK = 2, HV = 4, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64, MAX_CTX = 128;

    GgufWriter gw;
    gw.add_str("general.architecture", "qwen35moe");
    gw.add_u32("qwen35moe.block_count", NL + 1);
    gw.add_u32("qwen35moe.nextn_predict_layers", 1);
    gw.add_u32("qwen35moe.embedding_length", N);
    gw.add_u32("qwen35moe.expert_count", NEXP);
    gw.add_u32("qwen35moe.expert_used_count", TOPK);
    gw.add_u32("qwen35moe.expert_feed_forward_length", NFF);
    gw.add_u32("qwen35moe.expert_shared_feed_forward_length", NFF);
    gw.add_u32("qwen35moe.full_attention_interval", 4);
    gw.add_u32("qwen35moe.attention.head_count", H);
    gw.add_u32("qwen35moe.attention.head_count_kv", HKV);
    gw.add_u32("qwen35moe.attention.key_length", D);
    gw.add_u32("qwen35moe.attention.value_length", D);
    gw.add_f32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    gw.add_u32("qwen35moe.rope.dimension_count", NROT);
    gw.add_u32_arr("qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    gw.add_f32("qwen35moe.rope.freq_base", 10000000.0f);
    gw.add_u32("qwen35moe.ssm.conv_kernel", 4);
    gw.add_u32("qwen35moe.ssm.group_count", HK);
    gw.add_u32("qwen35moe.ssm.time_step_rank", HV);
    gw.add_u32("qwen35moe.ssm.state_size", S);
    gw.add_u32("qwen35moe.ssm.inner_size", ZV);

    gw.add_tensor("token_embd.weight", {N, NVOCAB}, ref::Q8_0, 1, 0.12f);
    gw.add_ones_f32("output_norm.weight", N, 1.05f);
    gw.add_tensor("output.weight", {N, NVOCAB}, ref::Q6_K, 2, 0.1f);

    uint64_t seed = 50;
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        gw.add_ones_f32(b + "attn_norm.weight", N, 1.0f);
        gw.add_ones_f32(b + "post_attention_norm.weight", N, 1.0f);
        if (l % 4 != 3) {
            gw.add_tensor(b + "attn_qkv.weight", {N, CONV}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_gate.weight", {N, ZV}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "ssm_alpha.weight", {N, HV}, ref::F32, seed++, 0.04f);
            gw.add_tensor(b + "ssm_beta.weight", {N, HV}, ref::F32, seed++, 0.04f);
            gw.add_tensor(b + "ssm_conv1d.weight", {4, CONV}, ref::F32, seed++, 0.05f);
            gw.add_ones_f32(b + "ssm_dt.bias", HV, 0.1f);
            gw.add_ones_f32(b + "ssm_a", HV, -0.5f);
            gw.add_ones_f32(b + "ssm_norm.weight", S, 1.0f);
            gw.add_tensor(b + "ssm_out.weight", {ZV, N}, ref::Q5_K, seed++, 0.04f);
        } else {
            gw.add_tensor(b + "attn_q.weight", {N, H * 2 * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_k.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_v.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_ones_f32(b + "attn_q_norm.weight", D, 1.0f);
            gw.add_ones_f32(b + "attn_k_norm.weight", D, 1.0f);
            gw.add_tensor(b + "attn_output.weight", {H * D, N}, ref::Q8_0, seed++, 0.04f);
        }
        gw.add_tensor(b + "ffn_gate_inp.weight", {N, NEXP}, ref::BF16, seed++, 0.1f);
        gw.add_tensor(b + "ffn_gate_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_exps.weight", {NFF, N, NEXP}, l == 3 ? ref::Q6_K : ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_gate_inp_shexp.weight", {N}, ref::F32, seed++, 0.05f);
        gw.add_tensor(b + "ffn_gate_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_shexp.weight", {NFF, N}, ref::Q8_0, seed++, 0.03f);
    }
    gw.add_ones_f32("blk.4.nextn.enorm.weight", N, 1.0f);

    const std::string path = "q36_prompt_gemm_synth.gguf";
    if (!gw.write(path)) return 2;

    constexpr int NTOK = 75;
    std::vector<int32_t> tokens(NTOK);
    for (int i = 0; i < NTOK; ++i) tokens[i] = (i * 37 + 11) % NVOCAB;

    std::vector<float> ref63, ref74;
    std::string err;
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = false;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "ref open: %s\n", err.c_str()); return 1; }
        for (int t = 0; t < NTOK; ++t) {
            const float* lg = nullptr;
            if (!eng.forward(tokens[t], lg, err)) { std::fprintf(stderr, "ref fwd %d: %s\n", t, err.c_str()); return 1; }
            if (t == 63) ref63.assign(lg, lg + NVOCAB);
            if (t == 74) ref74.assign(lg, lg + NVOCAB);
        }
    }

    auto check_logits = [&](const char* label, const float* got, const std::vector<float>& ref_v) -> bool {
        Err e;
        for (int i = 0; i < NVOCAB; ++i) e.add(got[i], ref_v[i]);
        const int ag = argmax(got, NVOCAB), ar = argmax(ref_v.data(), NVOCAB);
        std::printf("%s: rel L2=%.3e, argmax=%d (ref %d)\n", label, e.l2(), ag, ar);
        return e.l2() < 2e-2 && ag == ar;
    };

    // Resident T=64 + tail 11, and single-call T=75
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = false;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "resident open: %s\n", err.c_str()); return 1; }
        const float* lg = nullptr;
        if (!eng.forward_chunk(0, tokens.data(), 64, nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "resident T=64: %s\n", err.c_str());
            return 1;
        }
        if (!check_logits("resident T=64", lg, ref63)) return 1;
        if (!eng.forward_chunk(0, tokens.data() + 64, NTOK - 64, nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "resident tail: %s\n", err.c_str());
            return 1;
        }
        if (!check_logits("resident tail 64+11", lg, ref74)) return 1;

        eng.reset();
        if (!eng.forward_chunk(0, tokens.data(), NTOK, nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "resident single T=75: %s\n", err.c_str());
            return 1;
        }
        if (!check_logits("resident single T=75", lg, ref74)) return 1;
    }

    // Streamed arena_gb=0.10 (fits >= 256 expert slots per layer for T=64 across Q4_K/Q6_K classes)
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = true;
        opt.arena_gb = 0.10;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "stream open: %s\n", err.c_str()); return 1; }
        const float* lg = nullptr;
        if (!eng.forward_chunk(0, tokens.data(), 64, nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "stream T=64: %s\n", err.c_str());
            return 1;
        }
        if (!check_logits("stream T=64", lg, ref63)) return 1;
        if (!eng.forward_chunk(0, tokens.data() + 64, NTOK - 64, nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "stream tail: %s\n", err.c_str());
            return 1;
        }
        if (!check_logits("stream tail 64+11", lg, ref74)) return 1;
    }

    std::remove(path.c_str());
    return 0;
}
