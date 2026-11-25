// metal/tests/q36_engine_test.cpp - TDD Cycle 4: Q36MetalEngine end-to-end synthetic qwen35moe GGUF parity & reset test.
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Minimal GGUF v3 writer for synthetic qwen35moe tests.
struct GgufWriter {
    struct Kv {
        std::string key;
        uint32_t type;  // 4=u32, 6=f32, 8=str, 9=arr(u32)
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
        const uint64_t cols = dims[0];
        const int rows = (int)(n / cols);
        const size_t rb = ref::row_bytes(type, (int)cols);
        std::vector<uint8_t> buf(rb * rows);
        ref::random_blocks(type, buf.data(), rows, (int)cols, seed, scale);
        tens.push_back({name, std::move(dims), type, std::move(buf)});
    }

    void add_ones_f32(const std::string& name, uint64_t n, float val = 1.0f) {
        std::vector<uint8_t> buf(4 * n);
        float* p = reinterpret_cast<float*>(buf.data());
        for (uint64_t i = 0; i < n; ++i) p[i] = val;
        tens.push_back({name, {n}, ref::F32, std::move(buf)});
    }

    void add_f32_custom(const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& vals) {
        std::vector<uint8_t> buf(4 * vals.size());
        std::memcpy(buf.data(), vals.data(), buf.size());
        tens.push_back({name, std::move(dims), ref::F32, std::move(buf)});
    }

    bool write(const std::string& path) const {
        std::vector<uint8_t> hdr;
        auto pu32 = [&](uint32_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
        auto pu64 = [&](uint64_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 8); };
        auto pf32 = [&](float v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
        auto pstr = [&](const std::string& s) {
            pu64(s.size());
            hdr.insert(hdr.end(), s.begin(), s.end());
        };

        pu32(0x46554747u);  // GGUF
        pu32(3);
        pu64(tens.size());
        pu64(kvs.size());

        for (const auto& kv : kvs) {
            pstr(kv.key);
            pu32(kv.type);
            if (kv.type == 4) pu32(kv.u);
            else if (kv.type == 6) pf32(kv.f);
            else if (kv.type == 8) pstr(kv.s);
            else if (kv.type == 9) {
                pu32(4);
                pu64(kv.arr.size());
                for (uint32_t x : kv.arr) pu32(x);
            }
        }

        const uint64_t align = 32;
        uint64_t data_off = 0;
        for (const auto& t : tens) {
            pstr(t.name);
            pu32((uint32_t)t.dims.size());
            for (uint64_t d : t.dims) pu64(d);
            pu32((uint32_t)t.type);
            pu64(data_off);
            data_off = (data_off + t.data.size() + align - 1) / align * align;
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

std::vector<double> deq_tensor(const GgufWriter::Ten& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    std::vector<float> tmp(n);
    ref::dequant(t.type, t.data.data(), tmp.data(), n);
    return std::vector<double>(tmp.begin(), tmp.end());
}

std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x, int n_out, int n_in) {
    std::vector<double> y(n_out, 0.0);
    for (int r = 0; r < n_out; ++r) {
        double s = 0;
        for (int i = 0; i < n_in; ++i) s += W[(size_t)r * n_in + i] * x[i];
        y[r] = s;
    }
    return y;
}

std::vector<double> rmsnorm(const std::vector<double>& x, const std::vector<double>& w, double eps) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double r = 1.0 / std::sqrt(ss / x.size() + eps);
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * r * w[i];
    return y;
}

}  // namespace

int main() {
    // 1. Reject non-qwen35moe architecture
    {
        GgufWriter bad;
        bad.add_str("general.architecture", "llama");
        bad.add_ones_f32("dummy", 32);
        const std::string bad_path = "q36_bad_arch.gguf";
        if (!bad.write(bad_path)) return 2;

        strata::metal_q36::Q36MetalEngine eng;
        std::string err;
        if (eng.open(bad_path, {}, err)) {
            std::fprintf(stderr, "FAIL: expected open() to reject llama architecture\n");
            return 1;
        }
        std::printf("reject non-qwen35moe: ok (%s)\n", err.c_str());
    }

    // 2. Synthetic qwen35moe model (4 trunk layers + 1 MTP layer ignored, 256 experts top-8)
    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256, TOPK = 8, NVOCAB = 256;
    constexpr int HK = 16, HV = 16, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64, MAX_CTX = 32;
    constexpr double EPS = 1e-6, ROPE_BASE = 10000000.0;

    GgufWriter gw;
    gw.add_str("general.architecture", "qwen35moe");
    gw.add_u32("qwen35moe.block_count", NL + 1);  // +1 MTP layer (blk.4.*) to verify trunk skipping
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
    gw.add_f32("qwen35moe.attention.layer_norm_rms_epsilon", (float)EPS);
    gw.add_u32("qwen35moe.rope.dimension_count", NROT);
    gw.add_u32_arr("qwen35moe.rope.dimension_sections", {11, 11, 10, 0});
    gw.add_f32("qwen35moe.rope.freq_base", (float)ROPE_BASE);
    gw.add_u32("qwen35moe.ssm.conv_kernel", 4);
    gw.add_u32("qwen35moe.ssm.group_count", HK);
    gw.add_u32("qwen35moe.ssm.time_step_rank", HV);
    gw.add_u32("qwen35moe.ssm.state_size", S);
    gw.add_u32("qwen35moe.ssm.inner_size", ZV);

    gw.add_tensor("token_embd.weight", {N, NVOCAB}, ref::Q8_0, 1, 0.15f);
    gw.add_ones_f32("output_norm.weight", N, 1.05f);
    gw.add_tensor("output.weight", {N, NVOCAB}, ref::Q6_K, 2, 0.1f);

    uint64_t seed = 10;
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        gw.add_ones_f32(b + "attn_norm.weight", N, 1.0f + 0.02f * l);
        gw.add_ones_f32(b + "post_attention_norm.weight", N, 1.0f - 0.02f * l);

        if (l % 4 != 3) {
            gw.add_tensor(b + "attn_qkv.weight", {N, CONV}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_gate.weight", {N, ZV}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "ssm_alpha.weight", {N, HV}, ref::F32, seed++, 0.04f);
            gw.add_tensor(b + "ssm_beta.weight", {N, HV}, ref::F32, seed++, 0.04f);
            gw.add_tensor(b + "ssm_conv1d.weight", {4, CONV}, ref::F32, seed++, 0.05f);
            gw.add_ones_f32(b + "ssm_dt.bias", HV, 0.1f);
            gw.add_ones_f32(b + "ssm_a", HV, -0.5f);
            gw.add_ones_f32(b + "ssm_norm.weight", S, 1.0f);
            gw.add_tensor(b + "ssm_out.weight", {ZV, N}, ref::Q4_K, seed++, 0.04f);
        } else {
            gw.add_tensor(b + "attn_q.weight", {N, H * 2 * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_k.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_v.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_ones_f32(b + "attn_q_norm.weight", D, 1.0f);
            gw.add_ones_f32(b + "attn_k_norm.weight", D, 1.0f);
            gw.add_tensor(b + "attn_output.weight", {H * D, N}, ref::Q4_K, seed++, 0.04f);
        }

        gw.add_tensor(b + "ffn_gate_inp.weight", {N, NEXP}, ref::F32, seed++, 0.1f);
        gw.add_tensor(b + "ffn_gate_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        const int dn_tp = (l % 4 == 3) ? ref::Q6_K : ref::Q4_K;
        gw.add_tensor(b + "ffn_down_exps.weight", {NFF, N, NEXP}, dn_tp, seed++, 0.03f);

        gw.add_tensor(b + "ffn_gate_inp_shexp.weight", {N}, ref::F32, seed++, 0.05f);
        gw.add_tensor(b + "ffn_gate_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_shexp.weight", {NFF, N}, ref::Q8_0, seed++, 0.03f);
    }
    // Dummy MTP layer blk.4.* (must be ignored by Phase 1)
    gw.add_ones_f32("blk.4.nextn.enorm.weight", N, 1.0f);

    const std::string synth_path = "q36_synth_test.gguf";
    if (!gw.write(synth_path)) return 2;

    strata::metal_q36::Q36MetalEngine eng;
    std::string err;
    if (!eng.open(synth_path, {MAX_CTX, true}, err)) {
        std::fprintf(stderr, "open failed: %s\n", err.c_str());
        return 1;
    }
    if (eng.n_vocab() != NVOCAB || eng.n_trunk_layers() != NL) {
        std::fprintf(stderr, "metadata mismatch: vocab=%lld layers=%lld\n",
                     (long long)eng.n_vocab(), (long long)eng.n_trunk_layers());
        return 1;
    }

    auto find_ten = [&](const std::string& name) -> const GgufWriter::Ten& {
        for (const auto& t : gw.tens) if (t.name == name) return t;
        std::abort();
    };

    // CPU double-precision reference state
    std::vector<std::vector<double>> gdn_S(NL, std::vector<double>((size_t)S * HV * S, 0.0));
    std::vector<std::vector<double>> gdn_H(NL, std::vector<double>((size_t)CONV * 3, 0.0));
    std::vector<std::vector<double>> kv_K(NL, std::vector<double>((size_t)HKV * MAX_CTX * D, 0.0));
    std::vector<std::vector<double>> kv_V(NL, std::vector<double>((size_t)HKV * MAX_CTX * D, 0.0));

    const auto emb_all = deq_tensor(find_ten("token_embd.weight"));
    const auto out_norm = deq_tensor(find_ten("output_norm.weight"));
    const auto out_w = deq_tensor(find_ten("output.weight"));

    const int tokens[] = {7, 42, 19};
    std::vector<float> first_tok_logits;
    int fails = 0;

    for (int pos = 0; pos < 3; ++pos) {
        const int tok = tokens[pos];
        const float* gpu_logits = nullptr;
        if (!eng.forward(tok, gpu_logits, err)) {
            std::fprintf(stderr, "forward failed at pos %d: %s\n", pos, err.c_str());
            return 1;
        }
        if (pos == 0) first_tok_logits.assign(gpu_logits, gpu_logits + NVOCAB);

        std::vector<double> x(emb_all.begin() + (size_t)tok * N, emb_all.begin() + (size_t)(tok + 1) * N);
        for (int l = 0; l < NL; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const auto h = rmsnorm(x, deq_tensor(find_ten(b + "attn_norm.weight")), EPS);
            std::vector<double> y(N, 0.0);

            if (l % 4 != 3) {
                const auto qkv = matvec(deq_tensor(find_ten(b + "attn_qkv.weight")), h, CONV, N);
                const auto z = matvec(deq_tensor(find_ten(b + "attn_gate.weight")), h, ZV, N);
                const auto al = matvec(deq_tensor(find_ten(b + "ssm_alpha.weight")), h, HV, N);
                const auto be = matvec(deq_tensor(find_ten(b + "ssm_beta.weight")), h, HV, N);
                const auto cw = deq_tensor(find_ten(b + "ssm_conv1d.weight"));
                const auto dt = deq_tensor(find_ten(b + "ssm_dt.bias"));
                const auto sa = deq_tensor(find_ten(b + "ssm_a"));
                const auto gm = deq_tensor(find_ten(b + "ssm_norm.weight"));

                std::vector<double> cv(CONV);
                for (int c = 0; c < CONV; ++c) {
                    double s = gdn_H[l][c * 3] * cw[c * 4] + gdn_H[l][c * 3 + 1] * cw[c * 4 + 1] +
                               gdn_H[l][c * 3 + 2] * cw[c * 4 + 2] + qkv[c] * cw[c * 4 + 3];
                    cv[c] = s / (1.0 + std::exp(-s));
                    gdn_H[l][c * 3] = gdn_H[l][c * 3 + 1];
                    gdn_H[l][c * 3 + 1] = gdn_H[l][c * 3 + 2];
                    gdn_H[l][c * 3 + 2] = qkv[c];
                }
                for (int hh = 0; hh < 2 * HK; ++hh) {
                    double ss = 0;
                    for (int i = 0; i < S; ++i) ss += cv[hh * S + i] * cv[hh * S + i];
                    const double rs = 1.0 / std::sqrt(ss / S + EPS / S);
                    for (int i = 0; i < S; ++i) cv[hh * S + i] = cv[hh * S + i] * rs / std::sqrt((double)S);
                }
                std::vector<double> gdn_o(ZV);
                for (int vh = 0; vh < HV; ++vh) {
                    const int kh = vh % HK;
                    const double* q = &cv[kh * S];
                    const double* k = &cv[HK * S + kh * S];
                    const double* v = &cv[2 * HK * S + vh * S];
                    const double av = al[vh] + dt[vh];
                    const double sp = av > 20.0 ? av : std::log1p(std::exp(av));
                    const double g = std::exp(sp * sa[vh]);
                    const double bval = 1.0 / (1.0 + std::exp(-be[vh]));
                    std::vector<double> o(S);
                    for (int jj = 0; jj < S; ++jj) {
                        double kv = 0;
                        for (int i = 0; i < S; ++i) kv += gdn_S[l][((size_t)i * HV + vh) * S + jj] * k[i];
                        const double delta = (v[jj] - g * kv) * bval;
                        double acc = 0;
                        for (int i = 0; i < S; ++i) {
                            double& sv = gdn_S[l][((size_t)i * HV + vh) * S + jj];
                            sv = g * sv + k[i] * delta;
                            acc += sv * q[i];
                        }
                        o[jj] = acc / std::sqrt((double)S);
                    }
                    double ss = 0;
                    for (int jj = 0; jj < S; ++jj) ss += o[jj] * o[jj];
                    const double rs = 1.0 / std::sqrt(ss / S + EPS);
                    for (int jj = 0; jj < S; ++jj) {
                        const double zz = z[vh * S + jj];
                        gdn_o[vh * S + jj] = o[jj] * rs * gm[jj] * (zz / (1.0 + std::exp(-zz)));
                    }
                }
                y = matvec(deq_tensor(find_ten(b + "ssm_out.weight")), gdn_o, N, ZV);
            } else {
                const auto qg = matvec(deq_tensor(find_ten(b + "attn_q.weight")), h, H * 2 * D, N);
                const auto kx = matvec(deq_tensor(find_ten(b + "attn_k.weight")), h, HKV * D, N);
                const auto vx = matvec(deq_tensor(find_ten(b + "attn_v.weight")), h, HKV * D, N);
                const auto qw = deq_tensor(find_ten(b + "attn_q_norm.weight"));
                const auto kw = deq_tensor(find_ten(b + "attn_k_norm.weight"));
                std::vector<double> q(H * D), gate(H * D);
                const int half = NROT / 2;
                for (int hd = 0; hd < H + HKV; ++hd) {
                    const bool isq = hd < H;
                    const double* src = isq ? &qg[hd * 2 * D] : &kx[(hd - H) * D];
                    double ss = 0;
                    for (int i = 0; i < D; ++i) ss += src[i] * src[i];
                    const double rs = 1.0 / std::sqrt(ss / D + EPS);
                    std::vector<double> buf(D);
                    for (int i = 0; i < D; ++i) buf[i] = src[i] * rs * (isq ? qw[i] : kw[i]);
                    double* dst = isq ? &q[hd * D] : &kv_K[l][((size_t)(hd - H) * MAX_CTX + pos) * D];
                    for (int i = 0; i < D; ++i) {
                        double v = buf[i];
                        if (i < NROT) {
                            const int p = i < half ? i : i - half;
                            const double theta = (double)pos * std::pow(ROPE_BASE, -2.0 * p / NROT);
                            const double cs = std::cos(theta), sn = std::sin(theta);
                            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
                        }
                        dst[i] = v;
                    }
                    if (isq) for (int i = 0; i < D; ++i) gate[hd * D + i] = qg[hd * 2 * D + D + i];
                    else for (int i = 0; i < D; ++i) kv_V[l][((size_t)(hd - H) * MAX_CTX + pos) * D + i] = vx[(hd - H) * D + i];
                }
                std::vector<double> att(H * D);
                for (int h0 = 0; h0 < H; ++h0) {
                    const int kvh = h0 / (H / HKV);
                    std::vector<double> sc(pos + 1);
                    double mx = -1e30;
                    for (int p = 0; p <= pos; ++p) {
                        double s = 0;
                        for (int i = 0; i < D; ++i) s += q[h0 * D + i] * kv_K[l][((size_t)kvh * MAX_CTX + p) * D + i];
                        sc[p] = s / std::sqrt((double)D);
                        mx = std::max(mx, sc[p]);
                    }
                    double sum = 0;
                    for (int p = 0; p <= pos; ++p) { sc[p] = std::exp(sc[p] - mx); sum += sc[p]; }
                    for (int i = 0; i < D; ++i) {
                        double acc = 0;
                        for (int p = 0; p <= pos; ++p) acc += sc[p] * kv_V[l][((size_t)kvh * MAX_CTX + p) * D + i];
                        att[h0 * D + i] = (acc / sum) / (1.0 + std::exp(-gate[h0 * D + i]));
                    }
                }
                y = matvec(deq_tensor(find_ten(b + "attn_output.weight")), att, N, H * D);
            }

            for (int i = 0; i < N; ++i) x[i] += y[i];
            const auto h2 = rmsnorm(x, deq_tensor(find_ten(b + "post_attention_norm.weight")), EPS);

            auto rl = matvec(deq_tensor(find_ten(b + "ffn_gate_inp.weight")), h2, NEXP, N);
            const double rmx = *std::max_element(rl.begin(), rl.end());
            double rsum = 0;
            for (double& v : rl) { v = std::exp(v - rmx); rsum += v; }
            for (double& v : rl) v /= rsum;
            std::vector<int> idx(NEXP);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](int a, int c) {
                return rl[a] != rl[c] ? rl[a] > rl[c] : a < c;
            });
            double wsum = 0;
            for (int k = 0; k < TOPK; ++k) wsum += rl[idx[k]];
            wsum = std::max(wsum, 6.103515625e-5);

            const auto eg_all = deq_tensor(find_ten(b + "ffn_gate_exps.weight"));
            const auto eu_all = deq_tensor(find_ten(b + "ffn_up_exps.weight"));
            const auto ed_all = deq_tensor(find_ten(b + "ffn_down_exps.weight"));
            for (int k = 0; k < TOPK; ++k) {
                const int e = idx[k];
                const double wk = rl[e] / wsum;
                std::vector<double> eh(NFF);
                for (int r = 0; r < NFF; ++r) {
                    double gv = 0, uv = 0;
                    for (int i = 0; i < N; ++i) {
                        gv += eg_all[((size_t)e * NFF + r) * N + i] * h2[i];
                        uv += eu_all[((size_t)e * NFF + r) * N + i] * h2[i];
                    }
                    eh[r] = (gv / (1.0 + std::exp(-gv))) * uv;
                }
                for (int i = 0; i < N; ++i) {
                    double dv = 0;
                    for (int r = 0; r < NFF; ++r) dv += ed_all[((size_t)e * N + i) * NFF + r] * eh[r];
                    x[i] += wk * dv;
                }
            }

            const auto sgi = deq_tensor(find_ten(b + "ffn_gate_inp_shexp.weight"));
            double sg_dot = 0;
            for (int i = 0; i < N; ++i) sg_dot += sgi[i] * h2[i];
            const double sg_s = 1.0 / (1.0 + std::exp(-sg_dot));
            const auto sg_v = matvec(deq_tensor(find_ten(b + "ffn_gate_shexp.weight")), h2, NFF, N);
            const auto su_v = matvec(deq_tensor(find_ten(b + "ffn_up_shexp.weight")), h2, NFF, N);
            std::vector<double> sh_h(NFF);
            for (int r = 0; r < NFF; ++r) sh_h[r] = (sg_v[r] / (1.0 + std::exp(-sg_v[r]))) * su_v[r];
            const auto sh_out = matvec(deq_tensor(find_ten(b + "ffn_down_shexp.weight")), sh_h, N, NFF);
            for (int i = 0; i < N; ++i) x[i] += sg_s * sh_out[i];
        }

        const auto xn = rmsnorm(x, out_norm, EPS);
        const auto ref_logits = matvec(out_w, xn, NVOCAB, N);

        Err el;
        int gpu_arg = 0, ref_arg = 0;
        double max_abs = 0, max_ref = 0;
        for (int i = 0; i < NVOCAB; ++i) {
            el.add(gpu_logits[i], ref_logits[i]);
            max_abs = std::max(max_abs, std::abs((double)gpu_logits[i] - ref_logits[i]));
            max_ref = std::max(max_ref, std::abs(ref_logits[i]));
            if (gpu_logits[i] > gpu_logits[gpu_arg]) gpu_arg = i;
            if (ref_logits[i] > ref_logits[ref_arg]) ref_arg = i;
        }
        const bool ok = el.l2() < 1e-4 && gpu_arg == ref_arg;
        fails += !ok;
        std::printf("q36_engine_test pos=%d tok=%d: logits rel L2 %.2e (max_abs %.2e / max_ref %.2e) argmax gpu=%d ref=%d  %s\n",
                    pos, tok, el.l2(), max_abs, max_ref, gpu_arg, ref_arg, ok ? "PASS" : "FAIL");
    }

    // 3. Reset check: pos resets to 0 and produces bitwise-identical logits for token 7
    eng.reset();
    const float* reset_logits = nullptr;
    if (!eng.forward(tokens[0], reset_logits, err)) {
        std::fprintf(stderr, "forward after reset failed: %s\n", err.c_str());
        return 1;
    }
    const bool reset_ok = std::memcmp(first_tok_logits.data(), reset_logits, 4 * NVOCAB) == 0 && eng.pos() == 1;
    fails += !reset_ok;
    std::printf("q36_engine_test reset: bitwise identical=%s pos=%d  %s\n",
                std::memcmp(first_tok_logits.data(), reset_logits, 4 * NVOCAB) == 0 ? "yes" : "no",
                eng.pos(), reset_ok ? "PASS" : "FAIL");

    std::remove("q36_bad_arch.gguf");
    std::remove(synth_path.c_str());
    return fails ? 1 : 0;
}
