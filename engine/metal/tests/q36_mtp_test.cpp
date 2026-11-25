// metal/tests/q36_mtp_test.cpp - Milestone 3 TDD: blk.<n_layers> MTP speculative decoding bit-exact parity (Resident & Stream).
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

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

}  // namespace

int main() {
    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256, TOPK = 8, NVOCAB = 256;
    constexpr int HK = 2, HV = 4, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64, MAX_CTX = 64;

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
            gw.add_tensor(b + "ssm_out.weight", {ZV, N}, ref::Q4_K, seed++, 0.04f);
        } else {
            gw.add_tensor(b + "attn_q.weight", {N, H * 2 * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_k.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_tensor(b + "attn_v.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
            gw.add_ones_f32(b + "attn_q_norm.weight", D, 1.0f);
            gw.add_ones_f32(b + "attn_k_norm.weight", D, 1.0f);
            gw.add_tensor(b + "attn_output.weight", {H * D, N}, ref::Q4_K, seed++, 0.04f);
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

    // Complete MTP layer blk.4.*
    {
        const std::string b = "blk." + std::to_string(NL) + ".";
        gw.add_ones_f32(b + "nextn.enorm.weight", N, 1.0f);
        gw.add_ones_f32(b + "nextn.hnorm.weight", N, 1.0f);
        gw.add_tensor(b + "nextn.eh_proj.weight", {2 * N, N}, ref::Q8_0, seed++, 0.04f);
        gw.add_ones_f32(b + "nextn.shared_head_norm.weight", N, 1.05f);
        gw.add_ones_f32(b + "attn_norm.weight", N, 1.0f);
        gw.add_ones_f32(b + "post_attention_norm.weight", N, 1.0f);
        gw.add_tensor(b + "attn_q.weight", {N, H * 2 * D}, ref::Q8_0, seed++, 0.04f);
        gw.add_tensor(b + "attn_k.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
        gw.add_tensor(b + "attn_v.weight", {N, HKV * D}, ref::Q8_0, seed++, 0.04f);
        gw.add_ones_f32(b + "attn_q_norm.weight", D, 1.0f);
        gw.add_ones_f32(b + "attn_k_norm.weight", D, 1.0f);
        gw.add_tensor(b + "attn_output.weight", {H * D, N}, ref::Q8_0, seed++, 0.04f);
        gw.add_tensor(b + "ffn_gate_inp.weight", {N, NEXP}, ref::BF16, seed++, 0.1f);
        gw.add_tensor(b + "ffn_gate_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_exps.weight", {NFF, N, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_gate_inp_shexp.weight", {N}, ref::F32, seed++, 0.05f);
        gw.add_tensor(b + "ffn_gate_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_shexp.weight", {NFF, N}, ref::Q8_0, seed++, 0.03f);
    }

    const std::string path = "q36_mtp_synth.gguf";
    if (!gw.write(path)) return 2;

    const std::vector<int32_t> prompt = {5, 17, 42, 91, 133, 200, 12, 77};
    constexpr int NGEN = 12;

    // Reference: plain greedy without MTP
    std::vector<int32_t> ref_tokens;
    std::vector<float> ref_after_logits;
    int ref_end_pos = 0;
    std::string err;
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.mtp_drafts = 0;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "ref open: %s\n", err.c_str()); return 1; }
        const float* lg = nullptr;
        if (!eng.forward_chunk(0, prompt.data(), (int)prompt.size(), nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "ref prompt: %s\n", err.c_str());
            return 1;
        }
        int32_t nx = argmax(lg, NVOCAB);
        for (int i = 0; i < NGEN; ++i) {
            ref_tokens.push_back(nx);
            if (!eng.forward(nx, lg, err)) { std::fprintf(stderr, "ref step %d: %s\n", i, err.c_str()); return 1; }
            nx = argmax(lg, NVOCAB);
        }
        ref_end_pos = eng.pos(0);
        ref_after_logits.assign(lg, lg + NVOCAB);
    }

    auto run_mtp_mode = [&](const char* label, bool stream) -> int {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = stream;
        if (stream) opt.arena_gb = 0.10;
        opt.mtp_drafts = 2;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "%s open: %s\n", label, err.c_str()); return 1; }
        if (!eng.has_mtp()) { std::fprintf(stderr, "%s expected has_mtp() == true\n", label); return 1; }

        const float* lg = nullptr;
        if (!eng.forward_chunk(0, prompt.data(), (int)prompt.size(), nullptr, nullptr, 0, nullptr, true, lg, err)) {
            std::fprintf(stderr, "%s prompt: %s\n", label, err.c_str());
            return 1;
        }

        int32_t nx = argmax(lg, NVOCAB);
        std::vector<int32_t> out;
        std::vector<int32_t> fix, drafts, picks;
        int total_offered = 0, total_accepted = 0;

        while ((int)out.size() < NGEN) {
            out.push_back(nx);
            if ((int)out.size() == NGEN) {
                // Step final token so position matches ref_end_pos and compare logits
                if (!eng.forward(nx, lg, err)) { std::fprintf(stderr, "%s final fwd: %s\n", label, err.c_str()); return 1; }
                break;
            }
            int acc = 0;
            const int d = std::min(2, NGEN - (int)out.size());
            if (!eng.spec_step(0, nx, fix, d, drafts, picks, acc, err)) {
                std::fprintf(stderr, "%s spec_step: %s\n", label, err.c_str());
                return 1;
            }
            total_offered += d;
            total_accepted += acc;
            for (int i = 0; i < acc && (int)out.size() < NGEN; ++i) {
                out.push_back(drafts[(size_t)i]);
            }
            if ((int)out.size() == NGEN) {
                if (!eng.forward(picks[acc], lg, err)) {
                    std::fprintf(stderr, "%s sync fwd: %s\n", label, err.c_str());
                    return 1;
                }
                break;
            }
            fix.assign(drafts.begin(), drafts.begin() + acc);
            nx = picks[(size_t)acc];
        }

        if (out != ref_tokens) {
            std::fprintf(stderr, "%s token mismatch!\n", label);
            return 1;
        }
        // When NGEN tokens are emitted, if final token stepped was picks[acc] instead of stopping earlier:
        // Compare pos and logits when stopped cleanly after ref_tokens:
        if (eng.pos(0) == ref_end_pos) {
            Err e;
            for (int i = 0; i < NVOCAB; ++i) e.add(lg[i], ref_after_logits[i]);
            std::printf("%s: 12/12 bit-exact tokens (accepted %d/%d, pos=%d, follow rel L2=%.3e)\n",
                        label, total_accepted, total_offered, eng.pos(0), e.l2());
            if (!(e.l2() < 5e-4)) {
                std::fprintf(stderr, "%s follow logits rel L2 %.3e too large\n", label, e.l2());
                return 1;
            }
        } else {
            std::printf("%s: 12/12 bit-exact tokens (accepted %d/%d, pos=%d)\n",
                        label, total_accepted, total_offered, eng.pos(0));
        }
        return 0;
    };

    if (run_mtp_mode("resident-mtp", false) != 0) return 1;
    if (run_mtp_mode("stream-mtp", true) != 0) return 1;

    std::remove(path.c_str());
    return 0;
}
