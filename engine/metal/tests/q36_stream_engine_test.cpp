// metal/tests/q36_stream_engine_test.cpp - TDD Cycle 6: Q36MetalEngine Resident vs SSD Streaming exact parity.
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
    constexpr int HK = 16, HV = 16, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64, MAX_CTX = 32;

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

    gw.add_tensor("token_embd.weight", {N, NVOCAB}, ref::Q8_0, 1, 0.15f);
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
        gw.add_tensor(b + "ffn_gate_inp.weight", {N, NEXP}, ref::F32, seed++, 0.1f);
        gw.add_tensor(b + "ffn_gate_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_exps.weight", {NFF, N, NEXP}, l == 3 ? ref::Q6_K : ref::Q4_K, seed++, 0.03f);
        gw.add_tensor(b + "ffn_gate_inp_shexp.weight", {N}, ref::F32, seed++, 0.05f);
        gw.add_tensor(b + "ffn_gate_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_up_shexp.weight", {N, NFF}, ref::Q8_0, seed++, 0.03f);
        gw.add_tensor(b + "ffn_down_shexp.weight", {NFF, N}, ref::Q8_0, seed++, 0.03f);
    }
    gw.add_ones_f32("blk.4.nextn.enorm.weight", N, 1.0f);

    const std::string path = "q36_stream_eng_synth.gguf";
    if (!gw.write(path)) return 2;

    const int tokens[] = {7, 42, 19};
    std::vector<std::vector<float>> res_logits(3);
    std::string err;

    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = false;
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "resident open: %s\n", err.c_str()); return 1; }
        for (int p = 0; p < 3; ++p) {
            const float* lg = nullptr;
            if (!eng.forward(tokens[p], lg, err)) { std::fprintf(stderr, "resident fwd: %s\n", err.c_str()); return 1; }
            res_logits[p].assign(lg, lg + NVOCAB);
        }
    }

    int fails = 0;
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.stream = true;
        opt.arena_gb = 0.016;  // 16 MiB tight arena
        if (!eng.open(path, opt, err)) { std::fprintf(stderr, "stream open: %s\n", err.c_str()); return 1; }

        const auto st0 = eng.stream_stats();
        if (!st0.streamed || st0.arena_bytes == 0 || st0.arena_bytes > (20ull << 20)) {
            std::fprintf(stderr, "FAIL: unexpected stream stats streamed=%d arena=%llu\n",
                         st0.streamed ? 1 : 0, (unsigned long long)st0.arena_bytes);
            return 1;
        }

        std::vector<float> first_stream;
        for (int p = 0; p < 3; ++p) {
            const float* lg = nullptr;
            if (!eng.forward(tokens[p], lg, err)) { std::fprintf(stderr, "stream fwd: %s\n", err.c_str()); return 1; }
            if (p == 0) first_stream.assign(lg, lg + NVOCAB);
            Err e;
            for (int i = 0; i < NVOCAB; ++i) e.add(lg[i], res_logits[p][i]);
            const int ag_s = argmax(lg, NVOCAB), ag_r = argmax(res_logits[p].data(), NVOCAB);
            const bool ok = e.l2() < 1e-6 && ag_s == ag_r;
            fails += !ok;
            std::printf("q36_stream_engine_test pos=%d: rel L2 %.2e argmax stream=%d res=%d  %s\n",
                        p, e.l2(), ag_s, ag_r, ok ? "PASS" : "FAIL");
        }

        const auto st1 = eng.stream_stats();
        if (st1.reads == 0 || st1.pre_issued == 0) {
            std::fprintf(stderr, "FAIL: expected SSD reads > 0 and pre_issued > 0 (reads=%llu pre_issued=%llu)\n",
                         (unsigned long long)st1.reads, (unsigned long long)st1.pre_issued);
            return 1;
        }
        std::printf("stream stats: dense=%llu arena=%llu reads=%llu hits=%llu misses=%llu pre_hits=%llu pre_issued=%llu\n",
                    (unsigned long long)st1.dense_bytes, (unsigned long long)st1.arena_bytes,
                    (unsigned long long)st1.reads, (unsigned long long)st1.hits, (unsigned long long)st1.misses,
                    (unsigned long long)st1.pre_hits, (unsigned long long)st1.pre_issued);

        eng.reset();
        const float* lg_rst = nullptr;
        if (!eng.forward(tokens[0], lg_rst, err)) return 1;
        const bool rst_ok = std::memcmp(first_stream.data(), lg_rst, 4 * NVOCAB) == 0;
        fails += !rst_ok;
        std::printf("q36_stream_engine_test reset: bitwise=%s  %s\n", rst_ok ? "yes" : "no", rst_ok ? "PASS" : "FAIL");

        const std::string prof_path = "q36_test.strp";
        if (!eng.save_expert_profile(prof_path, err)) {
            std::fprintf(stderr, "FAIL: save_expert_profile: %s\n", err.c_str());
            return 1;
        }

        strata::metal_q36::Q36MetalEngine eng_prof;
        strata::metal_q36::Options popt;
        popt.ctx = MAX_CTX;
        popt.silent = true;
        popt.stream = true;
        popt.arena_gb = 0.016;
        popt.profile_path = prof_path;
        if (!eng_prof.open(path, popt, err)) {
            std::fprintf(stderr, "FAIL: profile open: %s\n", err.c_str());
            std::remove(prof_path.c_str());
            return 1;
        }
        for (int p = 0; p < 3; ++p) {
            const float* lg = nullptr;
            if (!eng_prof.forward(tokens[p], lg, err)) return 1;
            Err e;
            for (int i = 0; i < NVOCAB; ++i) e.add(lg[i], res_logits[p][i]);
            const int ag_s = argmax(lg, NVOCAB), ag_r = argmax(res_logits[p].data(), NVOCAB);
            const bool ok = e.l2() < 1e-6 && ag_s == ag_r;
            fails += !ok;
            std::printf("q36_stream_engine_test profile pos=%d: rel L2 %.2e argmax=%d/%d  %s\n",
                        p, e.l2(), ag_s, ag_r, ok ? "PASS" : "FAIL");
        }
        std::remove(prof_path.c_str());
    }

    std::remove(path.c_str());
    return fails ? 1 : 0;
}
