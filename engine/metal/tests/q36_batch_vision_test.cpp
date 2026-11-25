// metal/tests/q36_batch_vision_test.cpp - TDD tests for Q36 batched prefill, GDN checkpoints, --batch slots, and --vision M-RoPE.
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

int argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
}

double rel_l2(const float* a, const float* b, int n) {
    Err e;
    for (int i = 0; i < n; ++i) e.add(a[i], b[i]);
    return e.l2();
}

}  // namespace

int main() {
    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256, TOPK = 8, NVOCAB = 256;
    constexpr int HK = 16, HV = 16, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64, MAX_CTX = 64;

    GgufWriter gw;
    gw.add_str("general.architecture", "qwen35moe");
    gw.add_u32("qwen35moe.block_count", NL);
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

    uint64_t seed = 100;
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

    const std::string path = "q36_batch_vision_synth.gguf";
    if (!gw.write(path)) return 2;

    int fails = 0;
    std::string err;

    // 1. Batched prefill parity (Resident & Stream) against sequential forward()
    const std::vector<int32_t> prompt = {5, 17, 42, 88, 123, 7, 19, 64, 91, 12, 200};
    std::vector<float> seq_logits(NVOCAB);
    {
        strata::metal_q36::Q36MetalEngine ref_eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        if (!ref_eng.open(path, opt, err)) { std::fprintf(stderr, "ref open: %s\n", err.c_str()); return 1; }
        const float* lg = nullptr;
        for (int32_t tok : prompt) {
            if (!ref_eng.forward(tok, lg, err)) { std::fprintf(stderr, "ref fwd: %s\n", err.c_str()); return 1; }
        }
        seq_logits.assign(lg, lg + NVOCAB);

        for (bool do_stream : {false, true}) {
            strata::metal_q36::Q36MetalEngine bat;
            strata::metal_q36::Options bopt;
            bopt.ctx = MAX_CTX;
            bopt.silent = true;
            bopt.stream = do_stream;
            if (do_stream) bopt.arena_gb = 0.016;
            if (!bat.open(path, bopt, err)) { std::fprintf(stderr, "bat open(stream=%d): %s\n", do_stream ? 1 : 0, err.c_str()); return 1; }
            const float* blg = nullptr;
            if (!bat.forward_chunk(0, prompt.data(), (int)prompt.size(), nullptr, nullptr, 0, nullptr, true, blg, err)) {
                std::fprintf(stderr, "forward_chunk(stream=%d): %s\n", do_stream ? 1 : 0, err.c_str());
                return 1;
            }
            const double d = rel_l2(blg, seq_logits.data(), NVOCAB);
            const int ag_b = argmax(blg, NVOCAB), ag_r = argmax(seq_logits.data(), NVOCAB);
            const bool ok = d < 1e-5 && ag_b == ag_r && bat.pos(0) == (int)prompt.size();
            fails += !ok;
            std::printf("Test1 prefill(stream=%d): rel L2 %.2e argmax=%d/%d pos=%d  %s\n",
                        do_stream ? 1 : 0, d, ag_b, ag_r, bat.pos(0), ok ? "PASS" : "FAIL");
        }
    }

    // 2. GDN checkpoints
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX;
        opt.silent = true;
        opt.ckpt_count = 8;
        if (!eng.open(path, opt, err)) return 1;

        std::vector<int32_t> p12(12);
        for (int i = 0; i < 12; ++i) p12[i] = (i * 17 + 3) % NVOCAB;
        const float* lg = nullptr;
        for (int c = 0; c < 3; ++c) {
            if (!eng.forward_chunk(0, p12.data() + c * 4, 4, nullptr, nullptr, 0, nullptr, c == 2, lg, err)) return 1;
            eng.save_checkpoint(0);
        }
        const std::vector<int32_t> tail4 = {31, 32, 33, 34};
        if (!eng.forward_chunk(0, tail4.data(), 4, nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;

        const int cpos = eng.checkpoint_at_or_below(0, 9);
        const bool restored = (cpos == 8) && eng.restore_checkpoint(0, 8);
        const std::vector<int32_t> branch = {99, 100, 101};
        if (!eng.forward_chunk(0, branch.data(), 3, nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;
        std::vector<float> branched(lg, lg + NVOCAB);

        eng.reset();
        std::vector<int32_t> ref_seq(p12.begin(), p12.begin() + 8);
        ref_seq.insert(ref_seq.end(), branch.begin(), branch.end());
        if (!eng.forward_chunk(0, ref_seq.data(), (int)ref_seq.size(), nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;

        const double d = rel_l2(branched.data(), lg, NVOCAB);
        const bool ok = restored && d < 1e-6 && argmax(branched.data(), NVOCAB) == argmax(lg, NVOCAB);
        fails += !ok;
        std::printf("Test2 checkpoints: cpos=%d restored=%d rel L2 %.2e  %s\n",
                    cpos, restored ? 1 : 0, d, ok ? "PASS" : "FAIL");
    }

    // 3. Multi-slot parallel batch + copy_session
    {
        constexpr int NS = 3;
        const std::vector<int32_t> p[NS] = {
            {7, 13, 29},
            {11, 42, 73, 105},
            {99, 3, 55, 18, 210}
        };
        const std::vector<int32_t> step_toks[NS] = {
            {8, 9, 10, 11},
            {21, 22, 23, 24},
            {60, 61, 62, 63}
        };

        std::vector<std::vector<float>> ref_lg[NS];
        for (int s = 0; s < NS; ++s) {
            strata::metal_q36::Q36MetalEngine re;
            strata::metal_q36::Options ro;
            ro.ctx = MAX_CTX; ro.silent = true;
            if (!re.open(path, ro, err)) return 1;
            const float* lg = nullptr;
            if (!re.forward_chunk(0, p[s].data(), (int)p[s].size(), nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;
            for (int k = 0; k < 4; ++k) {
                if (!re.forward(step_toks[s][k], lg, err)) return 1;
                ref_lg[s].emplace_back(lg, lg + NVOCAB);
            }
        }

        strata::metal_q36::Q36MetalEngine be;
        strata::metal_q36::Options bo;
        bo.ctx = MAX_CTX; bo.silent = true; bo.n_slots = NS;
        if (!be.open(path, bo, err)) { std::fprintf(stderr, "batch open: %s\n", err.c_str()); return 1; }
        for (int s = 0; s < NS; ++s) {
            const float* lg = nullptr;
            if (!be.forward_chunk(s, p[s].data(), (int)p[s].size(), nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;
        }
        bool ok = true;
        double max_d = 0.0;
        for (int k = 0; k < 4; ++k) {
            strata::metal_q36::Q36MetalEngine::SlotStep steps[NS] = {
                {0, step_toks[0][k]}, {1, step_toks[1][k]}, {2, step_toks[2][k]}
            };
            std::vector<const float*> outs;
            if (!be.step_slots(steps, NS, outs, err)) { std::fprintf(stderr, "step_slots: %s\n", err.c_str()); return 1; }
            for (int s = 0; s < NS; ++s) {
                const double d = rel_l2(outs[s], ref_lg[s][k].data(), NVOCAB);
                max_d = std::max(max_d, d);
                if (d >= 1e-5 || argmax(outs[s], NVOCAB) != argmax(ref_lg[s][k].data(), NVOCAB)) ok = false;
            }
        }
        // copy_session(0, 2) parity check
        if (!be.copy_session(0, 2)) ok = false;
        const float *lg0 = nullptr, *lg2 = nullptr;
        if (!be.forward_chunk(0, &step_toks[0][0], 1, nullptr, nullptr, 0, nullptr, true, lg0, err)) return 1;
        std::vector<float> v0(lg0, lg0 + NVOCAB);
        if (!be.forward_chunk(2, &step_toks[0][0], 1, nullptr, nullptr, 0, nullptr, true, lg2, err)) return 1;
        const double copy_d = rel_l2(lg2, v0.data(), NVOCAB);
        if (copy_d >= 1e-6) ok = false;
        fails += !ok;
        std::printf("Test3 batch(slots=%d): max rel L2 %.2e copy_session rel L2 %.2e  %s\n",
                    be.n_slots(), max_d, copy_d, ok ? "PASS" : "FAIL");
    }

    // 4. Vision embeddings + M-RoPE 3D parity (batched chunk vs single-token forward_chunk)
    {
        const std::vector<int32_t> toks = {10, 248056, 248056, 248056, 248056, 25, 30};
        const std::vector<int32_t> cell_row = {-1, 0, 1, 2, 3, -1, -1};
        // 2x2 image at p=1 -> mrope coords: (0,0,0), (1,1,1), (1,1,2), (1,2,1), (1,2,2), (3,3,3), (4,4,4)
        const std::vector<int32_t> mrope = {
            0, 0, 0,
            1, 1, 1,
            1, 1, 2,
            1, 2, 1,
            1, 2, 2,
            3, 3, 3,
            4, 4, 4
        };
        std::vector<float> img_rows(4 * N);
        for (int i = 0; i < 4 * N; ++i) img_rows[i] = 0.05f * std::sin(0.13f * i + 0.7f);

        strata::metal_q36::Q36MetalEngine e1, e2, etext;
        strata::metal_q36::Options opt;
        opt.ctx = MAX_CTX; opt.silent = true;
        if (!e1.open(path, opt, err) || !e2.open(path, opt, err) || !etext.open(path, opt, err)) return 1;

        const float *lg1 = nullptr, *lg2 = nullptr, *lgt = nullptr;
        if (!e1.forward_chunk(0, toks.data(), (int)toks.size(), mrope.data(), img_rows.data(), 4, cell_row.data(), true, lg1, err))
            return 1;
        std::vector<float> v_bat(lg1, lg1 + NVOCAB);

        for (size_t i = 0; i < toks.size(); ++i) {
            const float* rptr = cell_row[i] >= 0 ? img_rows.data() : nullptr;
            const int cr = cell_row[i];
            if (!e2.forward_chunk(0, &toks[i], 1, &mrope[i * 3], rptr, cell_row[i] >= 0 ? 4 : 0, &cr, i + 1 == toks.size(), lg2, err))
                return 1;
        }
        const std::vector<int32_t> txt_toks = {10, 1, 2, 3, 4, 25, 30};
        if (!etext.forward_chunk(0, txt_toks.data(), (int)txt_toks.size(), nullptr, nullptr, 0, nullptr, true, lgt, err))
            return 1;

        const double d_seq = rel_l2(v_bat.data(), lg2, NVOCAB);
        const double d_txt = rel_l2(v_bat.data(), lgt, NVOCAB);
        const bool ok = std::isfinite(v_bat[0]) && d_seq < 1e-5 && d_txt > 1e-3 &&
                        argmax(v_bat.data(), NVOCAB) == argmax(lg2, NVOCAB);
        fails += !ok;
        std::printf("Test4 vision M-RoPE 3D: vs 1-tok rel L2 %.2e vs text-only %.2e  %s\n",
                    d_seq, d_txt, ok ? "PASS" : "FAIL");
    }

    std::remove(path.c_str());
    return fails ? 1 : 0;
}
