// metal/tests/q36_serve_test.cpp - TDD test for strata-q36-metal --serve protocol.
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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

struct Proc {
    pid_t pid = -1;
    int in_fd = -1;
    FILE* out_f = nullptr;
    std::vector<std::string> argv_storage;
};

bool spawn_serve(const std::vector<std::string>& args, Proc& p) {
    int in_pipe[2]{-1, -1}, out_pipe[2]{-1, -1};
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        std::vector<char*> argv;
        for (const auto& s : args) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        execv(args[0].c_str(), argv.data());
        _exit(127);
    }
    close(in_pipe[0]);
    close(out_pipe[1]);
    p.pid = pid;
    p.in_fd = in_pipe[1];
    p.out_f = fdopen(out_pipe[0], "r");
    return p.out_f != nullptr;
}

bool read_line(FILE* f, std::string& line) {
    line.clear();
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        line += buf;
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
    }
    return !line.empty();
}

bool write_all(int fd, const std::string& s) {
    const char* ptr = s.data();
    size_t rem = s.size();
    while (rem > 0) {
        ssize_t n = write(fd, ptr, rem);
        if (n <= 0) return false;
        ptr += n;
        rem -= (size_t)n;
    }
    return true;
}

int argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
}

bool run_serve_check(const std::string& bin, const std::string& gguf_path,
                     const std::vector<int>& expected_greedy, int nvocab, bool stream) {
    std::vector<std::string> args = {
        bin, "--serve", "--native", gguf_path, "--max-context", "128", "--unknown-flag", "foo"
    };
    if (stream) {
        args.push_back("--stream");
        args.push_back("--arena-gb");
        args.push_back("0.016");
    }

    Proc p;
    if (!spawn_serve(args, p)) {
        std::fprintf(stderr, "FAIL: spawn_serve failed\n");
        return false;
    }

    std::string line;
    if (!read_line(p.out_f, line) || line != "READY 128 stop") {
        std::fprintf(stderr, "FAIL: expected 'READY 128 stop', got '%s'\n", line.c_str());
        close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
        return false;
    }

    auto read_gen = [&](std::vector<int>& toks, std::string& done_line) -> bool {
        toks.clear();
        while (read_line(p.out_f, line)) {
            if (line.rfind("PP ", 0) == 0) continue;
            if (line.rfind("T ", 0) == 0) {
                toks.push_back(std::atoi(line.c_str() + 2));
            } else if (line.rfind("DONE ", 0) == 0) {
                done_line = line;
                return true;
            } else {
                std::fprintf(stderr, "FAIL: unexpected protocol line: '%s'\n", line.c_str());
                return false;
            }
        }
        std::fprintf(stderr, "FAIL: EOF before DONE\n");
        return false;
    };

    if (!write_all(p.in_fd, "GEN 4 7,42,19\n")) return false;
    std::vector<int> t1;
    std::string done1;
    if (!read_gen(t1, done1) || t1 != expected_greedy) {
        std::fprintf(stderr, "FAIL: greedy mismatch (stream=%d) got %zu tokens, done='%s'\n",
                     stream ? 1 : 0, t1.size(), done1.c_str());
        close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
        return false;
    }

    // Verify DONE fields: produced prompt_len prompt_ms decode_ms finish acc off reuse 0 0 0 0 0.0 fresh 0
    {
        std::istringstream iss(done1);
        std::string tag, finish;
        long long produced = -1, prompt_len = -1, acc = -1, off = -1, reuse = -1, fresh = -1;
        double pms = 0, dms = 0, d = 0;
        int z0 = -1, z1 = -1, z2 = -1, z3 = -1, z4 = -1;
        iss >> tag >> produced >> prompt_len >> pms >> dms >> finish >> acc >> off >> reuse >> z0 >> z1 >> z2 >> z3 >> d >> fresh >> z4;
        if (produced != 4 || prompt_len != 3 || reuse != 0 || fresh != 3) {
            std::fprintf(stderr, "FAIL: unexpected DONE1 '%s'\n", done1.c_str());
            close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
            return false;
        }
    }

    // Prefix-extending sampled request: live holds [7, 42, 19, t1[0], t1[1], t1[2]] (len 6)
    std::ostringstream req2;
    req2 << "GEN 3 temperature=0.8 top_k=20 top_p=0.95 penalty_last_n=64 penalty_present=1.5 seed=42 "
         << "7,42,19," << t1[0] << "," << t1[1] << "," << t1[2] << ",11\n";
    if (!write_all(p.in_fd, req2.str())) return false;
    std::vector<int> t2;
    std::string done2;
    if (!read_gen(t2, done2) || t2.size() != 3) {
        std::fprintf(stderr, "FAIL: sampled request failed (stream=%d) got %zu tokens\n",
                     stream ? 1 : 0, t2.size());
        close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
        return false;
    }
    for (int id : t2) {
        if (id < 0 || id >= nvocab) {
            std::fprintf(stderr, "FAIL: invalid sampled token id %d\n", id);
            close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
            return false;
        }
    }
    {
        std::istringstream iss(done2);
        std::string tag, finish;
        long long produced = -1, prompt_len = -1, acc = -1, off = -1, reuse = -1, fresh = -1;
        double pms = 0, dms = 0, d = 0;
        int z0 = -1, z1 = -1, z2 = -1, z3 = -1, z4 = -1;
        iss >> tag >> produced >> prompt_len >> pms >> dms >> finish >> acc >> off >> reuse >> z0 >> z1 >> z2 >> z3 >> d >> fresh >> z4;
        if (produced != 3 || prompt_len != 7 || reuse != 6 || fresh != 1) {
            std::fprintf(stderr, "FAIL: unexpected prefix reuse in DONE2 '%s'\n", done2.c_str());
            close(p.in_fd); fclose(p.out_f); int st = 0; waitpid(p.pid, &st, 0);
            return false;
        }
    }

    close(p.in_fd);
    fclose(p.out_f);
    int st = 0;
    waitpid(p.pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        std::fprintf(stderr, "FAIL: serve process exited with %d\n", st);
        return false;
    }
    std::printf("q36_serve_test stream=%d: greedy=%d,%d,%d,%d sampled=%d,%d,%d  PASS\n",
                stream ? 1 : 0, t1[0], t1[1], t1[2], t1[3], t2[0], t2[1], t2[2]);
    return true;
}

}  // namespace

int main() {
    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256, TOPK = 8, NVOCAB = 256;
    constexpr int HK = 16, HV = 16, S = 128, CONV = (2 * HK + HV) * S, ZV = HV * S;
    constexpr int H = 4, HKV = 2, D = 256, NROT = 64;

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

    const std::string path = "q36_serve_synth.gguf";
    if (!gw.write(path)) return 2;

    std::vector<int> expected_greedy;
    {
        strata::metal_q36::Q36MetalEngine eng;
        strata::metal_q36::Options opt;
        opt.ctx = 128;
        opt.silent = true;
        std::string err;
        if (!eng.open(path, opt, err)) return 1;
        const float* lg = nullptr;
        for (int tok : {7, 42, 19}) if (!eng.forward(tok, lg, err)) return 1;
        int nx = argmax(lg, NVOCAB);
        for (int step = 0; step < 4; ++step) {
            expected_greedy.push_back(nx);
            if (step + 1 == 4) break;
            if (!eng.forward(nx, lg, err)) return 1;
            nx = argmax(lg, NVOCAB);
        }
    }

    const std::string bin = STRATA_Q36_METAL_BIN;
    const bool ok0 = run_serve_check(bin, path, expected_greedy, NVOCAB, false);
    const bool ok1 = run_serve_check(bin, path, expected_greedy, NVOCAB, true);
    std::remove(path.c_str());
    return (ok0 && ok1) ? 0 : 1;
}
