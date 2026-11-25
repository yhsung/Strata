// metal/tests/q36_serve_batch_vision_test.cpp - TDD tests for --serve checkpoints, --batch, and --vision.
#include "q36/q36_metal.h"
#include "ref_quant.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
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
};

bool spawn_serve(const std::vector<std::string>& args, const std::vector<std::string>& extra_env, Proc& p) {
    int in_pipe[2]{-1, -1}, out_pipe[2]{-1, -1};
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        for (const auto& kv : extra_env) {
            size_t eq = kv.find('=');
            if (eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
        }
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

int close_proc(Proc& p) {
    close(p.in_fd);
    fclose(p.out_f);
    int st = 0;
    waitpid(p.pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

struct DoneInfo {
    long long produced = -1, prompt_len = -1, reuse = -1, fresh = -1;
    std::string finish;
};

bool parse_done(const std::string& line, DoneInfo& out) {
    std::istringstream iss(line);
    std::string tag;
    long long acc = 0, off = 0;
    double pms = 0, dms = 0, d = 0;
    int z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0;
    return (bool)(iss >> tag >> out.produced >> out.prompt_len >> pms >> dms >> out.finish >> acc >> off >> out.reuse
                      >> z0 >> z1 >> z2 >> z3 >> d >> out.fresh >> z4) && tag == "DONE";
}

int argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
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

    uint64_t seed = 200;
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

    const std::string path = "q36_serve_bv_synth.gguf";
    const std::string sve_path = "q36_serve_bv.sve";
    if (!gw.write(path)) return 2;

    {
        std::ofstream sve(sve_path, std::ios::binary);
        const int32_t hdr[5] = {0x31455653, 4, 2, 2, N};
        sve.write((const char*)hdr, sizeof(hdr));
        std::vector<float> pix(4 * N);
        for (int i = 0; i < 4 * N; ++i) pix[i] = 0.04f * std::cos(0.11f * i);
        sve.write((const char*)pix.data(), pix.size() * sizeof(float));
    }

    const std::string bin = STRATA_Q36_METAL_BIN;
    int fails = 0;

    // 1. Checkpoint partial-prefix reuse
    {
        const std::vector<int32_t> p12 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const std::vector<int32_t> p_div = {1, 2, 3, 4, 5, 6, 7, 8, 9, 77, 78, 79};

        std::vector<int> exp_div;
        {
            strata::metal_q36::Q36MetalEngine re;
            strata::metal_q36::Options ro;
            ro.ctx = 128; ro.silent = true;
            std::string err;
            if (!re.open(path, ro, err)) return 1;
            const float* lg = nullptr;
            if (!re.forward_chunk(0, p_div.data(), (int)p_div.size(), nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;
            int nx = argmax(lg, NVOCAB);
            for (int k = 0; k < 4; ++k) {
                exp_div.push_back(nx);
                if (k + 1 == 4) break;
                if (!re.forward(nx, lg, err)) return 1;
                nx = argmax(lg, NVOCAB);
            }
        }

        Proc p;
        if (!spawn_serve({bin, "--serve", "--native", path, "--max-context", "128"},
                         {"STRATA_Q36_CKPT_STEP=4", "STRATA_Q36_CKPT_N=8"}, p))
            return 1;
        std::string line;
        if (!read_line(p.out_f, line) || line != "READY 128 stop") {
            std::fprintf(stderr, "FAIL Test1 READY: '%s'\n", line.c_str());
            close_proc(p); return 1;
        }
        auto read_gen = [&](std::vector<int>& toks, DoneInfo& di) -> bool {
            toks.clear();
            while (read_line(p.out_f, line)) {
                if (line.rfind("PP ", 0) == 0) continue;
                if (line.rfind("T ", 0) == 0) toks.push_back(std::atoi(line.c_str() + 2));
                else if (line.rfind("DONE ", 0) == 0) return parse_done(line, di);
                else { std::fprintf(stderr, "unexpected: %s\n", line.c_str()); return false; }
            }
            return false;
        };

        std::ostringstream req1;
        req1 << "GEN 4 ";
        for (size_t i = 0; i < p12.size(); ++i) req1 << (i ? "," : "") << p12[i];
        req1 << "\n";
        write_all(p.in_fd, req1.str());
        std::vector<int> t1;
        DoneInfo d1;
        if (!read_gen(t1, d1) || d1.reuse != 0 || d1.fresh != 12) {
            std::fprintf(stderr, "FAIL Test1 req1 reuse=%lld fresh=%lld\n", d1.reuse, d1.fresh);
            close_proc(p); return 1;
        }

        std::ostringstream req2;
        req2 << "GEN 4 ";
        for (size_t i = 0; i < p_div.size(); ++i) req2 << (i ? "," : "") << p_div[i];
        req2 << "\n";
        write_all(p.in_fd, req2.str());
        std::vector<int> t2;
        DoneInfo d2;
        const bool ok = read_gen(t2, d2) && d2.reuse == 8 && d2.fresh == 4 && t2 == exp_div && close_proc(p) == 0;
        fails += !ok;
        std::printf("Test1 serve ckpt reuse: reuse=%lld fresh=%lld tok_match=%d  %s\n",
                    d2.reuse, d2.fresh, (t2 == exp_div) ? 1 : 0, ok ? "PASS" : "FAIL");
    }

    // 2. --batch 3 parallel BGEN / BT / BSTOP / BDONE
    {
        const std::vector<int32_t> p0 = {5, 19, 31, 47};
        const std::vector<int32_t> p1 = {101, 102, 103, 104, 105};
        std::vector<int> exp0;
        {
            strata::metal_q36::Q36MetalEngine re;
            strata::metal_q36::Options ro;
            ro.ctx = 128; ro.silent = true;
            std::string err;
            if (!re.open(path, ro, err)) return 1;
            const float* lg = nullptr;
            if (!re.forward_chunk(0, p0.data(), (int)p0.size(), nullptr, nullptr, 0, nullptr, true, lg, err)) return 1;
            int nx = argmax(lg, NVOCAB);
            for (int k = 0; k < 6; ++k) {
                exp0.push_back(nx);
                if (k + 1 == 6) break;
                if (!re.forward(nx, lg, err)) return 1;
                nx = argmax(lg, NVOCAB);
            }
        }

        Proc p;
        if (!spawn_serve({bin, "--serve", "--native", path, "--max-context", "128", "--batch", "3"}, {}, p))
            return 1;
        std::string info_line, ready_line;
        if (!read_line(p.out_f, info_line) || info_line != "INFO batch_slots=3 slot_cache=1" ||
            !read_line(p.out_f, ready_line) || ready_line != "READY 128 stop") {
            std::fprintf(stderr, "FAIL Test2 banner: '%s' / '%s'\n", info_line.c_str(), ready_line.c_str());
            close_proc(p); return 1;
        }

        std::map<int, std::vector<int>> got;
        auto read_badm = [&](int slot, int& cont) -> bool {
            std::string line;
            while (read_line(p.out_f, line)) {
                if (line.rfind("PP ", 0) == 0 || line.rfind("DONE ", 0) == 0) continue;
                if (line.rfind("T ", 0) == 0) {
                    got[slot].push_back(std::atoi(line.c_str() + 2));
                } else if (line.rfind("BADM ", 0) == 0) {
                    int s = -1;
                    if (std::sscanf(line.c_str(), "BADM %d %d", &s, &cont) == 2 && s == slot) return true;
                } else if (line.rfind("BT ", 0) == 0) {
                    int s = -1, y = -1;
                    if (std::sscanf(line.c_str(), "BT %d %d", &s, &y) == 2) got[s].push_back(y);
                } else {
                    std::fprintf(stderr, "FAIL Test2 unexpected during adm: '%s'\n", line.c_str());
                    return false;
                }
            }
            return false;
        };

        write_all(p.in_fd, "BGEN 0 6 5,19,31,47\n");
        int c0 = 0, c1 = 0;
        if (!read_badm(0, c0) || c0 != 1) { close_proc(p); return 1; }
        write_all(p.in_fd, "BGEN 1 6 101,102,103,104,105\n");
        if (!read_badm(1, c1) || c1 != 1) { close_proc(p); return 1; }

        bool sent_bstop = false;
        std::map<int, std::string> finish;
        std::string line;
        while ((finish.count(0) == 0 || finish.count(1) == 0) && read_line(p.out_f, line)) {
            if (line.rfind("BT ", 0) == 0) {
                int s = -1, y = -1;
                std::sscanf(line.c_str(), "BT %d %d", &s, &y);
                got[s].push_back(y);
                if (s == 1 && !sent_bstop) {
                    write_all(p.in_fd, "BSTOP 1\n");
                    sent_bstop = true;
                }
            } else if (line.rfind("BDONE ", 0) == 0) {
                int s = -1; long long prod = 0; char fin[32]{}; double ms = 0;
                if (std::sscanf(line.c_str(), "BDONE %d %lld %31s %lf", &s, &prod, fin, &ms) == 4) {
                    finish[s] = fin;
                }
            }
        }
        const int rc = close_proc(p);
        const bool ok = (rc == 0) && (finish[0] == "length") && (finish[1] == "cancel") && (got[0] == exp0);
        fails += !ok;
        std::printf("Test2 batch BGEN: slot0_match=%d (len=%zu) slot1_fin=%s  %s\n",
                    (got[0] == exp0) ? 1 : 0, got[0].size(), finish[1].c_str(), ok ? "PASS" : "FAIL");
    }

    // 3. --vision GENI / BGENI
    {
        Proc p_novis;
        if (!spawn_serve({bin, "--serve", "--native", path, "--max-context", "128"}, {}, p_novis)) return 1;
        std::string line;
        if (!read_line(p_novis.out_f, line) || line != "READY 128 stop") { close_proc(p_novis); return 1; }
        write_all(p_novis.in_fd, "GENI 2 " + sve_path + " 10,248056,248056,248056,248056,25\n");
        const bool novis_err = read_line(p_novis.out_f, line) && line.rfind("ERR ", 0) == 0 && close_proc(p_novis) == 0;

        Proc p;
        if (!spawn_serve({bin, "--serve", "--native", path, "--max-context", "128", "--batch", "2", "--vision"}, {}, p))
            return 1;
        std::string info_line, ready_line;
        if (!read_line(p.out_f, info_line) || info_line != "INFO batch_slots=2 slot_cache=1" ||
            !read_line(p.out_f, ready_line) || ready_line != "READY 128 stop") {
            close_proc(p); return 1;
        }
        write_all(p.in_fd, "GENI 2 img_keys=deadbeef " + sve_path + " 10,248056,248056,248056,248056,25\n");
        std::vector<int> tv1;
        DoneInfo dv1;
        bool ok1 = false;
        while (read_line(p.out_f, line)) {
            if (line.rfind("PP ", 0) == 0) continue;
            if (line.rfind("T ", 0) == 0) tv1.push_back(std::atoi(line.c_str() + 2));
            else if (line.rfind("DONE ", 0) == 0) { ok1 = parse_done(line, dv1); break; }
            else break;
        }

        // BGENI on slot 0 extending the exact same image + prefix should reuse 6 tokens!
        write_all(p.in_fd, "BGENI 0 2 img_keys=deadbeef " + sve_path + " 10,248056,248056,248056,248056,25,30\n");
        std::vector<int> tv2;
        DoneInfo dv2;
        bool got_done2 = false, got_badm = false, got_bdone = false;
        while (read_line(p.out_f, line)) {
            if (line.rfind("PP ", 0) == 0) continue;
            if (line.rfind("T ", 0) == 0) tv2.push_back(std::atoi(line.c_str() + 2));
            else if (line.rfind("DONE ", 0) == 0) got_done2 = parse_done(line, dv2);
            else if (line.rfind("BADM ", 0) == 0) got_badm = true;
            else if (line.rfind("BT ", 0) == 0) tv2.push_back(std::atoi(line.c_str() + 5));
            else if (line.rfind("BDONE ", 0) == 0) { got_bdone = true; break; }
            else break;
        }
        const int rc = close_proc(p);
        const bool ok = novis_err && ok1 && dv1.reuse == 0 && dv1.fresh == 6 && tv1.size() == 2 &&
                        got_done2 && got_badm && got_bdone && dv2.reuse == 6 && dv2.fresh == 1 && tv2.size() == 2 && rc == 0;
        fails += !ok;
        std::printf("Test3 vision GENI/BGENI: novis_err=%d geni=(r=%lld,f=%lld) bgeni=(r=%lld,f=%lld)  %s\n",
                    novis_err ? 1 : 0, dv1.reuse, dv1.fresh, dv2.reuse, dv2.fresh, ok ? "PASS" : "FAIL");
    }

    std::remove(path.c_str());
    std::remove(sve_path.c_str());
    return fails ? 1 : 0;
}
