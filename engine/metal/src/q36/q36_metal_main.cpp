// metal/src/q36/q36_metal_main.cpp - CLI & --serve runner for Qwen3.6-35B-A3B Metal engine.
#include "q36/q36_metal.h"
#include "mtl_rt.h"
#include "strata/qwen35/sampling.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool is_eos(int32_t t) {
    return t == 248044 || t == 248046;  // <|endoftext|>, <|im_end|>
}

int32_t argmax_logits(const float* lg, int64_t nv) {
    int32_t best = 0;
    float bv = lg[0];
    for (int64_t i = 1; i < nv; ++i) {
        if (lg[i] > bv) {
            bv = lg[i];
            best = (int32_t)i;
        }
    }
    return best;
}

int32_t sample_logits(std::vector<float>& lg, const strata::qwen35::Sampling& s,
                      const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) {
        return (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int)lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return lg[(size_t)a] > lg[(size_t)b]; });
    idx.resize((size_t)kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t)idx[0]];
    double sum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        sum += (p[i] = std::exp((lg[(size_t)idx[i]] - mx) / s.temperature));
    }
    size_t keep = idx.size();
    double cum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) {
            keep = i;
            break;
        }
        cum += p[i] / sum;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0.0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0.0) return (int32_t)idx[i];
    }
    return (int32_t)idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t)x);
        s = e;
    }
    return v;
}

int run_serve(strata::metal_q36::Q36MetalEngine& eng, int ctx) {
    const int64_t nv = eng.n_vocab();
    const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
    std::vector<int32_t> live;
    std::vector<float> last_logits((size_t)nv, 0.0f);

    std::printf("READY %d stop\n", ctx);
    std::fflush(stdout);

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(std::move(l));
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });

    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) break;
            line = std::move(lines.front());
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }

        const char* p = line.c_str() + 4;
        while (*p == ' ') ++p;
        char* end_max = nullptr;
        const long long max_new = std::strtoll(p, &end_max, 10);
        if (end_max == p) {
            std::printf("ERR bad max_new\n");
            std::fflush(stdout);
            continue;
        }
        p = end_max;

        strata::qwen35::Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t)(p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) {
                p = st0;
                break;
            }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }

        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t)ids.size() + max_new > ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds context (%d)\n",
                        ids.size(), max_new, ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad_tok = false;
        for (int32_t t : ids) {
            if (t < 0 || t >= (int32_t)nv) {
                bad_tok = true;
                break;
            }
        }
        if (bad_tok) {
            std::printf("ERR token id outside vocabulary\n");
            std::fflush(stdout);
            continue;
        }

        stop.store(false);
        size_t reuse = 0;
        if (!live.empty() && ids.size() >= live.size() &&
            std::equal(live.begin(), live.end(), ids.begin())) {
            reuse = live.size();
        } else {
            eng.reset();
            live.clear();
        }

        const size_t fresh = ids.size() - reuse;
        const auto t_pf0 = Clock::now();
        bool cancelled = false;
        bool fwd_err = false;
        std::string err;
        const float* logits = nullptr;
        size_t done = 0;
        for (; done < fresh; ++done) {
            if (stop.load()) {
                cancelled = true;
                break;
            }
            const int32_t tok = ids[reuse + done];
            if (!eng.forward(tok, logits, err)) {
                fwd_err = true;
                break;
            }
            live.push_back(tok);
            if (((done + 1) % 32 == 0) || (done + 1 == fresh)) {
                const double pms = ms_since(t_pf0);
                std::printf("PP %zu %zu %.0f %.1f\n", done + 1, fresh, pms,
                            (double)(done + 1) / std::max(1e-9, pms / 1000.0));
                std::fflush(stdout);
            }
        }
        if (fwd_err) {
            eng.reset();
            live.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (fresh > 0 && !cancelled && logits != nullptr) {
            std::memcpy(last_logits.data(), logits, 4 * (size_t)nv);
        }
        const double prompt_ms = ms_since(t_pf0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %zu 0\n",
                        ids.size(), prompt_ms, reuse, done);
            std::fflush(stdout);
            continue;
        }

        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, pen_prompt);
        std::vector<float> scratch_lg;
        const auto t_dec0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";

        for (;;) {
            int32_t nx = 0;
            if (!smp.needs_logits()) {
                nx = argmax_logits(last_logits.data(), nv);
            } else {
                scratch_lg = last_logits;
                nx = sample_logits(scratch_lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);

            if (is_eos(nx)) {
                finish = "stop";
                break;
            }
            if (produced >= max_new) {
                finish = "length";
                break;
            }
            if (stop.load()) {
                finish = "cancel";
                break;
            }
            if (eng.pos() + 1 > ctx) {
                finish = "length";
                break;
            }
            if (!eng.forward(nx, logits, err)) {
                eng.reset();
                live.clear();
                finish = "cancel";
                std::fprintf(stderr, "strata-q36-metal serve decode error: %s\n", err.c_str());
                break;
            }
            live.push_back(nx);
            std::memcpy(last_logits.data(), logits, 4 * (size_t)nv);
        }

        const double decode_ms = ms_since(t_dec0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n",
                    produced, ids.size(), prompt_ms, decode_ms, finish, reuse, fresh);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "strata-q36-metal serve: prompt %zu tok (%zu reused + %zu fresh) in %.0f ms (%.1f tok/s), "
                     "gen %lld tok in %.0f ms (%.1f tok/s, %s)\n",
                     ids.size(), reuse, fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double)fresh / prompt_ms : 0.0,
                     produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double)produced / decode_ms : 0.0,
                     finish);
    }

    reader.join();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tokens_csv, dump_logits;
    bool serve_mode = false;
    int n_gen = 16;
    int ctx = -1;
    bool stream = false;
    double arena_gb = -1.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--serve") serve_mode = true;
        else if ((a == "--model" || a == "--native") && i + 1 < argc) model_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if ((a == "--ctx" || a == "--max-context") && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
        else if (a == "--stream") stream = true;
        else if (a == "--arena-gb" && i + 1 < argc) { arena_gb = std::atof(argv[++i]); stream = true; }
        else if (serve_mode && a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "strata-q36-metal: ignoring unsupported serve flag %s\n", a.c_str());
            if (i + 1 < argc && argv[i + 1][0] != '-') ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36-metal [--serve] --model <gguf> [--tokens <id,id,...>] [--n N] [--ctx C] [--stream] [--arena-gb G]\n");
            return 2;
        }
    }

    if (ctx <= 0) ctx = serve_mode ? 8192 : 2048;

    if (model_path.empty() || (!serve_mode && tokens_csv.empty())) {
        std::fprintf(stderr, "strata-q36-metal: --model/--native%s required\n",
                     serve_mode ? " is" : " and --tokens are");
        return 2;
    }

    strata::metal_q36::Q36MetalEngine eng;
    strata::metal_q36::Options opt;
    opt.ctx = ctx;
    opt.stream = stream;
    opt.arena_gb = arena_gb;
    std::string err;
    const double t_load0 = mtl::now_s();
    if (!eng.open(model_path, opt, err)) {
        std::fprintf(stderr, "strata-q36-metal: %s\n", err.c_str());
        return 1;
    }
    const double t_load = mtl::now_s() - t_load0;

    if (serve_mode) {
        return run_serve(eng, ctx);
    }

    std::vector<int32_t> prompt;
    std::stringstream ss(tokens_csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) prompt.push_back(std::atoi(item.c_str()));
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "strata-q36-metal: empty prompt tokens\n");
        return 2;
    }

    const double t_pf0 = mtl::now_s();
    const float* logits = nullptr;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!eng.forward(prompt[i], logits, err)) {
            std::fprintf(stderr, "strata-q36-metal: forward error: %s\n", err.c_str());
            return 1;
        }
    }
    const double t_pf = mtl::now_s() - t_pf0;

    std::ofstream dump;
    if (!dump_logits.empty()) dump.open(dump_logits, std::ios::binary);

    std::vector<int32_t> gen;
    const double t_dec0 = mtl::now_s();
    int32_t next_tok = argmax_logits(logits, eng.n_vocab());
    for (int step = 0; step < n_gen; ++step) {
        gen.push_back(next_tok);
        if (dump) dump.write((const char*)logits, 4 * eng.n_vocab());
        if (step + 1 == n_gen) break;
        if (!eng.forward(next_tok, logits, err)) {
            std::fprintf(stderr, "strata-q36-metal: decode error: %s\n", err.c_str());
            return 1;
        }
        next_tok = argmax_logits(logits, eng.n_vocab());
    }
    const double t_dec = mtl::now_s() - t_dec0;

    std::printf("gen:");
    for (int32_t id : gen) std::printf(" %d", id);
    std::printf("\n");
    std::fprintf(stderr, "timing: load=%.2fs prefill=%zu tok in %.2fs (%.1f tok/s) decode=%d tok in %.2fs (%.1f tok/s)\n",
                 t_load, prompt.size(), t_pf, prompt.size() / std::max(t_pf, 1e-6),
                 n_gen, t_dec, (n_gen > 1 ? (n_gen - 1) : 1) / std::max(t_dec, 1e-6));
    const auto st = eng.stream_stats();
    if (st.streamed) {
        const double total_acc = (double)(st.hits + st.misses);
        const double hit_pct = total_acc > 0 ? 100.0 * (double)st.hits / total_acc : 0.0;
        std::fprintf(stderr,
                     "stream: dense=%.2f MiB arena=%.2f MiB contig=%s hits=%llu misses=%llu (hit=%.1f%%) reads=%llu read_mb=%.2f\n",
                     st.dense_bytes / (1024.0 * 1024.0), st.arena_bytes / (1024.0 * 1024.0),
                     st.using_contig ? "yes" : "no",
                     (unsigned long long)st.hits, (unsigned long long)st.misses, hit_pct,
                     (unsigned long long)st.reads, st.bytes_read / (1024.0 * 1024.0));
    }
    return 0;
}
