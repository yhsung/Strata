// metal/src/q36/q36_metal_main.cpp - CLI runner for Qwen3.6-35B-A3B Metal engine.
#include "q36/q36_metal.h"
#include "mtl_rt.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model_path, tokens_csv, dump_logits;
    int n_gen = 16;
    int ctx = 2048;
    bool stream = false;
    double arena_gb = -1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
        else if (a == "--stream") stream = true;
        else if (a == "--arena-gb" && i + 1 < argc) { arena_gb = std::atof(argv[++i]); stream = true; }
        else {
            std::fprintf(stderr,
                         "usage: strata-q36-metal --model <gguf> --tokens <id,id,...> [--n N] [--ctx C] [--stream] [--arena-gb G]\n");
            return 2;
        }
    }
    if (model_path.empty() || tokens_csv.empty()) {
        std::fprintf(stderr, "strata-q36-metal: --model and --tokens are required\n");
        return 2;
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

    const double t_pf0 = mtl::now_s();
    const float* logits = nullptr;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!eng.forward(prompt[i], logits, err)) {
            std::fprintf(stderr, "strata-q36-metal: forward error: %s\n", err.c_str());
            return 1;
        }
    }
    const double t_pf = mtl::now_s() - t_pf0;

    auto argmax = [&](const float* lg) -> int32_t {
        int32_t best = 0;
        float bv = lg[0];
        for (int64_t i = 1; i < eng.n_vocab(); ++i) {
            if (lg[i] > bv) { bv = lg[i]; best = (int32_t)i; }
        }
        return best;
    };

    std::ofstream dump;
    if (!dump_logits.empty()) dump.open(dump_logits, std::ios::binary);

    std::vector<int32_t> gen;
    const double t_dec0 = mtl::now_s();
    int32_t next_tok = argmax(logits);
    for (int step = 0; step < n_gen; ++step) {
        gen.push_back(next_tok);
        if (dump) dump.write((const char*)logits, 4 * eng.n_vocab());
        if (step + 1 == n_gen) break;
        if (!eng.forward(next_tok, logits, err)) {
            std::fprintf(stderr, "strata-q36-metal: decode error: %s\n", err.c_str());
            return 1;
        }
        next_tok = argmax(logits);
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
