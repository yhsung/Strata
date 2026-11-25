// metal/src/q36/q36_metal.h - Qwen3.6-35B-A3B (qwen35moe) Metal engine (Resident GGUF & SSD ExpertStream).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::metal_q36 {

struct Options {
    int ctx = 4096;
    bool silent = false;
    bool stream = false;
    double arena_gb = -1.0;
};

struct StreamStats {
    bool streamed = false;
    bool using_contig = false;
    uint64_t dense_bytes = 0;
    uint64_t arena_bytes = 0;
    uint64_t reads = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_read = 0;
};

class Q36MetalEngine {
public:
    Q36MetalEngine();
    ~Q36MetalEngine();

    bool open(const std::string& gguf_path, const Options& opt, std::string& err);
    void reset();
    bool forward(int32_t token, const float*& logits_out, std::string& err);
    int32_t greedy_next(int32_t token, std::string& err);

    int64_t n_vocab() const;
    int64_t n_trunk_layers() const;
    int pos() const;
    StreamStats stream_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal_q36
