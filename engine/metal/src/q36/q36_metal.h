// metal/src/q36/q36_metal.h - Qwen3.6-35B-A3B (qwen35moe) Metal engine (Resident GGUF & SSD ExpertStream).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::metal_q36 {

struct Options {
    int ctx = 4096;
    bool silent = false;
    bool stream = false;
    double arena_gb = -1.0;
    int n_slots = 1;
    int ckpt_count = 8;
    int mtp_drafts = 0;
    std::string profile_path;
};

struct StreamStats {
    bool streamed = false;
    bool using_contig = false;
    uint64_t dense_bytes = 0;
    uint64_t arena_bytes = 0;
    uint64_t reads = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t pre_hits = 0;
    uint64_t pre_issued = 0;
    uint64_t bytes_read = 0;
};

class Q36MetalEngine {
public:
    struct SlotStep {
        int slot = 0;
        int32_t token = 0;
    };

    Q36MetalEngine();
    ~Q36MetalEngine();

    bool open(const std::string& gguf_path, const Options& opt, std::string& err);
    void reset();
    void reset_slot(int slot);

    bool forward(int32_t token, const float*& logits_out, std::string& err);
    bool forward_chunk(int slot, const int32_t* tokens, int n_tok,
                       const int32_t* mrope_3d, const float* img_rows, int n_img_rows,
                       const int32_t* cell_row, bool need_logits,
                       const float*& logits_out, std::string& err);
    bool step_slots(const SlotStep* steps, int n_steps, std::vector<const float*>& logits_out, std::string& err);
    bool has_mtp() const;
    bool spec_step(int slot, int32_t x_tok, const std::vector<int32_t>& fix, int n_drafts,
                   std::vector<int32_t>& drafts, std::vector<int32_t>& picks,
                   int& accepted, std::string& err);

    int32_t greedy_next(int32_t token, std::string& err);

    void save_checkpoint(int slot);
    int checkpoint_at_or_below(int slot, int max_pos) const;
    bool restore_checkpoint(int slot, int target_pos);
    int checkpoint_count(int slot) const;
    bool copy_session(int src_slot, int dst_slot);

    int64_t n_vocab() const;
    int64_t n_embd() const;
    int64_t n_trunk_layers() const;
    int n_slots() const;
    int pos() const;
    int pos(int slot) const;
    int32_t mrope_delta(int slot) const;
    void set_mrope_delta(int slot, int32_t delta);
    StreamStats stream_stats() const;
    bool save_expert_profile(const std::string& path, std::string& err) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::metal_q36
