// metal/src/q36/q36_metal.cpp - Qwen3.6-35B-A3B (qwen35moe) Metal forward engine (Resident & SSD ExpertStream).
#include "q36/q36_metal.h"

#include "engine/arena_budget.h"
#include "engine/stream.h"
#include "gguf_lite.h"
#include "moe_tiles.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace strata::metal_q36 {

namespace {

constexpr int kMaxT = 8;
constexpr int kPromptT = 64;

struct GdnArgs {
    uint32_t h_k, h_v, channels, commit;
    float eps;
    uint32_t plane;
};

struct GdnPArgs {
    uint32_t n_tok, h_k, h_v, channels;
    float eps;
    uint32_t f32;
};

struct DqArgs {
    uint32_t n_in, n_out, row_bytes, perm;
};

struct GemmArgs {
    int M, N, K;
    int pad;
};

struct MmPlanArgs {
    uint32_t n_ent, topk, n_expert, tm, n_blocks, gu_gx, dn_gx;
};

struct MmArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff;
};

struct Q36AttnArgs {
    uint32_t H, HKV, D, n_rot;
    int32_t pos_t, pos_h, pos_w, kv_pos;
    uint32_t n_pos, ctx, plane;
    float base, eps, scale;
};

struct Q36RowMeta {
    int32_t pos_t, pos_h, pos_w, kv_pos;
    uint32_t n_pos, slot, pad0, pad1;
};

struct Q36NormArgs {
    uint32_t n, plane;
    float eps;
};

struct Q36EmbedArgs {
    uint32_t row, n_embd, row_bytes, type;
};

struct GemvArgs {
    uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride;
};

struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};

struct GlueArgs {
    uint32_t n, n_tok, n_embd, topk;
};

constexpr int P_INTS = 16 + 128 + 128 + 4 + 128 * 3;

int unit_of(int t) {
    switch (t) {
        case ref::Q2_0:
        case ref::NVFP4:
        case ref::IQ3_XXS:
        case ref::IQ3_S:
        case ref::BF16:
        case ref::F32: return 16;
        default: return 32;
    }
}

bool dq_supported(int t) {
    switch (t) {
        case ref::F32:
        case ref::BF16:
        case ref::Q8_0:
        case ref::Q4_0:
        case ref::Q5_0:
        case ref::IQ4_NL:
        case ref::IQ4_XS:
        case ref::Q2_0:
        case ref::IQ3_XXS:
        case ref::IQ3_S:
        case ref::Q2_K:
        case ref::Q3_K:
        case ref::Q4_K:
        case ref::Q5_K:
        case ref::Q6_K:
        case ref::NVFP4: return true;
        default: return false;
    }
}

int tile_sg(const char* name) {
    int tm = 0, tn = 0, tk = 0, sg = 2;
    if (std::sscanf(name, "_%d_%d_%d_%d", &tm, &tn, &tk, &sg) == 4) return sg;
    return 2;
}

int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K:
        case ref::Q5_K: return 2;
        case ref::Q6_K:
        case ref::Q3_K:
        case ref::Q2_K: return 3;
        default: return 0;
    }
}

uint32_t row_bytes_u(const gguf::Tensor& t) {
    return (uint32_t)ref::row_bytes(t.type, (int)t.shape[0]);
}

struct LayerRef {
    bool is_attn = false;
    const gguf::Tensor* attn_norm = nullptr;
    const gguf::Tensor* post_norm = nullptr;

    // GDN
    const gguf::Tensor* attn_qkv = nullptr;
    const gguf::Tensor* attn_gate = nullptr;
    const gguf::Tensor* ssm_alpha = nullptr;
    const gguf::Tensor* ssm_beta = nullptr;
    const gguf::Tensor* ssm_conv = nullptr;
    const gguf::Tensor* ssm_dt = nullptr;
    const gguf::Tensor* ssm_a = nullptr;
    const gguf::Tensor* ssm_norm = nullptr;
    const gguf::Tensor* ssm_out = nullptr;
    int gdn_idx = -1;

    // Attention
    const gguf::Tensor* attn_q = nullptr;
    const gguf::Tensor* attn_k = nullptr;
    const gguf::Tensor* attn_v = nullptr;
    const gguf::Tensor* attn_q_norm = nullptr;
    const gguf::Tensor* attn_k_norm = nullptr;
    const gguf::Tensor* attn_output = nullptr;
    int attn_idx = -1;

    // MoE
    const gguf::Tensor* gate_inp = nullptr;
    const gguf::Tensor* gate_exps = nullptr;
    const gguf::Tensor* up_exps = nullptr;
    const gguf::Tensor* down_exps = nullptr;
    const gguf::Tensor* shexp_gate_inp = nullptr;
    const gguf::Tensor* shexp_gate = nullptr;
    const gguf::Tensor* shexp_up = nullptr;
    const gguf::Tensor* shexp_down = nullptr;
};

struct Ckpt {
    int pos = 0;
    int32_t mrope_delta = 0;
    std::vector<int> gdn_par;
    std::vector<float> state_blob;
    std::vector<float> hist_blob;
    std::vector<float> hlast;
};

struct SlotState {
    int cur_pos = 0;
    int32_t mrope_delta = 0;
    std::vector<int> gdn_par;
    std::vector<float> hlast;
    std::vector<Ckpt> ckpts;
};

}  // namespace

struct Q36MetalEngine::Impl {
    gguf::File f;
    mtl::Context cx;
    mtl::Buffer W;
    strata::metal_engine::ExpertStream stream;
    bool streamed = false;
    bool has_mtp = false;
    uint64_t dense_bytes = 0;

    int lib_ops = -1, lib_gdn = -1, lib_attn = -1, lib_moe = -1, lib_mm = -1, lib_gdn_p = -1;
    bool prompt_gemm_ready = false;
    mtl::Pipeline p_embed, p_embed_batch, p_rms, p_add_rms, p_add;
    mtl::Pipeline p_rms_f16, p_add_rms_f16, p_rms_f32, p_mtp_cat;
    mtl::Pipeline p_attn_prep_batch, p_attn_dec_batch;
    mtl::Pipeline p_swiglu, p_comb, p_moe_route;
    mtl::Pipeline p_act_f16, p_gemm64;
    mtl::Pipeline p_gdn_pre, p_gdn_scan, p_gdn_post;
    mtl::Pipeline p_mm_clear, p_mm_hist, p_mm_scan, p_mm_place, p_mm_gather, p_mm_comb;
    mtl::Pipeline p_route_topk, p_swiglu_h, p_moe_out;
    std::map<std::pair<int, int>, mtl::Pipeline> p_gemv;
    std::map<std::pair<int, int>, mtl::Pipeline> p_router;
    std::map<std::pair<int, bool>, mtl::Pipeline> p_gdn;
    std::map<int, mtl::Pipeline> p_moe_gu;
    std::map<int, mtl::Pipeline> p_moe_dn;
    std::map<int, mtl::Pipeline> p_moe_gu_res;
    std::map<int, mtl::Pipeline> p_moe_dn_res;
    std::map<int, mtl::Pipeline> p_dq_f16;
    std::map<int, mtl::Pipeline> p_mm_gu2;
    std::map<int, mtl::Pipeline> p_mm_dn2;

    // Model dimensions
    int64_t n_embd = 0, n_vocab = 0, n_layers = 0, attn_interval = 4;
    int64_t n_expert = 0, top_k = 0, n_ff = 0, n_ff_sh = 0;
    int64_t H = 0, HKV = 0, D = 0, n_rot = 0;
    int64_t h_k = 0, h_v = 0, ssm_s = 0, conv_ch = 0, zv = 0, conv_k = 4;
    int n_gdn = 0, n_attn = 0;
    size_t gdn_state_floats = 0, gdn_hist_floats = 0;
    float eps = 1e-6f, rope_base = 10000000.0f;
    int max_ctx = 4096;
    int n_slots = 1;
    int ckpt_count = 8;
    int stream_T = 1;

    const gguf::Tensor* tok_embd = nullptr;
    const gguf::Tensor* out_norm = nullptr;
    const gguf::Tensor* out_w = nullptr;
    std::vector<LayerRef> layers;

    // MTP layer (blk.<n_layers>)
    const gguf::Tensor* mtp_enorm = nullptr;
    const gguf::Tensor* mtp_hnorm = nullptr;
    const gguf::Tensor* mtp_eh_proj = nullptr;
    const gguf::Tensor* mtp_shnorm = nullptr;
    LayerRef mtp_layer;
    mtl::Buffer mtp_exps_buf;
    uint64_t mtp_gate_off = 0, mtp_up_off = 0, mtp_down_off = 0;

    // State buffers (contiguous across n_slots)
    std::vector<SlotState> slots;
    std::vector<mtl::Buffer> gdn_state;  // each: n_slots * gdn_state_floats
    std::vector<mtl::Buffer> gdn_hist;   // each: n_slots * 2 * gdn_hist_floats
    std::vector<mtl::Buffer> kv_k;       // each: n_slots * HKV * max_ctx * D (+1 if has_mtp)
    std::vector<mtl::Buffer> kv_v;

    // Scratch buffers (up to kPromptT rows)
    mtl::Buffer x_buf, y_buf, xp_emb, xp_gdn, xp_att, xp_sh;
    mtl::Buffer xp_2n, hmain_buf, hmtp_buf, mtp_seeds_buf;
    std::vector<mtl::Buffer> v_qkv, v_al, v_be;
    mtl::Buffer qkv_buf, z_buf, al_buf, be_buf;
    mtl::Buffer qg_buf, kx_buf, vx_buf, q_buf, gate_buf;
    mtl::Buffer rt_logits, rt_ids, rt_w, rt_plan, rt_cnt;
    mtl::Buffer pred_logits;
    mtl::Buffer moe_h, moe_ye, sh_g, sh_u, sh_y, sh_gate, logits_buf;
    mtl::Buffer xp_f16, mm_wh;
    mtl::Buffer mm_gdn_qn, mm_gdn_kn, mm_gdn_vc, mm_gdn_gb, mm_gdn_kq, mm_gdn_o;
    mtl::Buffer mm_bcnt, mm_boff, mm_eoff, mm_tiles, mm_meta, mm_stok, mm_sw, mm_pos, mm_xs, mm_hs;
    mtl::Buffer tok_ids_buf, cell_row_buf, row_meta_buf, rinfo_buf, img_rows_buf;
    std::vector<uint64_t> exp_counts;
    bool stream_use_pred = false;

    const mtl::Pipeline& dq_pipe(int type) {
        auto it = p_dq_f16.find(type);
        if (it == p_dq_f16.end()) {
            const std::string name = "dq_f16_" + std::to_string(type);
            it = p_dq_f16.emplace(type, cx.pipeline(lib_mm, name)).first;
            mtl::require_fit(it->second, 64, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& mm_gu2_pipe(int type) {
        auto it = p_mm_gu2.find(type);
        if (it == p_mm_gu2.end()) {
            const auto tl = strata::metal_engine::gu2_tile(type);
            const std::string name = "mm_gu2_" + std::to_string(type) + tl.name;
            mtl::Pipeline p = streamed ? cx.pipeline(lib_mm, name, mtl::Constants().b(20, true))
                                       : cx.pipeline(lib_mm, name);
            it = p_mm_gu2.emplace(type, std::move(p)).first;
            mtl::require_fit(it->second, 32 * tile_sg(tl.name), name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& mm_dn2_pipe(int type) {
        auto it = p_mm_dn2.find(type);
        if (it == p_mm_dn2.end()) {
            const auto tl = strata::metal_engine::dn2_tile(type);
            const std::string name = "mm_down2_" + std::to_string(type) + tl.name;
            mtl::Pipeline p = streamed ? cx.pipeline(lib_mm, name, mtl::Constants().b(20, true))
                                       : cx.pipeline(lib_mm, name);
            it = p_mm_dn2.emplace(type, std::move(p)).first;
            mtl::require_fit(it->second, 32 * tile_sg(tl.name), name.c_str());
        }
        return it->second;
    }

    bool can_use_prompt_gemm(int T) const {
        if (!prompt_gemm_ready || T < 16 || T > kPromptT) return false;
        if (streamed && stream.min_class_slots() < (int)std::min<int64_t>(n_expert, (int64_t)T * top_k)) {
            return false;
        }
        return true;
    }

    void append_dense_mm(std::vector<mtl::Dispatch>& ds, const gguf::Tensor* t,
                       const mtl::Buffer& xh_in, uint32_t n_in, uint32_t n_out,
                       const mtl::Buffer& out, int n_tok, size_t out_off = 0) {
        DqArgs da{n_in, n_out, row_bytes_u(*t), 0u};
        mtl::Dispatch d_dq;
        d_dq.p = &dq_pipe(t->type);
        d_dq.args = {mtl::Arg::B(W, t->offset), mtl::Arg::B(mm_wh), mtl::Arg::V(da)};
        d_dq.grid[0] = (n_in / (uint32_t)unit_of(t->type) + 63u) / 64u;
        d_dq.grid[1] = n_out;
        d_dq.tg[0] = 64;
        ds.push_back(std::move(d_dq));

        GemmArgs ga{n_tok, (int)n_out, (int)n_in, 0};
        mtl::Dispatch d_gm;
        d_gm.p = &p_gemm64;
        d_gm.args = {mtl::Arg::B(xh_in), mtl::Arg::B(mm_wh), mtl::Arg::B(out, out_off), mtl::Arg::V(ga)};
        d_gm.grid[0] = (n_out + 63u) / 64u;
        d_gm.grid[1] = (uint32_t)(n_tok + 63) / 64u;
        d_gm.tg[0] = 128;
        ds.push_back(std::move(d_gm));
    }

    const mtl::Pipeline& gemv_pipe(int type, int n_tok) {
        const auto key = std::make_pair(type, n_tok);
        auto it = p_gemv.find(key);
        if (it == p_gemv.end()) {
            const std::string name = "gemv_" + std::to_string(type) + "_32_1_4_1";
            it = p_gemv.emplace(key, cx.pipeline(lib_moe, name, mtl::Constants().i(0, n_tok))).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline* router_pipe(int type, int n_tok) {
        const auto key = std::make_pair(type, n_tok);
        auto it = p_router.find(key);
        if (it == p_router.end()) {
            const std::string name = "router_route_" + std::to_string(type);
            it = p_router.emplace(key, cx.pipeline(lib_moe, name, mtl::Constants().i(0, n_tok))).first;
        }
        return it->second.max_threads >= 512 ? &it->second : nullptr;
    }

    const mtl::Pipeline& gdn_pipe(int n_tok, bool rows_mode) {
        const auto key = std::make_pair(n_tok, rows_mode);
        auto it = p_gdn.find(key);
        if (it == p_gdn.end()) {
            mtl::Constants c = mtl::Constants().i(2, n_tok).b(31, rows_mode).b(32, true);
            it = p_gdn.emplace(key, cx.pipeline(lib_gdn, "gdn_step", c)).first;
            mtl::require_fit(it->second, 512, "gdn_step");
        }
        return it->second;
    }

    const mtl::Pipeline& moe_gu_pipe(int type) {
        auto it = p_moe_gu.find(type);
        if (it == p_moe_gu.end()) {
            const std::string name = "moe_gu_" + std::to_string(type) + "_32_1_4";
            mtl::Pipeline p = streamed ? cx.pipeline(lib_moe, name, mtl::Constants().b(20, true))
                                       : cx.pipeline(lib_moe, name);
            it = p_moe_gu.emplace(type, std::move(p)).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& moe_dn_pipe(int type) {
        auto it = p_moe_dn.find(type);
        if (it == p_moe_dn.end()) {
            const std::string name = "moe_down_" + std::to_string(type) + "_8_1_4";
            mtl::Pipeline p = streamed ? cx.pipeline(lib_moe, name, mtl::Constants().b(20, true))
                                       : cx.pipeline(lib_moe, name);
            it = p_moe_dn.emplace(type, std::move(p)).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& moe_gu_res_pipe(int type) {
        auto it = p_moe_gu_res.find(type);
        if (it == p_moe_gu_res.end()) {
            const std::string name = "moe_gu_" + std::to_string(type) + "_32_1_4";
            it = p_moe_gu_res.emplace(type, cx.pipeline(lib_moe, name)).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& moe_dn_res_pipe(int type) {
        auto it = p_moe_dn_res.find(type);
        if (it == p_moe_dn_res.end()) {
            const std::string name = "moe_down_" + std::to_string(type) + "_8_1_4";
            it = p_moe_dn_res.emplace(type, cx.pipeline(lib_moe, name)).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    void at_cut(int layer) {
        const int32_t* ids_l = rt_ids.as<int32_t>() + (size_t)layer * kPromptT * top_k;
        const int n_ids = stream_T * (int)top_k;
        if (!exp_counts.empty()) {
            for (int i = 0; i < n_ids; ++i) {
                const int32_t e = ids_l[i];
                if (e >= 0 && e < n_expert) {
                    exp_counts[(size_t)layer * (size_t)n_expert + (size_t)e]++;
                }
            }
        }
        const float* pred_ptr = (stream_use_pred && layer + 1 < n_layers && pred_logits.ptr)
                                    ? (pred_logits.as<float>() + (size_t)layer * kPromptT * 512)
                                    : nullptr;
        stream.at_layer(layer, stream_T, (int)top_k, ids_l, pred_ptr, nullptr);
    }

    mtl::Dispatch disp_gemv(const gguf::Tensor* t, const mtl::Buffer& xp, uint32_t plane,
                            const mtl::Buffer& out, int n_tok, uint32_t y_stride = 0,
                            size_t out_off = 0) {
        const uint32_t n_in = (uint32_t)t->shape[0];
        const uint32_t n_out = t->shape.size() > 1 ? (uint32_t)t->shape[1] : 1u;
        const int k = xkind(t->type);
        GemvArgs ga{n_in, n_out, (uint32_t)n_tok, row_bytes_u(*t), n_in, y_stride ? y_stride : n_out};
        mtl::Dispatch d;
        d.p = &gemv_pipe(t->type, n_tok);
        d.args = {mtl::Arg::B(W, t->offset), mtl::Arg::B(xp, (size_t)4 * k * plane),
                  mtl::Arg::B(xp, (size_t)4 * 4 * plane), mtl::Arg::B(out, out_off), mtl::Arg::V(ga)};
        d.grid[0] = (n_out + 3) / 4;
        d.tg[0] = 128;
        return d;
    }

    void reset_slot_state(int slot) {
        SlotState& ss = slots[(size_t)slot];
        ss.cur_pos = 0;
        ss.mrope_delta = 0;
        std::fill(ss.gdn_par.begin(), ss.gdn_par.end(), 0);
        std::fill(ss.hlast.begin(), ss.hlast.end(), 0.0f);
        ss.ckpts.clear();
        for (int gi = 0; gi < n_gdn; ++gi) {
            std::memset(gdn_state[(size_t)gi].as<float>() + (size_t)slot * gdn_state_floats,
                        0, gdn_state_floats * sizeof(float));
            std::memset(gdn_hist[(size_t)gi].as<float>() + (size_t)slot * 2 * gdn_hist_floats,
                        0, 2 * gdn_hist_floats * sizeof(float));
        }
    }

    void ensure_img_buf(int n_img_rows) {
        const size_t need = std::max<size_t>(4 * (size_t)n_embd, (size_t)n_img_rows * (size_t)n_embd * 4);
        if (img_rows_buf.bytes < need) {
            img_rows_buf = cx.alloc(need);
        }
    }

    bool run_layers_batch(int n_tok, const int* step_slots_arr, bool rows_mode,
                          bool need_logits, int logits_row, std::string& err,
                          bool verify_mode = false) {
        (void)err;
        std::vector<mtl::Dispatch> ds;
        ds.reserve((size_t)n_layers * 16 + 8);
        std::vector<int> cuts;
        if (streamed) cuts.reserve((size_t)n_layers);

        const uint32_t N = (uint32_t)n_embd;
        const uint32_t NFF = (uint32_t)n_ff;
        const uint32_t NFF_SH = (uint32_t)n_ff_sh;
        const uint32_t NEXP = (uint32_t)n_expert;
        const uint32_t TOPK = (uint32_t)top_k;
        const uint32_t T = (uint32_t)n_tok;
        const uint32_t emb_plane = T * N;

        // Embed batch (tokens or vision embeddings)
        {
            Q36EmbedArgs ea{0u, N, row_bytes_u(*tok_embd), (uint32_t)tok_embd->type};
            mtl::Dispatch d;
            d.p = &p_embed_batch;
            d.args = {mtl::Arg::B(W, tok_embd->offset), mtl::Arg::B(x_buf),
                      mtl::Arg::B(tok_ids_buf), mtl::Arg::B(img_rows_buf),
                      mtl::Arg::B(cell_row_buf), mtl::Arg::V(ea)};
            d.grid[0] = (N / 32 + 7) / 8;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }

        for (int l = 0; l < n_layers; ++l) {
            const LayerRef& L = layers[(size_t)l];
            Q36NormArgs na{N, emb_plane, eps};
            {
                mtl::Dispatch d;
                d.p = &p_rms;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, L.attn_norm->offset),
                          mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
                d.grid[0] = 1;
                d.grid[1] = T;
                d.tg[0] = 256;
                ds.push_back(std::move(d));
            }

            if (!L.is_attn) {
                const int gi = L.gdn_idx;
                const mtl::Buffer& b_qkv = verify_mode ? v_qkv[(size_t)gi] : qkv_buf;
                const mtl::Buffer& b_al = verify_mode ? v_al[(size_t)gi] : al_buf;
                const mtl::Buffer& b_be = verify_mode ? v_be[(size_t)gi] : be_buf;
                auto d_qkv = disp_gemv(L.attn_qkv, xp_emb, emb_plane, b_qkv, n_tok);
                auto d_z = disp_gemv(L.attn_gate, xp_emb, emb_plane, z_buf, n_tok);
                auto d_al = disp_gemv(L.ssm_alpha, xp_emb, emb_plane, b_al, n_tok);
                auto d_be = disp_gemv(L.ssm_beta, xp_emb, emb_plane, b_be, n_tok);
                d_z.barrier = false;
                d_al.barrier = false;
                d_be.barrier = false;
                ds.push_back(std::move(d_qkv));
                ds.push_back(std::move(d_z));
                ds.push_back(std::move(d_al));
                ds.push_back(std::move(d_be));

                const uint32_t gdn_plane = T * (uint32_t)zv;
                mtl::Dispatch d_gdn;
                if (rows_mode) {
                    const size_t ri_off = (size_t)gi * (8 + 4 * kMaxT) * 4;
                    GdnArgs ga{(uint32_t)h_k, (uint32_t)h_v, (uint32_t)conv_ch, 1u, eps, gdn_plane};
                    d_gdn.p = &gdn_pipe(1, true);
                    d_gdn.args = {mtl::Arg::B(gdn_state[(size_t)gi]),
                                  mtl::Arg::B(gdn_hist[(size_t)gi]),
                                  mtl::Arg::B(b_qkv),
                                  mtl::Arg::B(b_al),
                                  mtl::Arg::B(b_be),
                                  mtl::Arg::B(z_buf),
                                  mtl::Arg::B(W, L.ssm_conv->offset),
                                  mtl::Arg::B(W, L.ssm_dt->offset),
                                  mtl::Arg::B(W, L.ssm_a->offset),
                                  mtl::Arg::B(W, L.ssm_norm->offset),
                                  mtl::Arg::B(xp_gdn),
                                  mtl::Arg::V(ga),
                                  mtl::Arg::B(gdn_hist[(size_t)gi]),
                                  mtl::Arg::B(rinfo_buf, ri_off)};
                    d_gdn.grid[0] = (uint32_t)h_v;
                    d_gdn.grid[1] = T;
                    d_gdn.tg[0] = 512;
                    for (int r = 0; r < n_tok; ++r) {
                        const int s = step_slots_arr[r];
                        slots[(size_t)s].gdn_par[(size_t)gi] = 1 - slots[(size_t)s].gdn_par[(size_t)gi];
                    }
                } else {
                    const int s = step_slots_arr[0];
                    const int par = slots[(size_t)s].gdn_par[(size_t)gi];
                    if (!verify_mode) slots[(size_t)s].gdn_par[(size_t)gi] = 1 - par;
                    const size_t st_off = (size_t)s * gdn_state_floats * 4;
                    const size_t h_in_off = ((size_t)s * 2 + (size_t)par) * gdn_hist_floats * 4;
                    const size_t h_out_off = ((size_t)s * 2 + (size_t)(1 - par)) * gdn_hist_floats * 4;

                    GdnArgs ga{(uint32_t)h_k, (uint32_t)h_v, (uint32_t)conv_ch, verify_mode ? 0u : 1u, eps, gdn_plane};
                    d_gdn.p = &gdn_pipe(n_tok, false);
                    d_gdn.args = {mtl::Arg::B(gdn_state[(size_t)gi], st_off),
                                  mtl::Arg::B(gdn_hist[(size_t)gi], h_in_off),
                                  mtl::Arg::B(b_qkv),
                                  mtl::Arg::B(b_al),
                                  mtl::Arg::B(b_be),
                                  mtl::Arg::B(z_buf),
                                  mtl::Arg::B(W, L.ssm_conv->offset),
                                  mtl::Arg::B(W, L.ssm_dt->offset),
                                  mtl::Arg::B(W, L.ssm_a->offset),
                                  mtl::Arg::B(W, L.ssm_norm->offset),
                                  mtl::Arg::B(xp_gdn),
                                  mtl::Arg::V(ga),
                                  mtl::Arg::B(gdn_hist[(size_t)gi], h_out_off)};
                    d_gdn.grid[0] = (uint32_t)h_v;
                    d_gdn.tg[0] = 512;
                }
                ds.push_back(std::move(d_gdn));
                ds.push_back(disp_gemv(L.ssm_out, xp_gdn, gdn_plane, y_buf, n_tok));
            } else {
                auto d_q = disp_gemv(L.attn_q, xp_emb, emb_plane, qg_buf, n_tok);
                auto d_k = disp_gemv(L.attn_k, xp_emb, emb_plane, kx_buf, n_tok);
                auto d_v = disp_gemv(L.attn_v, xp_emb, emb_plane, vx_buf, n_tok);
                d_k.barrier = false;
                d_v.barrier = false;
                ds.push_back(std::move(d_q));
                ds.push_back(std::move(d_k));
                ds.push_back(std::move(d_v));

                const int ai = L.attn_idx;
                const uint32_t att_plane = T * (uint32_t)(H * D);
                Q36AttnArgs aa{(uint32_t)H, (uint32_t)HKV, (uint32_t)D, (uint32_t)n_rot,
                               0, 0, 0, 0, 0, (uint32_t)max_ctx, att_plane,
                               rope_base, eps, 1.0f / std::sqrt((float)D)};

                mtl::Dispatch d_prep;
                d_prep.p = &p_attn_prep_batch;
                d_prep.args = {mtl::Arg::B(qg_buf), mtl::Arg::B(kx_buf), mtl::Arg::B(vx_buf),
                               mtl::Arg::B(W, L.attn_q_norm->offset), mtl::Arg::B(W, L.attn_k_norm->offset),
                               mtl::Arg::B(q_buf), mtl::Arg::B(gate_buf),
                               mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                               mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
                d_prep.grid[0] = (uint32_t)(H + HKV);
                d_prep.grid[1] = T;
                d_prep.tg[0] = 256;
                ds.push_back(std::move(d_prep));

                mtl::Dispatch d_dec;
                d_dec.p = &p_attn_dec_batch;
                d_dec.args = {mtl::Arg::B(q_buf), mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                              mtl::Arg::B(gate_buf), mtl::Arg::B(xp_att),
                              mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
                d_dec.grid[0] = (uint32_t)H;
                d_dec.grid[1] = T;
                d_dec.tg[0] = 256;
                ds.push_back(std::move(d_dec));

                ds.push_back(disp_gemv(L.attn_output, xp_att, att_plane, y_buf, n_tok));
            }

            // x += y; h2 = rms(x) * post_attention_norm (+ xprep)
            {
                mtl::Dispatch d;
                d.p = &p_add_rms;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::B(W, L.post_norm->offset),
                          mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
                d.grid[0] = 1;
                d.grid[1] = T;
                d.tg[0] = 256;
                ds.push_back(std::move(d));
            }

            // MoE routing + plan
            const uint32_t gu_rb = row_bytes_u(*L.gate_exps);
            const uint32_t dn_rb = row_bytes_u(*L.down_exps);
            MoeArgs ma{
                streamed ? 0ull : L.gate_exps->offset,
                streamed ? stream.up_in_slot(l) : L.up_exps->offset,
                streamed ? stream.down_in_slot(l) : L.down_exps->offset,
                streamed ? 0ull : (uint64_t)gu_rb * NFF,
                streamed ? 0ull : (uint64_t)dn_rb * N,
                gu_rb, dn_rb, N, NFF, T, TOPK, NEXP, NFF / 4, N / 16, 0};
            const size_t ids_off = streamed ? (size_t)4 * (size_t)l * kPromptT * TOPK : 0;

            if (const mtl::Pipeline* rp = (L.gate_inp->type == ref::F32 || L.gate_inp->type == ref::BF16)
                                                ? router_pipe(L.gate_inp->type, n_tok)
                                                : nullptr) {
                GemvArgs ga_rt{N, NEXP, T, row_bytes_u(*L.gate_inp), N, 512};
                mtl::Dispatch d_rr;
                d_rr.p = rp;
                d_rr.args = {mtl::Arg::B(W, L.gate_inp->offset), mtl::Arg::B(xp_emb),
                             mtl::Arg::B(xp_emb, (size_t)4 * 4 * emb_plane), mtl::Arg::B(rt_logits),
                             mtl::Arg::V(ga_rt), mtl::Arg::B(rt_ids, ids_off), mtl::Arg::B(rt_w),
                             mtl::Arg::B(rt_plan), mtl::Arg::V(ma), mtl::Arg::B(rt_cnt)};
                d_rr.grid[0] = NEXP / 16;
                d_rr.tg[0] = 512;
                ds.push_back(std::move(d_rr));
            } else {
                ds.push_back(disp_gemv(L.gate_inp, xp_emb, emb_plane, rt_logits, n_tok, 512));
                mtl::Dispatch d_rt;
                d_rt.p = &p_moe_route;
                d_rt.args = {mtl::Arg::B(rt_logits), mtl::Arg::B(rt_ids, ids_off), mtl::Arg::B(rt_w),
                             mtl::Arg::B(rt_plan), mtl::Arg::V(ma)};
                d_rt.grid[0] = 1;
                d_rt.tg[0] = 512;
                ds.push_back(std::move(d_rt));
            }
            if (streamed) {
                if (l + 1 < n_layers) {
                    mtl::Dispatch d_pred = disp_gemv(layers[(size_t)l + 1].gate_inp, xp_emb, emb_plane,
                                                     pred_logits, n_tok, 512,
                                                     (size_t)4 * (size_t)l * kPromptT * 512);
                    d_pred.barrier = false;
                    ds.push_back(std::move(d_pred));
                }
                cuts.push_back((int)ds.size());
            }

            // Routed experts gate/up + down
            {
                const int gk = xkind(L.gate_exps->type);
                const mtl::Buffer& w0 = streamed ? stream.chunk(0) : W;
                const mtl::Buffer& w1 = streamed ? stream.chunk(1) : W;
                const mtl::Buffer& w2 = streamed ? stream.chunk(2) : W;
                const mtl::Buffer& w3 = streamed ? stream.chunk(3) : W;
                const mtl::Arg smap_arg = streamed ? mtl::Arg::B(stream.smap(), (size_t)4 * (size_t)l * 512)
                                                   : mtl::Arg::B(rt_plan);

                mtl::Dispatch d_gu;
                d_gu.p = &moe_gu_pipe(L.gate_exps->type);
                d_gu.args = {mtl::Arg::B(w0), mtl::Arg::B(xp_emb, (size_t)4 * gk * emb_plane),
                             mtl::Arg::B(moe_h), mtl::Arg::B(rt_plan), mtl::Arg::V(ma),
                             mtl::Arg::B(xp_emb, (size_t)4 * 4 * emb_plane), smap_arg,
                             mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
                d_gu.tg[0] = 128;
                d_gu.indirect = &rt_plan;
                d_gu.ind_off = 4 * 4;
                ds.push_back(std::move(d_gu));

                mtl::Dispatch d_dn;
                d_dn.p = &moe_dn_pipe(L.down_exps->type);
                d_dn.args = {mtl::Arg::B(w0), mtl::Arg::B(moe_h), mtl::Arg::B(moe_ye),
                             mtl::Arg::B(rt_plan), mtl::Arg::V(ma), mtl::Arg::B(rt_plan),
                             smap_arg, mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
                d_dn.tg[0] = 128;
                d_dn.indirect = &rt_plan;
                d_dn.ind_off = 4 * 8;
                ds.push_back(std::move(d_dn));
            }

            // Shared expert
            {
                const uint32_t sh_plane = T * NFF_SH;
                auto d_sg = disp_gemv(L.shexp_gate, xp_emb, emb_plane, sh_g, n_tok);
                auto d_su = disp_gemv(L.shexp_up, xp_emb, emb_plane, sh_u, n_tok);
                auto d_sgi = disp_gemv(L.shexp_gate_inp, xp_emb, emb_plane, sh_gate, n_tok);
                d_sg.barrier = false;
                d_su.barrier = false;
                d_sgi.barrier = false;
                ds.push_back(std::move(d_sg));
                ds.push_back(std::move(d_su));
                ds.push_back(std::move(d_sgi));

                GlueArgs gl_sw{NFF_SH, T, N, TOPK};
                mtl::Dispatch d_sw;
                d_sw.p = &p_swiglu;
                d_sw.args = {mtl::Arg::B(sh_g), mtl::Arg::B(sh_u), mtl::Arg::B(xp_sh), mtl::Arg::V(gl_sw)};
                d_sw.grid[0] = NFF_SH / 16;
                d_sw.grid[1] = T;
                d_sw.tg[0] = 16;
                ds.push_back(std::move(d_sw));

                ds.push_back(disp_gemv(L.shexp_down, xp_sh, sh_plane, sh_y, n_tok));
            }

            // Combine routed + shared -> y_buf, then x_buf += y_buf
            {
                GlueArgs gl_c{N, T, N, TOPK};
                mtl::Dispatch d_comb;
                d_comb.p = &p_comb;
                d_comb.args = {mtl::Arg::B(moe_ye), mtl::Arg::B(rt_plan), mtl::Arg::B(sh_y),
                               mtl::Arg::B(sh_gate), mtl::Arg::B(y_buf), mtl::Arg::V(gl_c)};
                d_comb.grid[0] = (N * T + 127) / 128;
                d_comb.tg[0] = 128;
                ds.push_back(std::move(d_comb));

                const uint32_t total_n = N * T;
                mtl::Dispatch d_add;
                d_add.p = &p_add;
                d_add.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::V(total_n)};
                d_add.grid[0] = (total_n + 255) / 256;
                d_add.tg[0] = 256;
                ds.push_back(std::move(d_add));
            }
        }

        // Final RMSNorm + LM Head (+ hmain capture when MTP enabled)
        if (has_mtp) {
            Q36NormArgs hna{N, 0u, eps};
            mtl::Dispatch d_hm;
            d_hm.p = &p_rms_f32;
            d_hm.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, out_norm->offset),
                         mtl::Arg::B(hmain_buf), mtl::Arg::V(hna)};
            d_hm.grid[0] = 1;
            d_hm.grid[1] = T;
            d_hm.tg[0] = 256;
            ds.push_back(std::move(d_hm));
        }
        if (need_logits) {
            if (logits_row >= 0) {
                Q36NormArgs na{N, N, eps};
                mtl::Dispatch d_rms;
                d_rms.p = &p_rms;
                d_rms.args = {mtl::Arg::B(x_buf, (size_t)4 * (size_t)logits_row * N),
                              mtl::Arg::B(W, out_norm->offset), mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
                d_rms.grid[0] = 1;
                d_rms.tg[0] = 256;
                ds.push_back(std::move(d_rms));

                ds.push_back(disp_gemv(out_w, xp_emb, N, logits_buf, 1));
            } else {
                Q36NormArgs na{N, emb_plane, eps};
                mtl::Dispatch d_rms;
                d_rms.p = &p_rms;
                d_rms.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, out_norm->offset),
                              mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
                d_rms.grid[0] = 1;
                d_rms.grid[1] = T;
                d_rms.tg[0] = 256;
                ds.push_back(std::move(d_rms));

                ds.push_back(disp_gemv(out_w, xp_emb, emb_plane, logits_buf, n_tok));
            }
        }

        stream_T = n_tok;
        stream_use_pred = true;
        if (streamed) {
            cx.run_graph_seg(ds, cuts, [](void* u, int s) { ((Impl*)u)->at_cut(s); }, this);
        } else {
            cx.run_graph(ds);
        }
        return true;
    }

    bool run_prompt_gemm_chunk(int slot, int n_tok, bool need_logits, std::string& err) {
        (void)err;
        std::vector<mtl::Dispatch> ds;
        ds.reserve((size_t)n_layers * 28 + 8);
        std::vector<int> cuts;
        if (streamed) cuts.reserve((size_t)n_layers);

        const uint32_t N = (uint32_t)n_embd;
        const uint32_t NFF = (uint32_t)n_ff;
        const uint32_t NFF_SH = (uint32_t)n_ff_sh;
        const uint32_t NEXP = (uint32_t)n_expert;
        const uint32_t TOPK = (uint32_t)top_k;
        const uint32_t T = (uint32_t)n_tok;

        // Embed batch (tokens or vision embeddings) -> x_buf (fp32)
        {
            Q36EmbedArgs ea{0u, N, row_bytes_u(*tok_embd), (uint32_t)tok_embd->type};
            mtl::Dispatch d;
            d.p = &p_embed_batch;
            d.args = {mtl::Arg::B(W, tok_embd->offset), mtl::Arg::B(x_buf),
                      mtl::Arg::B(tok_ids_buf), mtl::Arg::B(img_rows_buf),
                      mtl::Arg::B(cell_row_buf), mtl::Arg::V(ea)};
            d.grid[0] = (N / 32 + 7) / 8;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }

        for (int l = 0; l < n_layers; ++l) {
            const LayerRef& L = layers[(size_t)l];
            Q36NormArgs na{N, 0u, eps};
            {
                mtl::Dispatch d;
                d.p = &p_rms_f16;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, L.attn_norm->offset),
                          mtl::Arg::B(xp_f16), mtl::Arg::V(na)};
                d.grid[0] = 1;
                d.grid[1] = T;
                d.tg[0] = 256;
                ds.push_back(std::move(d));
            }

            if (!L.is_attn) {
                append_dense_mm(ds, L.attn_qkv, xp_f16, N, (uint32_t)conv_ch, qkv_buf, n_tok);
                append_dense_mm(ds, L.attn_gate, xp_f16, N, (uint32_t)zv, z_buf, n_tok);
                append_dense_mm(ds, L.ssm_alpha, xp_f16, N, (uint32_t)h_v, al_buf, n_tok);
                append_dense_mm(ds, L.ssm_beta, xp_f16, N, (uint32_t)h_v, be_buf, n_tok);

                const int gi = L.gdn_idx;
                const int par = slots[(size_t)slot].gdn_par[(size_t)gi];
                slots[(size_t)slot].gdn_par[(size_t)gi] = 1 - par;
                const size_t st_off = (size_t)slot * gdn_state_floats * 4;
                const size_t h_in_off = ((size_t)slot * 2 + (size_t)par) * gdn_hist_floats * 4;
                const size_t h_out_off = ((size_t)slot * 2 + (size_t)(1 - par)) * gdn_hist_floats * 4;

                GdnPArgs gpa{T, (uint32_t)h_k, (uint32_t)h_v, (uint32_t)conv_ch, eps, 0u};
                {
                    mtl::Dispatch d;
                    d.p = &p_gdn_pre;
                    d.args = {mtl::Arg::B(qkv_buf), mtl::Arg::B(gdn_hist[(size_t)gi], h_in_off),
                              mtl::Arg::B(W, L.ssm_conv->offset), mtl::Arg::B(al_buf),
                              mtl::Arg::B(be_buf), mtl::Arg::B(W, L.ssm_dt->offset),
                              mtl::Arg::B(W, L.ssm_a->offset), mtl::Arg::B(mm_gdn_qn),
                              mtl::Arg::B(mm_gdn_kn), mtl::Arg::B(mm_gdn_vc),
                              mtl::Arg::B(mm_gdn_gb), mtl::Arg::B(mm_gdn_kq),
                              mtl::Arg::B(gdn_hist[(size_t)gi], h_out_off), mtl::Arg::V(gpa)};
                    d.grid[0] = (uint32_t)(h_k + h_v / 2);
                    d.grid[1] = T;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_gdn_scan;
                    d.args = {mtl::Arg::B(gdn_state[(size_t)gi], st_off), mtl::Arg::B(mm_gdn_qn),
                              mtl::Arg::B(mm_gdn_kn), mtl::Arg::B(mm_gdn_vc),
                              mtl::Arg::B(mm_gdn_gb), mtl::Arg::B(mm_gdn_kq),
                              mtl::Arg::B(mm_gdn_o), mtl::Arg::V(gpa)};
                    d.grid[0] = (uint32_t)h_v;
                    d.tg[0] = 128;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_gdn_post;
                    d.args = {mtl::Arg::B(mm_gdn_o), mtl::Arg::B(z_buf),
                              mtl::Arg::B(W, L.ssm_norm->offset), mtl::Arg::B(xp_f16),
                              mtl::Arg::B(mm_gdn_o), mtl::Arg::V(gpa)};
                    d.grid[0] = (uint32_t)h_v;
                    d.grid[1] = T;
                    d.tg[0] = 128;
                    ds.push_back(std::move(d));
                }
                append_dense_mm(ds, L.ssm_out, xp_f16, (uint32_t)zv, N, y_buf, n_tok);
            } else {
                append_dense_mm(ds, L.attn_q, xp_f16, N, (uint32_t)(H * 2 * D), qg_buf, n_tok);
                append_dense_mm(ds, L.attn_k, xp_f16, N, (uint32_t)(HKV * D), kx_buf, n_tok);
                append_dense_mm(ds, L.attn_v, xp_f16, N, (uint32_t)(HKV * D), vx_buf, n_tok);

                const int ai = L.attn_idx;
                const uint32_t att_plane = T * (uint32_t)(H * D);
                Q36AttnArgs aa{(uint32_t)H, (uint32_t)HKV, (uint32_t)D, (uint32_t)n_rot,
                               0, 0, 0, 0, 0, (uint32_t)max_ctx, att_plane,
                               rope_base, eps, 1.0f / std::sqrt((float)D)};

                mtl::Dispatch d_prep;
                d_prep.p = &p_attn_prep_batch;
                d_prep.args = {mtl::Arg::B(qg_buf), mtl::Arg::B(kx_buf), mtl::Arg::B(vx_buf),
                               mtl::Arg::B(W, L.attn_q_norm->offset), mtl::Arg::B(W, L.attn_k_norm->offset),
                               mtl::Arg::B(q_buf), mtl::Arg::B(gate_buf),
                               mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                               mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
                d_prep.grid[0] = (uint32_t)(H + HKV);
                d_prep.grid[1] = T;
                d_prep.tg[0] = 256;
                ds.push_back(std::move(d_prep));

                mtl::Dispatch d_dec;
                d_dec.p = &p_attn_dec_batch;
                d_dec.args = {mtl::Arg::B(q_buf), mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                              mtl::Arg::B(gate_buf), mtl::Arg::B(xp_att),
                              mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
                d_dec.grid[0] = (uint32_t)H;
                d_dec.grid[1] = T;
                d_dec.tg[0] = 256;
                ds.push_back(std::move(d_dec));

                const uint32_t n_act = T * (uint32_t)(H * D);
                mtl::Dispatch d_a16;
                d_a16.p = &p_act_f16;
                d_a16.args = {mtl::Arg::B(xp_att), mtl::Arg::B(xp_f16), mtl::Arg::V(n_act)};
                d_a16.grid[0] = (n_act + 255u) / 256u;
                d_a16.tg[0] = 256;
                ds.push_back(std::move(d_a16));

                append_dense_mm(ds, L.attn_output, xp_f16, (uint32_t)(H * D), N, y_buf, n_tok);
            }

            // x += y; xp_emb = rms(x) * post_norm (fp32 for mm_gather), xp_f16 (fp16 for dense_mm)
            {
                mtl::Dispatch d;
                d.p = &p_add_rms_f16;
                d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::B(W, L.post_norm->offset),
                          mtl::Arg::B(xp_emb), mtl::Arg::B(xp_f16), mtl::Arg::V(na)};
                d.grid[0] = 1;
                d.grid[1] = T;
                d.tg[0] = 256;
                ds.push_back(std::move(d));
            }

            // MoE routing + cut (stream_use_pred stays false so ExpertStream::ahead is not called on T > 8)
            const size_t ids_off = streamed ? (size_t)4 * (size_t)l * kPromptT * TOPK : 0;
            append_dense_mm(ds, L.gate_inp, xp_f16, N, NEXP, rt_logits, n_tok);
            {
                mtl::Dispatch d;
                d.p = &p_route_topk;
                d.args = {mtl::Arg::B(rt_logits), mtl::Arg::B(rt_ids, ids_off), mtl::Arg::B(rt_w),
                          mtl::Arg::V(T), mtl::Arg::V(NEXP)};
                d.grid[0] = (T + 7u) / 8u;
                d.tg[0] = 256;
                ds.push_back(std::move(d));
            }
            if (streamed) cuts.push_back((int)ds.size());

            // Routed MoE plan + mm_gather + mm_gu2 + mm_down2 + mm_combine
            {
                const auto gu_t = strata::metal_engine::gu2_tile(L.gate_exps->type);
                const auto dn_t = strata::metal_engine::dn2_tile(L.down_exps->type);
                const uint32_t tm = 16u;
                const uint32_t n_ent = T * TOPK;
                const uint32_t n_blocks = (n_ent + 1023u) / 1024u;
                const uint32_t maxrows = n_ent + NEXP * tm;
                const uint32_t gu_rb = row_bytes_u(*L.gate_exps);
                const uint32_t dn_rb = row_bytes_u(*L.down_exps);

                MmPlanArgs pa{n_ent, TOPK, NEXP, tm, n_blocks,
                              (uint32_t)strata::metal_engine::moe_grid_x((int)NFF, gu_t.tn),
                              (uint32_t)strata::metal_engine::moe_grid_x((int)N, dn_t.tn)};
                MmArgs ma{
                    streamed ? 0ull : L.gate_exps->offset,
                    streamed ? stream.up_in_slot(l) : L.up_exps->offset,
                    streamed ? stream.down_in_slot(l) : L.down_exps->offset,
                    streamed ? 0ull : (uint64_t)gu_rb * NFF,
                    streamed ? 0ull : (uint64_t)dn_rb * N,
                    gu_rb, dn_rb, N, NFF};

                {
                    mtl::Dispatch d;
                    d.p = &p_mm_clear;
                    d.args = {mtl::Arg::B(mm_stok), mtl::Arg::V(maxrows)};
                    d.grid[0] = (maxrows + 255u) / 256u;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_mm_hist;
                    d.args = {mtl::Arg::B(rt_ids, ids_off), mtl::Arg::B(mm_bcnt), mtl::Arg::V(pa)};
                    d.grid[0] = n_blocks;
                    d.tg[0] = 1024;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_mm_scan;
                    d.args = {mtl::Arg::B(mm_bcnt), mtl::Arg::B(mm_boff), mtl::Arg::B(mm_eoff),
                              mtl::Arg::B(mm_tiles), mtl::Arg::B(mm_meta), mtl::Arg::V(pa)};
                    d.grid[0] = 1;
                    d.tg[0] = NEXP;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_mm_place;
                    d.args = {mtl::Arg::B(rt_ids, ids_off), mtl::Arg::B(rt_w), mtl::Arg::B(mm_boff),
                              mtl::Arg::B(mm_eoff), mtl::Arg::B(mm_stok), mtl::Arg::B(mm_sw),
                              mtl::Arg::B(mm_pos), mtl::Arg::V(pa)};
                    d.grid[0] = n_blocks;
                    d.tg[0] = 1024;
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &p_mm_gather;
                    d.args = {mtl::Arg::B(xp_emb), mtl::Arg::B(mm_stok), mtl::Arg::B(mm_xs), mtl::Arg::V(N)};
                    d.indirect = &mm_meta;
                    d.ind_off = 32;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }

                const mtl::Buffer& w0 = streamed ? stream.chunk(0) : W;
                const mtl::Buffer& w1 = streamed ? stream.chunk(1) : W;
                const mtl::Buffer& w2 = streamed ? stream.chunk(2) : W;
                const mtl::Buffer& w3 = streamed ? stream.chunk(3) : W;
                const mtl::Arg smap_arg = streamed ? mtl::Arg::B(stream.smap(), (size_t)4 * (size_t)l * 512)
                                                   : mtl::Arg::B(mm_meta);
                {
                    mtl::Dispatch d;
                    d.p = &mm_gu2_pipe(L.gate_exps->type);
                    d.args = {mtl::Arg::B(w0), mtl::Arg::B(mm_xs), mtl::Arg::B(mm_hs),
                              mtl::Arg::B(mm_tiles), mtl::Arg::V(ma), smap_arg,
                              mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
                    d.indirect = &mm_meta;
                    d.ind_off = 8;
                    d.tg[0] = 32 * (uint32_t)tile_sg(gu_t.name);
                    ds.push_back(std::move(d));
                }
                {
                    mtl::Dispatch d;
                    d.p = &mm_dn2_pipe(L.down_exps->type);
                    d.args = {mtl::Arg::B(w0), mtl::Arg::B(mm_hs), mtl::Arg::B(moe_ye),
                              mtl::Arg::B(mm_tiles), mtl::Arg::V(ma), smap_arg,
                              mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
                    d.indirect = &mm_meta;
                    d.ind_off = 20;
                    d.tg[0] = 32 * (uint32_t)tile_sg(dn_t.name);
                    ds.push_back(std::move(d));
                }
                {
                    const uint32_t ca[4] = {T, N, TOPK, 0u};
                    mtl::Dispatch d;
                    d.p = &p_mm_comb;
                    d.args = {mtl::Arg::B(moe_ye), mtl::Arg::B(mm_pos), mtl::Arg::B(mm_sw),
                              mtl::Arg::B(y_buf), mtl::Arg::V(ca)};
                    d.grid[0] = (T * N + 255u) / 256u;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
            }

            // Shared expert -> sh_y, moe_out -> y_buf, q36_add -> x_buf
            {
                append_dense_mm(ds, L.shexp_gate, xp_f16, N, NFF_SH, sh_g, n_tok);
                append_dense_mm(ds, L.shexp_up, xp_f16, N, NFF_SH, sh_u, n_tok);
                append_dense_mm(ds, L.shexp_gate_inp, xp_f16, N, 1u, sh_gate, n_tok);
                {
                    const uint32_t n_sw = T * NFF_SH;
                    mtl::Dispatch d;
                    d.p = &p_swiglu_h;
                    d.args = {mtl::Arg::B(sh_g), mtl::Arg::B(sh_u), mtl::Arg::B(xp_f16), mtl::Arg::V(n_sw)};
                    d.grid[0] = (n_sw + 255u) / 256u;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
                append_dense_mm(ds, L.shexp_down, xp_f16, NFF_SH, N, sh_y, n_tok);
                {
                    const uint32_t ma_out[2] = {T, N};
                    mtl::Dispatch d;
                    d.p = &p_moe_out;
                    d.args = {mtl::Arg::B(y_buf), mtl::Arg::B(sh_y), mtl::Arg::B(sh_gate),
                              mtl::Arg::B(y_buf), mtl::Arg::V(ma_out)};
                    d.grid[0] = (T * N + 255u) / 256u;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
                {
                    const uint32_t total_n = T * N;
                    mtl::Dispatch d;
                    d.p = &p_add;
                    d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::V(total_n)};
                    d.grid[0] = (total_n + 255u) / 256u;
                    d.tg[0] = 256;
                    ds.push_back(std::move(d));
                }
            }
        }

        if (has_mtp) {
            Q36NormArgs hna{N, 0u, eps};
            mtl::Dispatch d_hm;
            d_hm.p = &p_rms_f32;
            d_hm.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, out_norm->offset),
                         mtl::Arg::B(hmain_buf), mtl::Arg::V(hna)};
            d_hm.grid[0] = 1;
            d_hm.grid[1] = T;
            d_hm.tg[0] = 256;
            ds.push_back(std::move(d_hm));
        }
        if (need_logits) {
            Q36NormArgs na{N, N, eps};
            mtl::Dispatch d_rms;
            d_rms.p = &p_rms;
            d_rms.args = {mtl::Arg::B(x_buf, (size_t)4 * (size_t)(n_tok - 1) * N),
                          mtl::Arg::B(W, out_norm->offset), mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
            d_rms.grid[0] = 1;
            d_rms.tg[0] = 256;
            ds.push_back(std::move(d_rms));

            ds.push_back(disp_gemv(out_w, xp_emb, N, logits_buf, 1));
        }

        stream_T = n_tok;
        stream_use_pred = false;
        if (streamed) {
            cx.run_graph_seg(ds, cuts, [](void* u, int s) { ((Impl*)u)->at_cut(s); }, this);
        } else {
            cx.run_graph(ds);
        }
        return true;
    }

    bool forward_mtp_chunk(int slot, int n_tok, int p0, const int32_t* toks,
                           const int32_t* mrope_3d, const int32_t* crs,
                           const mtl::Buffer& seeds_buf, size_t seeds_off,
                           bool draft, std::string& err) {
        (void)err;
        if (!has_mtp || n_tok <= 0 || n_tok > kMaxT) { err = "invalid MTP chunk"; return false; }
        SlotState& ss = slots[(size_t)slot];

        int32_t* tid_p = tok_ids_buf.as<int32_t>();
        int32_t* cr_p = cell_row_buf.as<int32_t>();
        Q36RowMeta* meta_p = row_meta_buf.as<Q36RowMeta>();
        for (int t = 0; t < n_tok; ++t) {
            tid_p[t] = toks[t];
            cr_p[t] = crs ? crs[t] : -1;
            const int kvp = p0 + t;
            const int32_t pt = mrope_3d ? mrope_3d[3 * t + 0] : (kvp + ss.mrope_delta);
            const int32_t ph = mrope_3d ? mrope_3d[3 * t + 1] : pt;
            const int32_t pw = mrope_3d ? mrope_3d[3 * t + 2] : pt;
            meta_p[t] = {pt, ph, pw, kvp, (uint32_t)(kvp + 1), (uint32_t)slot, 0u, 0u};
        }

        const uint32_t N = (uint32_t)n_embd;
        const uint32_t NFF = (uint32_t)n_ff;
        const uint32_t NFF_SH = (uint32_t)n_ff_sh;
        const uint32_t NEXP = (uint32_t)n_expert;
        const uint32_t TOPK = (uint32_t)top_k;
        const uint32_t T = (uint32_t)n_tok;
        const uint32_t emb_plane = T * N;
        const uint32_t cat_plane = T * 2u * N;
        const LayerRef& L = mtp_layer;

        std::vector<mtl::Dispatch> ds;
        ds.reserve(32);

        // 1. Embed token -> y_buf
        {
            Q36EmbedArgs ea{0u, N, row_bytes_u(*tok_embd), (uint32_t)tok_embd->type};
            mtl::Dispatch d;
            d.p = &p_embed_batch;
            d.args = {mtl::Arg::B(W, tok_embd->offset), mtl::Arg::B(y_buf),
                      mtl::Arg::B(tok_ids_buf), mtl::Arg::B(img_rows_buf),
                      mtl::Arg::B(cell_row_buf), mtl::Arg::V(ea)};
            d.grid[0] = (N / 32 + 7) / 8;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }

        // 2. [rms(e)*enorm | rms(h)*hnorm] -> xp_2n, eh_proj -> x_buf
        {
            Q36NormArgs na{N, cat_plane, eps};
            mtl::Dispatch d;
            d.p = &p_mtp_cat;
            d.args = {mtl::Arg::B(y_buf), mtl::Arg::B(seeds_buf, seeds_off),
                      mtl::Arg::B(W, mtp_enorm->offset), mtl::Arg::B(W, mtp_hnorm->offset),
                      mtl::Arg::B(xp_2n), mtl::Arg::V(na)};
            d.grid[0] = 1;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }
        ds.push_back(disp_gemv(mtp_eh_proj, xp_2n, cat_plane, x_buf, n_tok));

        // 3. Full attention layer at kv_k[n_attn] / kv_v[n_attn]
        {
            Q36NormArgs na{N, emb_plane, eps};
            mtl::Dispatch d;
            d.p = &p_rms;
            d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, L.attn_norm->offset),
                      mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
            d.grid[0] = 1;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }
        {
            auto d_q = disp_gemv(L.attn_q, xp_emb, emb_plane, qg_buf, n_tok);
            auto d_k = disp_gemv(L.attn_k, xp_emb, emb_plane, kx_buf, n_tok);
            auto d_v = disp_gemv(L.attn_v, xp_emb, emb_plane, vx_buf, n_tok);
            d_k.barrier = false;
            d_v.barrier = false;
            ds.push_back(std::move(d_q));
            ds.push_back(std::move(d_k));
            ds.push_back(std::move(d_v));

            const int ai = n_attn;
            const uint32_t att_plane = T * (uint32_t)(H * D);
            Q36AttnArgs aa{(uint32_t)H, (uint32_t)HKV, (uint32_t)D, (uint32_t)n_rot,
                           0, 0, 0, 0, 0, (uint32_t)max_ctx, att_plane,
                           rope_base, eps, 1.0f / std::sqrt((float)D)};

            mtl::Dispatch d_prep;
            d_prep.p = &p_attn_prep_batch;
            d_prep.args = {mtl::Arg::B(qg_buf), mtl::Arg::B(kx_buf), mtl::Arg::B(vx_buf),
                           mtl::Arg::B(W, L.attn_q_norm->offset), mtl::Arg::B(W, L.attn_k_norm->offset),
                           mtl::Arg::B(q_buf), mtl::Arg::B(gate_buf),
                           mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                           mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
            d_prep.grid[0] = (uint32_t)(H + HKV);
            d_prep.grid[1] = T;
            d_prep.tg[0] = 256;
            ds.push_back(std::move(d_prep));

            mtl::Dispatch d_dec;
            d_dec.p = &p_attn_dec_batch;
            d_dec.args = {mtl::Arg::B(q_buf), mtl::Arg::B(kv_k[(size_t)ai]), mtl::Arg::B(kv_v[(size_t)ai]),
                          mtl::Arg::B(gate_buf), mtl::Arg::B(xp_att),
                          mtl::Arg::B(row_meta_buf), mtl::Arg::V(aa)};
            d_dec.grid[0] = (uint32_t)H;
            d_dec.grid[1] = T;
            d_dec.tg[0] = 256;
            ds.push_back(std::move(d_dec));

            ds.push_back(disp_gemv(L.attn_output, xp_att, att_plane, y_buf, n_tok));
        }

        // 4. Post-attention RMSNorm + Resident MoE + Shared Expert
        {
            Q36NormArgs na{N, emb_plane, eps};
            mtl::Dispatch d;
            d.p = &p_add_rms;
            d.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::B(W, L.post_norm->offset),
                      mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
            d.grid[0] = 1;
            d.grid[1] = T;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }
        {
            const uint32_t gu_rb = row_bytes_u(*L.gate_exps);
            const uint32_t dn_rb = row_bytes_u(*L.down_exps);
            MoeArgs ma{
                streamed ? mtp_gate_off : L.gate_exps->offset,
                streamed ? mtp_up_off : L.up_exps->offset,
                streamed ? mtp_down_off : L.down_exps->offset,
                (uint64_t)gu_rb * NFF,
                (uint64_t)dn_rb * N,
                gu_rb, dn_rb, N, NFF, T, TOPK, NEXP, NFF / 4, N / 16, 0};

            if (const mtl::Pipeline* rp = (L.gate_inp->type == ref::F32 || L.gate_inp->type == ref::BF16)
                                                ? router_pipe(L.gate_inp->type, n_tok)
                                                : nullptr) {
                GemvArgs ga_rt{N, NEXP, T, row_bytes_u(*L.gate_inp), N, 512};
                mtl::Dispatch d_rr;
                d_rr.p = rp;
                d_rr.args = {mtl::Arg::B(W, L.gate_inp->offset), mtl::Arg::B(xp_emb),
                             mtl::Arg::B(xp_emb, (size_t)4 * 4 * emb_plane), mtl::Arg::B(rt_logits),
                             mtl::Arg::V(ga_rt), mtl::Arg::B(rt_ids), mtl::Arg::B(rt_w),
                             mtl::Arg::B(rt_plan), mtl::Arg::V(ma), mtl::Arg::B(rt_cnt)};
                d_rr.grid[0] = NEXP / 16;
                d_rr.tg[0] = 512;
                ds.push_back(std::move(d_rr));
            } else {
                ds.push_back(disp_gemv(L.gate_inp, xp_emb, emb_plane, rt_logits, n_tok, 512));
                mtl::Dispatch d_rt;
                d_rt.p = &p_moe_route;
                d_rt.args = {mtl::Arg::B(rt_logits), mtl::Arg::B(rt_ids), mtl::Arg::B(rt_w),
                             mtl::Arg::B(rt_plan), mtl::Arg::V(ma)};
                d_rt.grid[0] = 1;
                d_rt.tg[0] = 512;
                ds.push_back(std::move(d_rt));
            }

            const int gk = xkind(L.gate_exps->type);
            const mtl::Buffer& ew = streamed ? mtp_exps_buf : W;
            mtl::Dispatch d_gu;
            d_gu.p = &moe_gu_res_pipe(L.gate_exps->type);
            d_gu.args = {mtl::Arg::B(ew), mtl::Arg::B(xp_emb, (size_t)4 * gk * emb_plane),
                         mtl::Arg::B(moe_h), mtl::Arg::B(rt_plan), mtl::Arg::V(ma),
                         mtl::Arg::B(xp_emb, (size_t)4 * 4 * emb_plane), mtl::Arg::B(rt_plan),
                         mtl::Arg::B(ew), mtl::Arg::B(ew), mtl::Arg::B(ew)};
            d_gu.tg[0] = 128;
            d_gu.indirect = &rt_plan;
            d_gu.ind_off = 4 * 4;
            ds.push_back(std::move(d_gu));

            mtl::Dispatch d_dn;
            d_dn.p = &moe_dn_res_pipe(L.down_exps->type);
            d_dn.args = {mtl::Arg::B(ew), mtl::Arg::B(moe_h), mtl::Arg::B(moe_ye),
                         mtl::Arg::B(rt_plan), mtl::Arg::V(ma), mtl::Arg::B(rt_plan),
                         mtl::Arg::B(rt_plan), mtl::Arg::B(ew), mtl::Arg::B(ew), mtl::Arg::B(ew)};
            d_dn.tg[0] = 128;
            d_dn.indirect = &rt_plan;
            d_dn.ind_off = 4 * 8;
            ds.push_back(std::move(d_dn));
        }
        {
            const uint32_t sh_plane = T * NFF_SH;
            auto d_sg = disp_gemv(L.shexp_gate, xp_emb, emb_plane, sh_g, n_tok);
            auto d_su = disp_gemv(L.shexp_up, xp_emb, emb_plane, sh_u, n_tok);
            auto d_sgi = disp_gemv(L.shexp_gate_inp, xp_emb, emb_plane, sh_gate, n_tok);
            d_sg.barrier = false;
            d_su.barrier = false;
            d_sgi.barrier = false;
            ds.push_back(std::move(d_sg));
            ds.push_back(std::move(d_su));
            ds.push_back(std::move(d_sgi));

            GlueArgs gl_sw{NFF_SH, T, N, TOPK};
            mtl::Dispatch d_sw;
            d_sw.p = &p_swiglu;
            d_sw.args = {mtl::Arg::B(sh_g), mtl::Arg::B(sh_u), mtl::Arg::B(xp_sh), mtl::Arg::V(gl_sw)};
            d_sw.grid[0] = NFF_SH / 16;
            d_sw.grid[1] = T;
            d_sw.tg[0] = 16;
            ds.push_back(std::move(d_sw));

            ds.push_back(disp_gemv(L.shexp_down, xp_sh, sh_plane, sh_y, n_tok));
        }
        {
            GlueArgs gl_c{N, T, N, TOPK};
            mtl::Dispatch d_comb;
            d_comb.p = &p_comb;
            d_comb.args = {mtl::Arg::B(moe_ye), mtl::Arg::B(rt_plan), mtl::Arg::B(sh_y),
                           mtl::Arg::B(sh_gate), mtl::Arg::B(y_buf), mtl::Arg::V(gl_c)};
            d_comb.grid[0] = (N * T + 127) / 128;
            d_comb.tg[0] = 128;
            ds.push_back(std::move(d_comb));

            const uint32_t total_n = N * T;
            mtl::Dispatch d_add;
            d_add.p = &p_add;
            d_add.args = {mtl::Arg::B(x_buf), mtl::Arg::B(y_buf), mtl::Arg::V(total_n)};
            d_add.grid[0] = (total_n + 255) / 256;
            d_add.tg[0] = 256;
            ds.push_back(std::move(d_add));
        }

        // 5. shared_head_norm -> hmtp_buf (+ last row logits if draft)
        {
            Q36NormArgs hna{N, 0u, eps};
            mtl::Dispatch d_hm;
            d_hm.p = &p_rms_f32;
            d_hm.args = {mtl::Arg::B(x_buf), mtl::Arg::B(W, mtp_shnorm->offset),
                         mtl::Arg::B(hmtp_buf), mtl::Arg::V(hna)};
            d_hm.grid[0] = 1;
            d_hm.grid[1] = T;
            d_hm.tg[0] = 256;
            ds.push_back(std::move(d_hm));
        }
        if (draft) {
            Q36NormArgs na{N, N, eps};
            mtl::Dispatch d_rms;
            d_rms.p = &p_rms;
            d_rms.args = {mtl::Arg::B(x_buf, (size_t)4 * (size_t)(n_tok - 1) * N),
                          mtl::Arg::B(W, mtp_shnorm->offset), mtl::Arg::B(xp_emb), mtl::Arg::V(na)};
            d_rms.grid[0] = 1;
            d_rms.tg[0] = 256;
            ds.push_back(std::move(d_rms));

            ds.push_back(disp_gemv(out_w, xp_emb, N, logits_buf, 1));
        }

        cx.run_graph(ds);
        return true;
    }

    void commit_gdn(int slot, int n_keep) {
        std::vector<mtl::Dispatch> ds;
        ds.reserve((size_t)n_gdn);
        for (int l = 0; l < n_layers; ++l) {
            const LayerRef& L = layers[(size_t)l];
            if (L.is_attn) continue;
            const int gi = L.gdn_idx;
            const int par = slots[(size_t)slot].gdn_par[(size_t)gi];
            slots[(size_t)slot].gdn_par[(size_t)gi] = 1 - par;
            const size_t st_off = (size_t)slot * gdn_state_floats * 4;
            const size_t h_in_off = ((size_t)slot * 2 + (size_t)par) * gdn_hist_floats * 4;
            const size_t h_out_off = ((size_t)slot * 2 + (size_t)(1 - par)) * gdn_hist_floats * 4;

            GdnArgs ga{(uint32_t)h_k, (uint32_t)h_v, (uint32_t)conv_ch, 1u, eps, 0u};
            mtl::Dispatch d_gdn;
            d_gdn.p = &gdn_pipe(n_keep, false);
            d_gdn.args = {mtl::Arg::B(gdn_state[(size_t)gi], st_off),
                          mtl::Arg::B(gdn_hist[(size_t)gi], h_in_off),
                          mtl::Arg::B(v_qkv[(size_t)gi]),
                          mtl::Arg::B(v_al[(size_t)gi]),
                          mtl::Arg::B(v_be[(size_t)gi]),
                          mtl::Arg::B(z_buf),
                          mtl::Arg::B(W, L.ssm_conv->offset),
                          mtl::Arg::B(W, L.ssm_dt->offset),
                          mtl::Arg::B(W, L.ssm_a->offset),
                          mtl::Arg::B(W, L.ssm_norm->offset),
                          mtl::Arg::B(xp_gdn),
                          mtl::Arg::V(ga),
                          mtl::Arg::B(gdn_hist[(size_t)gi], h_out_off)};
            d_gdn.grid[0] = (uint32_t)h_v;
            d_gdn.tg[0] = 512;
            ds.push_back(std::move(d_gdn));
        }
        cx.run_graph(ds);
    }
};

Q36MetalEngine::Q36MetalEngine() = default;
Q36MetalEngine::~Q36MetalEngine() = default;

bool Q36MetalEngine::open(const std::string& gguf_path, const Options& opt, std::string& err) {
    impl_ = std::make_unique<Impl>();
    Impl& m = *impl_;
    m.max_ctx = opt.ctx;
    m.n_slots = std::max(1, std::min(kMaxT, opt.n_slots));
    m.ckpt_count = std::max(0, opt.ckpt_count);

    if (!m.f.open(gguf_path, err)) return false;
    auto it_arch = m.f.meta_str.find("general.architecture");
    if (it_arch == m.f.meta_str.end() || it_arch->second != "qwen35moe") {
        err = "unsupported architecture '" + (it_arch == m.f.meta_str.end() ? "" : it_arch->second) +
              "': Q36MetalEngine supports qwen35moe only";
        return false;
    }

    auto req_num = [&](const std::string& key, double& out) -> bool {
        auto it = m.f.meta_num.find(key);
        if (it == m.f.meta_num.end()) {
            err = "missing GGUF metadata key: " + key;
            return false;
        }
        out = it->second;
        return true;
    };

    double v = 0;
    if (!req_num("qwen35moe.block_count", v)) return false;
    const int64_t block_count = (int64_t)v;
    int64_t nextn = 0;
    if (auto it = m.f.meta_num.find("qwen35moe.nextn_predict_layers"); it != m.f.meta_num.end())
        nextn = (int64_t)it->second;
    m.n_layers = block_count - nextn;
    if (m.n_layers <= 0) { err = "invalid qwen35moe trunk layer count"; return false; }

    if (!req_num("qwen35moe.embedding_length", v)) return false; m.n_embd = (int64_t)v;
    if (!req_num("qwen35moe.expert_count", v)) return false; m.n_expert = (int64_t)v;
    if (!req_num("qwen35moe.expert_used_count", v)) return false; m.top_k = (int64_t)v;
    if (!req_num("qwen35moe.expert_feed_forward_length", v)) return false; m.n_ff = (int64_t)v;
    if (!req_num("qwen35moe.expert_shared_feed_forward_length", v)) return false; m.n_ff_sh = (int64_t)v;
    if (!req_num("qwen35moe.full_attention_interval", v)) return false; m.attn_interval = (int64_t)v;
    if (!req_num("qwen35moe.attention.head_count", v)) return false; m.H = (int64_t)v;
    if (!req_num("qwen35moe.attention.head_count_kv", v)) return false; m.HKV = (int64_t)v;
    if (!req_num("qwen35moe.attention.key_length", v)) return false; m.D = (int64_t)v;
    if (!req_num("qwen35moe.attention.layer_norm_rms_epsilon", v)) return false; m.eps = (float)v;
    if (!req_num("qwen35moe.rope.dimension_count", v)) return false; m.n_rot = (int64_t)v;
    if (!req_num("qwen35moe.rope.freq_base", v)) return false; m.rope_base = (float)v;
    if (!req_num("qwen35moe.ssm.group_count", v)) return false; m.h_k = (int64_t)v;
    if (!req_num("qwen35moe.ssm.time_step_rank", v)) return false; m.h_v = (int64_t)v;
    if (!req_num("qwen35moe.ssm.state_size", v)) return false; m.ssm_s = (int64_t)v;
    if (!req_num("qwen35moe.ssm.inner_size", v)) return false; m.zv = (int64_t)v;
    if (!req_num("qwen35moe.ssm.conv_kernel", v)) return false; m.conv_k = (int64_t)v;
    m.conv_ch = 2 * m.ssm_s * m.h_k + m.ssm_s * m.h_v;

    if (m.D != 256 || m.ssm_s != 128 || m.conv_k != 4) {
        err = "unsupported qwen35moe head_dim / ssm_state_size / conv_kernel";
        return false;
    }
    if (m.n_expert > 512 || m.top_k > 16) {
        err = "unsupported expert_count / top_k";
        return false;
    }

    auto req_ten = [&](const std::string& name) -> const gguf::Tensor* {
        const gguf::Tensor* t = m.f.find(name);
        if (!t) err = "missing GGUF tensor: " + name;
        return t;
    };

    m.tok_embd = req_ten("token_embd.weight");
    m.out_norm = req_ten("output_norm.weight");
    m.out_w = req_ten("output.weight");
    if (!m.tok_embd || !m.out_norm || !m.out_w) return false;
    m.n_vocab = (int64_t)m.out_w->shape[1];

    m.layers.resize((size_t)m.n_layers);
    m.n_gdn = 0;
    m.n_attn = 0;
    for (int l = 0; l < m.n_layers; ++l) {
        LayerRef& L = m.layers[(size_t)l];
        const std::string b = "blk." + std::to_string(l) + ".";
        L.is_attn = (l % m.attn_interval) == (m.attn_interval - 1);
        L.attn_norm = req_ten(b + "attn_norm.weight");
        L.post_norm = req_ten(b + "post_attention_norm.weight");
        if (!L.attn_norm || !L.post_norm) return false;

        if (!L.is_attn) {
            L.gdn_idx = m.n_gdn++;
            L.attn_qkv = req_ten(b + "attn_qkv.weight");
            L.attn_gate = req_ten(b + "attn_gate.weight");
            L.ssm_alpha = req_ten(b + "ssm_alpha.weight");
            L.ssm_beta = req_ten(b + "ssm_beta.weight");
            L.ssm_conv = req_ten(b + "ssm_conv1d.weight");
            L.ssm_dt = req_ten(b + "ssm_dt.bias");
            L.ssm_a = req_ten(b + "ssm_a");
            L.ssm_norm = req_ten(b + "ssm_norm.weight");
            L.ssm_out = req_ten(b + "ssm_out.weight");
            if (!L.attn_qkv || !L.attn_gate || !L.ssm_alpha || !L.ssm_beta ||
                !L.ssm_conv || !L.ssm_dt || !L.ssm_a || !L.ssm_norm || !L.ssm_out)
                return false;
        } else {
            L.attn_idx = m.n_attn++;
            L.attn_q = req_ten(b + "attn_q.weight");
            L.attn_k = req_ten(b + "attn_k.weight");
            L.attn_v = req_ten(b + "attn_v.weight");
            L.attn_q_norm = req_ten(b + "attn_q_norm.weight");
            L.attn_k_norm = req_ten(b + "attn_k_norm.weight");
            L.attn_output = req_ten(b + "attn_output.weight");
            if (!L.attn_q || !L.attn_k || !L.attn_v || !L.attn_q_norm || !L.attn_k_norm || !L.attn_output)
                return false;
        }

        L.gate_inp = req_ten(b + "ffn_gate_inp.weight");
        L.gate_exps = req_ten(b + "ffn_gate_exps.weight");
        L.up_exps = req_ten(b + "ffn_up_exps.weight");
        L.down_exps = req_ten(b + "ffn_down_exps.weight");
        L.shexp_gate_inp = req_ten(b + "ffn_gate_inp_shexp.weight");
        L.shexp_gate = req_ten(b + "ffn_gate_shexp.weight");
        L.shexp_up = req_ten(b + "ffn_up_shexp.weight");
        L.shexp_down = req_ten(b + "ffn_down_shexp.weight");
        if (!L.gate_inp || !L.gate_exps || !L.up_exps || !L.down_exps ||
            !L.shexp_gate_inp || !L.shexp_gate || !L.shexp_up || !L.shexp_down)
            return false;
    }

    if (opt.mtp_drafts > 0 && nextn > 0) {
        const std::string b = "blk." + std::to_string(m.n_layers) + ".";
        m.mtp_eh_proj = m.f.find(b + "nextn.eh_proj.weight");
        m.mtp_enorm = m.f.find(b + "nextn.enorm.weight");
        m.mtp_hnorm = m.f.find(b + "nextn.hnorm.weight");
        m.mtp_shnorm = m.f.find(b + "nextn.shared_head_norm.weight");
        LayerRef& M = m.mtp_layer;
        M.is_attn = true;
        M.attn_idx = m.n_attn;
        M.attn_norm = m.f.find(b + "attn_norm.weight");
        M.post_norm = m.f.find(b + "post_attention_norm.weight");
        M.attn_q = m.f.find(b + "attn_q.weight");
        M.attn_k = m.f.find(b + "attn_k.weight");
        M.attn_v = m.f.find(b + "attn_v.weight");
        M.attn_q_norm = m.f.find(b + "attn_q_norm.weight");
        M.attn_k_norm = m.f.find(b + "attn_k_norm.weight");
        M.attn_output = m.f.find(b + "attn_output.weight");
        M.gate_inp = m.f.find(b + "ffn_gate_inp.weight");
        M.gate_exps = m.f.find(b + "ffn_gate_exps.weight");
        M.up_exps = m.f.find(b + "ffn_up_exps.weight");
        M.down_exps = m.f.find(b + "ffn_down_exps.weight");
        M.shexp_gate_inp = m.f.find(b + "ffn_gate_inp_shexp.weight");
        M.shexp_gate = m.f.find(b + "ffn_gate_shexp.weight");
        M.shexp_up = m.f.find(b + "ffn_up_shexp.weight");
        M.shexp_down = m.f.find(b + "ffn_down_shexp.weight");
        if (m.mtp_eh_proj && m.mtp_enorm && m.mtp_hnorm && m.mtp_shnorm &&
            M.attn_norm && M.post_norm && M.attn_q && M.attn_k && M.attn_v &&
            M.attn_q_norm && M.attn_k_norm && M.attn_output &&
            M.gate_inp && M.gate_exps && M.up_exps && M.down_exps &&
            M.shexp_gate_inp && M.shexp_gate && M.shexp_up && M.shexp_down) {
            m.has_mtp = true;
        }
    }

    const char* env_s = std::getenv("STRATA_METAL_STREAM");
    if (opt.stream || opt.arena_gb > 0.0) m.streamed = true;
    else if (env_s && std::string(env_s) == "1") m.streamed = true;
    else if (env_s && std::string(env_s) == "0") m.streamed = false;
    else m.streamed = strata::metal_engine::ExpertStream::wanted(m.f, m.cx);

    if (m.streamed) {
        if (!m.stream.open(m.f, (int)m.n_layers, err)) return false;
        if (m.has_mtp) {
            auto align256 = [](size_t x) -> size_t { return (x + 255u) & ~255ull; };
            const size_t g_sz = (size_t)m.mtp_layer.gate_exps->bytes;
            const size_t u_sz = (size_t)m.mtp_layer.up_exps->bytes;
            const size_t d_sz = (size_t)m.mtp_layer.down_exps->bytes;
            m.mtp_gate_off = 0;
            m.mtp_up_off = align256(g_sz);
            m.mtp_down_off = align256(m.mtp_up_off + u_sz);
            m.mtp_exps_buf = m.cx.alloc(m.mtp_down_off + d_sz);
            std::memcpy(m.mtp_exps_buf.as<uint8_t>() + m.mtp_gate_off,
                        (const uint8_t*)m.f.map + m.mtp_layer.gate_exps->offset, g_sz);
            std::memcpy(m.mtp_exps_buf.as<uint8_t>() + m.mtp_up_off,
                        (const uint8_t*)m.f.map + m.mtp_layer.up_exps->offset, u_sz);
            std::memcpy(m.mtp_exps_buf.as<uint8_t>() + m.mtp_down_off,
                        (const uint8_t*)m.f.map + m.mtp_layer.down_exps->offset, d_sz);
        }
        std::vector<std::string> skip;
        for (const auto& kv : m.f.tensors) {
            if (kv.first.rfind("blk.", 0) == 0) {
                const size_t dot = kv.first.find('.', 4);
                if (dot != std::string::npos) {
                    const int blk_idx = std::atoi(kv.first.substr(4, dot - 4).c_str());
                    if (blk_idx > m.n_layers || (!m.has_mtp && blk_idx >= m.n_layers)) {
                        skip.push_back(kv.first);
                    }
                }
            }
        }
        if (!m.stream.load_dense(m.f, m.cx, skip, m.W, err)) return false;
        m.dense_bytes = m.W.bytes;
        if (m.f.map && m.f.map_bytes > 0) {
            ::munmap(m.f.map, m.f.map_bytes);
            m.f.map = nullptr;
            m.f.map_bytes = 0;
        }
    } else {
        size_t max_span = 0;
        for (const auto& kv : m.f.tensors) max_span = std::max(max_span, (size_t)kv.second.bytes);
        m.W = m.cx.wrap(m.f.map, m.f.map_bytes, max_span);
    }

    m.lib_ops = m.cx.compile({"q36_ops.metal"}, "#define NO_GGML_COMMON 1\n");
    m.lib_gdn = m.cx.compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    m.lib_attn = m.cx.compile({"q36_attn.metal"}, "#define NO_GGML_COMMON 1\n");
    m.lib_moe = m.cx.compile(
        {"quant.metal", "units.metal", "moe.metal", "gemv.metal", "router.metal", "glue.metal"},
        "#define GEO_TOPK " + std::to_string(m.top_k) + "\n#define STRATA_KQ_DOWN 1\n");

    m.p_embed = m.cx.pipeline(m.lib_ops, "q36_embed_row");
    m.p_embed_batch = m.cx.pipeline(m.lib_ops, "q36_embed_batch");
    m.p_rms = m.cx.pipeline(m.lib_ops, "q36_rms_prep");
    m.p_add_rms = m.cx.pipeline(m.lib_ops, "q36_add_rms_prep");
    m.p_add = m.cx.pipeline(m.lib_ops, "q36_add");
    m.p_rms_f16 = m.cx.pipeline(m.lib_ops, "q36_rms_f16");
    m.p_add_rms_f16 = m.cx.pipeline(m.lib_ops, "q36_add_rms_f16");
    m.p_rms_f32 = m.cx.pipeline(m.lib_ops, "q36_rms_f32");
    m.p_mtp_cat = m.cx.pipeline(m.lib_ops, "q36_mtp_concat_prep");
    m.p_attn_prep_batch = m.cx.pipeline(m.lib_attn, "q36_attn_prep_batch");
    m.p_attn_dec_batch = m.cx.pipeline(m.lib_attn, "q36_attn_decode_batch");
    m.p_swiglu = m.cx.pipeline(m.lib_moe, "swiglu_prep");
    m.p_comb = m.cx.pipeline(m.lib_moe, "moe_combine_sh");
    m.p_moe_route = m.cx.pipeline(m.lib_moe, "moe_route");

    mtl::require_fit(m.p_embed_batch, 256, "q36_embed_batch");
    mtl::require_fit(m.p_rms, 256, "q36_rms_prep");
    mtl::require_fit(m.p_add_rms, 256, "q36_add_rms_prep");
    mtl::require_fit(m.p_rms_f16, 256, "q36_rms_f16");
    mtl::require_fit(m.p_add_rms_f16, 256, "q36_add_rms_f16");
    mtl::require_fit(m.p_rms_f32, 256, "q36_rms_f32");
    mtl::require_fit(m.p_mtp_cat, 256, "q36_mtp_concat_prep");
    mtl::require_fit(m.p_attn_prep_batch, 256, "q36_attn_prep_batch");
    mtl::require_fit(m.p_attn_dec_batch, 256, "q36_attn_decode_batch");
    // Pre-compile primary GDN pipelines cleanly at startup.
    m.gdn_pipe(1, false);
    if (m.n_slots > 1) m.gdn_pipe(1, true);

    bool pg_ok = (m.n_embd % 128 == 0) && (m.zv % 128 == 0) && ((m.H * m.D) % 128 == 0) &&
                 (m.n_ff % 128 == 0) && (m.n_ff_sh % 128 == 0) &&
                 (m.n_expert == 256 || m.n_expert == 512) && (m.h_v % 2 == 0);
    bool kq_dn = false;
    size_t max_layer_dense_elems = 0;
    auto track_dense = [&](const gguf::Tensor* t) {
        if (!t) return;
        if (!dq_supported(t->type)) pg_ok = false;
        const size_t n_in = (size_t)t->shape[0];
        const size_t n_out = t->shape.size() > 1 ? (size_t)t->shape[1] : 1u;
        if (n_in % (size_t)unit_of(t->type) != 0) pg_ok = false;
        max_layer_dense_elems = std::max(max_layer_dense_elems, n_in * n_out);
    };
    for (const LayerRef& L : m.layers) {
        if (!L.is_attn) {
            track_dense(L.attn_qkv);
            track_dense(L.attn_gate);
            track_dense(L.ssm_alpha);
            track_dense(L.ssm_beta);
            track_dense(L.ssm_out);
        } else {
            track_dense(L.attn_q);
            track_dense(L.attn_k);
            track_dense(L.attn_v);
            track_dense(L.attn_output);
        }
        track_dense(L.gate_inp);
        track_dense(L.shexp_gate);
        track_dense(L.shexp_up);
        track_dense(L.shexp_gate_inp);
        track_dense(L.shexp_down);
        if (L.gate_exps->type != L.up_exps->type) pg_ok = false;
        const auto gu_t = strata::metal_engine::gu2_tile(L.gate_exps->type);
        const auto dn_t = strata::metal_engine::dn2_tile(L.down_exps->type);
        if (m.n_ff % gu_t.tn != 0 || m.n_embd % dn_t.tn != 0) pg_ok = false;
        const int dt = L.down_exps->type;
        if (dt == ref::Q2_K || dt == ref::Q3_K || dt == ref::Q4_K || dt == ref::Q5_K ||
            dt == ref::Q6_K || dt == ref::IQ3_S || dt == ref::IQ4_XS) {
            kq_dn = true;
        }
    }
    m.prompt_gemm_ready = pg_ok;
    if (m.prompt_gemm_ready) {
        const std::string mm_pre = "#define GEO_TOPK " + std::to_string(m.top_k) + "\n" +
                                   (kq_dn ? "#define STRATA_KQ_DOWN 1\n" : "");
        m.lib_mm = m.cx.compile(
            {"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "gr.metal", "prompt.metal"},
            mm_pre);
        m.lib_gdn_p = m.cx.compile({"gdn_prompt.metal"}, "#define NO_GGML_COMMON 1\n");

        m.p_act_f16 = m.cx.pipeline(m.lib_mm, "act_f16");
        m.p_gemm64 = m.cx.pipeline(m.lib_mm, "gemm_f16_64_64_4");
        m.p_gdn_pre = m.cx.pipeline(m.lib_gdn_p, "gdn_pre");
        m.p_gdn_scan = m.cx.pipeline(m.lib_gdn_p, "gdn_scan_4");
        m.p_gdn_post = m.cx.pipeline(m.lib_gdn_p, "gdn_post", mtl::Constants().b(32, true));
        m.p_mm_clear = m.cx.pipeline(m.lib_mm, "mm_clear");
        m.p_mm_hist = m.cx.pipeline(m.lib_mm, "mm_hist");
        m.p_mm_scan = m.cx.pipeline(m.lib_mm, "mm_scan");
        m.p_mm_place = m.cx.pipeline(m.lib_mm, "mm_place");
        m.p_mm_gather = m.cx.pipeline(m.lib_mm, "mm_gather");
        m.p_mm_comb = m.cx.pipeline(m.lib_mm, "mm_combine");
        m.p_route_topk = m.cx.pipeline(m.lib_mm, "route_topk");
        m.p_swiglu_h = m.cx.pipeline(m.lib_mm, "swiglu_h");
        m.p_moe_out = m.cx.pipeline(m.lib_mm, "moe_out");

        mtl::require_fit(m.p_gemm64, 128, "gemm_f16_64_64_4");
        mtl::require_fit(m.p_gdn_pre, 256, "gdn_pre");
        mtl::require_fit(m.p_gdn_scan, 128, "gdn_scan_4");
        mtl::require_fit(m.p_gdn_post, 128, "gdn_post");
        mtl::require_fit(m.p_mm_hist, 1024, "mm_hist");
        mtl::require_fit(m.p_mm_scan, (int)m.n_expert, "mm_scan");
        mtl::require_fit(m.p_mm_place, 1024, "mm_place");
        mtl::require_fit(m.p_route_topk, 256, "route_topk");

        for (const LayerRef& L : m.layers) {
            if (!L.is_attn) {
                m.dq_pipe(L.attn_qkv->type);
                m.dq_pipe(L.attn_gate->type);
                m.dq_pipe(L.ssm_alpha->type);
                m.dq_pipe(L.ssm_beta->type);
                m.dq_pipe(L.ssm_out->type);
            } else {
                m.dq_pipe(L.attn_q->type);
                m.dq_pipe(L.attn_k->type);
                m.dq_pipe(L.attn_v->type);
                m.dq_pipe(L.attn_output->type);
            }
            m.dq_pipe(L.gate_inp->type);
            m.dq_pipe(L.shexp_gate->type);
            m.dq_pipe(L.shexp_up->type);
            m.dq_pipe(L.shexp_gate_inp->type);
            m.dq_pipe(L.shexp_down->type);
            m.mm_gu2_pipe(L.gate_exps->type);
            m.mm_dn2_pipe(L.down_exps->type);
        }
    }

    m.gdn_state_floats = (size_t)m.ssm_s * m.h_v * m.ssm_s;
    m.gdn_hist_floats = (size_t)m.conv_ch * 3;
    m.gdn_state.resize((size_t)m.n_gdn);
    m.gdn_hist.resize((size_t)m.n_gdn);
    for (int i = 0; i < m.n_gdn; ++i) {
        m.gdn_state[(size_t)i] = m.cx.alloc(4 * (size_t)m.n_slots * m.gdn_state_floats);
        m.gdn_hist[(size_t)i] = m.cx.alloc(4 * (size_t)m.n_slots * 2 * m.gdn_hist_floats);
    }

    const int n_kv_layers = m.n_attn + (m.has_mtp ? 1 : 0);
    m.kv_k.resize((size_t)n_kv_layers);
    m.kv_v.resize((size_t)n_kv_layers);
    for (int i = 0; i < n_kv_layers; ++i) {
        m.kv_k[(size_t)i] = m.cx.alloc(4 * (size_t)m.n_slots * m.HKV * m.max_ctx * m.D);
        m.kv_v[(size_t)i] = m.cx.alloc(4 * (size_t)m.n_slots * m.HKV * m.max_ctx * m.D);
    }

    m.slots.resize((size_t)m.n_slots);
    for (int s = 0; s < m.n_slots; ++s) {
        m.slots[(size_t)s].gdn_par.assign((size_t)m.n_gdn, 0);
        if (m.has_mtp) m.slots[(size_t)s].hlast.assign((size_t)m.n_embd, 0.0f);
    }

    auto xp_bytes = [](int64_t n) -> size_t { return 4 * (size_t)(4 * n + n / 16) * kMaxT; };
    if (m.has_mtp) {
        m.xp_2n = m.cx.alloc(xp_bytes(2 * m.n_embd));
        m.hmain_buf = m.cx.alloc(4 * (size_t)m.n_embd * kPromptT);
        m.hmtp_buf = m.cx.alloc(4 * (size_t)m.n_embd * kMaxT);
        m.mtp_seeds_buf = m.cx.alloc(4 * (size_t)m.n_embd * kPromptT);
        m.v_qkv.resize((size_t)m.n_gdn);
        m.v_al.resize((size_t)m.n_gdn);
        m.v_be.resize((size_t)m.n_gdn);
        for (int gi = 0; gi < m.n_gdn; ++gi) {
            m.v_qkv[(size_t)gi] = m.cx.alloc(4 * (size_t)m.conv_ch * kMaxT);
            m.v_al[(size_t)gi] = m.cx.alloc(4 * (size_t)m.h_v * kMaxT);
            m.v_be[(size_t)gi] = m.cx.alloc(4 * (size_t)m.h_v * kMaxT);
        }
    }
    const size_t max_mm_rows = m.prompt_gemm_ready ? (size_t)kPromptT * m.top_k + (size_t)m.n_expert * 16 : 0;
    m.x_buf = m.cx.alloc(4 * (size_t)m.n_embd * kPromptT);
    m.y_buf = m.cx.alloc(4 * (size_t)m.n_embd * kPromptT);
    m.xp_emb = m.cx.alloc(std::max<size_t>(xp_bytes(m.n_embd), (size_t)4 * kPromptT * (size_t)m.n_embd));
    m.xp_gdn = m.cx.alloc(xp_bytes(m.zv));
    m.xp_att = m.cx.alloc(std::max<size_t>(xp_bytes(m.H * m.D), (size_t)4 * kPromptT * (size_t)(m.H * m.D)));
    m.xp_sh = m.cx.alloc(xp_bytes(m.n_ff_sh));

    m.qkv_buf = m.cx.alloc(4 * (size_t)m.conv_ch * kPromptT);
    m.z_buf = m.cx.alloc(4 * (size_t)m.zv * kPromptT);
    m.al_buf = m.cx.alloc(4 * (size_t)m.h_v * kPromptT);
    m.be_buf = m.cx.alloc(4 * (size_t)m.h_v * kPromptT);

    m.qg_buf = m.cx.alloc(4 * (size_t)m.H * 2 * m.D * kPromptT);
    m.kx_buf = m.cx.alloc(4 * (size_t)m.HKV * m.D * kPromptT);
    m.vx_buf = m.cx.alloc(4 * (size_t)m.HKV * m.D * kPromptT);
    m.q_buf = m.cx.alloc(4 * (size_t)m.H * m.D * kPromptT);
    m.gate_buf = m.cx.alloc(4 * (size_t)m.H * m.D * kPromptT);

    m.rt_logits = m.cx.alloc(4 * 512 * kPromptT);
    m.rt_ids = m.cx.alloc(4 * (size_t)(m.streamed ? m.n_layers : 1) * kPromptT * m.top_k);
    m.rt_w = m.cx.alloc(4 * (size_t)kPromptT * m.top_k);
    m.rt_plan = m.cx.alloc(4 * P_INTS);
    m.rt_cnt = m.cx.alloc(16);
    m.moe_h = m.cx.alloc(4 * (size_t)kMaxT * m.top_k * m.n_ff);
    m.moe_ye = m.cx.alloc(4 * std::max<size_t>((size_t)kMaxT * (size_t)m.top_k, max_mm_rows) * (size_t)m.n_embd);
    m.sh_g = m.cx.alloc(4 * (size_t)m.n_ff_sh * kPromptT);
    m.sh_u = m.cx.alloc(4 * (size_t)m.n_ff_sh * kPromptT);
    m.sh_y = m.cx.alloc(4 * (size_t)m.n_embd * kPromptT);
    m.sh_gate = m.cx.alloc(16 * kPromptT);
    m.logits_buf = m.cx.alloc(4 * (size_t)m.n_vocab * kMaxT);

    if (m.prompt_gemm_ready) {
        const size_t xh_max = (size_t)std::max({m.n_embd, m.zv, m.H * m.D, m.n_ff_sh});
        const size_t n_ent_max = (size_t)kPromptT * m.top_k;
        const size_t n_blk_max = (n_ent_max + 1023) / 1024;
        m.xp_f16 = m.cx.alloc(2 * (size_t)kPromptT * xh_max);
        m.mm_wh = m.cx.alloc(2 * std::max<size_t>(1, max_layer_dense_elems));
        m.mm_gdn_qn = m.cx.alloc(4 * (size_t)kPromptT * m.h_k * 128);
        m.mm_gdn_kn = m.cx.alloc(4 * (size_t)kPromptT * m.h_k * 128);
        m.mm_gdn_vc = m.cx.alloc(4 * (size_t)kPromptT * m.h_v * 128);
        m.mm_gdn_gb = m.cx.alloc(8 * (size_t)kPromptT * m.h_v);
        m.mm_gdn_kq = m.cx.alloc(4 * (size_t)kPromptT * m.h_k);
        m.mm_gdn_o = m.cx.alloc(4 * (size_t)kPromptT * m.h_v * 128);
        m.mm_bcnt = m.cx.alloc(4 * n_blk_max * (size_t)m.n_expert);
        m.mm_boff = m.cx.alloc(4 * n_blk_max * (size_t)m.n_expert);
        m.mm_eoff = m.cx.alloc(4 * (size_t)m.n_expert);
        m.mm_tiles = m.cx.alloc(8 * (max_mm_rows / 16 + 4));
        m.mm_meta = m.cx.alloc(64);
        m.mm_stok = m.cx.alloc(4 * max_mm_rows);
        m.mm_sw = m.cx.alloc(4 * max_mm_rows);
        m.mm_pos = m.cx.alloc(4 * n_ent_max);
        m.mm_xs = m.cx.alloc(2 * max_mm_rows * (size_t)m.n_embd);
        m.mm_hs = m.cx.alloc(2 * max_mm_rows * (size_t)m.n_ff);
    }

    m.tok_ids_buf = m.cx.alloc(4 * kPromptT);
    m.cell_row_buf = m.cx.alloc(4 * kPromptT);
    m.row_meta_buf = m.cx.alloc(sizeof(Q36RowMeta) * kPromptT);
    m.rinfo_buf = m.cx.alloc(4 * (size_t)std::max(1, m.n_gdn) * (8 + 4 * kMaxT));
    m.img_rows_buf = m.cx.alloc(4 * (size_t)m.n_embd);

    if (m.streamed) {
        m.pred_logits = m.cx.alloc(4 * (size_t)m.n_layers * kPromptT * 512);
        m.exp_counts.assign((size_t)m.n_layers * (size_t)m.n_expert, 0);
        uint64_t budget = 0;
        if (opt.arena_gb > 0.0) {
            budget = (uint64_t)(opt.arena_gb * (double)(1ULL << 30));
        } else {
            uint64_t phys = 0;
            size_t plen = sizeof(phys);
            sysctlbyname("hw.memsize", &phys, &plen, nullptr, 0);
            strata::metal_engine::ArenaMem am;
            am.phys = (double)phys;
            am.working_set = (double)m.cx.recommended_working_set();
            am.allocated = (double)m.cx.allocated();
            vm_statistics64_data_t vs{};
            mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
            if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &cnt) == KERN_SUCCESS) {
                const double pg = (double)vm_page_size;
                am.available = strata::metal_engine::arena_avail(
                    vs.free_count * pg, vs.purgeable_count * pg, vs.inactive_count * pg,
                    vs.speculative_count * pg, vs.external_page_count * pg);
            } else {
                am.available = (double)phys - (double)m.cx.allocated();
            }
            if (const char* e = std::getenv("STRATA_METAL_ARENA_GB")) am.arena_gb = std::atof(e);
            if (const char* e = std::getenv("STRATA_METAL_KEEP_FREE_GB")) am.keep_gb = std::atof(e);
            budget = (uint64_t)strata::metal_engine::arena_budget(am);
        }
        std::string prof = opt.profile_path;
        if (prof.empty()) {
            if (const char* e = std::getenv("STRATA_METAL_EXPERT_PROFILE")) prof = e;
        }
        // min_slots=64 accommodates a full T=8 chunk activating up to 8*8 distinct experts in a layer.
        if (!m.stream.make_arena(m.cx, budget, prof, 64, err)) return false;
    }

    if (!opt.silent) {
        std::fprintf(stderr, "strata-q36-metal: %s (%zu layers, hidden %lld, %lld experts top-%lld, %.2f GiB, slots=%d, mode=%s) ready\n",
                     m.cx.device_name().c_str(), m.layers.size(), (long long)m.n_embd,
                     (long long)m.n_expert, (long long)m.top_k, m.f.file_bytes / double(1 << 30),
                     m.n_slots, m.streamed ? "stream" : "resident");
    }
    return true;
}

void Q36MetalEngine::reset() {
    reset_slot(0);
}

void Q36MetalEngine::reset_slot(int slot) {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return;
    impl_->reset_slot_state(slot);
}

bool Q36MetalEngine::forward(int32_t token, const float*& logits_out, std::string& err) {
    return forward_chunk(0, &token, 1, nullptr, nullptr, 0, nullptr, true, logits_out, err);
}

bool Q36MetalEngine::forward_chunk(int slot, const int32_t* tokens, int n_tok,
                                   const int32_t* mrope_3d, const float* img_rows, int n_img_rows,
                                   const int32_t* cell_row, bool need_logits,
                                   const float*& logits_out, std::string& err) {
    Impl& m = *impl_;
    if (slot < 0 || slot >= m.n_slots) { err = "invalid slot"; return false; }
    if (n_tok <= 0) { err = "n_tok must be positive"; return false; }
    SlotState& ss = m.slots[(size_t)slot];
    if (ss.cur_pos + n_tok > m.max_ctx) { err = "context limit reached"; return false; }

    for (int i = 0; i < n_tok; ++i) {
        const int cr = cell_row ? cell_row[i] : -1;
        if (cr >= 0) {
            if (!img_rows || cr >= n_img_rows) { err = "invalid vision cell_row"; return false; }
        } else if (tokens[i] < 0 || tokens[i] >= m.n_vocab) {
            err = "token id out of range";
            return false;
        }
    }

    if (img_rows && n_img_rows > 0) {
        m.ensure_img_buf(n_img_rows);
        std::memcpy(m.img_rows_buf.as<float>(), img_rows,
                    (size_t)n_img_rows * (size_t)m.n_embd * sizeof(float));
    }

    int off = 0;
    while (off < n_tok) {
        const int rem = n_tok - off;
        const bool use_pg = m.can_use_prompt_gemm(std::min(kPromptT, rem));
        const int T = use_pg ? std::min(kPromptT, rem) : std::min(kMaxT, rem);
        const bool last_chunk = (off + T == n_tok);

        int32_t* tid_p = m.tok_ids_buf.as<int32_t>();
        int32_t* cr_p = m.cell_row_buf.as<int32_t>();
        Q36RowMeta* meta_p = m.row_meta_buf.as<Q36RowMeta>();

        for (int t = 0; t < T; ++t) {
            const int idx = off + t;
            tid_p[t] = tokens[idx];
            cr_p[t] = cell_row ? cell_row[idx] : -1;
            const int kvp = ss.cur_pos + t;
            const int32_t pt = mrope_3d ? mrope_3d[3 * idx + 0] : (kvp + ss.mrope_delta);
            const int32_t ph = mrope_3d ? mrope_3d[3 * idx + 1] : pt;
            const int32_t pw = mrope_3d ? mrope_3d[3 * idx + 2] : pt;
            meta_p[t] = {pt, ph, pw, kvp, (uint32_t)(kvp + 1), (uint32_t)slot, 0u, 0u};
        }

        if (use_pg) {
            if (!m.run_prompt_gemm_chunk(slot, T, last_chunk && need_logits, err)) {
                return false;
            }
        } else {
            const int slot_arr[1] = {slot};
            if (!m.run_layers_batch(T, slot_arr, false, last_chunk && need_logits, T - 1, err)) {
                return false;
            }
        }
        if (m.has_mtp) {
            const size_t N = (size_t)m.n_embd;
            float* dst_seeds = m.mtp_seeds_buf.as<float>();
            std::memcpy(dst_seeds, ss.hlast.data(), N * sizeof(float));
            if (T > 1) {
                std::memcpy(dst_seeds + N, m.hmain_buf.as<float>(), (size_t)(T - 1) * N * sizeof(float));
            }
            for (int sub_off = 0; sub_off < T; sub_off += kMaxT) {
                const int sub_T = std::min(kMaxT, T - sub_off);
                if (!m.forward_mtp_chunk(
                        slot, sub_T, ss.cur_pos + sub_off, tokens + off + sub_off,
                        mrope_3d ? mrope_3d + 3 * (off + sub_off) : nullptr,
                        cell_row ? cell_row + off + sub_off : nullptr,
                        m.mtp_seeds_buf, (size_t)sub_off * N * sizeof(float), false, err)) {
                    return false;
                }
            }
            std::memcpy(ss.hlast.data(), m.hmain_buf.as<float>() + (size_t)(T - 1) * N, N * sizeof(float));
        }
        ss.cur_pos += T;
        off += T;
    }

    logits_out = (need_logits ? m.logits_buf.as<float>() : nullptr);
    return true;
}

bool Q36MetalEngine::step_slots(const SlotStep* steps, int n_steps,
                                std::vector<const float*>& logits_out, std::string& err) {
    Impl& m = *impl_;
    if (n_steps <= 0 || n_steps > m.n_slots) { err = "invalid n_steps"; return false; }

    int32_t* tid_p = m.tok_ids_buf.as<int32_t>();
    int32_t* cr_p = m.cell_row_buf.as<int32_t>();
    Q36RowMeta* meta_p = m.row_meta_buf.as<Q36RowMeta>();
    int slot_arr[kMaxT]{};

    for (int r = 0; r < n_steps; ++r) {
        const int s = steps[r].slot;
        if (s < 0 || s >= m.n_slots) { err = "invalid slot in step_slots"; return false; }
        for (int u = 0; u < r; ++u) {
            if (steps[u].slot == s) { err = "duplicate slot in step_slots"; return false; }
        }
        const SlotState& ss = m.slots[(size_t)s];
        if (ss.cur_pos >= m.max_ctx) { err = "context limit reached"; return false; }
        if (steps[r].token < 0 || steps[r].token >= m.n_vocab) { err = "token id out of range"; return false; }

        slot_arr[r] = s;
        tid_p[r] = steps[r].token;
        cr_p[r] = -1;
        const int32_t mp = ss.cur_pos + ss.mrope_delta;
        meta_p[r] = {mp, mp, mp, ss.cur_pos, (uint32_t)(ss.cur_pos + 1), (uint32_t)s, 0u, 0u};
    }

    for (int gi = 0; gi < m.n_gdn; ++gi) {
        uint32_t* ri = m.rinfo_buf.as<uint32_t>() + (size_t)gi * (8 + 4 * kMaxT);
        ri[5] = (uint32_t)m.gdn_state_floats;
        ri[6] = (uint32_t)m.gdn_hist_floats;
        for (int r = 0; r < n_steps; ++r) {
            const int s = slot_arr[r];
            ri[8 + 4 * r + 0] = (uint32_t)s;
            ri[8 + 4 * r + 1] = (uint32_t)m.slots[(size_t)s].cur_pos;
            ri[8 + 4 * r + 2] = (uint32_t)m.slots[(size_t)s].gdn_par[(size_t)gi];
            ri[8 + 4 * r + 3] = 0u;
        }
    }

    if (!m.run_layers_batch(n_steps, slot_arr, true, true, -1, err)) {
        return false;
    }

    logits_out.resize((size_t)n_steps);
    float* lg_base = m.logits_buf.as<float>();
    for (int r = 0; r < n_steps; ++r) {
        if (m.has_mtp) {
            std::memcpy(m.slots[(size_t)slot_arr[r]].hlast.data(),
                        m.hmain_buf.as<float>() + (size_t)r * (size_t)m.n_embd,
                        (size_t)m.n_embd * sizeof(float));
        }
        m.slots[(size_t)slot_arr[r]].cur_pos++;
        logits_out[(size_t)r] = lg_base + (size_t)r * (size_t)m.n_vocab;
    }
    return true;
}

bool Q36MetalEngine::has_mtp() const {
    return impl_ && impl_->has_mtp;
}

bool Q36MetalEngine::spec_step(int slot, int32_t x_tok, const std::vector<int32_t>& fix, int n_drafts,
                               std::vector<int32_t>& drafts, std::vector<int32_t>& picks,
                               int& accepted, std::string& err) {
    Impl& m = *impl_;
    accepted = 0;
    drafts.clear();
    picks.clear();
    if (!m.has_mtp) { err = "MTP not loaded"; return false; }
    if (slot < 0 || slot >= m.n_slots) { err = "invalid slot in spec_step"; return false; }
    if (n_drafts < 0 || n_drafts + 1 > kMaxT) { err = "invalid n_drafts in spec_step"; return false; }
    if ((int)fix.size() + 1 > kMaxT) { err = "invalid fix size in spec_step"; return false; }
    SlotState& ss = m.slots[(size_t)slot];
    if (ss.cur_pos + n_drafts + 1 > m.max_ctx) { err = "context limit reached"; return false; }
    if (x_tok < 0 || x_tok >= m.n_vocab) { err = "token id out of range"; return false; }
    for (int32_t ft : fix) {
        if (ft < 0 || ft >= m.n_vocab) { err = "fix token id out of range"; return false; }
    }

    auto argmax_fn = [](const float* p, int64_t n) -> int32_t {
        int32_t best = 0;
        float bv = p[0];
        for (int64_t i = 1; i < n; ++i) {
            if (p[i] > bv) { bv = p[i]; best = (int32_t)i; }
        }
        return best;
    };

    const size_t N = (size_t)m.n_embd;
    const int T0 = (int)fix.size() + 1;
    std::vector<int32_t> tok0(fix.begin(), fix.end());
    tok0.push_back(x_tok);

    float* seeds_p = m.mtp_seeds_buf.as<float>();
    if (!fix.empty()) {
        std::memcpy(seeds_p, m.hmain_buf.as<float>(), fix.size() * N * sizeof(float));
    }
    std::memcpy(seeds_p + fix.size() * N, ss.hlast.data(), N * sizeof(float));

    const int p0 = ss.cur_pos - (int)fix.size();
    if (!m.forward_mtp_chunk(slot, T0, p0, tok0.data(), nullptr, nullptr, m.mtp_seeds_buf, 0, n_drafts > 0, err)) {
        return false;
    }

    drafts.assign((size_t)n_drafts, 0);
    if (n_drafts > 0) {
        drafts[0] = argmax_fn(m.logits_buf.as<float>(), m.n_vocab);
    }
    int last_row = T0 - 1;
    for (int d = 1; d < n_drafts; ++d) {
        if (!m.forward_mtp_chunk(slot, 1, ss.cur_pos + d, &drafts[(size_t)d - 1], nullptr, nullptr,
                                 m.hmtp_buf, (size_t)last_row * N * sizeof(float), true, err)) {
            return false;
        }
        last_row = 0;
        drafts[(size_t)d] = argmax_fn(m.logits_buf.as<float>(), m.n_vocab);
    }

    // Main trunk verify window of T = n_drafts + 1 rows: [x_tok, drafts...]
    const int T = n_drafts + 1;
    int32_t* tid_p = m.tok_ids_buf.as<int32_t>();
    int32_t* cr_p = m.cell_row_buf.as<int32_t>();
    Q36RowMeta* meta_p = m.row_meta_buf.as<Q36RowMeta>();
    for (int t = 0; t < T; ++t) {
        tid_p[t] = (t == 0) ? x_tok : drafts[(size_t)t - 1];
        cr_p[t] = -1;
        const int kvp = ss.cur_pos + t;
        const int32_t mp = kvp + ss.mrope_delta;
        meta_p[t] = {mp, mp, mp, kvp, (uint32_t)(kvp + 1), (uint32_t)slot, 0u, 0u};
    }

    const int slot_arr[1] = {slot};
    if (!m.run_layers_batch(T, slot_arr, false, true, -1, err, true)) {
        return false;
    }

    picks.resize((size_t)T);
    const float* lg = m.logits_buf.as<float>();
    for (int t = 0; t < T; ++t) {
        picks[(size_t)t] = argmax_fn(lg + (size_t)t * (size_t)m.n_vocab, m.n_vocab);
    }

    int a = 0;
    while (a < n_drafts && picks[(size_t)a] == drafts[(size_t)a]) ++a;

    m.commit_gdn(slot, a + 1);
    std::memcpy(ss.hlast.data(), m.hmain_buf.as<float>() + (size_t)a * N, N * sizeof(float));
    ss.cur_pos += (a + 1);
    accepted = a;
    return true;
}

void Q36MetalEngine::save_checkpoint(int slot) {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return;
    Impl& m = *impl_;
    if (m.ckpt_count <= 0) return;
    SlotState& ss = m.slots[(size_t)slot];

    Ckpt c;
    c.pos = ss.cur_pos;
    c.mrope_delta = ss.mrope_delta;
    c.gdn_par = ss.gdn_par;
    c.hlast = ss.hlast;
    c.state_blob.resize((size_t)m.n_gdn * m.gdn_state_floats);
    c.hist_blob.resize((size_t)m.n_gdn * 2 * m.gdn_hist_floats);
    for (int gi = 0; gi < m.n_gdn; ++gi) {
        const float* st_src = m.gdn_state[(size_t)gi].as<float>() + (size_t)slot * m.gdn_state_floats;
        std::memcpy(c.state_blob.data() + (size_t)gi * m.gdn_state_floats,
                    st_src, m.gdn_state_floats * sizeof(float));
        const float* h_src = m.gdn_hist[(size_t)gi].as<float>() + (size_t)slot * 2 * m.gdn_hist_floats;
        std::memcpy(c.hist_blob.data() + (size_t)gi * 2 * m.gdn_hist_floats,
                    h_src, 2 * m.gdn_hist_floats * sizeof(float));
    }

    for (auto& existing : ss.ckpts) {
        if (existing.pos == c.pos) {
            existing = std::move(c);
            return;
        }
    }
    ss.ckpts.push_back(std::move(c));
    std::sort(ss.ckpts.begin(), ss.ckpts.end(), [](const Ckpt& a, const Ckpt& b) { return a.pos < b.pos; });
    while ((int)ss.ckpts.size() > m.ckpt_count) {
        size_t victim = 1;
        int best_span = std::numeric_limits<int>::max();
        for (size_t i = 1; i + 1 < ss.ckpts.size(); ++i) {
            const int span = ss.ckpts[i + 1].pos - ss.ckpts[i - 1].pos;
            if (span < best_span) {
                best_span = span;
                victim = i;
            }
        }
        ss.ckpts.erase(ss.ckpts.begin() + (long)victim);
    }
}

int Q36MetalEngine::checkpoint_at_or_below(int slot, int max_pos) const {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return 0;
    const SlotState& ss = impl_->slots[(size_t)slot];
    int best = 0;
    for (const auto& c : ss.ckpts) {
        if (c.pos <= max_pos && c.pos > best) best = c.pos;
    }
    return best;
}

bool Q36MetalEngine::restore_checkpoint(int slot, int target_pos) {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return false;
    Impl& m = *impl_;
    SlotState& ss = m.slots[(size_t)slot];
    for (const auto& c : ss.ckpts) {
        if (c.pos == target_pos) {
            ss.cur_pos = c.pos;
            ss.mrope_delta = c.mrope_delta;
            ss.gdn_par = c.gdn_par;
            ss.hlast = c.hlast;
            for (int gi = 0; gi < m.n_gdn; ++gi) {
                float* st_dst = m.gdn_state[(size_t)gi].as<float>() + (size_t)slot * m.gdn_state_floats;
                std::memcpy(st_dst, c.state_blob.data() + (size_t)gi * m.gdn_state_floats,
                            m.gdn_state_floats * sizeof(float));
                float* h_dst = m.gdn_hist[(size_t)gi].as<float>() + (size_t)slot * 2 * m.gdn_hist_floats;
                std::memcpy(h_dst, c.hist_blob.data() + (size_t)gi * 2 * m.gdn_hist_floats,
                            2 * m.gdn_hist_floats * sizeof(float));
            }
            ss.ckpts.erase(std::remove_if(ss.ckpts.begin(), ss.ckpts.end(),
                                          [&](const Ckpt& k) { return k.pos > target_pos; }),
                           ss.ckpts.end());
            return true;
        }
    }
    return false;
}

int Q36MetalEngine::checkpoint_count(int slot) const {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return 0;
    return (int)impl_->slots[(size_t)slot].ckpts.size();
}

bool Q36MetalEngine::copy_session(int src_slot, int dst_slot) {
    if (!impl_ || src_slot < 0 || src_slot >= impl_->n_slots ||
        dst_slot < 0 || dst_slot >= impl_->n_slots || src_slot == dst_slot)
        return false;
    Impl& m = *impl_;
    const SlotState& src = m.slots[(size_t)src_slot];
    SlotState& dst = m.slots[(size_t)dst_slot];

    for (int gi = 0; gi < m.n_gdn; ++gi) {
        std::memcpy(m.gdn_state[(size_t)gi].as<float>() + (size_t)dst_slot * m.gdn_state_floats,
                    m.gdn_state[(size_t)gi].as<float>() + (size_t)src_slot * m.gdn_state_floats,
                    m.gdn_state_floats * sizeof(float));
        std::memcpy(m.gdn_hist[(size_t)gi].as<float>() + (size_t)dst_slot * 2 * m.gdn_hist_floats,
                    m.gdn_hist[(size_t)gi].as<float>() + (size_t)src_slot * 2 * m.gdn_hist_floats,
                    2 * m.gdn_hist_floats * sizeof(float));
    }

    const size_t kv_stride = (size_t)m.HKV * m.max_ctx * m.D;
    const size_t copy_len = (size_t)src.cur_pos * (size_t)m.D * sizeof(float);
    if (copy_len > 0) {
        const int n_kv_layers = m.n_attn + (m.has_mtp ? 1 : 0);
        for (int ai = 0; ai < n_kv_layers; ++ai) {
            float* ks = m.kv_k[(size_t)ai].as<float>() + (size_t)src_slot * kv_stride;
            float* kd = m.kv_k[(size_t)ai].as<float>() + (size_t)dst_slot * kv_stride;
            float* vs = m.kv_v[(size_t)ai].as<float>() + (size_t)src_slot * kv_stride;
            float* vd = m.kv_v[(size_t)ai].as<float>() + (size_t)dst_slot * kv_stride;
            for (int kh = 0; kh < m.HKV; ++kh) {
                const size_t hoff = (size_t)kh * (size_t)m.max_ctx * (size_t)m.D;
                std::memcpy(kd + hoff, ks + hoff, copy_len);
                std::memcpy(vd + hoff, vs + hoff, copy_len);
            }
        }
    }

    dst.cur_pos = src.cur_pos;
    dst.mrope_delta = src.mrope_delta;
    dst.gdn_par = src.gdn_par;
    dst.hlast = src.hlast;
    dst.ckpts = src.ckpts;
    return true;
}

int32_t Q36MetalEngine::greedy_next(int32_t token, std::string& err) {
    const float* lg = nullptr;
    if (!forward(token, lg, err)) return -1;
    const int64_t nv = impl_->n_vocab;
    int32_t best = 0;
    float bv = lg[0];
    for (int64_t i = 1; i < nv; ++i) {
        if (lg[i] > bv) { bv = lg[i]; best = (int32_t)i; }
    }
    return best;
}

int64_t Q36MetalEngine::n_vocab() const { return impl_ ? impl_->n_vocab : 0; }
int64_t Q36MetalEngine::n_embd() const { return impl_ ? impl_->n_embd : 0; }
int64_t Q36MetalEngine::n_trunk_layers() const { return impl_ ? impl_->n_layers : 0; }
int Q36MetalEngine::n_slots() const { return impl_ ? impl_->n_slots : 0; }
int Q36MetalEngine::pos() const { return pos(0); }
int Q36MetalEngine::pos(int slot) const {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return 0;
    return impl_->slots[(size_t)slot].cur_pos;
}
int32_t Q36MetalEngine::mrope_delta(int slot) const {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return 0;
    return impl_->slots[(size_t)slot].mrope_delta;
}
void Q36MetalEngine::set_mrope_delta(int slot, int32_t delta) {
    if (!impl_ || slot < 0 || slot >= impl_->n_slots) return;
    impl_->slots[(size_t)slot].mrope_delta = delta;
}

StreamStats Q36MetalEngine::stream_stats() const {
    StreamStats st;
    if (!impl_ || !impl_->streamed) return st;
    st.streamed = true;
    st.using_contig = impl_->stream.using_contig();
    st.dense_bytes = impl_->dense_bytes;
    st.arena_bytes = impl_->stream.arena_used_bytes();
    st.reads = impl_->stream.reads();
    st.hits = impl_->stream.hits();
    st.misses = impl_->stream.misses();
    st.pre_hits = impl_->stream.pre_hits();
    st.pre_issued = impl_->stream.pre_issued();
    st.bytes_read = impl_->stream.bytes_read();
    return st;
}

bool Q36MetalEngine::save_expert_profile(const std::string& path, std::string& err) const {
    if (!impl_ || !impl_->streamed) {
        err = "save_expert_profile requires streaming mode";
        return false;
    }
    const Impl& m = *impl_;
    struct Entry { uint16_t l, e; uint64_t cnt; };
    std::vector<Entry> items;
    for (int l = 0; l < m.n_layers; ++l) {
        for (int e = 0; e < m.n_expert; ++e) {
            const uint64_t c = m.exp_counts[(size_t)l * (size_t)m.n_expert + (size_t)e];
            if (c > 0) items.push_back({(uint16_t)l, (uint16_t)e, c});
        }
    }
    std::sort(items.begin(), items.end(), [](const Entry& a, const Entry& b) {
        return a.cnt != b.cnt ? a.cnt > b.cnt : (a.l != b.l ? a.l < b.l : a.e < b.e);
    });
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        err = "cannot open profile for write: " + path;
        return false;
    }
    const char magic[4] = {'S', 'T', 'R', 'P'};
    const uint32_t hdr[5] = {1u, (uint32_t)m.n_layers, (uint32_t)m.n_expert, 0u, (uint32_t)items.size()};
    bool ok = std::fwrite(magic, 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5;
    for (const Entry& it : items) {
        const uint16_t pair[2] = {it.l, it.e};
        if (ok && std::fwrite(pair, 2, 2, f) != 2) ok = false;
    }
    std::fclose(f);
    if (!ok) {
        err = "failed writing profile: " + path;
        return false;
    }
    return true;
}

}  // namespace strata::metal_q36
