// metal/src/q36/q36_metal.cpp - Qwen3.6-35B-A3B (qwen35moe) Metal forward engine (Resident & SSD ExpertStream).
#include "q36/q36_metal.h"

#include "engine/arena_budget.h"
#include "engine/stream.h"
#include "gguf_lite.h"
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
#include <map>
#include <vector>

namespace strata::metal_q36 {

namespace {

struct GdnArgs {
    uint32_t h_k, h_v, channels, commit;
    float eps;
    uint32_t plane;
};

struct Q36AttnArgs {
    uint32_t H, HKV, D, n_rot;
    int32_t pos_t, pos_h, pos_w, kv_pos;
    uint32_t n_pos, ctx, plane;
    float base, eps, scale;
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

}  // namespace

struct Q36MetalEngine::Impl {
    gguf::File f;
    mtl::Context cx;
    mtl::Buffer W;
    strata::metal_engine::ExpertStream stream;
    bool streamed = false;
    uint64_t dense_bytes = 0;

    int lib_ops = -1, lib_gdn = -1, lib_attn = -1, lib_moe = -1;
    mtl::Pipeline p_embed, p_rms, p_add_rms, p_add;
    mtl::Pipeline p_gdn;
    mtl::Pipeline p_attn_prep, p_attn_dec;
    mtl::Pipeline p_swiglu, p_comb, p_moe_route;
    std::map<int, mtl::Pipeline> p_gemv;
    std::map<int, mtl::Pipeline> p_router;
    std::map<int, mtl::Pipeline> p_moe_gu;
    std::map<int, mtl::Pipeline> p_moe_dn;

    // Model dimensions
    int64_t n_embd = 0, n_vocab = 0, n_layers = 0, attn_interval = 4;
    int64_t n_expert = 0, top_k = 0, n_ff = 0, n_ff_sh = 0;
    int64_t H = 0, HKV = 0, D = 0, n_rot = 0;
    int64_t h_k = 0, h_v = 0, ssm_s = 0, conv_ch = 0, zv = 0, conv_k = 4;
    float eps = 1e-6f, rope_base = 10000000.0f;
    int max_ctx = 4096;
    int cur_pos = 0;

    const gguf::Tensor* tok_embd = nullptr;
    const gguf::Tensor* out_norm = nullptr;
    const gguf::Tensor* out_w = nullptr;
    std::vector<LayerRef> layers;

    // State buffers
    std::vector<mtl::Buffer> gdn_state;
    std::vector<mtl::Buffer> gdn_hist[2];
    std::vector<int> gdn_par;
    std::vector<mtl::Buffer> kv_k;
    std::vector<mtl::Buffer> kv_v;

    // Scratch buffers (1-token decode)
    mtl::Buffer x_buf, y_buf, xp_emb, xp_gdn, xp_att, xp_sh;
    mtl::Buffer qkv_buf, z_buf, al_buf, be_buf;
    mtl::Buffer qg_buf, kx_buf, vx_buf, q_buf, gate_buf;
    mtl::Buffer rt_logits, rt_ids, rt_w, rt_plan, rt_cnt;
    mtl::Buffer moe_h, moe_ye, sh_g, sh_u, sh_y, sh_gate, logits_buf;

    const mtl::Pipeline& gemv_pipe(int type) {
        auto it = p_gemv.find(type);
        if (it == p_gemv.end()) {
            const std::string name = "gemv_" + std::to_string(type) + "_32_1_4_1";
            it = p_gemv.emplace(type, cx.pipeline(lib_moe, name, mtl::Constants().i(0, 1))).first;
            mtl::require_fit(it->second, 128, name.c_str());
        }
        return it->second;
    }

    const mtl::Pipeline& router_pipe(int type) {
        auto it = p_router.find(type);
        if (it == p_router.end()) {
            const std::string name = "router_route_" + std::to_string(type);
            it = p_router.emplace(type, cx.pipeline(lib_moe, name, mtl::Constants().i(0, 1))).first;
            mtl::require_fit(it->second, 512, name.c_str());
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

    void at_cut(int layer) {
        stream.at_layer(layer, 1, (int)top_k, rt_ids.as<int32_t>() + (size_t)layer * top_k, nullptr, nullptr);
    }

    mtl::Dispatch disp_gemv(const gguf::Tensor* t, const mtl::Buffer& xp, uint32_t plane,
                            const mtl::Buffer& out, uint32_t y_stride = 0) {
        const uint32_t n_in = (uint32_t)t->shape[0];
        const uint32_t n_out = t->shape.size() > 1 ? (uint32_t)t->shape[1] : 1u;
        const int k = xkind(t->type);
        GemvArgs ga{n_in, n_out, 1, row_bytes_u(*t), n_in, y_stride ? y_stride : n_out};
        mtl::Dispatch d;
        d.p = &gemv_pipe(t->type);
        d.args = {mtl::Arg::B(W, t->offset), mtl::Arg::B(xp, (size_t)4 * k * plane),
                  mtl::Arg::B(xp, (size_t)4 * 4 * plane), mtl::Arg::B(out), mtl::Arg::V(ga)};
        d.grid[0] = (n_out + 3) / 4;
        d.tg[0] = 128;
        return d;
    }
};

Q36MetalEngine::Q36MetalEngine() = default;
Q36MetalEngine::~Q36MetalEngine() = default;

bool Q36MetalEngine::open(const std::string& gguf_path, const Options& opt, std::string& err) {
    impl_ = std::make_unique<Impl>();
    Impl& m = *impl_;
    m.max_ctx = opt.ctx;

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
    int n_gdn = 0, n_attn = 0;
    for (int l = 0; l < m.n_layers; ++l) {
        LayerRef& L = m.layers[(size_t)l];
        const std::string b = "blk." + std::to_string(l) + ".";
        L.is_attn = (l % m.attn_interval) == (m.attn_interval - 1);
        L.attn_norm = req_ten(b + "attn_norm.weight");
        L.post_norm = req_ten(b + "post_attention_norm.weight");
        if (!L.attn_norm || !L.post_norm) return false;

        if (!L.is_attn) {
            L.gdn_idx = n_gdn++;
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
            L.attn_idx = n_attn++;
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

    const char* env_s = std::getenv("STRATA_METAL_STREAM");
    if (opt.stream || opt.arena_gb > 0.0) m.streamed = true;
    else if (env_s && std::string(env_s) == "1") m.streamed = true;
    else if (env_s && std::string(env_s) == "0") m.streamed = false;
    else m.streamed = strata::metal_engine::ExpertStream::wanted(m.f, m.cx);

    if (m.streamed) {
        if (!m.stream.open(m.f, (int)m.n_layers, err)) return false;
        std::vector<std::string> skip;
        for (const auto& kv : m.f.tensors) {
            if (kv.first.rfind("blk.", 0) == 0) {
                const size_t dot = kv.first.find('.', 4);
                if (dot != std::string::npos) {
                    const int blk_idx = std::atoi(kv.first.substr(4, dot - 4).c_str());
                    if (blk_idx >= m.n_layers) skip.push_back(kv.first);
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
    m.p_rms = m.cx.pipeline(m.lib_ops, "q36_rms_prep");
    m.p_add_rms = m.cx.pipeline(m.lib_ops, "q36_add_rms_prep");
    m.p_add = m.cx.pipeline(m.lib_ops, "q36_add");
    m.p_gdn = m.cx.pipeline(m.lib_gdn, "gdn_step", mtl::Constants().i(2, 1).b(32, true));
    m.p_attn_prep = m.cx.pipeline(m.lib_attn, "q36_attn_prep");
    m.p_attn_dec = m.cx.pipeline(m.lib_attn, "q36_attn_decode");
    m.p_swiglu = m.cx.pipeline(m.lib_moe, "swiglu_prep");
    m.p_comb = m.cx.pipeline(m.lib_moe, "moe_combine_sh");
    m.p_moe_route = m.cx.pipeline(m.lib_moe, "moe_route");

    mtl::require_fit(m.p_embed, 256, "q36_embed_row");
    mtl::require_fit(m.p_rms, 256, "q36_rms_prep");
    mtl::require_fit(m.p_add_rms, 256, "q36_add_rms_prep");
    mtl::require_fit(m.p_gdn, 512, "gdn_step");
    mtl::require_fit(m.p_attn_prep, 256, "q36_attn_prep");
    mtl::require_fit(m.p_attn_dec, 256, "q36_attn_decode");

    m.gdn_state.resize((size_t)n_gdn);
    m.gdn_hist[0].resize((size_t)n_gdn);
    m.gdn_hist[1].resize((size_t)n_gdn);
    m.gdn_par.assign((size_t)n_gdn, 0);
    for (int i = 0; i < n_gdn; ++i) {
        m.gdn_state[(size_t)i] = m.cx.alloc(4 * (size_t)m.ssm_s * m.h_v * m.ssm_s);
        m.gdn_hist[0][(size_t)i] = m.cx.alloc(4 * (size_t)m.conv_ch * 3);
        m.gdn_hist[1][(size_t)i] = m.cx.alloc(4 * (size_t)m.conv_ch * 3);
    }

    m.kv_k.resize((size_t)n_attn);
    m.kv_v.resize((size_t)n_attn);
    for (int i = 0; i < n_attn; ++i) {
        m.kv_k[(size_t)i] = m.cx.alloc(4 * (size_t)m.HKV * m.max_ctx * m.D);
        m.kv_v[(size_t)i] = m.cx.alloc(4 * (size_t)m.HKV * m.max_ctx * m.D);
    }

    auto xp_bytes = [](int64_t n) -> size_t { return 4 * (size_t)(4 * n + n / 16); };
    m.x_buf = m.cx.alloc(4 * (size_t)m.n_embd);
    m.y_buf = m.cx.alloc(4 * (size_t)m.n_embd);
    m.xp_emb = m.cx.alloc(xp_bytes(m.n_embd));
    m.xp_gdn = m.cx.alloc(xp_bytes(m.zv));
    m.xp_att = m.cx.alloc(xp_bytes(m.H * m.D));
    m.xp_sh = m.cx.alloc(xp_bytes(m.n_ff_sh));

    m.qkv_buf = m.cx.alloc(4 * (size_t)m.conv_ch);
    m.z_buf = m.cx.alloc(4 * (size_t)m.zv);
    m.al_buf = m.cx.alloc(4 * (size_t)m.h_v);
    m.be_buf = m.cx.alloc(4 * (size_t)m.h_v);

    m.qg_buf = m.cx.alloc(4 * (size_t)m.H * 2 * m.D);
    m.kx_buf = m.cx.alloc(4 * (size_t)m.HKV * m.D);
    m.vx_buf = m.cx.alloc(4 * (size_t)m.HKV * m.D);
    m.q_buf = m.cx.alloc(4 * (size_t)m.H * m.D);
    m.gate_buf = m.cx.alloc(4 * (size_t)m.H * m.D);

    m.rt_logits = m.cx.alloc(4 * 512);
    m.rt_ids = m.cx.alloc(4 * (size_t)(m.streamed ? m.n_layers : 1) * m.top_k);
    m.rt_w = m.cx.alloc(4 * (size_t)m.top_k);
    m.rt_plan = m.cx.alloc(4 * P_INTS);
    m.rt_cnt = m.cx.alloc(16);
    m.moe_h = m.cx.alloc(4 * (size_t)m.top_k * m.n_ff);
    m.moe_ye = m.cx.alloc(4 * (size_t)m.top_k * m.n_embd);
    m.sh_g = m.cx.alloc(4 * (size_t)m.n_ff_sh);
    m.sh_u = m.cx.alloc(4 * (size_t)m.n_ff_sh);
    m.sh_y = m.cx.alloc(4 * (size_t)m.n_embd);
    m.sh_gate = m.cx.alloc(16);
    m.logits_buf = m.cx.alloc(4 * (size_t)m.n_vocab);

    if (m.streamed) {
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
        if (!m.stream.make_arena(m.cx, budget, "", 16, err)) return false;
    }

    if (!opt.silent) {
        std::fprintf(stderr, "strata-q36-metal: %s (%zu layers, hidden %lld, %lld experts top-%lld, %.2f GiB, mode=%s) ready\n",
                     m.cx.device_name().c_str(), m.layers.size(), (long long)m.n_embd,
                     (long long)m.n_expert, (long long)m.top_k, m.f.file_bytes / double(1 << 30),
                     m.streamed ? "stream" : "resident");
    }
    return true;
}

void Q36MetalEngine::reset() {
    Impl& m = *impl_;
    m.cur_pos = 0;
    for (auto& b : m.gdn_state) std::memset(b.ptr, 0, b.bytes);
    for (auto& b : m.gdn_hist[0]) std::memset(b.ptr, 0, b.bytes);
    for (auto& b : m.gdn_hist[1]) std::memset(b.ptr, 0, b.bytes);
    std::fill(m.gdn_par.begin(), m.gdn_par.end(), 0);
}

bool Q36MetalEngine::forward(int32_t token, const float*& logits_out, std::string& err) {
    Impl& m = *impl_;
    if (token < 0 || token >= m.n_vocab) {
        err = "token id out of range";
        return false;
    }
    if (m.cur_pos >= m.max_ctx) {
        err = "context limit reached";
        return false;
    }

    std::vector<mtl::Dispatch> ds;
    ds.reserve((size_t)m.n_layers * 16 + 8);
    std::vector<int> cuts;
    if (m.streamed) cuts.reserve((size_t)m.n_layers);

    const uint32_t N = (uint32_t)m.n_embd;
    const uint32_t NFF = (uint32_t)m.n_ff;
    const uint32_t NFF_SH = (uint32_t)m.n_ff_sh;
    const uint32_t NEXP = (uint32_t)m.n_expert;
    const uint32_t TOPK = (uint32_t)m.top_k;

    // Embed token
    {
        Q36EmbedArgs ea{(uint32_t)token, N, row_bytes_u(*m.tok_embd), (uint32_t)m.tok_embd->type};
        mtl::Dispatch d;
        d.p = &m.p_embed;
        d.args = {mtl::Arg::B(m.W, m.tok_embd->offset), mtl::Arg::B(m.x_buf), mtl::Arg::V(ea)};
        d.grid[0] = (N / 32 + 7) / 8;
        d.tg[0] = 256;
        ds.push_back(std::move(d));
    }

    for (int l = 0; l < m.n_layers; ++l) {
        const LayerRef& L = m.layers[(size_t)l];
        Q36NormArgs na{N, N, m.eps};
        {
            mtl::Dispatch d;
            d.p = &m.p_rms;
            d.args = {mtl::Arg::B(m.x_buf), mtl::Arg::B(m.W, L.attn_norm->offset), mtl::Arg::B(m.xp_emb), mtl::Arg::V(na)};
            d.grid[0] = 1;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }

        if (!L.is_attn) {
            auto d_qkv = m.disp_gemv(L.attn_qkv, m.xp_emb, N, m.qkv_buf);
            auto d_z = m.disp_gemv(L.attn_gate, m.xp_emb, N, m.z_buf);
            auto d_al = m.disp_gemv(L.ssm_alpha, m.xp_emb, N, m.al_buf);
            auto d_be = m.disp_gemv(L.ssm_beta, m.xp_emb, N, m.be_buf);
            d_z.barrier = false;
            d_al.barrier = false;
            d_be.barrier = false;
            ds.push_back(std::move(d_qkv));
            ds.push_back(std::move(d_z));
            ds.push_back(std::move(d_al));
            ds.push_back(std::move(d_be));

            const int gi = L.gdn_idx;
            const int par = m.gdn_par[(size_t)gi];
            m.gdn_par[(size_t)gi] = 1 - par;

            GdnArgs ga{(uint32_t)m.h_k, (uint32_t)m.h_v, (uint32_t)m.conv_ch, 1u, m.eps, (uint32_t)m.zv};
            mtl::Dispatch d_gdn;
            d_gdn.p = &m.p_gdn;
            d_gdn.args = {mtl::Arg::B(m.gdn_state[(size_t)gi]),
                          mtl::Arg::B(m.gdn_hist[par][(size_t)gi]),
                          mtl::Arg::B(m.qkv_buf),
                          mtl::Arg::B(m.al_buf),
                          mtl::Arg::B(m.be_buf),
                          mtl::Arg::B(m.z_buf),
                          mtl::Arg::B(m.W, L.ssm_conv->offset),
                          mtl::Arg::B(m.W, L.ssm_dt->offset),
                          mtl::Arg::B(m.W, L.ssm_a->offset),
                          mtl::Arg::B(m.W, L.ssm_norm->offset),
                          mtl::Arg::B(m.xp_gdn),
                          mtl::Arg::V(ga),
                          mtl::Arg::B(m.gdn_hist[1 - par][(size_t)gi])};
            d_gdn.grid[0] = (uint32_t)m.h_v;
            d_gdn.tg[0] = 512;
            ds.push_back(std::move(d_gdn));

            ds.push_back(m.disp_gemv(L.ssm_out, m.xp_gdn, (uint32_t)m.zv, m.y_buf));
        } else {
            auto d_q = m.disp_gemv(L.attn_q, m.xp_emb, N, m.qg_buf);
            auto d_k = m.disp_gemv(L.attn_k, m.xp_emb, N, m.kx_buf);
            auto d_v = m.disp_gemv(L.attn_v, m.xp_emb, N, m.vx_buf);
            d_k.barrier = false;
            d_v.barrier = false;
            ds.push_back(std::move(d_q));
            ds.push_back(std::move(d_k));
            ds.push_back(std::move(d_v));

            const int ai = L.attn_idx;
            const uint32_t att_plane = (uint32_t)(m.H * m.D);
            Q36AttnArgs aa{(uint32_t)m.H, (uint32_t)m.HKV, (uint32_t)m.D, (uint32_t)m.n_rot,
                           m.cur_pos, m.cur_pos, m.cur_pos, m.cur_pos,
                           (uint32_t)(m.cur_pos + 1), (uint32_t)m.max_ctx, att_plane,
                           m.rope_base, m.eps, 1.0f / std::sqrt((float)m.D)};

            mtl::Dispatch d_prep;
            d_prep.p = &m.p_attn_prep;
            d_prep.args = {mtl::Arg::B(m.qg_buf), mtl::Arg::B(m.kx_buf), mtl::Arg::B(m.vx_buf),
                           mtl::Arg::B(m.W, L.attn_q_norm->offset), mtl::Arg::B(m.W, L.attn_k_norm->offset),
                           mtl::Arg::B(m.q_buf), mtl::Arg::B(m.gate_buf),
                           mtl::Arg::B(m.kv_k[(size_t)ai]), mtl::Arg::B(m.kv_v[(size_t)ai]), mtl::Arg::V(aa)};
            d_prep.grid[0] = (uint32_t)(m.H + m.HKV);
            d_prep.tg[0] = 256;
            ds.push_back(std::move(d_prep));

            mtl::Dispatch d_dec;
            d_dec.p = &m.p_attn_dec;
            d_dec.args = {mtl::Arg::B(m.q_buf), mtl::Arg::B(m.kv_k[(size_t)ai]), mtl::Arg::B(m.kv_v[(size_t)ai]),
                          mtl::Arg::B(m.gate_buf), mtl::Arg::B(m.xp_att), mtl::Arg::V(aa)};
            d_dec.grid[0] = (uint32_t)m.H;
            d_dec.tg[0] = 256;
            ds.push_back(std::move(d_dec));

            ds.push_back(m.disp_gemv(L.attn_output, m.xp_att, att_plane, m.y_buf));
        }

        // x += y; h2 = rms(x) * post_attention_norm (+ xprep)
        {
            mtl::Dispatch d;
            d.p = &m.p_add_rms;
            d.args = {mtl::Arg::B(m.x_buf), mtl::Arg::B(m.y_buf), mtl::Arg::B(m.W, L.post_norm->offset),
                      mtl::Arg::B(m.xp_emb), mtl::Arg::V(na)};
            d.grid[0] = 1;
            d.tg[0] = 256;
            ds.push_back(std::move(d));
        }

        // MoE routing + plan
        const uint32_t gu_rb = row_bytes_u(*L.gate_exps);
        const uint32_t dn_rb = row_bytes_u(*L.down_exps);
        MoeArgs ma{
            m.streamed ? 0ull : L.gate_exps->offset,
            m.streamed ? m.stream.up_in_slot(l) : L.up_exps->offset,
            m.streamed ? m.stream.down_in_slot(l) : L.down_exps->offset,
            m.streamed ? 0ull : (uint64_t)gu_rb * NFF,
            m.streamed ? 0ull : (uint64_t)dn_rb * N,
            gu_rb, dn_rb, N, NFF, 1, TOPK, NEXP, NFF / 4, N / 16, 0};
        const size_t ids_off = m.streamed ? (size_t)4 * (size_t)l * TOPK : 0;

        if (L.gate_inp->type == ref::F32 || L.gate_inp->type == ref::BF16) {
            GemvArgs ga_rt{N, NEXP, 1, row_bytes_u(*L.gate_inp), N, 512};
            mtl::Dispatch d_rr;
            d_rr.p = &m.router_pipe(L.gate_inp->type);
            d_rr.args = {mtl::Arg::B(m.W, L.gate_inp->offset), mtl::Arg::B(m.xp_emb),
                         mtl::Arg::B(m.xp_emb, (size_t)4 * 4 * N), mtl::Arg::B(m.rt_logits),
                         mtl::Arg::V(ga_rt), mtl::Arg::B(m.rt_ids, ids_off), mtl::Arg::B(m.rt_w),
                         mtl::Arg::B(m.rt_plan), mtl::Arg::V(ma), mtl::Arg::B(m.rt_cnt)};
            d_rr.grid[0] = NEXP / 16;
            d_rr.tg[0] = 512;
            ds.push_back(std::move(d_rr));
        } else {
            ds.push_back(m.disp_gemv(L.gate_inp, m.xp_emb, N, m.rt_logits, 512));
            mtl::Dispatch d_rt;
            d_rt.p = &m.p_moe_route;
            d_rt.args = {mtl::Arg::B(m.rt_logits), mtl::Arg::B(m.rt_ids, ids_off), mtl::Arg::B(m.rt_w),
                         mtl::Arg::B(m.rt_plan), mtl::Arg::V(ma)};
            d_rt.grid[0] = 1;
            d_rt.tg[0] = 512;
            ds.push_back(std::move(d_rt));
        }
        if (m.streamed) cuts.push_back((int)ds.size());

        // Routed experts gate/up + down
        {
            const int gk = xkind(L.gate_exps->type);
            const mtl::Buffer& w0 = m.streamed ? m.stream.chunk(0) : m.W;
            const mtl::Buffer& w1 = m.streamed ? m.stream.chunk(1) : m.W;
            const mtl::Buffer& w2 = m.streamed ? m.stream.chunk(2) : m.W;
            const mtl::Buffer& w3 = m.streamed ? m.stream.chunk(3) : m.W;
            const mtl::Arg smap_arg = m.streamed ? mtl::Arg::B(m.stream.smap(), (size_t)4 * (size_t)l * 512)
                                                 : mtl::Arg::B(m.rt_plan);

            mtl::Dispatch d_gu;
            d_gu.p = &m.moe_gu_pipe(L.gate_exps->type);
            d_gu.args = {mtl::Arg::B(w0), mtl::Arg::B(m.xp_emb, (size_t)4 * gk * N),
                         mtl::Arg::B(m.moe_h), mtl::Arg::B(m.rt_plan), mtl::Arg::V(ma),
                         mtl::Arg::B(m.xp_emb, (size_t)4 * 4 * N), smap_arg,
                         mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
            d_gu.tg[0] = 128;
            d_gu.indirect = &m.rt_plan;
            d_gu.ind_off = 4 * 4;
            ds.push_back(std::move(d_gu));

            mtl::Dispatch d_dn;
            d_dn.p = &m.moe_dn_pipe(L.down_exps->type);
            d_dn.args = {mtl::Arg::B(w0), mtl::Arg::B(m.moe_h), mtl::Arg::B(m.moe_ye),
                         mtl::Arg::B(m.rt_plan), mtl::Arg::V(ma), mtl::Arg::B(m.rt_plan),
                         smap_arg, mtl::Arg::B(w1), mtl::Arg::B(w2), mtl::Arg::B(w3)};
            d_dn.tg[0] = 128;
            d_dn.indirect = &m.rt_plan;
            d_dn.ind_off = 4 * 8;
            ds.push_back(std::move(d_dn));
        }

        // Shared expert (can overlap with routed experts after router)
        {
            auto d_sg = m.disp_gemv(L.shexp_gate, m.xp_emb, N, m.sh_g);
            auto d_su = m.disp_gemv(L.shexp_up, m.xp_emb, N, m.sh_u);
            auto d_sgi = m.disp_gemv(L.shexp_gate_inp, m.xp_emb, N, m.sh_gate);
            d_sg.barrier = false;
            d_su.barrier = false;
            d_sgi.barrier = false;
            ds.push_back(std::move(d_sg));
            ds.push_back(std::move(d_su));
            ds.push_back(std::move(d_sgi));

            GlueArgs gl_sw{NFF_SH, 1, N, TOPK};
            mtl::Dispatch d_sw;
            d_sw.p = &m.p_swiglu;
            d_sw.args = {mtl::Arg::B(m.sh_g), mtl::Arg::B(m.sh_u), mtl::Arg::B(m.xp_sh), mtl::Arg::V(gl_sw)};
            d_sw.grid[0] = NFF_SH / 16;
            d_sw.tg[0] = 16;
            ds.push_back(std::move(d_sw));

            ds.push_back(m.disp_gemv(L.shexp_down, m.xp_sh, NFF_SH, m.sh_y));
        }

        // Combine routed + shared -> y_buf, then x_buf += y_buf
        {
            GlueArgs gl_c{N, 1, N, TOPK};
            mtl::Dispatch d_comb;
            d_comb.p = &m.p_comb;
            d_comb.args = {mtl::Arg::B(m.moe_ye), mtl::Arg::B(m.rt_plan), mtl::Arg::B(m.sh_y),
                           mtl::Arg::B(m.sh_gate), mtl::Arg::B(m.y_buf), mtl::Arg::V(gl_c)};
            d_comb.grid[0] = (N + 127) / 128;
            d_comb.tg[0] = 128;
            ds.push_back(std::move(d_comb));

            mtl::Dispatch d_add;
            d_add.p = &m.p_add;
            d_add.args = {mtl::Arg::B(m.x_buf), mtl::Arg::B(m.y_buf), mtl::Arg::V(N)};
            d_add.grid[0] = (N + 255) / 256;
            d_add.tg[0] = 256;
            ds.push_back(std::move(d_add));
        }
    }

    // Final RMSNorm + output logits
    {
        Q36NormArgs na{N, N, m.eps};
        mtl::Dispatch d_rms;
        d_rms.p = &m.p_rms;
        d_rms.args = {mtl::Arg::B(m.x_buf), mtl::Arg::B(m.W, m.out_norm->offset), mtl::Arg::B(m.xp_emb), mtl::Arg::V(na)};
        d_rms.grid[0] = 1;
        d_rms.tg[0] = 256;
        ds.push_back(std::move(d_rms));

        ds.push_back(m.disp_gemv(m.out_w, m.xp_emb, N, m.logits_buf));
    }

    if (m.streamed) {
        m.cx.run_graph_seg(ds, cuts, [](void* u, int s) { ((Impl*)u)->at_cut(s); }, &m);
    } else {
        m.cx.run_graph(ds);
    }
    m.cur_pos++;
    logits_out = m.logits_buf.as<float>();
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
int64_t Q36MetalEngine::n_trunk_layers() const { return impl_ ? impl_->n_layers : 0; }
int Q36MetalEngine::pos() const { return impl_ ? impl_->cur_pos : 0; }

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
    st.bytes_read = impl_->stream.bytes_read();
    return st;
}

}  // namespace strata::metal_q36
