// metal/tests/q36_stream_test.cpp - TDD Cycle 5: parameterized ExpertStream (NL=4, NE=256) + stream_experts.py STRX.
#include "engine/stream.h"
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cstdio>
#include <cstdlib>
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
        std::string s;
    };
    struct Ten {
        std::string name;
        std::vector<uint64_t> dims;
        int type;
        std::vector<uint8_t> data;
    };
    std::vector<Kv> kvs;
    std::vector<Ten> tens;

    void add_u32(const std::string& k, uint32_t v) { kvs.push_back({k, 4, v, {}}); }
    void add_str(const std::string& k, const std::string& v) { kvs.push_back({k, 8, 0, v}); }
    void add_tensor(const std::string& name, std::vector<uint64_t> dims, int type, uint64_t seed) {
        uint64_t n = 1;
        for (uint64_t d : dims) n *= d;
        const int cols = (int)dims[0];
        const int rows = (int)(n / cols);
        std::vector<uint8_t> buf(ref::row_bytes(type, cols) * rows);
        ref::random_blocks(type, buf.data(), rows, cols, seed, 0.05f);
        tens.push_back({name, std::move(dims), type, std::move(buf)});
    }
    bool write(const std::string& path) const {
        std::vector<uint8_t> hdr;
        auto pu32 = [&](uint32_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
        auto pu64 = [&](uint64_t v) { hdr.insert(hdr.end(), (uint8_t*)&v, (uint8_t*)&v + 8); };
        auto pstr = [&](const std::string& s) { pu64(s.size()); hdr.insert(hdr.end(), s.begin(), s.end()); };
        pu32(0x46554747u); pu32(3); pu64(tens.size()); pu64(kvs.size());
        for (const auto& kv : kvs) {
            pstr(kv.key); pu32(kv.type);
            if (kv.type == 4) pu32(kv.u); else pstr(kv.s);
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

bool verify_expert_bytes(const gguf::File& f, const strata::metal_engine::ExpertStream& es,
                         int l, int e) {
    const uint32_t code = es.smap().as<uint32_t>()[l * 512 + e];
    if (code == 0xFFFFFFFFu) {
        std::fprintf(stderr, "missing smap entry at layer %d expert %d\n", l, e);
        return false;
    }
    const uint32_t chunk = code >> 28;
    const uint32_t page = code & 0x0FFFFFFFu;
    const uint8_t* slot_ptr = es.chunk((int)chunk).as<uint8_t>() + (uint64_t)page * 16384ull;

    const std::string b = "blk." + std::to_string(l) + ".";
    const auto* tg = f.find(b + "ffn_gate_exps.weight");
    const auto* tu = f.find(b + "ffn_up_exps.weight");
    const auto* td = f.find(b + "ffn_down_exps.weight");
    const uint64_t gs = tg->bytes / 256, us = tu->bytes / 256, ds = td->bytes / 256;
    const uint8_t* map = (const uint8_t*)f.map;
    if (std::memcmp(slot_ptr, map + tg->offset + e * gs, gs) != 0) return false;
    if (std::memcmp(slot_ptr + es.up_in_slot(l), map + tu->offset + e * us, us) != 0) return false;
    if (std::memcmp(slot_ptr + es.down_in_slot(l), map + td->offset + e * ds, ds) != 0) return false;
    return true;
}

}  // namespace

int main() {
    constexpr int NL = 4, N = 256, NFF = 256, NEXP = 256;
    const std::string path = "q36_stream_synth.gguf";
    const std::string strx_path = path + ".experts";
    std::remove(strx_path.c_str());

    GgufWriter gw;
    gw.add_str("general.architecture", "qwen35moe");
    gw.add_u32("qwen35moe.block_count", NL + 1);
    gw.add_u32("qwen35moe.nextn_predict_layers", 1);
    gw.add_tensor("token_embd.weight", {N, 64}, ref::Q8_0, 1);
    uint64_t seed = 10;
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        gw.add_tensor(b + "attn_norm.weight", {N}, ref::F32, seed++);
        gw.add_tensor(b + "ffn_gate_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++);
        gw.add_tensor(b + "ffn_up_exps.weight", {N, NFF, NEXP}, ref::Q4_K, seed++);
        gw.add_tensor(b + "ffn_down_exps.weight", {NFF, N, NEXP}, l == 3 ? ref::Q6_K : ref::Q4_K, seed++);
    }
    gw.add_tensor("blk.4.nextn.enorm.weight", {N}, ref::F32, seed++);
    if (!gw.write(path)) return 2;

    mtl::Context cx;
    std::string err;

    // 1. Stream directly from GGUF with tight 16 MiB budget (forces LRU eviction across 4x256 experts)
    {
        gguf::File f;
        if (!f.open(path, err)) { std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1; }
        strata::metal_engine::ExpertStream es;
        if (!es.open(f, NL, err)) { std::fprintf(stderr, "stream open: %s\n", err.c_str()); return 1; }

        mtl::Buffer W_dense;
        if (!es.load_dense(f, cx, {"blk.4.nextn.enorm.weight"}, W_dense, err)) {
            std::fprintf(stderr, "load_dense: %s\n", err.c_str());
            return 1;
        }
        if (f.find("blk.4.nextn.enorm.weight")->offset != 0) {
            std::fprintf(stderr, "expected skipped tensor offset == 0\n");
            return 1;
        }
        if (!es.make_arena(cx, 16ull << 20, "", 16, err)) {
            std::fprintf(stderr, "make_arena: %s\n", err.c_str());
            return 1;
        }

        // Re-open raw file map to compare against original GGUF bytes (since load_dense rewrote f.tensors offsets for dense)
        gguf::File f_orig;
        if (!f_orig.open(path, err)) return 1;

        for (int round = 0; round < 3; ++round) {
            for (int l = 0; l < NL; ++l) {
                int32_t ids[8];
                for (int k = 0; k < 8; ++k) ids[k] = (round * 37 + l * 13 + k * 7) % NEXP;
                es.at_layer(l, 1, 8, ids, nullptr);
                for (int k = 0; k < 8; ++k) {
                    if (!verify_expert_bytes(f_orig, es, l, ids[k])) {
                        std::fprintf(stderr, "FAIL: byte mismatch round %d layer %d expert %d\n", round, l, ids[k]);
                        return 1;
                    }
                }
            }
        }
        if (es.reads() == 0) {
            std::fprintf(stderr, "FAIL: expected >0 SSD reads\n");
            return 1;
        }
        std::printf("q36_stream_test (GGUF): arena=%llu bytes, reads=%llu, hits=%llu, misses=%llu  PASS\n",
                    (unsigned long long)es.arena_used_bytes(), (unsigned long long)es.reads(),
                    (unsigned long long)es.hits(), (unsigned long long)es.misses());
    }

    // 2. Generate .experts (STRX) via stream_experts.py and verify ExpertStream opens & serves it
    {
        const std::string tool = std::string(STRATA_METAL_ROOT_DIR) + "/tools/stream_experts.py";
        const std::string cmd = "UV_PYTHON_PREFERENCE=only-managed uv run --python 3.12 \"" + tool + "\" --gguf \"" + path + "\" --verify";
        if (std::system(cmd.c_str()) != 0) {
            std::fprintf(stderr, "FAIL: stream_experts.py failed\n");
            return 1;
        }
        gguf::File f;
        if (!f.open(path, err)) return 1;
        strata::metal_engine::ExpertStream es;
        if (!es.open(f, NL, err)) { std::fprintf(stderr, "stream open STRX: %s\n", err.c_str()); return 1; }
        if (!es.using_contig()) {
            std::fprintf(stderr, "FAIL: expected ExpertStream to detect and open .experts STRX file\n");
            return 1;
        }
        mtl::Buffer W_dense;
        if (!es.load_dense(f, cx, {"blk.4.nextn.enorm.weight"}, W_dense, err)) return 1;
        if (!es.make_arena(cx, 16ull << 20, "", 16, err)) return 1;

        gguf::File f_orig;
        if (!f_orig.open(path, err)) return 1;
        for (int l = 0; l < NL; ++l) {
            int32_t ids[8] = {0, 17, 63, 128, 199, 211, 240, 255};
            es.at_layer(l, 1, 8, ids, nullptr);
            for (int k = 0; k < 8; ++k) {
                if (!verify_expert_bytes(f_orig, es, l, ids[k])) {
                    std::fprintf(stderr, "FAIL: STRX byte mismatch layer %d expert %d\n", l, ids[k]);
                    return 1;
                }
            }
        }
        std::printf("q36_stream_test (STRX): reads=%llu  PASS\n", (unsigned long long)es.reads());
    }

    std::remove(strx_path.c_str());
    std::remove(path.c_str());
    return 0;
}
