// metal/src/q36/q36_metal_main.cpp - CLI & --serve/--batch/--vision runner for Qwen3.6-35B-A3B Metal engine.
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

constexpr int32_t kImagePad = 248056;  // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct ImgReq {
    std::vector<float> rows;
    std::vector<int32_t> cell_row;
    std::vector<int32_t> mrope;
    std::vector<ImgKey> imgs;
    int32_t delta = 0;
};

std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t)hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t)n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " +
                 std::to_string((long long)n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t)hdr[1] * (size_t)hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) {
            ve = "short embeddings file";
            break;
        }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;

    const int64_t n = (int64_t)ids.size();
    r.cell_row.assign((size_t)n, -1);
    r.mrope.assign((size_t)n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t)c * 3 + 0] = (int32_t)t;
        r.mrope[(size_t)c * 3 + 1] = (int32_t)h;
        r.mrope[(size_t)c * 3 + 2] = (int32_t)w;
    };

    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t)i] != kImagePad) {
            put(i, p, p, p);
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h = 0;
        {
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof(grid));
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t)im.n * (size_t)n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j) {
            if (i + j >= n || ids[(size_t)(i + j)] != kImagePad) {
                return "image " + std::to_string(k) + " has " + std::to_string((long long)im.n) +
                       " rows but fewer <|image_pad|> tokens";
            }
        }
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t)(i + j)] = (int32_t)((im.off / (size_t)n_embd) + (size_t)j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t)(n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t)(p - n);
    return "";
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

struct SessMeta {
    bool active = false;
    uint64_t used = 0;
    std::vector<int32_t> live;
    std::vector<ImgKey> imgs;
};

struct DecodeSlot {
    bool active = false;
    bool stop = false;
    int32_t x = 0;
    int64_t produced = 0;
    int64_t max_new = 0;
    strata::qwen35::Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
};

struct ServeRunner {
    strata::metal_q36::Q36MetalEngine& eng;
    int ctx = 8192;
    bool vision = false;
    int ckpt_step = 256;
    int ckpt_n = 12;
    int mtp_d = 0;
    int64_t nv = 0;
    bool pen_prompt = false;

    std::vector<SessMeta> sess;
    std::vector<DecodeSlot> slots;
    int n_active = 0;
    uint64_t use_seq = 0;
    std::vector<float> scratch_lg;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    ServeRunner(strata::metal_q36::Q36MetalEngine& e, int c, bool vis, int cs, int cn, int md)
        : eng(e), ctx(c), vision(vis), ckpt_step(std::max(1, cs)), ckpt_n(cn), mtp_d(md), nv(e.n_vocab()) {
        pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
        sess.resize((size_t)e.n_slots());
        slots.resize((size_t)e.n_slots());
        scratch_lg.resize((size_t)nv);
    }

    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int)slots.size()) slots[(size_t)b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    void window() {
        poll_control();
        std::vector<strata::metal_q36::Q36MetalEngine::SlotStep> steps;
        for (int b = 0; b < eng.n_slots(); ++b) {
            if (slots[(size_t)b].active) steps.push_back({b, slots[(size_t)b].x});
        }
        if (steps.empty()) return;

        std::vector<const float*> outs;
        std::string err;
        if (!eng.step_slots(steps.data(), (int)steps.size(), outs, err)) {
            std::fprintf(stderr, "strata-q36-metal batch step error: %s\n", err.c_str());
            for (const auto& st : steps) {
                DecodeSlot& sl = slots[(size_t)st.slot];
                SessMeta& ss = sess[(size_t)st.slot];
                std::printf("BDONE %d %lld cancel %.1f\n", st.slot, (long long)sl.produced, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            }
            std::fflush(stdout);
            return;
        }

        for (size_t r = 0; r < steps.size(); ++r) {
            const int b = steps[r].slot;
            DecodeSlot& sl = slots[(size_t)b];
            SessMeta& ss = sess[(size_t)b];
            ss.live.push_back(sl.x);
            int32_t y = 0;
            if (sl.smp.needs_logits()) {
                std::memcpy(scratch_lg.data(), outs[r], 4 * (size_t)nv);
                y = sample_logits(scratch_lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = argmax_logits(outs[r], nv);
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : eng.pos(b) + 1 > ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long)sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    size_t match_len(const SessMeta& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        size_t L = 0;
        const size_t n = std::min(ss.live.size(), ids.size());
        while (L < n && ss.live[L] == ids[L]) ++L;
        for (const ImgKey& k : imgs) {
            if (k.cell >= (int64_t)L) continue;
            if (k.cell + k.n > (int64_t)L) L = k.cell > 0 ? (size_t)k.cell : 0;
        }
        if (L == 0) return 0;
        std::vector<ImgKey> arriving, held;
        for (const ImgKey& k : imgs)
            if (k.cell + k.n <= (int64_t)L) arriving.push_back(k);
        for (const ImgKey& k : ss.imgs)
            if (k.cell + k.n <= (int64_t)L) held.push_back(k);
        return arriving == held ? L : 0;
    }

    size_t resume_len(int slot_idx, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const SessMeta& ss = sess[(size_t)slot_idx];
        size_t L = match_len(ss, ids, imgs);
        if (L == 0 || ids.empty()) return 0;
        if (ckpt_n <= 0) return (L == ss.live.size() && L < ids.size()) ? L : 0;
        const size_t last = ids.size() - 1;
        if (L > last) L = last;
        if (L == ss.live.size()) return L;
        const int p = eng.checkpoint_at_or_below(slot_idx, (int)L);
        return p > 0 ? (size_t)p : 0;
    }

    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < eng.n_slots(); ++i) {
            const SessMeta& ss = sess[(size_t)i];
            if (ss.active) continue;
            const size_t l = resume_len(i, ids, imgs);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < sess[(size_t)lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        const char* p = line.c_str() + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int)b;
            if (e1 == line.c_str() || b < 0 || b >= (long)slots.size() || slots[(size_t)b].active || eng.n_slots() < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d)\n", eng.n_slots() < 2 ? 0 : eng.n_slots());
                std::fflush(stdout);
                return;
            }
        }
        while (*p == ' ') ++p;
        char* end_max = nullptr;
        const long long max_new = std::strtoll(p, &end_max, 10);
        if (end_max == p) {
            std::printf("ERR bad max_new\n");
            std::fflush(stdout);
            return;
        }
        p = end_max;

        strata::qwen35::Sampling smp;
        std::vector<uint64_t> img_keys;
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
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t)x);
                    q = (*qe == ',') ? (qe + 1) : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }

        std::string emb_path;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb_path.assign(p, (size_t)(e - p));
            p = e;
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb_path.empty())) {
            std::printf("ERR bad request: max_new or ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t)ids.size() + max_new > ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds context (%d)\n",
                        ids.size(), max_new, ctx);
            std::fflush(stdout);
            return;
        }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb_path, ids, eng.n_embd(), img_keys, ir);
            if (!ve.empty()) {
                std::printf("ERR %s\n", ve.c_str());
                std::fflush(stdout);
                return;
            }
        }
        bool bad_tok = false;
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool is_img = geni && ir.cell_row[i] >= 0;
            if (!is_img && (ids[i] < 0 || ids[i] >= (int32_t)nv)) {
                bad_tok = true;
                break;
            }
        }
        if (bad_tok) {
            std::printf("ERR token id outside vocabulary\n");
            std::fflush(stdout);
            return;
        }

        stop.store(false);
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        SessMeta& ss = sess[(size_t)s];
        ss.used = ++use_seq;
        size_t reuse = resume_len(s, ids, ir.imgs);
        if (adm) {
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < eng.n_slots(); ++i) {
                if (i == s || sess[(size_t)i].active) continue;
                const size_t l = resume_len(i, ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0 && eng.copy_session(src, s)) {
                ss.live = sess[(size_t)src].live;
                ss.imgs = sess[(size_t)src].imgs;
                reuse = best;
            }
        }

        if (reuse == 0) {
            eng.reset_slot(s);
            ss.live.clear();
            ss.imgs.clear();
        } else if (reuse < ss.live.size()) {
            if (!eng.restore_checkpoint(s, (int)reuse)) {
                eng.reset_slot(s);
                ss.live.clear();
                ss.imgs.clear();
                reuse = 0;
            } else {
                ss.live.resize(reuse);
                std::vector<ImgKey> hold;
                for (const ImgKey& k : ss.imgs)
                    if (k.cell + k.n <= (int64_t)reuse) hold.push_back(k);
                ss.imgs.swap(hold);
            }
        }
        eng.set_mrope_delta(s, geni ? ir.delta : 0);

        const size_t fresh = ids.size() - reuse;
        const auto t_pf0 = Clock::now();
        bool cancelled = false;
        bool fwd_err = false;
        std::string err;
        const float* logits = nullptr;
        size_t done = 0;

        while (done < fresh) {
            if (stop.load()) {
                cancelled = true;
                break;
            }
            const size_t cur_pos = reuse + done;
            size_t step_n = std::min<size_t>(64, fresh - done);
            if (ckpt_n > 0) {
                const size_t next_ck = ((cur_pos / (size_t)ckpt_step) + 1) * (size_t)ckpt_step;
                if (cur_pos + step_n > next_ck) step_n = next_ck - cur_pos;
            }
            const bool last_pf = (done + step_n == fresh);
            const int32_t* mr_p = geni ? (ir.mrope.data() + cur_pos * 3) : nullptr;
            const float* im_p = geni ? ir.rows.data() : nullptr;
            const int n_im = geni ? (int)(ir.rows.size() / (size_t)eng.n_embd()) : 0;
            const int32_t* cr_p = geni ? (ir.cell_row.data() + cur_pos) : nullptr;

            if (!eng.forward_chunk(s, ids.data() + cur_pos, (int)step_n,
                                   mr_p, im_p, n_im, cr_p, last_pf, logits, err)) {
                fwd_err = true;
                break;
            }
            ss.live.insert(ss.live.end(), ids.begin() + (long)cur_pos, ids.begin() + (long)(cur_pos + step_n));
            done += step_n;

            const size_t new_pos = reuse + done;
            const bool mid_image = geni && (ir.cell_row[new_pos - 1] >= 0);
            if (ckpt_n > 0 && !mid_image &&
                ((new_pos % (size_t)ckpt_step == 0) || last_pf)) {
                eng.save_checkpoint(s);
            }

            const double pms = ms_since(t_pf0);
            std::printf("PP %zu %zu %.0f %.1f\n", done, fresh, pms,
                        (double)done / std::max(1e-9, pms / 1000.0));
            std::fflush(stdout);

            if (n_active > 0 && !last_pf) {
                window();
            } else {
                poll_control();
            }
        }

        if (fwd_err) {
            eng.reset_slot(s);
            ss.live.clear();
            ss.imgs.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            return;
        }

        ss.imgs.clear();
        bool img_cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t)ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t)ss.live.size()) img_cut = true;
        }
        if (cancelled && img_cut) {
            eng.reset_slot(s);
            ss.live.clear();
            ss.imgs.clear();
        }

        const double prompt_ms = ms_since(t_pf0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %zu 0\n",
                        ids.size(), prompt_ms, reuse, done);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }

        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, pen_prompt);
        auto pick_from = [&](const float* lg_ptr) -> int32_t {
            if (!smp.needs_logits()) return argmax_logits(lg_ptr, nv);
            std::memcpy(scratch_lg.data(), lg_ptr, 4 * (size_t)nv);
            return sample_logits(scratch_lg, smp, hist, rng);
        };

        const auto t_dec0 = Clock::now();
        int32_t nx = pick_from(logits);
        if (adm) {
            std::printf("T %d\n", nx);
            hist.push_back(nx);
            const long long produced = 1;
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : eng.pos(s) + 1 > ctx ? "length" : nullptr;
            const bool cont = (fin == nullptr);
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n",
                        produced, ids.size(), prompt_ms, ms_since(t_dec0),
                        cont ? "length" : fin, reuse, fresh);
            if (cont) {
                DecodeSlot& sl = slots[(size_t)slot];
                sl = DecodeSlot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = std::move(hist);
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata-q36-metal batch: slot %d prompt %zu tok (%zu reused + %zu fresh) in %.0f ms (%.1f tok/s)%s\n",
                         slot, ids.size(), reuse, fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double)fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }

        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        const bool use_mtp = eng.has_mtp() && mtp_d > 0 && !smp.needs_logits() && !adm && n_active == 0;
        std::vector<int32_t> fix, drafts, picks;
        auto emit = [&](int32_t t) -> bool {
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++produced;
            hist.push_back(t);
            if (is_eos(t)) { finish = "stop"; return false; }
            if (produced >= max_new) { finish = "length"; return false; }
            if (stop.load()) { finish = "cancel"; return false; }
            return true;
        };
        for (;;) {
            if (!emit(nx)) break;
            if (n_active > 0) window();

            if (use_mtp && n_active == 0 && eng.pos(s) + mtp_d + 1 <= ctx) {
                int acc = 0;
                if (!eng.spec_step(s, nx, fix, mtp_d, drafts, picks, acc, err)) {
                    eng.reset_slot(s);
                    ss.live.clear();
                    ss.imgs.clear();
                    finish = "cancel";
                    std::fprintf(stderr, "strata-q36-metal serve spec error: %s\n", err.c_str());
                    break;
                }
                offered += mtp_d;
                accepted += acc;
                ss.live.push_back(nx);
                for (int i = 0; i < acc; ++i) ss.live.push_back(drafts[(size_t)i]);
                fix.assign(drafts.begin(), drafts.begin() + acc);
                if (ckpt_n > 0 && (eng.pos(s) / ckpt_step) > ((eng.pos(s) - (acc + 1)) / ckpt_step)) {
                    eng.save_checkpoint(s);
                }
                bool ended = false;
                for (int i = 0; i < acc && !ended; ++i) {
                    if (!emit(drafts[(size_t)i])) ended = true;
                }
                if (ended) break;
                nx = picks[(size_t)acc];
                continue;
            }

            if (eng.pos(s) + 1 > ctx) { finish = "length"; break; }
            if (!eng.forward_chunk(s, &nx, 1, nullptr, nullptr, 0, nullptr, true, logits, err)) {
                eng.reset_slot(s);
                ss.live.clear();
                ss.imgs.clear();
                finish = "cancel";
                std::fprintf(stderr, "strata-q36-metal serve decode error: %s\n", err.c_str());
                break;
            }
            ss.live.push_back(nx);
            fix.clear();
            nx = pick_from(logits);
        }

        const double decode_ms = ms_since(t_dec0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %zu 0\n",
                    produced, ids.size(), prompt_ms, decode_ms, finish, accepted, offered, reuse, fresh);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "strata-q36-metal serve: session %d prompt %zu tok (%zu reused + %zu fresh) in %.0f ms (%.1f tok/s), "
                     "gen %lld tok in %.0f ms (%.1f tok/s, %s), drafts accepted %lld of %lld, %d ckpt\n",
                     s, ids.size(), reuse, fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double)fresh / prompt_ms : 0.0,
                     produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double)produced / decode_ms : 0.0,
                     finish, accepted, offered, eng.checkpoint_count(s));
    }

    int run() {
        if (eng.n_slots() > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", eng.n_slots());
        std::printf("READY %d stop\n", ctx);
        std::fflush(stdout);

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
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = std::move(lines.front());
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0 && eof) {
                    break;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int)slots.size()) slots[(size_t)b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
            } else {
                request(line);
            }
            if (n_active > 0) window();
        }

        reader.join();
        return 0;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tokens_csv, dump_logits, profile_path, save_profile_path;
    bool serve_mode = false;
    bool vision = false;
    int n_gen = 16;
    int ctx = -1;
    int n_slots = 1;
    int mtp_d = 0;
    bool stream = false;
    double arena_gb = -1.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--serve") serve_mode = true;
        else if ((a == "--model" || a == "--native") && i + 1 < argc) model_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if ((a == "--ctx" || a == "--max-context") && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if ((a == "--batch" || a == "--slots") && i + 1 < argc) n_slots = std::max(1, std::min(8, std::atoi(argv[++i])));
        else if (a == "--vision") vision = true;
        else if (a == "--mtp" && i + 1 < argc) mtp_d = std::max(0, std::atoi(argv[++i]));
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
        else if (a == "--stream") stream = true;
        else if (a == "--arena-gb" && i + 1 < argc) { arena_gb = std::atof(argv[++i]); stream = true; }
        else if (a == "--expert-profile" && i + 1 < argc) profile_path = argv[++i];
        else if (a == "--save-expert-profile" && i + 1 < argc) save_profile_path = argv[++i];
        else if (serve_mode && a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "strata-q36-metal: ignoring unsupported serve flag %s\n", a.c_str());
            if (i + 1 < argc && argv[i + 1][0] != '-') ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36-metal [--serve] --model <gguf> [--tokens <id,id,...>] [--n N] [--ctx C] [--batch S] [--mtp D] [--vision] [--stream] [--arena-gb G] [--expert-profile P] [--save-expert-profile P]\n");
            return 2;
        }
    }

    if (ctx <= 0) ctx = serve_mode ? 8192 : 2048;

    int ckpt_n = (n_slots == 1) ? 12 : 8;
    if (const char* e = std::getenv("STRATA_Q36_CKPT_N")) ckpt_n = std::max(0, std::atoi(e));
    int ckpt_step = 256;
    if (const char* e = std::getenv("STRATA_Q36_CKPT_STEP")) ckpt_step = std::max(1, std::atoi(e));

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
    opt.n_slots = n_slots;
    opt.ckpt_count = ckpt_n;
    opt.mtp_drafts = mtp_d;
    opt.profile_path = profile_path;
    std::string err;
    const double t_load0 = mtl::now_s();
    if (!eng.open(model_path, opt, err)) {
        std::fprintf(stderr, "strata-q36-metal: %s\n", err.c_str());
        return 1;
    }
    const double t_load = mtl::now_s() - t_load0;

    if (serve_mode) {
        ServeRunner runner(eng, ctx, vision, ckpt_step, ckpt_n, mtp_d);
        const int rc = runner.run();
        if (!save_profile_path.empty()) {
            if (!eng.save_expert_profile(save_profile_path, err))
                std::fprintf(stderr, "strata-q36-metal: save_expert_profile: %s\n", err.c_str());
        }
        return rc;
    }

    std::vector<int32_t> prompt = parse_ids(tokens_csv.c_str());
    if (prompt.empty()) {
        std::fprintf(stderr, "strata-q36-metal: empty prompt tokens\n");
        return 2;
    }

    const double t_pf0 = mtl::now_s();
    const float* logits = nullptr;
    if (!eng.forward_chunk(0, prompt.data(), (int)prompt.size(), nullptr, nullptr, 0, nullptr, true, logits, err)) {
        std::fprintf(stderr, "strata-q36-metal: forward error: %s\n", err.c_str());
        return 1;
    }
    const double t_pf = mtl::now_s() - t_pf0;

    std::ofstream dump;
    if (!dump_logits.empty()) dump.open(dump_logits, std::ios::binary);

    std::vector<int32_t> gen;
    const double t_dec0 = mtl::now_s();
    int32_t next_tok = argmax_logits(logits, eng.n_vocab());
    long long offered = 0, accepted = 0;
    if (eng.has_mtp() && mtp_d > 0) {
        std::vector<int32_t> fix, drafts, picks;
        while ((int)gen.size() < n_gen) {
            gen.push_back(next_tok);
            if ((int)gen.size() >= n_gen) break;
            if (eng.pos(0) + mtp_d + 1 <= ctx) {
                int acc = 0;
                if (!eng.spec_step(0, next_tok, fix, mtp_d, drafts, picks, acc, err)) {
                    std::fprintf(stderr, "strata-q36-metal: spec_step error: %s\n", err.c_str());
                    return 1;
                }
                offered += mtp_d;
                accepted += acc;
                for (int i = 0; i < acc && (int)gen.size() < n_gen; ++i) {
                    gen.push_back(drafts[(size_t)i]);
                }
                if ((int)gen.size() >= n_gen) break;
                fix.assign(drafts.begin(), drafts.begin() + acc);
                next_tok = picks[(size_t)acc];
            } else {
                if (!eng.forward(next_tok, logits, err)) {
                    std::fprintf(stderr, "strata-q36-metal: decode error: %s\n", err.c_str());
                    return 1;
                }
                fix.clear();
                next_tok = argmax_logits(logits, eng.n_vocab());
            }
        }
    } else {
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
    }
    const double t_dec = mtl::now_s() - t_dec0;

    std::printf("gen:");
    for (int32_t id : gen) std::printf(" %d", id);
    std::printf("\n");
    std::fprintf(stderr, "timing: load=%.2fs prefill=%zu tok in %.2fs (%.1f tok/s) decode=%d tok in %.2fs (%.1f tok/s)\n",
                 t_load, prompt.size(), t_pf, prompt.size() / std::max(t_pf, 1e-6),
                 n_gen, t_dec, (n_gen > 1 ? (n_gen - 1) : 1) / std::max(t_dec, 1e-6));
    if (offered > 0) {
        std::fprintf(stderr, "mtp: drafts accepted %lld of %lld (%.1f%%)\n",
                     accepted, offered, 100.0 * (double)accepted / (double)offered);
    }
    const auto st = eng.stream_stats();
    if (st.streamed) {
        const double total_acc = (double)(st.hits + st.misses);
        const double hit_pct = total_acc > 0 ? 100.0 * (double)st.hits / total_acc : 0.0;
        std::fprintf(stderr,
                     "stream: dense=%.2f MiB arena=%.2f MiB contig=%s hits=%llu misses=%llu (hit=%.1f%%) pre_hits=%llu pre_issued=%llu reads=%llu read_mb=%.2f\n",
                     st.dense_bytes / (1024.0 * 1024.0), st.arena_bytes / (1024.0 * 1024.0),
                     st.using_contig ? "yes" : "no",
                     (unsigned long long)st.hits, (unsigned long long)st.misses, hit_pct,
                     (unsigned long long)st.pre_hits, (unsigned long long)st.pre_issued,
                     (unsigned long long)st.reads, st.bytes_read / (1024.0 * 1024.0));
    }
    if (!save_profile_path.empty()) {
        if (!eng.save_expert_profile(save_profile_path, err)) {
            std::fprintf(stderr, "strata-q36-metal: save_expert_profile: %s\n", err.c_str());
            return 1;
        }
    }
    return 0;
}
