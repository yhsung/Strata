// metal/tests/q36_attn_parity.cpp - TDD Cycle 2: Qwen3.6 Dense GQA + output gate + per-head RMSNorm + IMROPE
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int H = 16, HKV = 2, D = 256, N_ROT = 64, MAX_CTX = 64;

struct Q36AttnArgs {
    uint32_t H, HKV, D, n_rot;
    int32_t pos_t, pos_h, pos_w, kv_pos;
    uint32_t n_pos, ctx, plane;
    float base, eps, scale;
};

struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"q36_attn.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());

    mtl::Pipeline p_prep = cx.pipeline(lib, "q36_attn_prep");
    mtl::Pipeline p_dec = cx.pipeline(lib, "q36_attn_decode");
    mtl::require_fit(p_prep, 256, "q36_attn_prep");
    mtl::require_fit(p_dec, 256, "q36_attn_decode");

    mtl::Buffer qg = cx.alloc(4 * H * 2 * D), kx = cx.alloc(4 * HKV * D), vx = cx.alloc(4 * HKV * D),
                qw = cx.alloc(4 * D), kw = cx.alloc(4 * D),
                q = cx.alloc(4 * H * D), gate = cx.alloc(4 * H * D),
                kc = cx.alloc(4 * (size_t)HKV * MAX_CTX * D),
                vc = cx.alloc(4 * (size_t)HKV * MAX_CTX * D);

    const uint32_t n_out = H * D;
    const uint32_t plane = n_out;
    mtl::Buffer att = cx.alloc(4 * (4 * plane + plane / 16));

    ref::Rng r(123);
    for (int i = 0; i < D; ++i) {
        qw.as<float>()[i] = 1.0f + 0.25f * r.sym();
        kw.as<float>()[i] = 1.0f + 0.25f * r.sym();
    }

    std::vector<double> kc_ref((size_t)HKV * MAX_CTX * D, 0.0);
    std::vector<double> vc_ref((size_t)HKV * MAX_CTX * D, 0.0);

    int fails = 0;
    const int test_positions[] = {0, 1, 7, 35};
    // Fill consecutive positions 0..35, checking parity at 0, 1, 7, 35
    for (int pos = 0; pos <= 35; ++pos) {
        for (int i = 0; i < H * 2 * D; ++i) qg.as<float>()[i] = r.sym();
        for (int i = 0; i < HKV * D; ++i) {
            kx.as<float>()[i] = r.sym();
            vx.as<float>()[i] = r.sym();
        }

        // Test IMROPE with distinct (pos_t, pos_h, pos_w) at pos==7, scalar pos otherwise
        const int pt = pos;
        const int ph = (pos == 7) ? 13 : pos;
        const int pw = (pos == 7) ? 29 : pos;

        Q36AttnArgs args{H, HKV, D, N_ROT, pt, ph, pw, pos, (uint32_t)(pos + 1), MAX_CTX, plane,
                         10000000.0f, 1e-6f, 1.0f / std::sqrt((float)D)};

        mtl::Dispatch d1;
        d1.p = &p_prep;
        d1.args = {mtl::Arg::B(qg), mtl::Arg::B(kx), mtl::Arg::B(vx), mtl::Arg::B(qw), mtl::Arg::B(kw),
                   mtl::Arg::B(q), mtl::Arg::B(gate), mtl::Arg::B(kc), mtl::Arg::B(vc), mtl::Arg::V(args)};
        d1.grid[0] = H + HKV;
        d1.tg[0] = 256;

        mtl::Dispatch d2;
        d2.p = &p_dec;
        d2.args = {mtl::Arg::B(q), mtl::Arg::B(kc), mtl::Arg::B(vc), mtl::Arg::B(gate), mtl::Arg::B(att),
                   mtl::Arg::V(args)};
        d2.grid[0] = H;
        d2.tg[0] = 256;

        cx.run_graph({d1, d2});

        // Double-precision CPU reference
        std::vector<double> q_ref(H * D), g_ref(H * D);
        const int half = N_ROT / 2;
        for (int head = 0; head < H + HKV; ++head) {
            const bool isq = head < H;
            const float* src = isq ? qg.as<float>() + head * 2 * D : kx.as<float>() + (head - H) * D;
            const float* nw = isq ? qw.as<float>() : kw.as<float>();
            double ss = 0;
            for (int i = 0; i < D; ++i) ss += (double)src[i] * src[i];
            const double rms = 1.0 / std::sqrt(ss / D + 1e-6);
            std::vector<double> buf(D);
            for (int i = 0; i < D; ++i) buf[i] = (double)src[i] * rms * nw[i];

            double* dst = isq ? &q_ref[head * D] : &kc_ref[((size_t)(head - H) * MAX_CTX + pos) * D];
            for (int i = 0; i < D; ++i) {
                double v = buf[i];
                if (i < N_ROT) {
                    const int p = i < half ? i : i - half;
                    const int mpos = (p % 3 == 0) ? pt : ((p % 3 == 1) ? ph : pw);
                    const double theta = (double)mpos * std::pow(10000000.0, -2.0 * p / N_ROT);
                    const double cs = std::cos(theta), sn = std::sin(theta);
                    v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
                }
                dst[i] = v;
            }
            if (isq) {
                for (int i = 0; i < D; ++i) g_ref[head * D + i] = qg.as<float>()[head * 2 * D + D + i];
            } else {
                for (int i = 0; i < D; ++i)
                    vc_ref[((size_t)(head - H) * MAX_CTX + pos) * D + i] = vx.as<float>()[(head - H) * D + i];
            }
        }

        std::vector<double> att_ref(H * D);
        for (int h = 0; h < H; ++h) {
            const int kvh = h / (H / HKV);
            std::vector<double> sc(pos + 1);
            double mx = -1e30;
            for (int p = 0; p <= pos; ++p) {
                double s = 0;
                for (int i = 0; i < D; ++i) s += q_ref[h * D + i] * kc_ref[((size_t)kvh * MAX_CTX + p) * D + i];
                sc[p] = s / std::sqrt((double)D);
                mx = std::max(mx, sc[p]);
            }
            double sum = 0;
            for (int p = 0; p <= pos; ++p) {
                sc[p] = std::exp(sc[p] - mx);
                sum += sc[p];
            }
            for (int i = 0; i < D; ++i) {
                double acc = 0;
                for (int p = 0; p <= pos; ++p) acc += sc[p] * vc_ref[((size_t)kvh * MAX_CTX + p) * D + i];
                const double sig = 1.0 / (1.0 + std::exp(-g_ref[h * D + i]));
                att_ref[h * D + i] = (acc / sum) * sig;
            }
        }

        if (std::find(std::begin(test_positions), std::end(test_positions), pos) != std::end(test_positions)) {
            Err eq, ek, ea, ex1, ex2, ex3, exsum;
            for (int i = 0; i < H * D; ++i) eq.add(q.as<float>()[i], q_ref[i]);
            for (int h = 0; h < HKV; ++h)
                for (int i = 0; i < D; ++i)
                    ek.add(kc.as<float>()[((size_t)h * MAX_CTX + pos) * D + i],
                           kc_ref[((size_t)h * MAX_CTX + pos) * D + i]);
            const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
            const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
            const float* ap = att.as<float>();
            for (uint32_t i = 0; i < n_out; ++i) {
                ea.add(ap[i], att_ref[i]);
                ex1.add(ap[plane + i], att_ref[i] * q2s[i & 7]);
                ex2.add(ap[2 * plane + i], att_ref[i] * p4s[i & 3]);
                ex3.add(ap[3 * plane + i], (i & 1) ? att_ref[i] * 0.00390625 : att_ref[i]);
            }
            for (uint32_t g = 0; g < n_out / 16; ++g) {
                double s = 0;
                for (int j = 0; j < 16; ++j) s += att_ref[g * 16 + j];
                exsum.add(ap[4 * plane + g], s);
            }
            const bool ok = eq.l2() < 1e-5 && ek.l2() < 1e-5 && ea.l2() < 1e-5 &&
                            ex1.l2() < 1e-5 && ex2.l2() < 1e-5 && ex3.l2() < 1e-5 && exsum.l2() < 1e-5;
            fails += !ok;
            std::printf("q36_attn_parity pos=%2d: q %.2e k %.2e att %.2e xprep(%.2e,%.2e,%.2e,sum %.2e)  %s\n",
                        pos, eq.l2(), ek.l2(), ea.l2(), ex1.l2(), ex2.l2(), ex3.l2(), exsum.l2(), ok ? "PASS" : "FAIL");
        }
    }
    return fails ? 1 : 0;
}
