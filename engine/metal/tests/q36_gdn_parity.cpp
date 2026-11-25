// metal/tests/q36_gdn_parity.cpp - TDD Cycle 1: Qwen3.6 GDN SiLU output gate (FC 32 = GDN_SILU_C)
// vs Flash-Next default sigmoid output gate (FC 32 unset).
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int S = 128, HK = 16, HV = 32, C = (2 * HK + HV) * S, MAXT = 4;
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct Err {
    double se = 0, sr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());

    const size_t st_elems = (size_t)S * HV * S;
    mtl::Buffer state = cx.alloc(4 * st_elems), st0 = cx.alloc(4 * st_elems),
                hist = cx.alloc(4 * C * 3), h0 = cx.alloc(4 * C * 3),
                qkv = cx.alloc(4 * C * MAXT), al = cx.alloc(4 * HV * MAXT),
                be = cx.alloc(4 * HV * MAXT), z = cx.alloc(4 * HV * S * MAXT),
                cw = cx.alloc(4 * C * 4), dt = cx.alloc(4 * HV),
                sa = cx.alloc(4 * HV), gm = cx.alloc(4 * S),
                y = cx.alloc(4 * HV * S * MAXT), hist2 = cx.alloc(4 * C * 3);

    ref::Rng r(42);
    for (size_t i = 0; i < st_elems; ++i) st0.as<float>()[i] = r.sym() * 0.1f;
    for (int i = 0; i < C * 3; ++i) h0.as<float>()[i] = r.sym();
    for (int i = 0; i < C * MAXT; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < HV * MAXT; ++i) { al.as<float>()[i] = r.sym() * 2.0f; be.as<float>()[i] = r.sym() * 2.0f; }
    for (int i = 0; i < HV * S * MAXT; ++i) z.as<float>()[i] = r.sym() * 2.0f;
    for (int i = 0; i < C * 4; ++i) cw.as<float>()[i] = r.sym() * 0.5f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -std::exp(r.sym()); }
    for (int i = 0; i < S; ++i) gm.as<float>()[i] = 1.0f + 0.2f * r.sym();

    auto run_case = [&](int nt, bool use_silu) -> bool {
        mtl::Constants c = mtl::Constants().i(2, nt);
        if (use_silu) c.b(32, true);
        mtl::Pipeline p = cx.pipeline(lib, "gdn_step", c);
        mtl::require_fit(p, 512, "q36_gdn_parity");

        std::memcpy(state.ptr, st0.ptr, 4 * st_elems);
        std::memcpy(hist.ptr, h0.ptr, 4 * C * 3);

        GdnArgs a{HK, HV, C, 1u, 1e-6f, 0};
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(state), mtl::Arg::B(hist), mtl::Arg::B(qkv), mtl::Arg::B(al),
                  mtl::Arg::B(be), mtl::Arg::B(z), mtl::Arg::B(cw), mtl::Arg::B(dt),
                  mtl::Arg::B(sa), mtl::Arg::B(gm), mtl::Arg::B(y), mtl::Arg::V(a),
                  mtl::Arg::B(hist2)};
        d.grid[0] = HV;
        d.tg[0] = 512;
        cx.run({d});

        std::vector<double> Sd(st_elems), Hd(C * 3);
        for (size_t i = 0; i < st_elems; ++i) Sd[i] = st0.as<float>()[i];
        for (int i = 0; i < C * 3; ++i) Hd[i] = h0.as<float>()[i];

        Err ey, es, eh;
        for (int t = 0; t < nt; ++t) {
            std::vector<double> cv(C);
            for (int ch = 0; ch < C; ++ch) {
                const double in = qkv.as<float>()[t * C + ch];
                double s = Hd[ch * 3] * cw.as<float>()[ch * 4] +
                           Hd[ch * 3 + 1] * cw.as<float>()[ch * 4 + 1] +
                           Hd[ch * 3 + 2] * cw.as<float>()[ch * 4 + 2] +
                           in * cw.as<float>()[ch * 4 + 3];
                cv[ch] = s / (1.0 + std::exp(-s));
                Hd[ch * 3] = Hd[ch * 3 + 1];
                Hd[ch * 3 + 1] = Hd[ch * 3 + 2];
                Hd[ch * 3 + 2] = in;
            }
            for (int hh = 0; hh < 2 * HK; ++hh) {
                double ss = 0;
                for (int i = 0; i < S; ++i) ss += cv[hh * S + i] * cv[hh * S + i];
                const double rs = 1.0 / std::sqrt(ss / S + 1e-6 / S);
                for (int i = 0; i < S; ++i) cv[hh * S + i] = cv[hh * S + i] * rs / std::sqrt((double)S);
            }
            for (int h = 0; h < HV; ++h) {
                const int hk = h % HK;
                const double* q = &cv[hk * S];
                const double* k = &cv[HK * S + hk * S];
                const double* v = &cv[2 * HK * S + h * S];
                const double av = (double)al.as<float>()[t * HV + h] + dt.as<float>()[h];
                const double sp = av > 20 ? av : std::log1p(std::exp(av));
                const double g = std::exp(sp * sa.as<float>()[h]);
                const double b = 1.0 / (1.0 + std::exp(-(double)be.as<float>()[t * HV + h]));
                std::vector<double> o(S);
                for (int jj = 0; jj < S; ++jj) {
                    double kv = 0;
                    for (int i = 0; i < S; ++i) kv += Sd[((size_t)i * HV + h) * S + jj] * k[i];
                    const double delta = (v[jj] - g * kv) * b;
                    double acc = 0;
                    for (int i = 0; i < S; ++i) {
                        double& sv = Sd[((size_t)i * HV + h) * S + jj];
                        sv = g * sv + k[i] * delta;
                        acc += sv * q[i];
                    }
                    o[jj] = acc / std::sqrt((double)S);
                }
                double ss = 0;
                for (int jj = 0; jj < S; ++jj) ss += o[jj] * o[jj];
                const double rs = 1.0 / std::sqrt(ss / S + 1e-6);
                for (int jj = 0; jj < S; ++jj) {
                    const double zz = z.as<float>()[t * HV * S + h * S + jj];
                    const double gz = use_silu ? (zz / (1.0 + std::exp(-zz))) : (1.0 / (1.0 + std::exp(-zz)));
                    ey.add(y.as<float>()[t * HV * S + h * S + jj], o[jj] * rs * gm.as<float>()[jj] * gz);
                }
            }
        }
        for (size_t i = 0; i < st_elems; ++i) es.add(state.as<float>()[i], Sd[i]);
        for (int i = 0; i < C * 3; ++i) eh.add(hist2.as<float>()[i], Hd[i]);

        const bool ok = ey.l2() < 1e-5 && es.l2() < 1e-5 && eh.l2() == 0.0;
        std::printf("q36_gdn_parity (%s, ntok=%d): y rel L2 %.2e, state %.2e, hist %.1e  %s\n",
                    use_silu ? "SiLU" : "sigmoid", nt, ey.l2(), es.l2(), eh.l2(), ok ? "PASS" : "FAIL");
        return ok;
    };

    int fails = 0;
    fails += !run_case(1, false);
    for (int nt = 1; nt <= MAXT; ++nt) fails += !run_case(nt, true);
    return fails ? 1 : 0;
}
