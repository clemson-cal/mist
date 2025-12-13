/**
================================================================================
Copyright 2023 - 2025, Jonathan Zrake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
*/

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include "mist/ascii_reader.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/core.hpp"
#include "mist/driver.hpp"
#include "mist/ndarray.hpp"
#include "mist/serialize.hpp"

using namespace mist;

// =============================================================================
// Type aliases
// =============================================================================

using cons_t = dvec_t<3>;  // conserved variables: (D, S, tau)
using prim_t = dvec_t<3>;  // primitive variables: (rho, u, p)

// =============================================================================
// Math utility functions
// =============================================================================

static constexpr double gamma_law = 4.0 / 3.0;

static inline auto min2(double a, double b) -> double {
    return a < b ? a : b;
}

static inline auto max2(double a, double b) -> double {
    return a > b ? a : b;
}

static inline auto min3(double a, double b, double c) -> double {
    return min2(a, min2(b, c));
}

static inline auto max3(double a, double b, double c) -> double {
    return max2(a, max2(b, c));
}

static inline auto sign(double x) -> double {
    return std::copysign(1.0, x);
}

static inline auto minabs(double a, double b, double c) -> double {
    return min3(std::fabs(a), std::fabs(b), std::fabs(c));
}

static inline auto plm_minmod(double yl, double yc, double yr, double plm_theta) -> double {
    auto a = (yc - yl) * plm_theta;
    auto b = (yr - yl) * 0.5;
    auto c = (yr - yc) * plm_theta;
    return 0.25 * std::fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

// =============================================================================
// SR Hydrodynamics functions
// =============================================================================

static auto gamma_beta_squared(prim_t p) -> double {
    return p[1] * p[1];
}

static auto momentum_squared(cons_t u) -> double {
    return u[1] * u[1];
}

static auto lorentz_factor(prim_t p) -> double {
    return std::sqrt(1.0 + gamma_beta_squared(p));
}

static auto beta(prim_t p) -> double {
    return p[1] / lorentz_factor(p);
}

static auto enthalpy_density(prim_t p) -> double {
    auto rho = p[0];
    auto pre = p[2];
    return rho + pre * (1.0 + 1.0 / (gamma_law - 1.0));
}

static auto prim_to_cons(prim_t p) -> cons_t {
    auto rho = p[0];
    auto pre = p[2];
    auto w = lorentz_factor(p);
    auto h = enthalpy_density(p) / rho;
    auto m = rho * w;
    auto u = cons_t{};
    u[0] = m;
    u[1] = m * (h * p[1]);
    u[2] = m * (h * w - 1.0) - pre;
    return u;
}

static auto cons_to_prim(cons_t cons, double p = 0.0) -> prim_t {
    auto newton_iter_max = 50;
    auto error_tolerance = 1e-12 * (cons[0] + cons[2]);
    auto gm = gamma_law;
    auto m = cons[0];
    auto tau = cons[2];
    auto ss = momentum_squared(cons);
    auto w0 = 0.0;

    for (int n = 0; n < newton_iter_max; ++n) {
        auto et = tau + p + m;
        auto b2 = min2(ss / et / et, 1.0 - 1e-10);
        auto w2 = 1.0 / (1.0 - b2);
        auto w = std::sqrt(w2);
        auto d = m / w;
        auto de = (tau + m * (1.0 - w) + p * (1.0 - w2)) / w2;
        auto dh = d + de + p;
        auto a2 = gm * p / dh;
        auto g = b2 * a2 - 1.0;
        auto f = de * (gm - 1.0) - p;

        if (std::fabs(f) < error_tolerance) {
            w0 = w;
            break;
        }
        p -= f / g;
    }
    return prim_t{m / w0, w0 * cons[1] / (tau + m + p), p};
}

static auto prim_and_cons_to_flux(prim_t p, cons_t u) -> cons_t {
    auto pre = p[2];
    auto vn = beta(p);
    auto f = cons_t{};
    f[0] = vn * u[0];
    f[1] = vn * u[1] + pre;
    f[2] = vn * u[2] + pre * vn;
    return f;
}

static auto sound_speed_squared(prim_t p) -> double {
    auto pre = p[2];
    auto rho_h = enthalpy_density(p);
    return gamma_law * pre / rho_h;
}

static auto outer_wavespeeds(prim_t p) -> dvec_t<2> {
    auto a2 = sound_speed_squared(p);
    auto uu = gamma_beta_squared(p);
    auto vn = beta(p);
    auto g2 = 1.0 + uu;
    auto s2 = a2 / g2 / (1.0 - a2);
    auto v2 = vn * vn;
    auto k0 = std::sqrt(s2 * (1.0 - v2 + s2));
    return dvec(vn - k0, vn + k0) / (1.0 + s2);
}

static auto riemann_hlle(prim_t pl, prim_t pr, cons_t ul, cons_t ur, double v_face = 0.0) -> cons_t {
    auto fl = prim_and_cons_to_flux(pl, ul);
    auto fr = prim_and_cons_to_flux(pr, ur);

    auto cl = std::sqrt(sound_speed_squared(pl));
    auto cr = std::sqrt(sound_speed_squared(pr));
    auto vl = beta(pl);
    auto vr = beta(pr);
    auto alm = (vl - cl) / (1.0 - vl * cl);
    auto alp = (vl + cl) / (1.0 + vl * cl);
    auto arm = (vr - cr) / (1.0 - vr * cr);
    auto arp = (vr + cr) / (1.0 + vr * cr);
    auto am = min2(alm, arm);
    auto ap = max2(alp, arp);

    if (v_face < am) {
        return fl - ul * v_face;
    }
    if (v_face > ap) {
        return fr - ur * v_face;
    }
    auto u_hll = (ur * ap - ul * am + (fl - fr)) / (ap - am);
    auto f_hll = (fl * ap - fr * am - (ul - ur) * ap * am) / (ap - am);
    return f_hll - u_hll * v_face;
}

static auto max_wavespeed(prim_t p) -> double {
    auto ws = outer_wavespeeds(p);
    return max2(std::fabs(ws[0]), std::fabs(ws[1]));
}

// =============================================================================
// Helper functions
// =============================================================================

static auto cell_center_x(int i, double x0, double dx) -> double {
    return x0 + (i + 0.5) * dx;
}

// =============================================================================
// Initial conditions
// =============================================================================

enum class initial_condition {
    sod,
    blast_wave,
    wind
};

auto to_string(initial_condition ic) -> const char* {
    switch (ic) {
        case initial_condition::sod: return "sod";
        case initial_condition::blast_wave: return "blast_wave";
        case initial_condition::wind: return "wind";
    }
    return "unknown";
}

auto from_string(std::type_identity<initial_condition>, const std::string& s) -> initial_condition {
    if (s == "sod") return initial_condition::sod;
    if (s == "blast_wave") return initial_condition::blast_wave;
    if (s == "wind") return initial_condition::wind;
    throw std::runtime_error("unknown initial condition: " + s);
}

static auto initial_primitive(initial_condition ic, double x) -> prim_t {
    switch (ic) {
        case initial_condition::sod:
            // Relativistic Sod shock tube
            if (x < 0.5) {
                return prim_t{1.0, 0.0, 1.0};  // left state: rho=1, u=0, p=1
            } else {
                return prim_t{0.125, 0.0, 0.1};  // right state: rho=0.125, u=0, p=0.1
            }
        case initial_condition::blast_wave:
            // Strong blast wave (ultra-relativistic)
            if (x < 0.5) {
                return prim_t{1.0, 0.0, 1000.0};
            } else {
                return prim_t{1.0, 0.0, 0.01};
            }
        case initial_condition::wind:
            // Simple wind (for testing)
            return prim_t{1.0, 1.0, 0.01};  // rho=1, u=1 (gamma*beta), p=0.01
    }
    return prim_t{1.0, 0.0, 1.0};
}

// =============================================================================
// Boundary conditions
// =============================================================================

enum class boundary_condition {
    outflow,
    reflecting,
    periodic
};

auto to_string(boundary_condition bc) -> const char* {
    switch (bc) {
        case boundary_condition::outflow: return "outflow";
        case boundary_condition::reflecting: return "reflecting";
        case boundary_condition::periodic: return "periodic";
    }
    return "unknown";
}

auto from_string(std::type_identity<boundary_condition>, const std::string& s) -> boundary_condition {
    if (s == "outflow") return boundary_condition::outflow;
    if (s == "reflecting") return boundary_condition::reflecting;
    if (s == "periodic") return boundary_condition::periodic;
    throw std::runtime_error("unknown boundary condition: " + s);
}

// =============================================================================
// 1D Special Relativistic Hydrodynamics Physics Module
// =============================================================================

struct srhd {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double plm_theta = 1.5;
        initial_condition ic = initial_condition::sod;
        boundary_condition bc_lo = boundary_condition::outflow;
        boundary_condition bc_hi = boundary_condition::outflow;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("ic", ic),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("plm_theta", plm_theta),
                field("ic", ic),
                field("bc_lo", bc_lo),
                field("bc_hi", bc_hi)
            );
        }
    };

    struct initial_t {
        unsigned int num_zones = 400;
        double x0 = 0.0;
        double x1 = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("x0", x0),
                field("x1", x1)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("x0", x0),
                field("x1", x1)
            );
        }
    };

    struct state_t {
        index_space_t<1> interior;
        cached_t<cons_t, 1> cons;  // conserved variables: (D, S, tau)
        double time;
        double dt;

        state_t() = default;

        state_t(index_space_t<1> s)
            : interior(s)
            , cons(cache(fill(expand(s, 2), cons_t{0.0, 0.0, 0.0}), memory::host, exec::cpu))
            , time(0.0)
            , dt(0.0)
        {
        }

        auto fields() const {
            return std::make_tuple(
                field("cons", cons),
                field("time", time),
                field("dt", dt)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("cons", cons),
                field("time", time),
                field("dt", dt)
            );
        }
    };

    using product_t = std::vector<cached_t<double, 1>>;

    struct exec_context_t {
        const config_t& config;
        const initial_t& initial;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini) {}
    };
};

// =============================================================================
// Custom serialization for srhd::state_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const srhd::state_t& s) {
    ar.begin_group();
    // Serialize only interior data (guard zones are reconstructed from boundary conditions)
    auto interior = cache(map(s.cons[s.interior], std::identity{}), memory::host, exec::cpu);
    serialize(ar, "cons", interior);
    serialize(ar, "time", s.time);
    serialize(ar, "dt", s.dt);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, srhd::state_t& s) -> bool {
    if (!ar.begin_group()) return false;
    auto interior_data = cached_t<cons_t, 1>{};
    double time = 0.0;
    double dt = 0.0;
    deserialize(ar, "cons", interior_data);
    deserialize(ar, "time", time);
    deserialize(ar, "dt", dt);
    ar.end_group();

    // Reconstruct state from serialized interior data
    s = srhd::state_t(space(interior_data));
    copy(s.cons[s.interior], interior_data);
    s.time = time;
    s.dt = dt;
    return true;
}

// =============================================================================
// Physics interface implementation
// =============================================================================

auto default_physics_config(std::type_identity<srhd>) -> srhd::config_t {
    return {.rk_order = 1, .cfl = 0.4, .plm_theta = 1.5};
}

auto default_initial_config(std::type_identity<srhd>) -> srhd::initial_t {
    return {.num_zones = 400, .x0 = 0.0, .x1 = 1.0};
}

auto initial_state(const srhd::exec_context_t& ctx) -> srhd::state_t {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto S = index_space(ivec(0), uvec(ini.num_zones));
    auto state = srhd::state_t(S);
    auto dx = (ini.x1 - ini.x0) / ini.num_zones;

    // Initialize conserved variables from primitives
    for_each(state.interior, [&](ivec_t<1> idx) {
        auto i = idx[0];
        auto x = cell_center_x(i, ini.x0, dx);
        auto prim = initial_primitive(cfg.ic, x);
        state.cons[i] = prim_to_cons(prim);
    });

    return state;
}

static void apply_boundary_conditions(srhd::state_t& state, const srhd::config_t& cfg) {
    auto i0 = start(state.interior)[0];
    auto i1 = upper(state.interior)[0] - 1;

    // Left boundary
    switch (cfg.bc_lo) {
        case boundary_condition::outflow:
            for (int g = 0; g < 2; ++g) {
                state.cons[i0 - 1 - g] = state.cons[i0];
            }
            break;
        case boundary_condition::reflecting:
            for (int g = 0; g < 2; ++g) {
                auto u = state.cons[i0 + g];
                state.cons[i0 - 1 - g] = cons_t{u[0], -u[1], u[2]};  // Flip momentum
            }
            break;
        case boundary_condition::periodic:
            for (int g = 0; g < 2; ++g) {
                state.cons[i0 - 1 - g] = state.cons[i1 - g];
            }
            break;
    }

    // Right boundary
    switch (cfg.bc_hi) {
        case boundary_condition::outflow:
            for (int g = 0; g < 2; ++g) {
                state.cons[i1 + 1 + g] = state.cons[i1];
            }
            break;
        case boundary_condition::reflecting:
            for (int g = 0; g < 2; ++g) {
                auto u = state.cons[i1 - g];
                state.cons[i1 + 1 + g] = cons_t{u[0], -u[1], u[2]};  // Flip momentum
            }
            break;
        case boundary_condition::periodic:
            for (int g = 0; g < 2; ++g) {
                state.cons[i1 + 1 + g] = state.cons[i0 + g];
            }
            break;
    }
}

static auto compute_dt(const srhd::state_t& state, const srhd::exec_context_t& ctx) -> double {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto dx = (ini.x1 - ini.x0) / ini.num_zones;
    auto max_a = 0.0;

    for_each(state.interior, [&](ivec_t<1> idx) {
        auto i = idx[0];
        auto prim = cons_to_prim(state.cons[i]);
        max_a = max2(max_a, max_wavespeed(prim));
    });

    return cfg.cfl * dx / max_a;
}

static auto get_prim(const srhd::state_t& state, int i) -> prim_t {
    return cons_to_prim(state.cons[i]);
}

static auto get_cons(const srhd::state_t& state, int i) -> cons_t {
    return state.cons[i];
}

static void set_cons(srhd::state_t& state, int i, cons_t u) {
    state.cons[i] = u;
}

void advance(srhd::state_t& state, const srhd::exec_context_t& ctx) {
    auto& ini = ctx.initial;
    auto& cfg = ctx.config;
    auto dx = (ini.x1 - ini.x0) / ini.num_zones;

    // Apply boundary conditions
    apply_boundary_conditions(state, cfg);

    // Compute timestep
    state.dt = compute_dt(state, ctx);
    auto dtdx = state.dt / dx;

    auto i0 = start(state.interior)[0];
    auto i1 = upper(state.interior)[0];

    if (cfg.rk_order == 1) {
        // Forward Euler with PLM reconstruction
        auto du = std::vector<cons_t>(ini.num_zones);

        // Compute flux differences
        for (int i = i0; i < i1; ++i) {
            // Get primitive states for reconstruction
            auto pim2 = get_prim(state, i - 2);
            auto pim1 = get_prim(state, i - 1);
            auto pi = get_prim(state, i);
            auto pip1 = get_prim(state, i + 1);
            auto pip2 = get_prim(state, i + 2);

            // PLM reconstruction at left face (i - 1/2)
            auto pl_l = prim_t{};
            auto pl_r = prim_t{};
            for (int q = 0; q < 3; ++q) {
                auto slope_l = plm_minmod(pim2[q], pim1[q], pi[q], cfg.plm_theta);
                auto slope_r = plm_minmod(pim1[q], pi[q], pip1[q], cfg.plm_theta);
                pl_l[q] = pim1[q] + 0.5 * slope_l;
                pl_r[q] = pi[q] - 0.5 * slope_r;
            }

            // PLM reconstruction at right face (i + 1/2)
            auto pr_l = prim_t{};
            auto pr_r = prim_t{};
            for (int q = 0; q < 3; ++q) {
                auto slope_l = plm_minmod(pim1[q], pi[q], pip1[q], cfg.plm_theta);
                auto slope_r = plm_minmod(pi[q], pip1[q], pip2[q], cfg.plm_theta);
                pr_l[q] = pi[q] + 0.5 * slope_l;
                pr_r[q] = pip1[q] - 0.5 * slope_r;
            }

            // Compute fluxes
            auto fl = riemann_hlle(pl_l, pl_r, prim_to_cons(pl_l), prim_to_cons(pl_r));
            auto fr = riemann_hlle(pr_l, pr_r, prim_to_cons(pr_l), prim_to_cons(pr_r));

            du[i - i0] = (fl - fr) * dtdx;
        }

        // Update conserved variables
        for (int i = i0; i < i1; ++i) {
            auto cons = get_cons(state, i);
            set_cons(state, i, cons + du[i - i0]);
        }

    } else {
        throw std::runtime_error("only rk_order=1 (forward Euler) is supported");
    }

    state.time += state.dt;
}

auto zone_count(const srhd::state_t& state, const srhd::exec_context_t& ctx) -> std::size_t {
    return ctx.initial.num_zones;
}

auto names_of_time(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"t"};
}

auto names_of_timeseries(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"time", "total_mass", "total_energy", "max_lorentz"};
}

auto names_of_products(std::type_identity<srhd>) -> std::vector<std::string> {
    return {"density", "velocity", "pressure", "lorentz_factor", "cell_x"};
}

auto get_time(const srhd::state_t& state, const std::string& name) -> double {
    if (name == "t") {
        return state.time;
    }
    throw std::runtime_error("unknown time variable: " + name);
}

auto get_timeseries(
    const srhd::config_t& cfg,
    const srhd::initial_t& ini,
    const srhd::state_t& state,
    const std::string& name
) -> double {
    if (name == "time") {
        return state.time;
    }

    auto dx = (ini.x1 - ini.x0) / ini.num_zones;
    auto total_mass = 0.0;
    auto total_energy = 0.0;
    auto max_lorentz = 1.0;

    for_each(state.interior, [&](ivec_t<1> idx) {
        auto i = idx[0];
        auto cons = get_cons(state, i);
        auto prim = cons_to_prim(cons);
        total_mass += cons[0] * dx;  // D * dx
        total_energy += cons[2] * dx;  // tau * dx
        max_lorentz = max2(max_lorentz, lorentz_factor(prim));
    });

    if (name == "total_mass") return total_mass;
    if (name == "total_energy") return total_energy;
    if (name == "max_lorentz") return max_lorentz;

    throw std::runtime_error("unknown timeseries column: " + name);
}

auto get_product(
    const srhd::state_t& state,
    const std::string& name,
    const srhd::exec_context_t& ctx
) -> srhd::product_t {
    auto& ini = ctx.initial;
    auto dx = (ini.x1 - ini.x0) / ini.num_zones;
    auto result = std::vector<cached_t<double, 1>>{};

    if (name == "density") {
        auto arr = cache(zeros<double>(state.interior), memory::host, exec::cpu);
        for_each(state.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto prim = get_prim(state, i);
            arr[i] = prim[0];  // rho
        });
        result.push_back(std::move(arr));
    } else if (name == "velocity") {
        auto arr = cache(zeros<double>(state.interior), memory::host, exec::cpu);
        for_each(state.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto prim = get_prim(state, i);
            arr[i] = beta(prim);  // v = u/gamma
        });
        result.push_back(std::move(arr));
    } else if (name == "pressure") {
        auto arr = cache(zeros<double>(state.interior), memory::host, exec::cpu);
        for_each(state.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto prim = get_prim(state, i);
            arr[i] = prim[2];  // p
        });
        result.push_back(std::move(arr));
    } else if (name == "lorentz_factor") {
        auto arr = cache(zeros<double>(state.interior), memory::host, exec::cpu);
        for_each(state.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            auto prim = get_prim(state, i);
            arr[i] = lorentz_factor(prim);
        });
        result.push_back(std::move(arr));
    } else if (name == "cell_x") {
        auto arr = cache(zeros<double>(state.interior), memory::host, exec::cpu);
        for_each(state.interior, [&](ivec_t<1> idx) {
            auto i = idx[0];
            arr[i] = cell_center_x(i, ini.x0, dx);
        });
        result.push_back(std::move(arr));
    } else {
        throw std::runtime_error("unknown product: " + name);
    }

    return result;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    mist::program_t<srhd> prog;
    prog.physics = default_physics_config(std::type_identity<srhd>{});
    prog.initial = default_initial_config(std::type_identity<srhd>{});

    auto final_state = mist::run(prog);

    return 0;
}
