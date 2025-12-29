#pragma once
#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>

namespace mist::tensor {

// ============================================================
// Fixed string for template parameters
// ============================================================

template<std::size_t N>
struct fixed_string {
    char data[N]{};
    constexpr fixed_string(char const (&s)[N]) {
        for (std::size_t i = 0; i < N; ++i) data[i] = s[i];
    }
    constexpr operator std::string_view() const { return {data, N - 1}; }
};

// ============================================================
// Type ordering for canonical forms
// ============================================================

template<typename T>
struct type_order;

template<typename T>
constexpr int type_order_v = type_order<T>::value;

// ============================================================
// Literals using integral_constant
// ============================================================

template<int N>
using lit = std::integral_constant<int, N>;

inline constexpr lit<0> zero;
inline constexpr lit<1> one;
inline constexpr lit<2> two;
inline constexpr lit<-1> neg_one;

template<int N>
struct type_order<lit<N>> : std::integral_constant<int, 2000 + N> {};

// ============================================================
// Zero detection
// ============================================================

template<typename T>
struct is_zero : std::false_type {};

template<int N>
    requires (N == 0)
struct is_zero<lit<N>> : std::true_type {};

template<typename T>
constexpr bool is_zero_v = is_zero<T>::value;

// ============================================================
// Coordinate symbols
// ============================================================

template<int I>
struct symb {
    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (std::is_convertible_v<
                          std::tuple_element_t<I, std::tuple<Args...>>,
                          std::string>) {
            return std::string(std::get<I>(std::tie(args...)));
        } else {
            return std::get<I>(std::tie(args...));
        }
    }
};

template<int I>
struct type_order<symb<I>> : std::integral_constant<int, I> {};

// ============================================================
// Named parameters
// ============================================================

template<fixed_string Name>
struct param {
    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (sizeof...(Args) > 0 &&
                      std::is_convertible_v<
                          std::tuple_element_t<0, std::tuple<Args...>>,
                          std::string>) {
            return std::string(Name);
        } else {
            static_assert(sizeof...(Args) == 0, "params need binding for numeric eval");
            return 0.0;
        }
    }
};

template<fixed_string Name>
struct type_order<param<Name>> : std::integral_constant<int, 3000> {};

// ============================================================
// Basis one-forms
// ============================================================

template<int I>
struct dx {
    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (std::is_arithmetic_v<
                          std::tuple_element_t<0, std::tuple<Args...>>>) {
            return 0.0;  // one-forms need contraction to evaluate
        } else {
            return "d" + std::string(std::get<I>(std::tie(args...)));
        }
    }
};

template<int I>
struct type_order<dx<I>> : std::integral_constant<int, 1000 + I> {};

// ============================================================
// Forward declarations
// ============================================================

template<typename... Ts>
struct mul_expr;

template<typename... Ts>
struct add_expr;

template<typename N, typename D>
struct div_expr;

template<typename E, typename F>
struct func_expr;

// ============================================================
// Check if arguments are string-like
// ============================================================

template<typename... Args>
constexpr bool is_string_args_v = sizeof...(Args) > 0 &&
    std::is_convertible_v<std::tuple_element_t<0, std::tuple<Args...>>, std::string>;

// ============================================================
// Expression evaluation helper
// ============================================================

template<typename E, typename... Args>
auto eval(E const& e, Args const&... args) {
    if constexpr (requires { e(args...); }) {
        auto result = e(args...);
        // Convert const char* to std::string for string operations
        if constexpr (is_string_args_v<Args...> &&
                      std::is_convertible_v<decltype(result), std::string> &&
                      !std::is_same_v<decltype(result), std::string>) {
            return std::string(result);
        } else {
            return result;
        }
    } else if constexpr (requires { E::value; }) {
        if constexpr (is_string_args_v<Args...>) {
            return std::to_string(E::value);
        } else {
            return E::value;
        }
    }
}

// ============================================================
// Multiplication expression (variadic, sorted)
// ============================================================

template<typename... Ts>
struct mul_expr {
    std::tuple<Ts...> factors;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (sizeof...(Args) > 0 &&
                      std::is_arithmetic_v<
                          std::tuple_element_t<0, std::tuple<Args...>>>) {
            return std::apply([&](auto const&... fs) {
                return (eval(fs, args...) * ...);
            }, factors);
        } else {
            return std::apply([&](auto const&... fs) {
                std::string result;
                bool first = true;
                ((result += (first ? (first = false, "") : " * ") +
                            eval(fs, args...)), ...);
                return "(" + result + ")";
            }, factors);
        }
    }
};

template<typename T>
struct mul_expr<T> {
    T factor;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        return eval(factor, args...);
    }
};

template<typename... Ts>
struct type_order<mul_expr<Ts...>> : std::integral_constant<int, 5000> {};

// ============================================================
// Addition expression (variadic, sorted, like terms collected)
// ============================================================

template<typename... Ts>
struct add_expr {
    std::tuple<Ts...> terms;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (sizeof...(Args) > 0 &&
                      std::is_arithmetic_v<
                          std::tuple_element_t<0, std::tuple<Args...>>>) {
            return std::apply([&](auto const&... ts) {
                return (eval(ts, args...) + ...);
            }, terms);
        } else {
            return std::apply([&](auto const&... ts) {
                std::string result;
                bool first = true;
                ((result += (first ? (first = false, "") : " + ") +
                            eval(ts, args...)), ...);
                return "(" + result + ")";
            }, terms);
        }
    }
};

template<typename... Ts>
struct type_order<add_expr<Ts...>> : std::integral_constant<int, 6000> {};

// ============================================================
// Division expression
// ============================================================

template<typename N, typename D>
struct div_expr {
    N num;
    D den;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        auto n = eval(num, args...);
        auto d = eval(den, args...);
        if constexpr (std::is_arithmetic_v<decltype(n)>) {
            return n / d;
        } else {
            return "(" + n + " / " + d + ")";
        }
    }
};

template<typename N, typename D>
struct type_order<div_expr<N, D>> : std::integral_constant<int, 7000> {};

// ============================================================
// Operators: multiplication
// ============================================================

// Simplification: multiply by one
template<typename R>
auto operator*(lit<1>, R r) { return r; }

template<typename L>
auto operator*(L l, lit<1>) { return l; }

inline auto operator*(lit<1>, lit<1>) { return one; }

// Simplification: multiply by zero
template<typename R>
auto operator*(lit<0>, R) { return zero; }

template<typename L>
auto operator*(L, lit<0>) { return zero; }

inline auto operator*(lit<0>, lit<0>) { return zero; }

inline auto operator*(lit<1>, lit<0>) { return zero; }
inline auto operator*(lit<0>, lit<1>) { return zero; }

// General multiplication with canonical ordering
template<typename A, typename B>
    requires (!std::same_as<A, lit<0>> && !std::same_as<B, lit<0>> &&
              !std::same_as<A, lit<1>> && !std::same_as<B, lit<1>>)
auto operator*(A a, B b) {
    if constexpr (type_order_v<A> <= type_order_v<B>) {
        return mul_expr<A, B>{{a, b}};
    } else {
        return mul_expr<B, A>{{b, a}};
    }
}

// ============================================================
// Operators: addition
// ============================================================

// Simplification: add zero
template<typename R>
auto operator+(lit<0>, R r) { return r; }

template<typename L>
auto operator+(L l, lit<0>) { return l; }

inline auto operator+(lit<0>, lit<0>) { return zero; }

// General addition with canonical ordering
template<typename A, typename B>
    requires (!std::same_as<A, lit<0>> && !std::same_as<B, lit<0>>)
auto operator+(A a, B b) {
    if constexpr (type_order_v<A> <= type_order_v<B>) {
        return add_expr<A, B>{{a, b}};
    } else {
        return add_expr<B, A>{{b, a}};
    }
}

// ============================================================
// Operators: subtraction and negation
// ============================================================

template<typename A, typename B>
auto operator-(A a, B b) {
    return a + neg_one * b;
}

template<typename A>
auto operator-(A a) {
    return neg_one * a;
}

// ============================================================
// Operators: division
// ============================================================

template<typename N, typename D>
auto operator/(N n, D d) {
    return div_expr<N, D>{n, d};
}

// ============================================================
// Function expressions
// ============================================================

struct sin_fn {
    static constexpr auto name = "sin";
    static auto eval(auto x) { return std::sin(x); }
};

struct cos_fn {
    static constexpr auto name = "cos";
    static auto eval(auto x) { return std::cos(x); }
};

struct tan_fn {
    static constexpr auto name = "tan";
    static auto eval(auto x) { return std::tan(x); }
};

struct sinh_fn {
    static constexpr auto name = "sinh";
    static auto eval(auto x) { return std::sinh(x); }
};

struct cosh_fn {
    static constexpr auto name = "cosh";
    static auto eval(auto x) { return std::cosh(x); }
};

struct tanh_fn {
    static constexpr auto name = "tanh";
    static auto eval(auto x) { return std::tanh(x); }
};

struct exp_fn {
    static constexpr auto name = "exp";
    static auto eval(auto x) { return std::exp(x); }
};

struct log_fn {
    static constexpr auto name = "log";
    static auto eval(auto x) { return std::log(x); }
};

struct sqrt_fn {
    static constexpr auto name = "sqrt";
    static auto eval(auto x) { return std::sqrt(x); }
};

template<typename E, typename F>
struct func_expr {
    E arg;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        auto v = tensor::eval(arg, args...);
        if constexpr (std::is_arithmetic_v<decltype(v)>) {
            return F::eval(v);
        } else {
            return std::string(F::name) + "(" + v + ")";
        }
    }
};

template<typename E, typename F>
struct type_order<func_expr<E, F>> : std::integral_constant<int, 4000 + type_order_v<E>> {};

template<typename E> auto sin(E e) { return func_expr<E, sin_fn>{e}; }
template<typename E> auto cos(E e) { return func_expr<E, cos_fn>{e}; }
template<typename E> auto tan(E e) { return func_expr<E, tan_fn>{e}; }
template<typename E> auto sinh(E e) { return func_expr<E, sinh_fn>{e}; }
template<typename E> auto cosh(E e) { return func_expr<E, cosh_fn>{e}; }
template<typename E> auto tanh(E e) { return func_expr<E, tanh_fn>{e}; }
template<typename E> auto exp(E e) { return func_expr<E, exp_fn>{e}; }
template<typename E> auto log(E e) { return func_expr<E, log_fn>{e}; }
template<typename E> auto sqrt(E e) { return func_expr<E, sqrt_fn>{e}; }

// ============================================================
// Differential operator: d(expr) -> one-form
// ============================================================

template<int I>
auto d(symb<I>) { return dx<I>{}; }

// ============================================================
// Tensor product of one-forms (symmetric, canonical order)
// ============================================================

template<int I, int J>
struct tens_basis {
    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (std::is_arithmetic_v<
                          std::tuple_element_t<0, std::tuple<Args...>>>) {
            return 1.0;
        } else {
            return "(d" + std::string(std::get<I>(std::tie(args...))) +
                   " \u2297 d" + std::string(std::get<J>(std::tie(args...))) + ")";
        }
    }
};

// Canonical tens_basis type: always I <= J
template<int I, int J>
using canonical_tens = tens_basis<(I < J ? I : J), (I < J ? J : I)>;

template<int I, int J>
auto tens(dx<I>, dx<J>) -> canonical_tens<I, J> {
    return {};
}

// Verify symmetry at type level
static_assert(std::same_as<decltype(tens(dx<0>{}, dx<1>{})),
                           decltype(tens(dx<1>{}, dx<0>{}))>);

// ============================================================
// Metric term: coefficient * basis (always canonical: I <= J)
// ============================================================

template<typename Coeff, int I, int J>
    requires (I <= J)
struct metric_term {
    Coeff coeff;
    static constexpr int i = I;
    static constexpr int j = J;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        auto c = tensor::eval(coeff, args...);
        auto b = tens_basis<I, J>{}(args...);
        if constexpr (std::is_arithmetic_v<decltype(c)>) {
            return c * b;
        } else {
            return "(" + c + " * " + b + ")";
        }
    }
};

// Canonical metric_term type
template<typename Coeff, int I, int J>
using canonical_metric_term = metric_term<Coeff, (I < J ? I : J), (I < J ? J : I)>;

template<typename Coeff, int I, int J>
auto operator*(Coeff c, tens_basis<I, J>) -> canonical_metric_term<Coeff, I, J> {
    return {c};
}

// ============================================================
// Extract indices from metric term
// ============================================================

template<typename T>
struct term_indices {
    static constexpr bool is_metric_term = false;
};

template<typename Coeff, int I, int J>
struct term_indices<metric_term<Coeff, I, J>> {
    static constexpr bool is_metric_term = true;
    static constexpr int i = metric_term<Coeff, I, J>::i;
    static constexpr int j = metric_term<Coeff, I, J>::j;
    using coeff = Coeff;
};

// ============================================================
// Metric expression: sum of metric terms
// ============================================================

template<typename... Terms>
struct metric_expr {
    std::tuple<Terms...> terms;

    template<typename... Args>
    auto operator()(Args const&... args) const {
        if constexpr (sizeof...(Args) > 0 &&
                      std::is_arithmetic_v<
                          std::tuple_element_t<0, std::tuple<Args...>>>) {
            return std::apply([&](auto const&... ts) {
                return (ts(args...) + ...);
            }, terms);
        } else {
            return std::apply([&](auto const&... ts) {
                std::string result;
                bool first = true;
                ((result += (first ? (first = false, "") : " + ") + ts(args...)), ...);
                return result;
            }, terms);
        }
    }

    template<int I, int J>
    auto get() const {
        constexpr int lo = (I < J) ? I : J;
        constexpr int hi = (I < J) ? J : I;
        return find_component<lo, hi, 0, Terms...>();
    }

private:
    template<int I, int J, std::size_t Idx>
    auto find_component() const {
        return lit<0>{};
    }

    template<int I, int J, std::size_t Idx, typename Head, typename... Tail>
    auto find_component() const {
        if constexpr (term_indices<Head>::is_metric_term &&
                      term_indices<Head>::i == I &&
                      term_indices<Head>::j == J) {
            return std::get<Idx>(terms).coeff;
        } else {
            return find_component<I, J, Idx + 1, Tail...>();
        }
    }
};

// ============================================================
// Metric term addition -> metric_expr
// ============================================================

template<typename Coeff1, int I1, int J1, typename Coeff2, int I2, int J2>
auto operator+(metric_term<Coeff1, I1, J1> a, metric_term<Coeff2, I2, J2> b) {
    return metric_expr<metric_term<Coeff1, I1, J1>,
                       metric_term<Coeff2, I2, J2>>{{a, b}};
}

template<typename... Terms, typename Coeff, int I, int J>
auto operator+(metric_expr<Terms...> m, metric_term<Coeff, I, J> t) {
    return metric_expr<Terms..., metric_term<Coeff, I, J>>{
        std::tuple_cat(m.terms, std::make_tuple(t))
    };
}

// ============================================================
// Concepts: Symmetric
// ============================================================

template<typename T, int I, int J>
using element_t = decltype(std::declval<T>().template get<I, J>());

template<typename T, int I, int J>
concept SymmetricPair = std::same_as<element_t<T, I, J>, element_t<T, J, I>>;

template<typename T>
concept Symmetric4x4 =
    SymmetricPair<T, 0, 1> &&
    SymmetricPair<T, 0, 2> &&
    SymmetricPair<T, 0, 3> &&
    SymmetricPair<T, 1, 2> &&
    SymmetricPair<T, 1, 3> &&
    SymmetricPair<T, 2, 3>;

// ============================================================
// Concepts: Diagonal
// ============================================================

template<typename T, int I, int J>
concept ZeroComponent = is_zero_v<element_t<T, I, J>>;

template<typename T>
concept Diagonal4x4 =
    ZeroComponent<T, 0, 1> &&
    ZeroComponent<T, 0, 2> &&
    ZeroComponent<T, 0, 3> &&
    ZeroComponent<T, 1, 2> &&
    ZeroComponent<T, 1, 3> &&
    ZeroComponent<T, 2, 3>;

// ============================================================
// Inverse metric (diagonal)
// ============================================================

template<typename Metric>
    requires Diagonal4x4<Metric>
struct inverse_metric {
    Metric const& g;

    template<int I, int J>
    auto get() const {
        if constexpr (I != J) {
            return lit<0>{};
        } else {
            return one / g.template get<I, I>();
        }
    }
};

template<Diagonal4x4 M>
auto inv(M const& m) {
    return inverse_metric<M>{m};
}

// ============================================================
// Partial derivatives
// ============================================================

// ∂_J(symb<I>) = δ^I_J
template<int J, int I>
auto partial(symb<I>) {
    if constexpr (I == J) {
        return one;
    } else {
        return lit<0>{};
    }
}

// ∂_J(constant) = 0
template<int J, typename T, T V>
auto partial(std::integral_constant<T, V>) {
    return lit<0>{};
}

// ∂_J(param) = 0
template<int J, fixed_string Name>
auto partial(param<Name>) {
    return lit<0>{};
}

// ∂_J(f * g) = (∂_J f) * g + f * (∂_J g)
template<int J, typename A, typename B>
auto partial(mul_expr<A, B> const& e) {
    auto [a, b] = e.factors;
    return partial<J>(a) * b + a * partial<J>(b);
}

// ∂_J(f + g) = ∂_J f + ∂_J g
template<int J, typename A, typename B>
auto partial(add_expr<A, B> const& e) {
    auto [a, b] = e.terms;
    return partial<J>(a) + partial<J>(b);
}

// ∂_J(f / g) = (∂_J f * g - f * ∂_J g) / g²
template<int J, typename N, typename D>
auto partial(div_expr<N, D> const& e) {
    return (partial<J>(e.num) * e.den - e.num * partial<J>(e.den)) / (e.den * e.den);
}

// Chain rule: ∂_J(sin(f)) = cos(f) * ∂_J(f)
template<int J, typename E>
auto partial(func_expr<E, sin_fn> const& e) {
    return cos(e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, cos_fn> const& e) {
    return neg_one * sin(e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, exp_fn> const& e) {
    return exp(e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, log_fn> const& e) {
    return (one / e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, sinh_fn> const& e) {
    return cosh(e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, cosh_fn> const& e) {
    return sinh(e.arg) * partial<J>(e.arg);
}

template<int J, typename E>
auto partial(func_expr<E, sqrt_fn> const& e) {
    return partial<J>(e.arg) / (two * sqrt(e.arg));
}

// ============================================================
// Metric derivatives: ∂_σ g_μν
// ============================================================

template<typename Metric>
struct metric_derivatives {
    Metric const& g;

    template<int Sigma, int Mu, int Nu>
    auto get() const {
        return partial<Sigma>(g.template get<Mu, Nu>());
    }
};

template<typename M>
auto d_metric(M const& m) {
    return metric_derivatives<M>{m};
}

// ============================================================
// Christoffel symbols: Γ^ρ_μν = ½ g^ρσ (∂_μ g_σν + ∂_ν g_σμ - ∂_σ g_μν)
// ============================================================

template<typename Metric, int Dim = 4>
struct christoffel {
    Metric const& g;

    template<int Rho, int Mu, int Nu>
    auto get() const {
        auto g_inv = inv(g);
        auto dg = d_metric(g);
        return (one / two) * sum_sigma<Rho, Mu, Nu, 0>(g_inv, dg);
    }

private:
    template<int Rho, int Mu, int Nu, int Sigma, typename GInv, typename DG>
    auto sum_sigma(GInv const& g_inv, DG const& dg) const {
        auto term = g_inv.template get<Rho, Sigma>() * (
            dg.template get<Mu, Sigma, Nu>() +
            dg.template get<Nu, Sigma, Mu>() -
            dg.template get<Sigma, Mu, Nu>()
        );

        if constexpr (Sigma + 1 < Dim) {
            return term + sum_sigma<Rho, Mu, Nu, Sigma + 1>(g_inv, dg);
        } else {
            return term;
        }
    }
};

template<typename M>
auto make_christoffel(M const& m) {
    return christoffel<M>{m};
}

} // namespace mist::tensor
