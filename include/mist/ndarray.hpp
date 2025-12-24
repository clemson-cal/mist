#pragma once

#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>
#include "core.hpp"

namespace mist {

// =============================================================================
// Memory and layout enums
// =============================================================================

enum class memory {
    host,
    device,
    managed
};

enum class layout {
    aos,
    soa
};

// =============================================================================
// Forward declarations
// =============================================================================

template<std::size_t S, typename F>
struct lazy_t;

template<typename T, std::size_t S>
struct array_t;

template<typename T, std::size_t S>
struct array_view_t;

template<typename T, std::size_t N, std::size_t S, layout L = layout::aos>
struct array_vec_t;

// Aliases for backward compatibility
template<typename T, std::size_t S>
using cached_t = array_t<T, S>;

template<typename T, std::size_t S>
using cached_view_t = array_view_t<T, S>;

template<typename T, std::size_t N, std::size_t S, layout L = layout::aos>
using cached_vec_t = array_vec_t<T, N, S, L>;

// =============================================================================
// NdArray concept and refinements
// =============================================================================

template<typename A>
concept NdArray = requires(const A& a, ivec_t<A::rank> idx) {
    typename A::value_type;
    { A::rank } -> std::convertible_to<std::size_t>;
    { space(a) } -> std::same_as<const index_space_t<A::rank>&>;
    { a(idx) } -> std::convertible_to<typename A::value_type>;
};

template<typename A>
concept CachedNdArray = NdArray<A> && requires(A& a, const A& ca) {
    { data(a) } -> std::convertible_to<typename A::value_type*>;
    { data(ca) } -> std::convertible_to<const typename A::value_type*>;
};

template<typename A>
concept WritableNdArray = NdArray<A> && requires(A& a, ivec_t<A::rank> idx, typename A::value_type v) {
    { a(idx) = v };
};

// =============================================================================
// Free functions for NdArray types
// =============================================================================

template<NdArray A>
auto start(const A& a) { return start(space(a)); }

template<NdArray A>
auto shape(const A& a) { return shape(space(a)); }

template<NdArray A>
auto size(const A& a) { return size(space(a)); }

template<NdArray A>
auto at(const A& a, const ivec_t<A::rank>& idx) { return a(idx); }

template<NdArray A>
auto at(A& a, const ivec_t<A::rank>& idx) -> decltype(a(idx)) { return a(idx); }

// =============================================================================
// lazy_t: Lazy array (space + function)
// =============================================================================

template<std::size_t S, typename F>
struct lazy_t {
    using value_type = std::invoke_result_t<F, ivec_t<S>>;
    static constexpr std::size_t rank = S;

    index_space_t<S> _space;
    F _func;

    MIST_HD auto operator()(const ivec_t<S>& idx) const { return _func(idx); }

    // Convenience for 1D arrays: accept integer index
    MIST_HD auto operator[](int i) const requires (S == 1) { return _func(ivec(i)); }
};

template<std::size_t S, typename F>
const index_space_t<S>& space(const lazy_t<S, F>& a) { return a._space; }

// -----------------------------------------------------------------------------
// lazy() constructor
// -----------------------------------------------------------------------------

template<std::size_t S, typename F>
auto lazy(const index_space_t<S>& space, F&& func) {
    return lazy_t<S, std::decay_t<F>>{space, std::forward<F>(func)};
}

// =============================================================================
// array_t: Owning memory-backed array
// =============================================================================

template<typename T, std::size_t S>
struct array_t {
    using value_type = T;
    static constexpr std::size_t rank = S;

    index_space_t<S> _space;
    T* _data;
    memory _location;

    // Default constructor: creates empty array
    array_t()
        : _space(index_space(ivec_t<S>{}, uvec_t<S>{})), _data(nullptr), _location(memory::host)
    {
    }

    // Constructor
    array_t(const index_space_t<S>& space, memory loc = memory::host)
        : _space(space), _data(nullptr), _location(loc)
    {
        std::size_t n = size(space);
        switch (_location) {
            case memory::host:
                _data = new T[n]();
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMalloc(&_data, n * sizeof(T));
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaMallocManaged(&_data, n * sizeof(T));
                #endif
                break;
        }
    }

    // Destructor
    ~array_t() {
        if (!_data) return;
        if (_location == memory::host) {
            delete[] _data;
        } else {
            #ifdef __CUDACC__
            cudaFree(_data);
            #endif
        }
    }

    // No copy
    array_t(const array_t&) = delete;
    array_t& operator=(const array_t&) = delete;

    // Move
    array_t(array_t&& other) noexcept
        : _space(other._space)
        , _data(other._data)
        , _location(other._location)
    {
        other._data = nullptr;
    }

    array_t& operator=(array_t&& other) noexcept {
        if (this != &other) {
            if (_data) {
                if (_location == memory::host) {
                    delete[] _data;
                } else {
                    #ifdef __CUDACC__
                    cudaFree(_data);
                    #endif
                }
            }
            _space = other._space;
            _data = other._data;
            _location = other._location;
            other._data = nullptr;
        }
        return *this;
    }

    // Element access
    MIST_HD T& operator()(const ivec_t<S>& idx) {
        return _data[ndoffset(_space, idx)];
    }

    MIST_HD const T& operator()(const ivec_t<S>& idx) const {
        return _data[ndoffset(_space, idx)];
    }

    // Convenience for 1D arrays: accept integer index
    MIST_HD T& operator[](int i) requires (S == 1) {
        return _data[ndoffset(_space, ivec(i))];
    }

    MIST_HD const T& operator[](int i) const requires (S == 1) {
        return _data[ndoffset(_space, ivec(i))];
    }

    // Subspace view via operator[]
    auto operator[](const index_space_t<S>& subspace) -> array_view_t<T, S>;
    auto operator[](const index_space_t<S>& subspace) const -> array_view_t<const T, S>;
};

template<typename T, std::size_t S>
const index_space_t<S>& space(const array_t<T, S>& a) { return a._space; }

template<typename T, std::size_t S>
T* data(array_t<T, S>& a) { return a._data; }

template<typename T, std::size_t S>
const T* data(const array_t<T, S>& a) { return a._data; }

template<typename T, std::size_t S>
memory location(const array_t<T, S>& a) { return a._location; }

// =============================================================================
// array_view_t: Non-owning view into array (supports strided access)
// =============================================================================

template<typename T, std::size_t S>
struct array_view_t {
    using value_type = T;
    static constexpr std::size_t rank = S;

    index_space_t<S> _space;   // The subspace this view represents
    index_space_t<S> _parent;  // The parent allocation's index space
    T* _data;                  // Pointer to first element of parent

    // Constructor for contiguous view (parent equals space)
    array_view_t(const index_space_t<S>& space, T* data)
        : _space(space), _parent(space), _data(data) {}

    // Constructor for subspace view
    array_view_t(const index_space_t<S>& space, const index_space_t<S>& parent, T* data)
        : _space(space), _parent(parent), _data(data) {}

    // Converting constructor: array_view_t<U> -> array_view_t<const U>
    template<typename U>
        requires std::is_same_v<T, const U>
    array_view_t(const array_view_t<U, S>& other)
        : _space(other._space), _parent(other._parent), _data(other._data) {}

    // Element access - constness is encoded in T, not in the view
    // Use array_view_t<const T, S> for read-only access
    MIST_HD T& operator()(const ivec_t<S>& idx) const {
        return _data[ndoffset(_parent, idx)];
    }

    // Convenience for 1D arrays: accept integer index
    MIST_HD T& operator[](int i) const requires (S == 1) {
        return _data[ndoffset(_parent, ivec(i))];
    }
};

template<typename T, std::size_t S>
const index_space_t<S>& space(const array_view_t<T, S>& a) { return a._space; }

template<typename T, std::size_t S>
const index_space_t<S>& parent(const array_view_t<T, S>& a) { return a._parent; }

template<typename T, std::size_t S>
T* data(const array_view_t<T, S>& a) { return a._data; }

// -----------------------------------------------------------------------------
// view() - create views into arrays
// -----------------------------------------------------------------------------

// View of entire array
template<typename T, std::size_t S>
auto view(array_t<T, S>& a) -> array_view_t<T, S> {
    return array_view_t<T, S>(a._space, a._data);
}

template<typename T, std::size_t S>
auto view(const array_t<T, S>& a) -> array_view_t<const T, S> {
    return array_view_t<const T, S>(a._space, a._data);
}

// View of subspace (parent space enables MPI subarray datatypes)
template<typename T, std::size_t S>
auto view(array_t<T, S>& a, const index_space_t<S>& subspace) -> array_view_t<T, S> {
    return array_view_t<T, S>(subspace, a._space, a._data);
}

template<typename T, std::size_t S>
auto view(const array_t<T, S>& a, const index_space_t<S>& subspace) -> array_view_t<const T, S> {
    return array_view_t<const T, S>(subspace, a._space, a._data);
}

// -----------------------------------------------------------------------------
// array_t::operator[] implementations (deferred due to array_view_t dependency)
// -----------------------------------------------------------------------------

template<typename T, std::size_t S>
auto array_t<T, S>::operator[](const index_space_t<S>& subspace) -> array_view_t<T, S> {
    return view(*this, subspace);
}

template<typename T, std::size_t S>
auto array_t<T, S>::operator[](const index_space_t<S>& subspace) const -> array_view_t<const T, S> {
    return view(*this, subspace);
}

// =============================================================================
// soa_ref_t: Proxy for SoA element access
// =============================================================================

template<typename T, std::size_t N, std::size_t S>
struct soa_ref_t {
    T* _data;
    index_space_t<S> _space;
    ivec_t<S> _idx;

    MIST_HD operator vec_t<T, N>() const {
        return ndread_soa<T, N>(_data, _space, _idx);
    }

    MIST_HD soa_ref_t& operator=(const vec_t<T, N>& value) {
        ndwrite_soa<T, N>(_data, _space, _idx, value);
        return *this;
    }
};

// =============================================================================
// array_vec_t: Vector-valued array with layout choice
// =============================================================================

template<typename T, std::size_t N, std::size_t S, layout L>
struct array_vec_t {
    using value_type = vec_t<T, N>;
    static constexpr std::size_t rank = S;
    static constexpr layout data_layout = L;

    index_space_t<S> _space;
    T* _data;
    memory _location;

    // Constructor
    array_vec_t(const index_space_t<S>& space, memory loc = memory::host)
        : _space(space), _data(nullptr), _location(loc)
    {
        std::size_t n = size(space) * N;
        switch (_location) {
            case memory::host:
                _data = new T[n]();
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMalloc(&_data, n * sizeof(T));
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaMallocManaged(&_data, n * sizeof(T));
                #endif
                break;
        }
    }

    // Destructor
    ~array_vec_t() {
        if (!_data) return;
        if (_location == memory::host) {
            delete[] _data;
        } else {
            #ifdef __CUDACC__
            cudaFree(_data);
            #endif
        }
    }

    // No copy
    array_vec_t(const array_vec_t&) = delete;
    array_vec_t& operator=(const array_vec_t&) = delete;

    // Move
    array_vec_t(array_vec_t&& other) noexcept
        : _space(other._space)
        , _data(other._data)
        , _location(other._location)
    {
        other._data = nullptr;
    }

    array_vec_t& operator=(array_vec_t&& other) noexcept {
        if (this != &other) {
            if (_data) {
                if (_location == memory::host) {
                    delete[] _data;
                } else {
                    #ifdef __CUDACC__
                    cudaFree(_data);
                    #endif
                }
            }
            _space = other._space;
            _data = other._data;
            _location = other._location;
            other._data = nullptr;
        }
        return *this;
    }

    // Element access - AoS
    MIST_HD vec_t<T, N>& aos_at(const ivec_t<S>& idx) {
        return reinterpret_cast<vec_t<T, N>*>(_data)[ndoffset(_space, idx)];
    }

    MIST_HD const vec_t<T, N>& aos_at(const ivec_t<S>& idx) const {
        return reinterpret_cast<const vec_t<T, N>*>(_data)[ndoffset(_space, idx)];
    }

    // Element access - dispatch by layout
    MIST_HD decltype(auto) operator()(const ivec_t<S>& idx) {
        if constexpr (L == layout::aos) {
            return aos_at(idx);
        } else {
            return soa_ref_t<T, N, S>{_data, _space, idx};
        }
    }

    MIST_HD decltype(auto) operator()(const ivec_t<S>& idx) const {
        if constexpr (L == layout::aos) {
            return aos_at(idx);
        } else {
            return ndread_soa<T, N>(_data, _space, idx);
        }
    }
};

template<typename T, std::size_t N, std::size_t S, layout L>
const index_space_t<S>& space(const array_vec_t<T, N, S, L>& a) { return a._space; }

template<typename T, std::size_t N, std::size_t S, layout L>
T* data(array_vec_t<T, N, S, L>& a) { return a._data; }

template<typename T, std::size_t N, std::size_t S, layout L>
const T* data(const array_vec_t<T, N, S, L>& a) { return a._data; }

template<typename T, std::size_t N, std::size_t S, layout L>
memory location(const array_vec_t<T, N, S, L>& a) { return a._location; }

// =============================================================================
// map: Lazy element-wise transform
// =============================================================================

// Lvalue source: capture by pointer
template<NdArray A, typename F>
auto map(const A& a, F func) {
    return lazy(space(a), [&a, func](const ivec_t<A::rank>& idx) {
        return func(a(idx));
    });
}

// Rvalue source: move into captures
template<NdArray A, typename F>
    requires (!std::is_lvalue_reference_v<A&&>)
auto map(A&& a, F func) {
    auto sp = space(a);
    return lazy(sp, [a = std::move(a), func](const ivec_t<A::rank>& idx) {
        return func(a(idx));
    });
}

// =============================================================================
// extract: Lazy view into subregion
// =============================================================================

// Lvalue source: capture by pointer
template<NdArray A>
auto extract(const A& a, const index_space_t<A::rank>& subspace) {
    return lazy(subspace, [&a](const ivec_t<A::rank>& idx) {
        return a(idx);
    });
}

// Rvalue source: move into captures
template<NdArray A>
    requires (!std::is_lvalue_reference_v<A&&>)
auto extract(A&& a, const index_space_t<A::rank>& subspace) {
    return lazy(subspace, [a = std::move(a)](const ivec_t<A::rank>& idx) {
        return a(idx);
    });
}

// =============================================================================
// insert: Lazy overlay of one array onto another
// =============================================================================

template<NdArray A, NdArray B>
    requires (A::rank == B::rank)
auto insert(const A& a, const B& b) {
    return lazy(space(a), [&a, &b](const ivec_t<A::rank>& idx) {
        if (contains(space(b), idx)) {
            return static_cast<typename A::value_type>(b(idx));
        }
        return a(idx);
    });
}

// =============================================================================
// cache: Materialize any NdArray to memory
// =============================================================================

namespace detail {

template<typename T>
void memcpy_host_to_host(T* dst, const T* src, std::size_t n) {
    std::memcpy(dst, src, n * sizeof(T));
}

template<typename T>
void memcpy_any(T* dst, const T* src, std::size_t n, memory dst_loc, memory src_loc) {
    #ifdef __CUDACC__
    cudaMemcpyKind kind;
    if (src_loc == memory::host && dst_loc == memory::host) {
        std::memcpy(dst, src, n * sizeof(T));
        return;
    }
    else if (src_loc == memory::host) {
        kind = cudaMemcpyHostToDevice;
    }
    else if (dst_loc == memory::host) {
        kind = cudaMemcpyDeviceToHost;
    }
    else {
        kind = cudaMemcpyDeviceToDevice;
    }
    cudaMemcpy(dst, src, n * sizeof(T), kind);
    #else
    (void)dst_loc;
    (void)src_loc;
    std::memcpy(dst, src, n * sizeof(T));
    #endif
}

} // namespace detail

// Primary template for lazy sources
template<NdArray A>
auto cache(const A& a, memory loc, exec e) {
    using T = typename A::value_type;
    constexpr auto S = A::rank;

    cached_t<T, S> result(space(a), loc);

    for_each(space(a), [&](const ivec_t<S>& idx) {
        result(idx) = a(idx);
    }, e);

    return result;
}

// Overload for cached_t source (bulk memcpy)
template<typename T, std::size_t S>
auto cache(const cached_t<T, S>& a, memory loc, exec /*e*/) {
    cached_t<T, S> result(space(a), loc);
    detail::memcpy_any(result._data, a._data, size(space(a)), loc, a._location);
    return result;
}

// Overload for cached_view_t source (bulk memcpy, assumes host)
template<typename T, std::size_t S>
auto cache(const cached_view_t<T, S>& a, memory loc, exec /*e*/) {
    cached_t<T, S> result(space(a), loc);
    memory src_loc = memory::host;  // view doesn't track location
    detail::memcpy_any(result._data, a._data, size(space(a)), loc, src_loc);
    return result;
}

// =============================================================================
// cache with layout for vector-valued arrays
// =============================================================================

template<layout L, NdArray A>
auto cache(const A& a, memory loc, exec e) {
    using V = typename A::value_type;

    // Check if value_type is vec_t
    if constexpr (requires { typename V::value_type; V::size(); }) {
        using T = typename V::value_type;
        constexpr std::size_t N = sizeof(V) / sizeof(T);
        constexpr auto S = A::rank;

        cached_vec_t<T, N, S, L> result(space(a), loc);

        for_each(space(a), [&](const ivec_t<S>& idx) {
            result(idx) = a(idx);
        }, e);

        return result;
    } else {
        // Fallback to scalar cache
        return cache(a, loc, e);
    }
}

// =============================================================================
// copy: Copy between arrays (requires matching spaces)
// =============================================================================

// Copy between views (requires matching index spaces)
template<typename T, std::size_t S>
void copy(array_view_t<T, S> dst, array_view_t<const T, S> src) {
    // Spaces must match exactly
    if (dst._space != src._space) {
        throw std::runtime_error("copy: index spaces must match");
    }
    for (auto idx : dst._space) {
        dst(idx) = src(idx);
    }
}

// Copy from const view to mutable view (same types)
template<typename T, std::size_t S>
void copy(array_view_t<T, S> dst, array_view_t<T, S> src) {
    copy(dst, array_view_t<const T, S>(src._space, src._data, src._strides));
}

// Copy from cached_t to view
template<typename T, std::size_t S>
void copy(array_view_t<T, S> dst, const cached_t<T, S>& src) {
    copy(dst, view(src));
}

// Copy from array_t to array_t (reallocates dst if needed)
template<typename T, std::size_t S>
void copy(array_t<T, S>& dst, const array_t<T, S>& src) {
    if (size(dst._space) != size(src._space)) {
        if (dst._data) {
            if (dst._location == memory::host) {
                delete[] dst._data;
            } else {
                #ifdef __CUDACC__
                cudaFree(dst._data);
                #endif
            }
        }
        std::size_t n = size(src._space);
        switch (dst._location) {
            case memory::host:
                dst._data = new T[n]();
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMalloc(&dst._data, n * sizeof(T));
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaMallocManaged(&dst._data, n * sizeof(T));
                #endif
                break;
        }
    }
    dst._space = src._space;
    detail::memcpy_any(dst._data, src._data, size(dst._space), dst._location, src._location);
}

// Copy from array_vec_t to array_vec_t (reallocates dst if needed)
template<typename T, std::size_t N, std::size_t S, layout L>
void copy(array_vec_t<T, N, S, L>& dst, const array_vec_t<T, N, S, L>& src) {
    if (size(dst._space) != size(src._space)) {
        if (dst._data) {
            if (dst._location == memory::host) {
                delete[] dst._data;
            } else {
                #ifdef __CUDACC__
                cudaFree(dst._data);
                #endif
            }
        }
        std::size_t n = size(src._space) * N;
        switch (dst._location) {
            case memory::host:
                dst._data = new T[n]();
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMalloc(&dst._data, n * sizeof(T));
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaMallocManaged(&dst._data, n * sizeof(T));
                #endif
                break;
        }
    }
    dst._space = src._space;
    detail::memcpy_any(dst._data, src._data, size(dst._space) * N, dst._location, src._location);
}

// =============================================================================
// copy_overlapping: Copy where index spaces overlap
// =============================================================================

// Copy overlapping region between views
template<typename T, std::size_t S>
void copy_overlapping(array_view_t<T, S> dst, array_view_t<T, S> src) {
    copy_overlapping(dst, array_view_t<const T, S>(src));
}

template<typename T, std::size_t S>
void copy_overlapping(array_view_t<T, S> dst, array_view_t<const T, S> src) {
    auto dst_space = space(dst);
    auto src_space = space(src);

    if (!overlaps(dst_space, src_space)) {
        return;
    }

    for (auto idx : dst_space) {
        if (contains(src_space, idx)) {
            dst(idx) = src(idx);
        }
    }
}

// Copy overlapping region between arrays
template<typename T, std::size_t S>
void copy_overlapping(array_t<T, S>& dst, const array_t<T, S>& src) {
    copy_overlapping(view(dst), view(src));
}

// =============================================================================
// safe_at: Host/device transparent access with proxy
// =============================================================================

template<typename T, std::size_t S>
struct safe_ref_t {
    cached_t<T, S>* _array;
    ivec_t<S> _idx;

    operator T() const {
        T value{};
        T* src = _array->_data + ndoffset(_array->_space, _idx);

        switch (_array->_location) {
            case memory::host:
                value = *src;
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMemcpy(&value, src, sizeof(T), cudaMemcpyDeviceToHost);
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaDeviceSynchronize();
                #endif
                value = *src;
                break;
        }
        return value;
    }

    safe_ref_t& operator=(const T& value) {
        T* dst = _array->_data + ndoffset(_array->_space, _idx);

        switch (_array->_location) {
            case memory::host:
                *dst = value;
                break;
            case memory::device:
                #ifdef __CUDACC__
                cudaMemcpy(dst, &value, sizeof(T), cudaMemcpyHostToDevice);
                #endif
                break;
            case memory::managed:
                #ifdef __CUDACC__
                cudaDeviceSynchronize();
                #endif
                *dst = value;
                break;
        }
        return *this;
    }
};

template<typename T, std::size_t S>
safe_ref_t<T, S> safe_at(cached_t<T, S>& a, const ivec_t<S>& idx) {
    return {&a, idx};
}

template<typename T, std::size_t S>
T safe_at(const cached_t<T, S>& a, const ivec_t<S>& idx) {
    T value;
    const T* src = a._data + ndoffset(a._space, idx);

    switch (a._location) {
        case memory::host:
            value = *src;
            break;
        case memory::device:
            #ifdef __CUDACC__
            cudaMemcpy(&value, src, sizeof(T), cudaMemcpyDeviceToHost);
            #endif
            break;
        case memory::managed:
            #ifdef __CUDACC__
            cudaDeviceSynchronize();
            #endif
            value = *src;
            break;
    }
    return value;
}

// =============================================================================
// Array constructors
// =============================================================================

// zeros: constant zero
template<typename T, std::size_t S>
auto zeros(const index_space_t<S>& space) {
    return lazy(space, [] MIST_HD (ivec_t<S>) { return T{0}; });
}

// ones: constant one
template<typename T, std::size_t S>
auto ones(const index_space_t<S>& space) {
    return lazy(space, [] MIST_HD (ivec_t<S>) { return T{1}; });
}

// fill: constant value
template<typename T, std::size_t S>
auto fill(const index_space_t<S>& space, T value) {
    return lazy(space, [value] MIST_HD (ivec_t<S>) { return value; });
}

// indices: multi-dimensional index at each position
template<std::size_t S>
auto indices(const index_space_t<S>& space) {
    return lazy(space, [] MIST_HD (ivec_t<S> idx) { return idx; });
}

// offsets: flat offset at each position
template<std::size_t S>
auto offsets(const index_space_t<S>& space) {
    return lazy(space, [space] MIST_HD (ivec_t<S> idx) { return ndoffset(space, idx); });
}

// range: 1D integer sequence [0, n)
inline auto range(std::size_t n) {
    auto space = index_space(ivec(0), uvec(static_cast<unsigned int>(n)));
    return lazy(space, [] MIST_HD (ivec_t<1> idx) { return idx[0]; });
}

// linspace: 1D evenly spaced values
inline auto linspace(double start, double stop, std::size_t n, bool endpoint = true) {
    auto space = index_space(ivec(0), uvec(static_cast<unsigned int>(n)));
    double step = (stop - start) / (endpoint ? (n - 1) : n);
    return lazy(space, [start, step] MIST_HD (ivec_t<1> idx) {
        return start + idx[0] * step;
    });
}

// coords: physical coordinates (origin + idx * delta)
template<std::size_t S>
auto coords(const index_space_t<S>& space, const dvec_t<S>& origin, const dvec_t<S>& delta) {
    return lazy(space, [origin, delta] MIST_HD (ivec_t<S> idx) {
        dvec_t<S> result{};
        for (std::size_t i = 0; i < S; ++i) {
            result._data[i] = origin._data[i] + idx._data[i] * delta._data[i];
        }
        return result;
    });
}

// // join: combine multiple arrays into one lazy array
// // Arrays are stored in std::array and looked up by index containment
// // The combined space is the bounding box of all input spaces
// namespace detail {
//     template<typename T, std::size_t N, std::size_t S>
//     index_space_t<S> bounding_box(const std::array<const cached_t<T, S>*, N>& arrays) {
//         ivec_t<S> lo = start(space(*arrays[0]));
//         ivec_t<S> hi = upper(space(*arrays[0]));
//         for (std::size_t n = 1; n < N; ++n) {
//             auto s = space(*arrays[n]);
//             for (std::size_t d = 0; d < S; ++d) {
//                 lo._data[d] = std::min(lo._data[d], start(s)._data[d]);
//                 hi._data[d] = std::max(hi._data[d], upper(s)._data[d]);
//             }
//         }
//         uvec_t<S> sh;
//         for (std::size_t d = 0; d < S; ++d) {
//             sh._data[d] = static_cast<unsigned int>(hi._data[d] - lo._data[d]);
//         }
//         return index_space(lo, sh);
//     }
// }
//
// template<typename T, std::size_t S, typename... Arrays>
//     requires (std::same_as<Arrays, cached_t<T, S>> && ...)
// auto join(const Arrays&... arrays) {
//     constexpr std::size_t N = sizeof...(Arrays);
//     auto ptrs = std::array<const cached_t<T, S>*, N>{&arrays...};
//     auto combined_space = detail::bounding_box(ptrs);
//
//     return lazy(combined_space, [ptrs](ivec_t<S> idx) {
//         for (std::size_t n = 0; n < N; ++n) {
//             if (contains(space(*ptrs[n]), idx)) {
//                 return (*ptrs[n])(idx);
//             }
//         }
//         return T{};  // Default if not found (shouldn't happen with proper usage)
//     });
// }

// =============================================================================
// Reduction operations
// =============================================================================

// min: minimum value in an array
template<NdArray A>
    requires std::is_arithmetic_v<typename A::value_type>
auto min(const A& array, exec e = exec::cpu) -> typename A::value_type {
    using T = typename A::value_type;
    constexpr auto S = A::rank;
    const auto sp = space(array);

    if (size(sp) == 0) {
        throw std::runtime_error("min: empty array");
    }

    const auto first_value = array(start(sp));
    return map_reduce(sp, first_value,
        [&array] MIST_HD (ivec_t<S> idx) { return array(idx); },
        [] MIST_HD (T a, T b) { return a < b ? a : b; },
        e
    );
}

// max: maximum value in an array
template<NdArray A>
    requires std::is_arithmetic_v<typename A::value_type>
auto max(const A& array, exec e = exec::cpu) -> typename A::value_type {
    using T = typename A::value_type;
    constexpr auto S = A::rank;
    const auto sp = space(array);

    if (size(sp) == 0) {
        throw std::runtime_error("max: empty array");
    }

    const auto first_value = array(start(sp));
    return map_reduce(sp, first_value,
        [&array] MIST_HD (ivec_t<S> idx) { return array(idx); },
        [] MIST_HD (T a, T b) { return a > b ? a : b; },
        e
    );
}

// sum: sum of all values in an array
template<NdArray A>
    requires std::is_arithmetic_v<typename A::value_type>
auto sum(const A& array, exec e = exec::cpu) -> typename A::value_type {
    using T = typename A::value_type;
    constexpr auto S = A::rank;
    const auto sp = space(array);

    return map_reduce(sp, T{0},
        [&array] MIST_HD (ivec_t<S> idx) { return array(idx); },
        [] MIST_HD (T a, T b) { return a + b; },
        e
    );
}

} // namespace mist
