#pragma once

// Core library

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

// CUDA compatibility macros
#ifdef __CUDACC__
#define MIST_HD __host__ __device__
#include <cub/cub.cuh>
#else
#define MIST_HD
#define __host__
#define __device__
#endif

namespace mist {

// =============================================================================
// Concepts
// =============================================================================

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept Numeric = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

template<typename F, typename T>
concept UnaryFunction = requires(F f, T t) {
    { f(t) };
};

// =============================================================================
// vec_t: Statically sized array type
// =============================================================================

template<Arithmetic T, std::size_t S>
    requires (S > 0)
struct vec_t {
    T _data[S];

    MIST_HD constexpr T& operator[](std::size_t i) { return _data[i]; }
    MIST_HD constexpr const T& operator[](std::size_t i) const { return _data[i]; }

    MIST_HD constexpr std::size_t size() const { return S; }

    // Spaceship operator for comparisons
    MIST_HD constexpr auto operator<=>(const vec_t&) const = default;
};

// Type aliases
template<std::size_t S> using dvec_t = vec_t<double, S>;
template<std::size_t S> using ivec_t = vec_t<int, S>;
template<std::size_t S> using uvec_t = vec_t<unsigned int, S>;

// =============================================================================
// vec_t free functions
// =============================================================================

// Get element at index (free function version)
template<Arithmetic T, std::size_t S>
MIST_HD constexpr T& at(vec_t<T, S>& v, std::size_t i) {
    return v._data[i];
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr const T& at(const vec_t<T, S>& v, std::size_t i) {
    return v._data[i];
}

// Get size (free function version)
template<Arithmetic T, std::size_t S>
MIST_HD constexpr std::size_t size(const vec_t<T, S>&) {
    return S;
}

// Get pointer to data
template<Arithmetic T, std::size_t S>
MIST_HD constexpr T* data(vec_t<T, S>& v) {
    return v._data;
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr const T* data(const vec_t<T, S>& v) {
    return v._data;
}

// Begin/end iterators for range-based for loops
template<Arithmetic T, std::size_t S>
MIST_HD constexpr T* begin(vec_t<T, S>& v) {
    return v._data;
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr const T* begin(const vec_t<T, S>& v) {
    return v._data;
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr T* end(vec_t<T, S>& v) {
    return v._data + S;
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr const T* end(const vec_t<T, S>& v) {
    return v._data + S;
}

// =============================================================================
// Constructors
// =============================================================================

// Generic vec constructor with type deduction
template<typename... Args>
    requires (sizeof...(Args) > 0) && (std::is_arithmetic_v<Args> && ...)
MIST_HD constexpr auto vec(Args... args) {
    using T = std::common_type_t<Args...>;
    return vec_t<T, sizeof...(Args)>{static_cast<T>(args)...};
}

// Typed constructors
template<typename... Args>
    requires (sizeof...(Args) > 0)
MIST_HD constexpr auto dvec(Args... args) {
    return vec_t<double, sizeof...(Args)>{static_cast<double>(args)...};
}

template<typename... Args>
    requires (sizeof...(Args) > 0)
MIST_HD constexpr auto ivec(Args... args) {
    return vec_t<int, sizeof...(Args)>{static_cast<int>(args)...};
}

template<typename... Args>
    requires (sizeof...(Args) > 0)
MIST_HD constexpr auto uvec(Args... args) {
    return vec_t<unsigned int, sizeof...(Args)>{static_cast<unsigned int>(args)...};
}

// Range constructor: generates [0, 1, 2, ..., S-1]
namespace detail {
    template<std::size_t S, std::size_t... Is>
    MIST_HD constexpr uvec_t<S> range_impl(std::index_sequence<Is...>) {
        return uvec_t<S>{static_cast<unsigned int>(Is)...};
    }
}

template<std::size_t S>
    requires (S > 0)
MIST_HD constexpr uvec_t<S> range() {
    return detail::range_impl<S>(std::make_index_sequence<S>{});
}

// =============================================================================
// Operators
// =============================================================================

// Addition: vec + vec
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto operator+(const vec_t<T, S>& a, const vec_t<U, S>& b) {
    using R = decltype(std::declval<T>() + std::declval<U>());
    vec_t<R, S> result{};
    for (std::size_t i = 0; i < S; ++i) {
        result._data[i] = a._data[i] + b._data[i];
    }
    return result;
}

// Subtraction: vec - vec
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto operator-(const vec_t<T, S>& a, const vec_t<U, S>& b) {
    using R = decltype(std::declval<T>() - std::declval<U>());
    vec_t<R, S> result{};
    for (std::size_t i = 0; i < S; ++i) {
        result._data[i] = a._data[i] - b._data[i];
    }
    return result;
}

// Multiplication: vec * scalar
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto operator*(const vec_t<T, S>& v, U scalar) {
    using R = decltype(std::declval<T>() * std::declval<U>());
    vec_t<R, S> result{};
    for (std::size_t i = 0; i < S; ++i) {
        result._data[i] = v._data[i] * scalar;
    }
    return result;
}

// Multiplication: scalar * vec
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto operator*(T scalar, const vec_t<U, S>& v) {
    return v * scalar;
}

// Division: vec / scalar
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto operator/(const vec_t<T, S>& v, U scalar) {
    using R = decltype(std::declval<T>() / std::declval<U>());
    vec_t<R, S> result{};
    for (std::size_t i = 0; i < S; ++i) {
        result._data[i] = v._data[i] / scalar;
    }
    return result;
}

// Compound assignment: vec += vec
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto& operator+=(vec_t<T, S>& a, const vec_t<U, S>& b) {
    for (std::size_t i = 0; i < S; ++i) {
        a._data[i] += b._data[i];
    }
    return a;
}

// Compound assignment: vec -= vec
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto& operator-=(vec_t<T, S>& a, const vec_t<U, S>& b) {
    for (std::size_t i = 0; i < S; ++i) {
        a._data[i] -= b._data[i];
    }
    return a;
}

// Compound assignment: vec *= scalar
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto& operator*=(vec_t<T, S>& v, U scalar) {
    for (std::size_t i = 0; i < S; ++i) {
        v._data[i] *= scalar;
    }
    return v;
}

// Compound assignment: vec /= scalar
template<Arithmetic T, Arithmetic U, std::size_t S>
MIST_HD constexpr auto& operator/=(vec_t<T, S>& v, U scalar) {
    for (std::size_t i = 0; i < S; ++i) {
        v._data[i] /= scalar;
    }
    return v;
}

// Dot product
template<Arithmetic T, Arithmetic U, std::size_t S>
constexpr auto dot(const vec_t<T, S>& a, const vec_t<U, S>& b) {
    using R = decltype(std::declval<T>() * std::declval<U>());
    R result{};
    for (std::size_t i = 0; i < S; ++i) {
        result += a._data[i] * b._data[i];
    }
    return result;
}

// Map function
template<Arithmetic T, std::size_t S, typename F>
    requires UnaryFunction<F, T>
constexpr auto map(const vec_t<T, S>& v, F&& func) {
    using R = decltype(func(std::declval<T>()));
    vec_t<R, S> result{};
    for (std::size_t i = 0; i < S; ++i) {
        result._data[i] = func(v._data[i]);
    }
    return result;
}

// Reduction functions
template<Arithmetic T, std::size_t S>
MIST_HD constexpr T sum(const vec_t<T, S>& v) {
    T result{};
    for (std::size_t i = 0; i < S; ++i) {
        result += v._data[i];
    }
    return result;
}

template<Arithmetic T, std::size_t S>
MIST_HD constexpr T product(const vec_t<T, S>& v) {
    T result = T(1);
    for (std::size_t i = 0; i < S; ++i) {
        result *= v._data[i];
    }
    return result;
}

template<std::size_t S>
MIST_HD constexpr bool any(const vec_t<bool, S>& v) {
    for (std::size_t i = 0; i < S; ++i) {
        if (v._data[i]) return true;
    }
    return false;
}

template<std::size_t S>
MIST_HD constexpr bool all(const vec_t<bool, S>& v) {
    for (std::size_t i = 0; i < S; ++i) {
        if (!v._data[i]) return false;
    }
    return true;
}

// =============================================================================
// index_space_t: Multi-dimensional index space
// =============================================================================

template<std::size_t S>
    requires (S > 0)
struct index_space_t {
    ivec_t<S> _start;
    uvec_t<S> _shape;

    constexpr auto operator<=>(const index_space_t&) const = default;
};

// Constructor for index_space_t
template<std::size_t S>
MIST_HD constexpr index_space_t<S> index_space(const ivec_t<S>& start, const uvec_t<S>& shape) {
    return index_space_t<S>{._start = start, ._shape = shape};
}

// Free functions for index_space_t
template<std::size_t S>
constexpr const ivec_t<S>& start(const index_space_t<S>& space) {
    return space._start;
}

template<std::size_t S>
constexpr const uvec_t<S>& shape(const index_space_t<S>& space) {
    return space._shape;
}

template<std::size_t S>
constexpr ivec_t<S> upper(const index_space_t<S>& space) {
    return space._start + map(space._shape, [](unsigned int v) { return static_cast<int>(v); });
}

template<std::size_t S>
MIST_HD constexpr ivec_t<S> clamp(const ivec_t<S>& index, const index_space_t<S>& space) {
    auto result = index;
    auto u = upper(space);
    for (std::size_t i = 0; i < S; ++i) {
        if (result._data[i] < space._start._data[i]) {
            result._data[i] = space._start._data[i];
        } else if (result._data[i] >= u._data[i]) {
            result._data[i] = u._data[i] - 1;
        }
    }
    return result;
}

template<std::size_t S>
MIST_HD constexpr unsigned int size(const index_space_t<S>& space) {
    unsigned int total = 1;
    for (std::size_t i = 0; i < S; ++i) {
        total *= space._shape._data[i];
    }
    return total;
}

template<std::size_t S>
MIST_HD constexpr bool contains(const index_space_t<S>& space, const ivec_t<S>& index) {
    for (std::size_t i = 0; i < S; ++i) {
        if (index._data[i] < space._start._data[i] ||
            index._data[i] >= space._start._data[i] + static_cast<int>(space._shape._data[i])) {
            return false;
        }
    }
    return true;
}

template<std::size_t S>
constexpr bool contains(const index_space_t<S>& space, const index_space_t<S>& other) {
    if (size(other) == 0) {
        return true;
    }
    ivec_t<S> other_front = other._start;
    ivec_t<S> other_back = other._start + map(other._shape, [](unsigned int v) { return static_cast<int>(v) - 1; });
    return contains(space, other_front) && contains(space, other_back);
}

template<std::size_t S>
constexpr bool overlaps(const index_space_t<S>& a, const index_space_t<S>& b) {
    for (std::size_t i = 0; i < S; ++i) {
        auto a_start = a._start._data[i];
        auto a_end = a_start + static_cast<int>(a._shape._data[i]);
        auto b_start = b._start._data[i];
        auto b_end = b_start + static_cast<int>(b._shape._data[i]);
        if (a_end <= b_start || b_end <= a_start) {
            return false;
        }
    }
    return size(a) > 0 && size(b) > 0;
}

template<std::size_t S>
constexpr index_space_t<S> subspace(const index_space_t<S>& space, unsigned int num_partitions, unsigned int which_partition, unsigned int axis) {
    auto large_partition_size = space._shape._data[axis] / num_partitions + 1;
    auto small_partition_size = space._shape._data[axis] / num_partitions;
    auto num_large_partitions = space._shape._data[axis] % num_partitions;
    auto n_large = which_partition < num_large_partitions ? which_partition : num_large_partitions;
    auto n_small = which_partition > n_large ? which_partition - n_large : 0;
    auto i0 = n_large * large_partition_size + n_small * small_partition_size;
    auto di = which_partition < num_large_partitions ? large_partition_size : small_partition_size;

    auto result = space;
    result._start._data[axis] = static_cast<int>(i0) + space._start._data[axis];
    result._shape._data[axis] = di;
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> subspace(const index_space_t<S>& space, const uvec_t<S>& shape, const uvec_t<S>& coords) {
    auto result = space;
    for (std::size_t axis = 0; axis < S; ++axis) {
        result = subspace(result, shape._data[axis], coords._data[axis], axis);
    }
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> shift(const index_space_t<S>& space, int amount, unsigned int axis) {
    auto result = space;
    result._start._data[axis] += amount;
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> nudge(const index_space_t<S>& space, const ivec_t<S>& lower, const ivec_t<S>& upper) {
    auto result = space;
    for (std::size_t axis = 0; axis < S; ++axis) {
        result._start._data[axis] += lower._data[axis];
        result._shape._data[axis] += static_cast<unsigned int>(upper._data[axis] - lower._data[axis]);
    }
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> contract(const index_space_t<S>& space, const uvec_t<S>& count) {
    ivec_t<S> lower{};
    ivec_t<S> upper{};
    for (std::size_t i = 0; i < S; ++i) {
        lower._data[i] = +static_cast<int>(count._data[i]);
        upper._data[i] = -static_cast<int>(count._data[i]);
    }
    return nudge(space, lower, upper);
}

template<std::size_t S>
constexpr index_space_t<S> contract(const index_space_t<S>& space, unsigned int count) {
    uvec_t<S> count_vec{};
    for (std::size_t i = 0; i < S; ++i) {
        count_vec._data[i] = count;
    }
    return contract(space, count_vec);
}

template<std::size_t S>
constexpr index_space_t<S> expand(const index_space_t<S>& space, const uvec_t<S>& count) {
    ivec_t<S> lower{};
    ivec_t<S> upper{};
    for (std::size_t i = 0; i < S; ++i) {
        lower._data[i] = -static_cast<int>(count._data[i]);
        upper._data[i] = +static_cast<int>(count._data[i]);
    }
    return nudge(space, lower, upper);
}

template<std::size_t S>
constexpr index_space_t<S> expand(const index_space_t<S>& space, unsigned int count) {
    uvec_t<S> count_vec{};
    for (std::size_t i = 0; i < S; ++i) {
        count_vec._data[i] = count;
    }
    return expand(space, count_vec);
}

template<std::size_t S>
constexpr index_space_t<S> translate(const index_space_t<S>& space, const ivec_t<S>& new_start) {
    auto result = space;
    result._start = new_start;
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> upper(const index_space_t<S>& space, unsigned int amount, unsigned int axis) {
    auto result = space;
    result._start._data[axis] = space._start._data[axis] + static_cast<int>(space._shape._data[axis]) - static_cast<int>(amount);
    result._shape._data[axis] = amount;
    return result;
}

template<std::size_t S>
constexpr index_space_t<S> lower(const index_space_t<S>& space, unsigned int amount, unsigned int axis) {
    auto result = space;
    result._shape._data[axis] = amount;
    return result;
}

// =============================================================================
// Multi-dimensional indexing
// =============================================================================

// Convert multi-dimensional index to flat offset (row-major ordering)
template<std::size_t S>
MIST_HD constexpr std::size_t ndoffset(const index_space_t<S>& space, const ivec_t<S>& index) {
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::size_t i = S; i > 0; --i) {
        offset += static_cast<std::size_t>(index._data[i - 1] - space._start._data[i - 1]) * stride;
        stride *= space._shape._data[i - 1];
    }
    return offset;
}

// Convert flat offset to multi-dimensional index (row-major ordering)
template<std::size_t S>
MIST_HD constexpr ivec_t<S> ndindex(const index_space_t<S>& space, std::size_t offset) {
    ivec_t<S> index{};
    for (std::size_t i = S; i > 0; --i) {
        index._data[i - 1] = space._start._data[i - 1] + static_cast<int>(offset % space._shape._data[i - 1]);
        offset /= space._shape._data[i - 1];
    }
    return index;
}

// Read scalar from buffer
template<typename T, std::size_t S>
MIST_HD constexpr T ndread(const T* data, const index_space_t<S>& space, const ivec_t<S>& index) {
    return data[ndoffset(space, index)];
}

// Write scalar to buffer
template<typename T, std::size_t S>
MIST_HD constexpr void ndwrite(T* data, const index_space_t<S>& space, const ivec_t<S>& index, T value) {
    data[ndoffset(space, index)] = value;
}

// Read vec_t from SoA buffer (component-major layout)
template<typename T, std::size_t N, std::size_t S>
    requires Arithmetic<T>
MIST_HD constexpr vec_t<T, N> ndread_soa(const T* data, const index_space_t<S>& space, const ivec_t<S>& index) {
    vec_t<T, N> result{};
    std::size_t offset = ndoffset(space, index);
    std::size_t stride = size(space);
    for (std::size_t i = 0; i < N; ++i) {
        result._data[i] = data[i * stride + offset];
    }
    return result;
}

// Write vec_t to SoA buffer (component-major layout)
template<typename T, std::size_t N, std::size_t S>
    requires Arithmetic<T>
MIST_HD constexpr void ndwrite_soa(T* data, const index_space_t<S>& space, const ivec_t<S>& index, const vec_t<T, N>& value) {
    std::size_t offset = ndoffset(space, index);
    std::size_t stride = size(space);
    for (std::size_t i = 0; i < N; ++i) {
        data[i * stride + offset] = value._data[i];
    }
}

// =============================================================================
// Iterator for index_space_t
// =============================================================================

template<std::size_t S>
class index_space_iterator {
    const index_space_t<S>* _space;
    std::size_t _offset;

public:
    using value_type = ivec_t<S>;
    using difference_type = std::ptrdiff_t;

    constexpr index_space_iterator(const index_space_t<S>* space, std::size_t offset)
        : _space(space), _offset(offset) {}

    constexpr ivec_t<S> operator*() const {
        return ndindex(*_space, _offset);
    }

    constexpr index_space_iterator& operator++() {
        ++_offset;
        return *this;
    }

    constexpr index_space_iterator operator++(int) {
        auto tmp = *this;
        ++_offset;
        return tmp;
    }

    constexpr bool operator==(const index_space_iterator& other) const {
        return _offset == other._offset;
    }

    constexpr bool operator!=(const index_space_iterator& other) const {
        return _offset != other._offset;
    }
};

template<std::size_t S>
constexpr index_space_iterator<S> begin(const index_space_t<S>& space) {
    return index_space_iterator<S>(&space, 0);
}

template<std::size_t S>
constexpr index_space_iterator<S> end(const index_space_t<S>& space) {
    return index_space_iterator<S>(&space, size(space));
}

// =============================================================================
// Execution policies
// =============================================================================

enum class exec {
    cpu,
    omp,
    gpu
};

// =============================================================================
// Parallel index space traversals
// =============================================================================

#ifdef __CUDACC__
template<std::size_t S, typename F>
__global__ void for_each_kernel(index_space_t<S> space, F func) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < size(space)) {
        func(ndindex(space, n));
    }
}

// Map kernel for map_reduce
template<std::size_t S, typename T, typename MapF>
__global__ void map_kernel(index_space_t<S> space, MapF map, T* d_output) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < size(space)) {
        d_output[n] = map(ndindex(space, n));
    }
}
#endif

template<std::size_t S, typename F>
void for_each(const index_space_t<S>& space, F&& func, exec e) {
    switch (e) {
        case exec::cpu: {
            for (std::size_t n = 0; n < size(space); ++n) {
                func(ndindex(space, n));
            }
            break;
        }
        case exec::omp: {
            #ifdef _OPENMP
            #pragma omp parallel for
            for (std::size_t n = 0; n < size(space); ++n) {
                func(ndindex(space, n));
            }
            #else
            throw std::runtime_error("unsupported exec::omp");
            #endif
            break;
        }
        case exec::gpu: {
            #ifdef __CUDACC__
            int blockSize = 256;
            int numBlocks = (size(space) + blockSize - 1) / blockSize;
            for_each_kernel<<<numBlocks, blockSize>>>(space, func);
            #else
            throw std::runtime_error("unsupported exec::gpu");
            #endif
            break;
        }
    }
}

// Default: CPU execution
template<std::size_t S, typename F>
void for_each(const index_space_t<S>& space, F&& func) {
    for_each(space, std::forward<F>(func), exec::cpu);
}

// =============================================================================
// Map-Reduce
// =============================================================================

template<std::size_t S, typename T, typename MapF, typename ReduceF>
T map_reduce(const index_space_t<S>& space, T init, MapF&& map, ReduceF&& reduce_op, exec e) {
    switch (e) {
        case exec::cpu: {
            T result = init;
            for (std::size_t n = 0; n < size(space); ++n) {
                result = reduce_op(result, map(ndindex(space, n)));
            }
            return result;
        }
        case exec::omp: {
            #ifdef _OPENMP
            T global_result = init;

            #pragma omp parallel
            {
                T local_result = init;

                #pragma omp for nowait
                for (std::size_t n = 0; n < size(space); ++n) {
                    local_result = reduce_op(local_result, map(ndindex(space, n)));
                }

                #pragma omp critical
                {
                    global_result = reduce_op(global_result, local_result);
                }
            }

            return global_result;
            #else
            throw std::runtime_error("unsupported exec::omp");
            #endif
        }
        case exec::gpu: {
            #ifdef __CUDACC__
            // Step 1: Map indices to values in device memory
            std::size_t n = size(space);
            T* d_mapped;
            cudaMalloc(&d_mapped, n * sizeof(T));

            // Launch map kernel
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;
            map_kernel<<<numBlocks, blockSize>>>(space, map, d_mapped);
            cudaDeviceSynchronize();

            // Step 2: Use CUB to reduce the mapped values
            T* d_result;
            cudaMalloc(&d_result, sizeof(T));

            void* d_temp_storage = nullptr;
            std::size_t temp_storage_bytes = 0;

            // Determine temporary storage requirements
            cub::DeviceReduce::Reduce(
                d_temp_storage, temp_storage_bytes,
                d_mapped, d_result, n, reduce_op, init
            );

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            // Run reduction
            cub::DeviceReduce::Reduce(
                d_temp_storage, temp_storage_bytes,
                d_mapped, d_result, n, reduce_op, init
            );

            // Copy result to host
            T result;
            cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

            // Cleanup
            cudaFree(d_mapped);
            cudaFree(d_result);
            cudaFree(d_temp_storage);

            return result;
            #else
            throw std::runtime_error("unsupported exec::gpu");
            #endif
        }
    }
}

// Default: CPU execution
template<std::size_t S, typename T, typename MapF, typename ReduceF>
T map_reduce(const index_space_t<S>& space, T init, MapF&& map, ReduceF&& reduce_op) {
    return map_reduce(space, init, std::forward<MapF>(map), std::forward<ReduceF>(reduce_op), exec::cpu);
}

} // namespace mist

#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

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
    T* _data;                   // Pointer to element at _space.start in parent
    uvec_t<S> _strides;         // Parent's strides for offset calculation

    // Constructor for contiguous view (strides derived from space)
    array_view_t(const index_space_t<S>& space, T* data)
        : _space(space), _data(data), _strides(compute_strides(shape(space))) {}

    // Constructor for strided view (subspace of parent)
    array_view_t(const index_space_t<S>& space, T* data, const uvec_t<S>& strides)
        : _space(space), _data(data), _strides(strides) {}

    // Converting constructor: array_view_t<U> -> array_view_t<const U>
    template<typename U>
        requires std::is_same_v<T, const U>
    array_view_t(const array_view_t<U, S>& other)
        : _space(other._space), _data(other._data), _strides(other._strides) {}

    MIST_HD T& operator()(const ivec_t<S>& idx) {
        return _data[strided_offset(idx)];
    }

    MIST_HD const T& operator()(const ivec_t<S>& idx) const {
        return _data[strided_offset(idx)];
    }

    // Convenience for 1D arrays: accept integer index
    MIST_HD T& operator[](int i) requires (S == 1) {
        return _data[strided_offset(ivec(i))];
    }

    MIST_HD const T& operator[](int i) const requires (S == 1) {
        return _data[strided_offset(ivec(i))];
    }

private:
    // Compute strides from shape (row-major)
    static constexpr uvec_t<S> compute_strides(const uvec_t<S>& shape) {
        uvec_t<S> strides{};
        strides._data[S - 1] = 1;
        for (std::size_t i = S - 1; i > 0; --i) {
            strides._data[i - 1] = strides._data[i] * shape._data[i];
        }
        return strides;
    }

    // Compute offset using stored strides
    MIST_HD constexpr std::size_t strided_offset(const ivec_t<S>& idx) const {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < S; ++i) {
            offset += static_cast<std::size_t>(idx._data[i] - _space._start._data[i]) * _strides._data[i];
        }
        return offset;
    }
};

template<typename T, std::size_t S>
const index_space_t<S>& space(const array_view_t<T, S>& a) { return a._space; }

template<typename T, std::size_t S>
T* data(array_view_t<T, S>& a) { return a._data; }

template<typename T, std::size_t S>
const T* data(const array_view_t<T, S>& a) { return a._data; }

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

// View of subspace (strided access into parent)
template<typename T, std::size_t S>
auto view(array_t<T, S>& a, const index_space_t<S>& subspace) -> array_view_t<T, S> {
    // Compute strides from parent's shape
    uvec_t<S> strides{};
    strides._data[S - 1] = 1;
    for (std::size_t i = S - 1; i > 0; --i) {
        strides._data[i - 1] = strides._data[i] * a._space._shape._data[i];
    }
    // Compute pointer to first element of subspace
    T* ptr = a._data + ndoffset(a._space, start(subspace));
    return array_view_t<T, S>(subspace, ptr, strides);
}

template<typename T, std::size_t S>
auto view(const array_t<T, S>& a, const index_space_t<S>& subspace) -> array_view_t<const T, S> {
    uvec_t<S> strides{};
    strides._data[S - 1] = 1;
    for (std::size_t i = S - 1; i > 0; --i) {
        strides._data[i - 1] = strides._data[i] * a._space._shape._data[i];
    }
    const T* ptr = a._data + ndoffset(a._space, start(subspace));
    return array_view_t<const T, S>(subspace, ptr, strides);
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

#include <concepts>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace mist {

// =============================================================================
// Field wrapper for named serialization
// =============================================================================

template<typename T>
struct field_t {
    const char* name;
    T& value;
};

template<typename T>
constexpr field_t<T> field(const char* name, T& value) {
    return field_t<T>{name, value};
}

template<typename T>
constexpr field_t<const T> field(const char* name, const T& value) {
    return field_t<const T>{name, value};
}

// =============================================================================
// Type traits for serialization
// =============================================================================

// Check if type is a vec_t
template<typename T>
struct is_vec : std::false_type {};

template<typename T, std::size_t N>
struct is_vec<vec_t<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

// Check if type is a std::vector
template<typename T>
struct is_std_vector : std::false_type {};

template<typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template<typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

// Get element type of std::vector
template<typename T>
struct vector_element_type { using type = void; };

template<typename T, typename A>
struct vector_element_type<std::vector<T, A>> { using type = T; };

template<typename T>
using vector_element_type_t = typename vector_element_type<T>::type;

// =============================================================================
// Serializable concept
// =============================================================================

template<typename T>
concept HasFields = requires(T t) {
    { t.fields() } -> std::same_as<decltype(t.fields())>;
};

template<typename T>
concept HasConstFields = requires(const T t) {
    { t.fields() } -> std::same_as<decltype(t.fields())>;
};

// =============================================================================
// Enum string conversion via ADL
// =============================================================================

// Concept to detect if a type has ADL to_string/from_string for enum conversion
template<typename E>
concept HasEnumStrings = std::is_enum_v<E> && requires(E e, const std::string& s) {
    { to_string(e) } -> std::convertible_to<const char*>;
    { from_string(std::type_identity<E>{}, s) } -> std::same_as<E>;
};

// =============================================================================
// Archive concepts
// =============================================================================

template<typename A>
concept ArchiveWriter = requires(A& ar, const char* name) {
    { ar.begin_named(name) } -> std::same_as<void>;
    { ar.write(int{}) } -> std::same_as<void>;
    { ar.write(double{}) } -> std::same_as<void>;
    { ar.write(std::string{}) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;
    { ar.end_group() } -> std::same_as<void>;
    { ar.begin_list() } -> std::same_as<void>;
    { ar.end_list() } -> std::same_as<void>;
};

template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d, std::string& s) {
    { ar.begin_named(name) } -> std::same_as<void>;
    { ar.read(i) } -> std::same_as<bool>;
    { ar.read(d) } -> std::same_as<bool>;
    { ar.read(s) } -> std::same_as<bool>;
    { ar.begin_group() } -> std::same_as<bool>;
    { ar.end_group() } -> std::same_as<void>;
    { ar.begin_list() } -> std::same_as<bool>;
    { ar.end_list() } -> std::same_as<void>;
    { ar.has_field(name) } -> std::same_as<bool>;
    { ar.count_items(name) } -> std::same_as<std::size_t>;
};

// =============================================================================
// Serialize declarations (two-arg: anonymous, three-arg: named)
// =============================================================================

// Named serialization - convenience wrapper
template<ArchiveWriter A, typename T>
void serialize(A& ar, const char* name, const T& value);

// Anonymous serialization - core implementations
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const T& value);

template<ArchiveWriter A>
void serialize(A& ar, const std::string& value);

template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const E& value);

template<ArchiveWriter A, typename T1, typename T2>
void serialize(A& ar, const std::pair<T1, T2>& value);

template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const vec_t<T, N>& value);

template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
    requires (!std::is_arithmetic_v<T>)
void serialize(A& ar, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::map<std::string, T>& value);

template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const T& value);

template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::optional<T>& value);

template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const T& value);

template<ArchiveWriter A, typename... Ts>
void serialize(A& ar, const std::variant<Ts...>& value);

// =============================================================================
// Deserialize declarations (two-arg: anonymous, three-arg: named)
// =============================================================================

// Named deserialization - convenience wrapper
template<ArchiveReader A, typename T>
auto deserialize(A& ar, const char* name, T& value) -> bool;

// Anonymous deserialization - core implementations
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, T& value) -> bool;

template<ArchiveReader A>
auto deserialize(A& ar, std::string& value) -> bool;

template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
auto deserialize(A& ar, E& value) -> bool;

template<ArchiveReader A, typename T1, typename T2>
auto deserialize(A& ar, std::pair<T1, T2>& value) -> bool;

template<ArchiveReader A, typename T, std::size_t N>
auto deserialize(A& ar, vec_t<T, N>& value) -> bool;

template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, std::vector<T>& value) -> bool;

template<ArchiveReader A, typename T>
    requires (!std::is_arithmetic_v<T>)
auto deserialize(A& ar, std::vector<T>& value) -> bool;

template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::map<std::string, T>& value) -> bool;

template<ArchiveReader A, typename T>
    requires HasFields<T>
auto deserialize(A& ar, T& value) -> bool;

template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::optional<T>& value) -> bool;

template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, T& value) -> bool;

template<ArchiveReader A, typename... Ts>
auto deserialize(A& ar, std::variant<Ts...>& value) -> bool;

// =============================================================================
// Named wrappers (three-arg versions)
// =============================================================================

template<ArchiveWriter A, typename T>
void serialize(A& ar, const char* name, const T& value) {
    ar.begin_named(name);
    serialize(ar, value);
}

template<ArchiveReader A, typename T>
auto deserialize(A& ar, const char* name, T& value) -> bool {
    ar.begin_named(name);
    return deserialize(ar, value);
}

// =============================================================================
// Serialize implementations (two-arg anonymous versions)
// =============================================================================

// Scalar types
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const T& value) {
    ar.write(value);
}

// std::string
template<ArchiveWriter A>
void serialize(A& ar, const std::string& value) {
    ar.write(value);
}

// Enums with ADL to_string/from_string
template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const E& value) {
    ar.write(std::string(to_string(value)));
}

// std::pair<T1, T2>
template<ArchiveWriter A, typename T1, typename T2>
void serialize(A& ar, const std::pair<T1, T2>& value) {
    ar.begin_group();
    serialize(ar, "first", value.first);
    serialize(ar, "second", value.second);
    ar.end_group();
}

// vec_t<T, N>
template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const vec_t<T, N>& value) {
    ar.write(value);
}

// std::vector<T> where T is arithmetic - use ar.write() for efficiency
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const std::vector<T>& value) {
    ar.write(value);
}

// std::vector<T> where T is not arithmetic - serialize each element
template<ArchiveWriter A, typename T>
    requires (!std::is_arithmetic_v<T>)
void serialize(A& ar, const std::vector<T>& value) {
    ar.begin_list();
    for (const auto& elem : value) {
        serialize(ar, elem);
    }
    ar.end_list();
}

// std::map<std::string, T>
template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::map<std::string, T>& value) {
    ar.begin_list();
    for (const auto& [key, val] : value) {
        ar.begin_group();
        serialize(ar, "key", key);
        serialize(ar, "value", val);
        ar.end_group();
    }
    ar.end_list();
}

// Compound types with fields()
template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const T& value) {
    ar.begin_group();
    std::apply([&ar](auto&&... fields) {
        (serialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
}

// std::optional<T>
template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::optional<T>& value) {
    ar.begin_group();
    bool has_value = value.has_value();
    serialize(ar, "has_value", has_value);
    if (has_value) {
        serialize(ar, "value", *value);
    }
    ar.end_group();
}

// CachedNdArray - dispatches to ar.write()
template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const T& value) {
    ar.write(value);
}

// std::variant<Ts...>
template<ArchiveWriter A, typename... Ts>
void serialize(A& ar, const std::variant<Ts...>& value) {
    ar.begin_group();
    serialize(ar, "index", value.index());
    std::visit([&ar](const auto& v) { serialize(ar, "value", v); }, value);
    ar.end_group();
}

// =============================================================================
// Deserialize implementations (two-arg anonymous versions)
// =============================================================================

// Scalar types
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, T& value) -> bool {
    return ar.read(value);
}

// std::string
template<ArchiveReader A>
auto deserialize(A& ar, std::string& value) -> bool {
    return ar.read(value);
}

// Enums with ADL to_string/from_string
template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
auto deserialize(A& ar, E& value) -> bool {
    std::string str;
    if (!ar.read(str)) return false;
    value = from_string(std::type_identity<E>{}, str);
    return true;
}

// std::pair<T1, T2>
template<ArchiveReader A, typename T1, typename T2>
auto deserialize(A& ar, std::pair<T1, T2>& value) -> bool {
    if (!ar.begin_group()) return false;
    deserialize(ar, "first", value.first);
    deserialize(ar, "second", value.second);
    ar.end_group();
    return true;
}

// vec_t<T, N>
template<ArchiveReader A, typename T, std::size_t N>
auto deserialize(A& ar, vec_t<T, N>& value) -> bool {
    return ar.read(value);
}

// std::vector<T> where T is arithmetic - use ar.read() for efficiency
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, std::vector<T>& value) -> bool {
    return ar.read(value);
}

// std::vector<T> where T is not arithmetic - deserialize each element
template<ArchiveReader A, typename T>
    requires (!std::is_arithmetic_v<T>)
auto deserialize(A& ar, std::vector<T>& value) -> bool {
    if (!ar.begin_list()) return false;
    value.clear();
    while (true) {
        T elem;
        if (!deserialize(ar, elem)) break;
        value.push_back(std::move(elem));
    }
    ar.end_list();
    return true;
}

// std::map<std::string, T>
template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::map<std::string, T>& value) -> bool {
    if (!ar.begin_list()) return false;
    value.clear();
    while (ar.begin_group()) {
        std::string key;
        T val;
        deserialize(ar, "key", key);
        deserialize(ar, "value", val);
        value[key] = std::move(val);
        ar.end_group();
    }
    ar.end_list();
    return true;
}

// Compound types with fields()
template<ArchiveReader A, typename T>
    requires HasFields<T>
auto deserialize(A& ar, T& value) -> bool {
    if (!ar.begin_group()) return false;
    std::apply([&ar](auto&&... fields) {
        (deserialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
    return true;
}

// std::optional<T>
template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::optional<T>& value) -> bool {
    if (!ar.begin_group()) return false;
    bool has_value = false;
    deserialize(ar, "has_value", has_value);
    if (has_value) {
        T temp;
        deserialize(ar, "value", temp);
        value = std::move(temp);
    } else {
        value = std::nullopt;
    }
    ar.end_group();
    return true;
}

// CachedNdArray - dispatches to ar.read()
template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, T& value) -> bool {
    return ar.read(value);
}

// std::variant<Ts...> - helper to deserialize by index
namespace detail {

template<ArchiveReader A, typename Variant, std::size_t I = 0>
auto deserialize_variant_by_index(A& ar, Variant& value, std::size_t index) -> bool {
    if constexpr (I >= std::variant_size_v<Variant>) {
        return false;
    } else {
        if (I == index) {
            std::variant_alternative_t<I, Variant> temp;
            if (!deserialize(ar, temp)) return false;
            value = std::move(temp);
            return true;
        }
        return deserialize_variant_by_index<A, Variant, I + 1>(ar, value, index);
    }
}

} // namespace detail

template<ArchiveReader A, typename... Ts>
auto deserialize(A& ar, std::variant<Ts...>& value) -> bool {
    if (!ar.begin_group()) return false;
    std::size_t index = 0;
    deserialize(ar, "index", index);
    ar.begin_named("value");
    auto result = detail::deserialize_variant_by_index(ar, value, index);
    ar.end_group();
    return result;
}

// =============================================================================
// Config field setter by path
// =============================================================================

namespace detail {

// Helper to parse string to target type
template<typename T>
void parse_and_assign(T& target, const std::string& value) {
    if constexpr (std::is_same_v<T, int>) {
        target = std::stoi(value);
    } else if constexpr (std::is_same_v<T, long>) {
        target = std::stol(value);
    } else if constexpr (std::is_same_v<T, long long>) {
        target = std::stoll(value);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long>) {
        target = std::stoul(value);
    } else if constexpr (std::is_same_v<T, float>) {
        target = std::stof(value);
    } else if constexpr (std::is_same_v<T, double>) {
        target = std::stod(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        target = (value == "true" || value == "1");
    } else if constexpr (std::is_same_v<T, std::string>) {
        target = value;
    } else if constexpr (HasEnumStrings<T>) {
        target = from_string(std::type_identity<T>{}, value);
    } else {
        throw std::runtime_error("unsupported type for set()");
    }
}

// Forward declaration
template<typename T>
void set_impl(T& obj, const std::string& path, const std::string& value);

// Set field on a leaf type (non-HasFields)
template<typename T>
void set_field(T& target, const std::string& rest, const std::string& value) {
    if (rest.empty()) {
        parse_and_assign(target, value);
    } else if constexpr (HasFields<T>) {
        set_impl(target, rest, value);
    } else {
        throw std::runtime_error("cannot descend into '" + rest + "': not a struct");
    }
}

// Helper to try setting a field if name matches
template<typename T>
void try_set_field(const char* name, T& target, const std::string& key,
                   const std::string& rest, const std::string& value, bool& found) {
    if (!found && std::string(name) == key) {
        set_field(target, rest, value);
        found = true;
    }
}

// Set field by path on a HasFields type
template<typename T>
void set_impl(T& obj, const std::string& path, const std::string& value) {
    auto dot = path.find('.');
    std::string key = path.substr(0, dot);
    std::string rest = (dot != std::string::npos) ? path.substr(dot + 1) : "";

    bool found = false;
    std::apply([&](auto&&... field) {
        (try_set_field(field.name, field.value, key, rest, value, found), ...);
    }, obj.fields());

    if (!found) {
        throw std::runtime_error("field not found: " + key);
    }
}

} // namespace detail

/**
 * Set a field in a struct by dot-separated path.
 *
 * Example:
 *   set(config, "physics.gamma", "1.33");
 *   set(config, "driver.t_final", "2.0");
 *   set(config, "mesh.boundary", "periodic");  // enum with to_string/from_string
 */
template<HasFields T>
void set(T& obj, const std::string& path, const std::string& value) {
    detail::set_impl(obj, path, value);
}

} // namespace mist

// Archive formats

#include <cctype>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mist {

// =============================================================================
// ASCII Reader - key-based lookup, missing fields return false
// =============================================================================

class ascii_reader {
public:
    explicit ascii_reader(std::istream& is) : is_(is) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // --- Scalars (return false if field missing) ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(T& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        value = read_number<T>();
        return true;
    }

    auto read(std::string& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        } else {
            skip_ws();
            if (peek() != '"') return false;
        }
        value = read_quoted_string();
        return true;
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    auto read(vec_t<T, N>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        for (std::size_t i = 0; i < N; ++i) {
            skip_ws();
            value[i] = read_number<T>();
            skip_ws();
            if (i < N - 1) expect(',');
        }
        expect(']');
        return true;
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(std::vector<T>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        value.clear();
        skip_ws();
        if (peek() != ']') {
            while (true) {
                skip_ws();
                value.push_back(read_number<T>());
                skip_ws();
                if (peek() == ',') { get(); continue; }
                if (peek() == ']') break;
                throw std::runtime_error("expected ',' or ']'");
            }
        }
        expect(']');
        return true;
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_data(T* ptr, std::size_t count) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        for (std::size_t i = 0; i < count; ++i) {
            skip_ws();
            ptr[i] = read_number<T>();
            skip_ws();
            if (i < count - 1) expect(',');
        }
        skip_ws();
        expect(']');
        return true;
    }

    // Overload for vec_t elements: read as flattened scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    auto read_data(vec_t<T, N>* ptr, std::size_t count) -> bool {
        return read_data(reinterpret_cast<T*>(ptr), count * N);
    }

    // --- CachedNdArray ---

    template<CachedNdArray T>
    auto read(T& arr) -> bool {
        if (!begin_group()) return false;
        using start_t = std::decay_t<decltype(start(arr))>;
        using shape_t = std::decay_t<decltype(shape(arr))>;
        start_t st;
        shape_t sh;
        begin_named("start");
        read(st);
        begin_named("shape");
        read(sh);
        arr = T(index_space(st, sh), memory::host);
        begin_named("data");
        read_data(data(arr), size(arr));
        end_group();
        return true;
    }

    // --- Groups ---

    auto begin_group() -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        } else {
            skip_ws();
            if (peek() != '{') return false;
        }
        expect('{');
        group_stack_.push_back(is_.tellg());
        return true;
    }

    void end_group() {
        skip_to_group_end();
        expect('}');
        if (!group_stack_.empty()) {
            group_stack_.pop_back();
        }
    }

    auto begin_list() -> bool { return begin_group(); }
    void end_list() { end_group(); }

    // --- Query ---

    auto has_field(const char* name) -> bool {
        auto pos = is_.tellg();
        bool found = seek_field(name);
        is_.seekg(pos);
        return found;
    }

    auto count_items(const char* name) -> std::size_t {
        auto pos = is_.tellg();
        if (!seek_field(name)) {
            is_.seekg(pos);
            return 0;
        }
        expect('{');

        std::size_t count = 0;
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '{') {
                get();
                if (depth == 0) count++;
                depth++;
            } else if (c == '}') {
                if (depth == 0) break;
                get();
                depth--;
            } else if (c == std::char_traits<char>::eof()) {
                break;
            } else {
                get();
            }
        }
        is_.seekg(pos);
        return count;
    }

    auto count_strings(const char* name) -> std::size_t {
        auto pos = is_.tellg();
        if (!seek_field(name)) {
            is_.seekg(pos);
            return 0;
        }
        expect('{');

        std::size_t count = 0;
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '}') {
                if (depth == 0) break;
                get();
                depth--;
            } else if (c == '{') {
                get();
                depth++;
            } else if (c == '"' && depth == 0) {
                count++;
                read_quoted_string();
            } else if (c == std::char_traits<char>::eof()) {
                break;
            } else {
                get();
            }
        }
        is_.seekg(pos);
        return count;
    }

private:
    std::istream& is_;
    std::vector<std::streampos> group_stack_;
    const char* pending_name_ = nullptr;

    auto peek() -> char { return static_cast<char>(is_.peek()); }
    auto get() -> char { return static_cast<char>(is_.get()); }

    void skip_ws() {
        while (is_) {
            while (is_ && std::isspace(peek())) get();
            if (peek() == '#') {
                while (is_ && get() != '\n') {}
            } else {
                break;
            }
        }
    }

    void expect(char c) {
        skip_ws();
        if (get() != c) {
            throw std::runtime_error(std::string("expected '") + c + "'");
        }
    }

    auto read_identifier() -> std::string {
        std::string s;
        while (is_ && (std::isalnum(peek()) || peek() == '_')) {
            s += get();
        }
        return s;
    }

    template<typename T>
    auto read_number() -> T {
        skip_ws();
        std::string token;
        while (is_) {
            char c = peek();
            if (std::isdigit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
                token += get();
            } else {
                break;
            }
        }
        T value;
        std::istringstream iss(token);
        iss >> value;
        if (iss.fail()) {
            throw std::runtime_error("failed to parse number: " + token);
        }
        return value;
    }

    auto read_quoted_string() -> std::string {
        skip_ws();
        expect('"');
        std::string result;
        while (is_) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char next = get();
                switch (next) {
                    case '\\': result += '\\'; break;
                    case '"':  result += '"'; break;
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    case 'r':  result += '\r'; break;
                    default:   result += next; break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    // Seek to a field by name within the current group
    auto seek_field(const char* name) -> bool {
        auto start = group_stack_.empty() ? std::streampos(0) : group_stack_.back();
        is_.seekg(start);

        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();

            if (c == '}') {
                if (depth == 0) return false;  // End of current group
                get();
                depth--;
            } else if (c == '{') {
                get();
                depth++;
            } else if (c == std::char_traits<char>::eof()) {
                return false;
            } else if (depth == 0 && (std::isalpha(c) || c == '_')) {
                auto id = read_identifier();
                if (id == name) {
                    skip_ws();
                    return true;
                }
                // Skip this field's value
                skip_field_value();
            } else {
                get();
            }
        }
        return false;
    }

    void skip_field_value() {
        skip_ws();
        char c = peek();
        if (c == '=') {
            get();
            skip_ws();
            c = peek();
            if (c == '"') {
                read_quoted_string();
            } else if (c == '[') {
                skip_bracketed();
            } else {
                // Skip scalar value
                while (is_ && !std::isspace(peek()) && peek() != '}' && peek() != '{') {
                    get();
                }
            }
        } else if (c == '{') {
            skip_braced();
        }
    }

    void skip_bracketed() {
        expect('[');
        int depth = 1;
        while (is_ && depth > 0) {
            char c = get();
            if (c == '[') depth++;
            else if (c == ']') depth--;
        }
    }

    void skip_braced() {
        expect('{');
        int depth = 1;
        while (is_ && depth > 0) {
            char c = get();
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
    }

    void skip_to_group_end() {
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '{') {
                get();
                depth++;
            } else if (c == '}') {
                if (depth == 0) return;
                get();
                depth--;
            } else if (c == std::char_traits<char>::eof()) {
                return;
            } else {
                get();
            }
        }
    }
};

} // namespace mist

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace mist {

// =============================================================================
// ASCII Writer - writes key-based format
// =============================================================================

class ascii_writer {
public:
    explicit ascii_writer(std::ostream& os, int indent_size = 4)
        : os_(os), indent_size_(indent_size), indent_level_(0) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // --- Scalars ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const T& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << format_value(value) << "\n";
    }

    void write(const std::string& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "\"" << escape(value) << "\"\n";
    }

    void write(const char* value) {
        write(std::string(value));
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    void write(const vec_t<T, N>& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < N; ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const std::vector<T>& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_data(const T* ptr, std::size_t count) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < count; ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(ptr[i]);
        }
        os_ << "]\n";
    }

    // Overload for vec_t elements: flatten to scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    void write_data(const vec_t<T, N>* ptr, std::size_t count) {
        write_data(reinterpret_cast<const T*>(ptr), count * N);
    }

    // --- CachedNdArray ---

    template<CachedNdArray T>
    void write(const T& arr) {
        begin_group();
        begin_named("start");
        write(start(arr));
        begin_named("shape");
        write(shape(arr));
        begin_named("data");
        write_data(data(arr), size(arr));
        end_group();
    }

    // --- Groups ---

    void begin_group() {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " ";
            pending_name_ = nullptr;
        }
        os_ << "{\n";
        indent_level_++;
    }

    void end_group() {
        indent_level_--;
        write_indent();
        os_ << "}\n";
    }

    void begin_list() { begin_group(); }
    void end_list() { end_group(); }

private:
    std::ostream& os_;
    int indent_size_;
    int indent_level_;
    const char* pending_name_ = nullptr;

    void write_indent() {
        for (int i = 0; i < indent_level_ * indent_size_; ++i) {
            os_ << ' ';
        }
    }

    template<typename T>
    static auto format_value(const T& value) -> std::string {
        if constexpr (std::is_floating_point_v<T>) {
            std::ostringstream oss;
            oss << std::setprecision(15) << value;
            auto s = oss.str();
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
                s += ".0";
            }
            return s;
        } else {
            return std::to_string(value);
        }
    }

    static auto escape(const std::string& s) -> std::string {
        std::string result;
        result.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '\\': result += "\\\\"; break;
                case '"':  result += "\\\""; break;
                case '\n': result += "\\n"; break;
                case '\t': result += "\\t"; break;
                case '\r': result += "\\r"; break;
                default:   result += c; break;
            }
        }
        return result;
    }
};

} // namespace mist

#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

namespace mist {

// =============================================================================
// Binary Format Type Tags
// =============================================================================

namespace binary_format {
    // Magic header to identify mist binary archives
    constexpr uint32_t MAGIC = 0x4D495354;  // "MIST" in ASCII
    constexpr uint8_t VERSION = 1;

    // Type tags
    constexpr uint8_t TYPE_INT32   = 0x01;
    constexpr uint8_t TYPE_INT64   = 0x02;
    constexpr uint8_t TYPE_FLOAT64 = 0x03;
    constexpr uint8_t TYPE_STRING  = 0x04;
    constexpr uint8_t TYPE_ARRAY   = 0x05;
    constexpr uint8_t TYPE_GROUP   = 0x06;
    constexpr uint8_t TYPE_LIST    = 0x07;  // Anonymous groups (vector of compounds)

    // Element type tags for arrays
    constexpr uint8_t ELEM_INT32   = 0x01;
    constexpr uint8_t ELEM_INT64   = 0x02;
    constexpr uint8_t ELEM_FLOAT64 = 0x03;

    template<typename T>
    constexpr uint8_t scalar_type_tag() {
        if constexpr (std::is_same_v<T, int32_t> || (std::is_same_v<T, int> && sizeof(int) == 4)) {
            return TYPE_INT32;
        } else if constexpr (std::is_same_v<T, int64_t> || (std::is_same_v<T, long> && sizeof(long) == 8)) {
            return TYPE_INT64;
        } else if constexpr (std::is_floating_point_v<T>) {
            return TYPE_FLOAT64;
        } else {
            static_assert(std::is_arithmetic_v<T>, "Unsupported scalar type");
            // Default to int64 for other integer types
            return TYPE_INT64;
        }
    }

    template<typename T>
    constexpr uint8_t element_type_tag() {
        if constexpr (std::is_floating_point_v<T>) {
            return ELEM_FLOAT64;
        } else if constexpr (sizeof(T) <= 4) {
            return ELEM_INT32;
        } else {
            return ELEM_INT64;
        }
    }
}

// =============================================================================
// Binary Writer (Self-Describing Format)
// =============================================================================
//
// Binary format specification:
// - Header: uint32 magic ("MIST") + uint8 version
// - Field name: uint64 length prefix + UTF-8 bytes
// - Scalars: name + type tag (1 byte) + value (as double/int64)
// - Strings: name + type tag + uint64 length + UTF-8 bytes
// - Arrays: name + type tag + element type tag + uint64 count + elements
// - Groups: name + type tag + uint64 field count + fields
// - Lists: name + type tag + uint64 item count + items (each is anonymous group)
//
// =============================================================================

class binary_writer {
public:
    explicit binary_writer(std::ostream& os, bool skip_header = false)
        : os_(os), header_written_(skip_header) {}

    // =========================================================================
    // Name context
    // =========================================================================

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // =========================================================================
    // Scalar types
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const T& value) {
        ensure_header();
        write_pending_name();

        // Write type tag and value (promote to standard sizes)
        if constexpr (std::is_floating_point_v<T>) {
            write_type_tag(binary_format::TYPE_FLOAT64);
            double v = static_cast<double>(value);
            write_raw(v);
        } else if constexpr (sizeof(T) <= 4) {
            write_type_tag(binary_format::TYPE_INT32);
            int32_t v = static_cast<int32_t>(value);
            write_raw(v);
        } else {
            write_type_tag(binary_format::TYPE_INT64);
            int64_t v = static_cast<int64_t>(value);
            write_raw(v);
        }
    }

    // =========================================================================
    // String type
    // =========================================================================

    void write(const std::string& value) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_STRING);

        uint64_t length = value.size();
        write_raw(length);
        if (length > 0) {
            os_.write(value.data(), static_cast<std::streamsize>(length));
        }
    }

    void write(const char* value) {
        write(std::string(value));
    }

    // =========================================================================
    // Arrays (fixed-size vec_t)
    // =========================================================================

    template<typename T, std::size_t N>
    void write(const vec_t<T, N>& value) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());

        uint64_t count = N;
        write_raw(count);

        // Write elements in standardized format
        for (std::size_t i = 0; i < N; ++i) {
            write_element(value[i]);
        }
    }

    // =========================================================================
    // Arrays (dynamic std::vector)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const std::vector<T>& value) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());

        uint64_t count = value.size();
        write_raw(count);

        for (const auto& elem : value) {
            write_element(elem);
        }
    }

    // =========================================================================
    // Bulk data (for ndarray)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_data(const T* ptr, std::size_t count) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());

        uint64_t n = count;
        write_raw(n);

        // Write raw bytes directly (no per-element conversion)
        os_.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(count * sizeof(T)));
    }

    // Overload for vec_t elements: flatten to scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    void write_data(const vec_t<T, N>* ptr, std::size_t count) {
        write_data(reinterpret_cast<const T*>(ptr), count * N);
    }

    // =========================================================================
    // CachedNdArray
    // =========================================================================

    template<CachedNdArray T>
    void write(const T& arr) {
        begin_group();
        begin_named("start");
        write(start(arr));
        begin_named("shape");
        write(shape(arr));
        begin_named("data");
        write_data(data(arr), size(arr));
        end_group();
    }

    // =========================================================================
    // Groups (named and anonymous)
    // =========================================================================

    void begin_group() {
        ensure_header();

        if (pending_name_) {
            write_name(pending_name_);
            pending_name_ = nullptr;
            write_type_tag(binary_format::TYPE_GROUP);
            // Increment parent's field count for named groups
            if (!field_counts_.empty()) {
                field_counts_.back()++;
            }
        } else {
            // Anonymous group within a list - increment parent's count
            if (!field_counts_.empty()) {
                field_counts_.back()++;
            }
            // Write GROUP type tag for anonymous groups
            write_raw(binary_format::TYPE_GROUP);
        }

        // Save position for field count backfill
        group_positions_.push_back(os_.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts_.push_back(0);
    }

    void begin_list() {
        ensure_header();

        if (pending_name_) {
            write_name(pending_name_);
            pending_name_ = nullptr;
            write_type_tag(binary_format::TYPE_LIST);
            // Increment parent's field count for named lists
            if (!field_counts_.empty()) {
                field_counts_.back()++;
            }
        } else {
            // Anonymous list - just write type tag
            if (!field_counts_.empty()) {
                field_counts_.back()++;
            }
            write_raw(binary_format::TYPE_LIST);
        }

        // Save position for item count backfill
        group_positions_.push_back(os_.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts_.push_back(0);
    }

    void end_list() {
        end_group();
    }

    void end_group() {
        if (group_positions_.empty()) {
            return;
        }

        std::streampos pos = group_positions_.back();
        group_positions_.pop_back();

        uint64_t count = field_counts_.back();
        field_counts_.pop_back();

        // Backfill the count
        std::streampos current_pos = os_.tellp();
        os_.seekp(pos);
        write_raw(count);
        os_.seekp(current_pos);
    }

private:
    std::ostream& os_;
    bool header_written_;
    std::vector<std::streampos> group_positions_;
    std::vector<uint64_t> field_counts_;
    const char* pending_name_ = nullptr;

    void ensure_header() {
        if (!header_written_) {
            write_raw(binary_format::MAGIC);
            write_raw(binary_format::VERSION);
            header_written_ = true;
        }
    }

    void write_pending_name() {
        if (pending_name_) {
            write_name(pending_name_);
            pending_name_ = nullptr;
        }
        // Increment field count for parent group
        if (!field_counts_.empty()) {
            field_counts_.back()++;
        }
    }

    void write_name(const char* name) {
        uint64_t length = std::strlen(name);
        write_raw(length);
        if (length > 0) {
            os_.write(name, static_cast<std::streamsize>(length));
        }
    }

    void write_type_tag(uint8_t tag) {
        write_raw(tag);
    }

    template<typename T>
    void write_raw(const T& value) {
        os_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    void write_element(const T& value) {
        if constexpr (std::is_floating_point_v<T>) {
            double v = static_cast<double>(value);
            write_raw(v);
        } else if constexpr (sizeof(T) <= 4) {
            int32_t v = static_cast<int32_t>(value);
            write_raw(v);
        } else {
            int64_t v = static_cast<int64_t>(value);
            write_raw(v);
        }
    }
};

} // namespace mist

namespace mist {

// =============================================================================
// Binary Reader - key-based lookup, missing fields return false
// =============================================================================
//
// Binary format specification:
// - Header: uint32 magic ("MIST") + uint8 version
// - Field name: uint64 length prefix + UTF-8 bytes
// - Scalars: name + type tag (1 byte) + value (as double/int64)
// - Strings: name + type tag + uint64 length + UTF-8 bytes
// - Arrays: name + type tag + element type tag + uint64 count + elements
// - Groups: name + type tag + uint64 field count + fields
// - Lists: name + type tag + uint64 item count + items
//
// =============================================================================

class binary_reader {
public:
    explicit binary_reader(std::istream& is, bool skip_header = false)
        : is_(is), header_read_(skip_header), base_position_(is.tellg()) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // --- Scalars ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(T& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        } else if (!check_list_item()) {
            return false;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag == binary_format::TYPE_FLOAT64) {
            double v;
            read_raw(v);
            value = static_cast<T>(v);
        } else if (type_tag == binary_format::TYPE_INT32) {
            int32_t v;
            read_raw(v);
            value = static_cast<T>(v);
        } else if (type_tag == binary_format::TYPE_INT64) {
            int64_t v;
            read_raw(v);
            value = static_cast<T>(v);
        } else {
            throw std::runtime_error("Expected scalar type");
        }
        return true;
    }

    auto read(std::string& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        } else if (!check_list_item()) {
            return false;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_STRING) {
            throw std::runtime_error("Expected string type");
        }
        value = read_string_data();
        return true;
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    auto read(vec_t<T, N>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type");
        }
        uint8_t elem_tag = read_type_tag();
        uint64_t count;
        read_raw(count);
        if (count != N) {
            throw std::runtime_error(
                "Array size mismatch: expected " + std::to_string(N) +
                ", got " + std::to_string(count));
        }
        for (std::size_t i = 0; i < N; ++i) {
            value[i] = read_element<T>(elem_tag);
        }
        return true;
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(std::vector<T>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type");
        }
        uint8_t elem_tag = read_type_tag();
        uint64_t count;
        read_raw(count);
        value.resize(count);
        for (auto& elem : value) {
            elem = read_element<T>(elem_tag);
        }
        return true;
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_data(T* ptr, std::size_t count) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type");
        }
        [[maybe_unused]] uint8_t elem_tag = read_type_tag();
        uint64_t n;
        read_raw(n);
        if (n != count) {
            throw std::runtime_error(
                "Data size mismatch: expected " + std::to_string(count) +
                ", got " + std::to_string(n));
        }
        is_.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(count * sizeof(T)));
        if (!is_) {
            throw std::runtime_error("Failed to read data");
        }
        return true;
    }

    // Overload for vec_t elements: read as flattened scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    auto read_data(vec_t<T, N>* ptr, std::size_t count) -> bool {
        return read_data(reinterpret_cast<T*>(ptr), count * N);
    }

    // --- CachedNdArray ---

    template<CachedNdArray T>
    auto read(T& arr) -> bool {
        if (!begin_group()) return false;
        using start_t = std::decay_t<decltype(start(arr))>;
        using shape_t = std::decay_t<decltype(shape(arr))>;
        start_t st;
        shape_t sh;
        begin_named("start");
        read(st);
        begin_named("shape");
        read(sh);
        arr = T(index_space(st, sh), memory::host);
        begin_named("data");
        read_data(data(arr), size(arr));
        end_group();
        return true;
    }

    // --- Groups ---

    auto begin_group() -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        } else if (!group_stack_.empty() && group_stack_.back().is_list) {
            // Inside a list - check if we've read all items
            if (group_stack_.back().remaining == 0) {
                return false;
            }
            group_stack_.back().remaining--;
        } else if (group_stack_.empty()) {
            // Root anonymous group - need to read header first
            ensure_header();
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_GROUP) {
            return false;
        }
        uint64_t field_count;
        read_raw(field_count);
        group_stack_.push_back({is_.tellg(), field_count, false});
        return true;
    }

    void end_group() {
        if (!group_stack_.empty()) {
            group_stack_.pop_back();
        }
    }

    auto begin_list() -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST) {
            return false;
        }
        uint64_t item_count;
        read_raw(item_count);
        group_stack_.push_back({is_.tellg(), item_count, true});
        return true;
    }

    void end_list() { end_group(); }

    // --- Query ---

    auto has_field(const char* name) -> bool {
        auto pos = is_.tellg();
        bool found = seek_field(name);
        is_.seekg(pos);
        return found;
    }

    auto count_items(const char* name) -> std::size_t {
        auto pos = is_.tellg();
        if (!seek_field(name)) {
            is_.seekg(pos);
            return 0;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST && type_tag != binary_format::TYPE_GROUP) {
            is_.seekg(pos);
            return 0;
        }
        uint64_t count;
        read_raw(count);
        is_.seekg(pos);
        return static_cast<std::size_t>(count);
    }

    auto count_strings(const char* name) -> std::size_t { return count_items(name); }

private:
    std::istream& is_;
    const char* pending_name_ = nullptr;

    struct group_info_t {
        std::streampos start;
        uint64_t remaining;
        bool is_list;
    };
    std::vector<group_info_t> group_stack_;
    bool header_read_ = false;
    std::streampos base_position_ = 0;

    // Check if we can read the next item in a list, decrementing remaining count
    auto check_list_item() -> bool {
        if (!group_stack_.empty() && group_stack_.back().is_list) {
            if (group_stack_.back().remaining == 0) {
                return false;
            }
            group_stack_.back().remaining--;
        }
        return true;
    }

    void ensure_header() {
        if (header_read_) return;
        is_.seekg(base_position_);
        uint32_t magic;
        read_raw(magic);
        if (magic != binary_format::MAGIC) {
            throw std::runtime_error("Invalid binary archive: bad magic number");
        }
        uint8_t version;
        read_raw(version);
        if (version != binary_format::VERSION) {
            throw std::runtime_error("Unsupported binary archive version: " + std::to_string(version));
        }
        header_read_ = true;
    }

    auto read_type_tag() -> uint8_t {
        uint8_t tag;
        read_raw(tag);
        return tag;
    }

    auto read_name() -> std::string {
        uint64_t length;
        read_raw(length);
        std::string name(length, '\0');
        if (length > 0) {
            is_.read(name.data(), static_cast<std::streamsize>(length));
            if (!is_) {
                throw std::runtime_error("Failed to read field name");
            }
        }
        return name;
    }

    auto read_string_data() -> std::string {
        uint64_t length;
        read_raw(length);
        std::string value(length, '\0');
        if (length > 0) {
            is_.read(value.data(), static_cast<std::streamsize>(length));
            if (!is_) {
                throw std::runtime_error("Failed to read string data");
            }
        }
        return value;
    }

    template<typename T>
    void read_raw(T& value) {
        is_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!is_) {
            throw std::runtime_error("Failed to read data from binary archive");
        }
    }

    template<typename T>
    auto read_element(uint8_t elem_tag) -> T {
        if (elem_tag == binary_format::ELEM_FLOAT64) {
            double v;
            read_raw(v);
            return static_cast<T>(v);
        } else if (elem_tag == binary_format::ELEM_INT32) {
            int32_t v;
            read_raw(v);
            return static_cast<T>(v);
        } else if (elem_tag == binary_format::ELEM_INT64) {
            int64_t v;
            read_raw(v);
            return static_cast<T>(v);
        }
        throw std::runtime_error("Unknown element type tag");
    }

    // Skip a field value (for seeking)
    void skip_field_value(uint8_t type_tag) {
        if (type_tag == binary_format::TYPE_FLOAT64) {
            double v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_INT32) {
            int32_t v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_INT64) {
            int64_t v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_STRING) {
            uint64_t len;
            read_raw(len);
            is_.seekg(len, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_ARRAY) {
            uint8_t elem_tag;
            read_raw(elem_tag);
            uint64_t count;
            read_raw(count);
            std::size_t elem_size = (elem_tag == binary_format::ELEM_INT32) ? 4 : 8;
            is_.seekg(count * elem_size, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_GROUP) {
            uint64_t field_count;
            read_raw(field_count);
            for (uint64_t i = 0; i < field_count; ++i) {
                read_name();  // skip field name
                uint8_t nested_tag = read_type_tag();
                skip_field_value(nested_tag);
            }
        } else if (type_tag == binary_format::TYPE_LIST) {
            uint64_t item_count;
            read_raw(item_count);
            for (uint64_t i = 0; i < item_count; ++i) {
                uint8_t item_tag = read_type_tag();
                skip_field_value(item_tag);
            }
        }
    }

    // Seek to a field by name within the current group
    auto seek_field(const char* name) -> bool {
        ensure_header();

        // Start position: base + 5 bytes (header) for root level, or group's start position
        auto start = group_stack_.empty() ? base_position_ + std::streamoff(5) : group_stack_.back().start;
        is_.seekg(start);

        // If in a list, we can't seek by name - just return false
        if (!group_stack_.empty() && group_stack_.back().is_list) {
            return false;
        }

        // Scan through fields in current group
        uint64_t fields_to_scan = group_stack_.empty() ? UINT64_MAX : group_stack_.back().remaining;

        for (uint64_t i = 0; i < fields_to_scan && is_; ++i) {
            auto field_name = read_name();
            if (field_name.empty() && !is_) {
                return false;  // EOF
            }
            if (field_name == name) {
                return true;
            }
            // Skip this field's value
            uint8_t type_tag = read_type_tag();
            skip_field_value(type_tag);
        }
        return false;
    }
};

} // namespace mist

// Parallelism

// Convenience header that includes all parallel execution components

#include <algorithm>
#include <atomic>
#include <concepts>
#include <functional>
#include <tuple>
#include <vector>

#include <chrono>
#include <concepts>
#include <map>
#include <string>

namespace mist {
namespace perf {

// =============================================================================
// Profiler concept
// =============================================================================

struct profile_entry_t {
    std::size_t count = 0;
    double time = 0.0;

    auto fields() const {
        return std::make_tuple(
            field("count", count),
            field("time", time)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("count", count),
            field("time", time)
        );
    }
};

template<typename P>
concept Profiler = requires(P& p, const P& cp, const std::string& name) {
    p.start();
    p.record(name);
    p.clear();
    { cp.data() } -> std::same_as<std::map<std::string, profile_entry_t>>;
};

// =============================================================================
// Null profiler: no-op implementation
// =============================================================================

struct null_profiler_t {
    void start() {}
    void record(const std::string&) {}
    void clear() {}
    auto data() const -> std::map<std::string, profile_entry_t> { return {}; }
};

// =============================================================================
// High-precision profiler using steady_clock
// =============================================================================

struct profiler_t {
    using clock_t = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    void start() {
        start_time = clock_t::now();
    }

    void record(const std::string& name) {
        auto now = clock_t::now();
        auto elapsed = std::chrono::duration<double>(now - start_time).count();
        auto& entry = records[name];
        entry.count += 1;
        entry.time += elapsed;
        start_time = now;
    }

    void clear() {
        records.clear();
    }

    auto data() const -> std::map<std::string, profile_entry_t> {
        return records;
    }

private:
    time_point_t start_time;
    std::map<std::string, profile_entry_t> records;
};

} // namespace perf
} // namespace mist

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace mist {
namespace parallel {

// =============================================================================
// Blocking queue (analogous to Rust channels)
// =============================================================================

template<typename T>
class blocking_queue {
public:
    void send(T item) {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.push(std::move(item));
        _cv.notify_one();
    }

    T recv() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return !_queue.empty(); });
        auto item = std::move(_queue.front());
        _queue.pop();
        return item;
    }

    std::optional<T> try_recv() {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_queue.empty()) {
            return std::nullopt;
        }
        auto item = std::move(_queue.front());
        _queue.pop();
        return item;
    }

private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cv;
};

} // namespace parallel
} // namespace mist

#include <concepts>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mist {
namespace parallel {

// =============================================================================
// Scheduler concept
// =============================================================================

template<typename S>
concept Scheduler = requires(S& scheduler) {
    { scheduler.spawn([](){}) } -> std::same_as<void>;
};

// =============================================================================
// Sequential scheduler (executes tasks immediately)
// =============================================================================

class sequential_scheduler_t {
public:
    template<typename F>
    void spawn(F&& task) {
        task();
    }
};

// =============================================================================
// Thread pool scheduler
// =============================================================================

class thread_pool_t {
public:
    explicit thread_pool_t(std::size_t num_threads) {
        _workers.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            _workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~thread_pool_t() {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F>
    void spawn(F&& task) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            auto shared_task = std::make_shared<std::decay_t<F>>(std::forward<F>(task));
            _tasks.push([shared_task]() mutable { (*shared_task)(); });
        }
        _cv.notify_one();
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this] {
                    return _stop || !_tasks.empty();
                });
                if (_stop && _tasks.empty()) {
                    return;
                }
                if (!_tasks.empty()) {
                    task = std::move(_tasks.front());
                    _tasks.pop();
                }
            }
            if (task) {
                task();
            }
        }
    }

    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _stop = false;
};

// Backward compatibility alias
using threadpool = thread_pool_t;

// =============================================================================
// Configurable scheduler (switches between sequential and thread pool)
// =============================================================================

class scheduler_t {
public:
    scheduler_t() = default;

    void set_num_threads(std::size_t n) {
        _pool.reset();
        _num_threads = n;
        if (n > 0) {
            _pool = std::make_unique<thread_pool_t>(n);
        }
    }

    std::size_t num_threads() const {
        return _num_threads;
    }

    template<typename F>
    void spawn(F&& task) {
        if (_pool) {
            _pool->spawn(std::forward<F>(task));
        } else {
            task();
        }
    }

private:
    std::size_t _num_threads = 0;
    std::unique_ptr<thread_pool_t> _pool;
};

} // namespace parallel
} // namespace mist

namespace mist {
namespace parallel {

// =============================================================================
// Stage concepts
// =============================================================================

// Check if a stage has a static name member
template<typename S>
concept HasName = requires { { S::name } -> std::convertible_to<const char*>; };

template<typename S>
auto stage_name(std::size_t index) -> std::string {
    if constexpr (HasName<S>) {
        return S::name;
    } else {
        return "stage_" + std::to_string(index);
    }
}

// Compute stage: per-peer transform (fully parallel)
template<typename S>
concept ComputeStage = requires { &S::value; };

// Exchange stage: peers request/provide data (barrier required)
template<typename S>
concept ExchangeStage = requires {
    typename S::space_t;
    typename S::buffer_t;
    &S::provides;
    &S::data;
};

// Reduce stage: fold across peers then broadcast result (barrier required)
template<typename S>
concept ReduceStage = requires { typename S::value_type; &S::init; &S::reduce; &S::finalize; };

// Type trait to extract Context type from a stage
namespace detail {
    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(const C&) const);

    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(C) const);

    template<typename C, typename S>
    C extract_context_from_value(C(S::*)(C) const);

    template<typename V, typename S, typename C>
    C extract_context_from_reduce(V(S::*)(V, const C&) const);
}

template<typename S>
struct stage_context;

template<ExchangeStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_provides(&S::provides));
};

template<ReduceStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_reduce(&S::reduce));
};

template<ComputeStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_value(&S::value));
};

template<typename S>
using stage_context_t = typename stage_context<S>::type;

// =============================================================================
// Pipeline: stores stage instances
// =============================================================================

template<typename... Stages>
struct pipeline_t {
    using context_t = stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>;
    std::tuple<Stages...> stages;

    template<std::size_t I>
    const auto& get() const { return std::get<I>(stages); }
};

namespace detail {
    template<typename T>
    struct is_pipeline_impl : std::false_type {};

    template<typename... Stages>
    struct is_pipeline_impl<pipeline_t<Stages...>> : std::true_type {};
}

template<typename T>
concept Pipeline = detail::is_pipeline_impl<T>::value;

template<typename... Stages>
auto pipeline(Stages... stages) -> pipeline_t<Stages...> {
    return {std::make_tuple(std::move(stages)...)};
}

// =============================================================================
// compose: fuse multiple compute stages into one
// =============================================================================

template<typename... Stages>
struct composed_t {
    std::tuple<Stages...> stages;

    template<typename Context>
    auto value(Context ctx) const -> Context {
        return apply_stages(std::move(ctx), std::index_sequence_for<Stages...>{});
    }

private:
    template<typename Context, std::size_t... Is>
    auto apply_stages(Context ctx, std::index_sequence<Is...>) const -> Context {
        ((ctx = std::get<Is>(stages).value(std::move(ctx))), ...);
        return ctx;
    }
};

template<typename... Stages>
auto compose(Stages... stages) -> composed_t<Stages...> {
    return {std::make_tuple(std::move(stages)...)};
}

// =============================================================================
// Execute: barrier-based pipeline execution
// =============================================================================

namespace detail {

// Execute a single Exchange stage across all contexts
template<ExchangeStage Stage, typename Context>
void execute_exchange(
    const Stage& stage,
    std::vector<Context>& contexts
) {
    struct request_t {
        std::size_t requester;
        typename Stage::buffer_t buffer;
        typename Stage::space_t requested_space;
    };

    // Collect all requests
    auto requests = std::vector<request_t>{};
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        stage.need(contexts[i], [&](auto buf) {
            requests.push_back({i, buf, space(buf)});
        });
    }

    // Route requests to providers
    for (auto& req : requests) {
        for (std::size_t j = 0; j < contexts.size(); ++j) {
            if (overlaps(req.requested_space, stage.provides(contexts[j]))) {
                copy_overlapping(req.buffer, stage.data(contexts[j]));
            }
        }
    }
}

// Execute a single Compute stage across all contexts (parallel)
template<ComputeStage Stage, typename Context, Scheduler Sched>
void execute_compute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched
) {
    auto queue = blocking_queue<std::pair<std::size_t, Context>>{};
    auto n = contexts.size();

    for (std::size_t i = 0; i < n; ++i) {
        sched.spawn([&queue, &stage, i, c = std::move(contexts[i])]() mutable {
            queue.send({i, stage.value(std::move(c))});
        });
    }

    for (std::size_t i = 0; i < n; ++i) {
        auto [idx, ctx] = queue.recv();
        contexts[idx] = std::move(ctx);
    }
}

// Execute a single Reduce stage: fold then broadcast
template<ReduceStage Stage, typename Context>
void execute_reduce(
    const Stage& stage,
    std::vector<Context>& contexts
) {
    auto acc = Stage::init();
    for (const auto& ctx : contexts) {
        acc = stage.reduce(acc, ctx);
    }
    for (auto& ctx : contexts) {
        stage.finalize(acc, ctx);
    }
}

} // namespace detail

// =============================================================================
// Barrier-based pipeline execution
// =============================================================================

namespace detail {

// Dispatch a stage by type
template<typename Stage, typename Context, Scheduler Sched>
void execute_stage(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched
) {
    if constexpr (ExchangeStage<Stage>) {
        execute_exchange(stage, contexts);
    } else if constexpr (ReduceStage<Stage>) {
        execute_reduce(stage, contexts);
    } else {
        execute_compute(stage, contexts, sched);
    }
}

} // namespace detail

// Execute pipeline with profiler
template<typename... Stages, Scheduler Sched, perf::Profiler Prof>
void execute(
    const pipeline_t<Stages...>& pipe,
    std::vector<stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>>& contexts,
    Sched& sched,
    Prof& profiler
) {
    std::size_t stage_index = 0;
    std::apply([&](const Stages&... stages) {
        ((profiler.start(),
          detail::execute_stage(stages, contexts, sched),
          profiler.record(stage_name<Stages>(stage_index++))), ...);
    }, pipe.stages);
}

// Convenience overload: execute a single stage without wrapping in pipeline
template<typename Stage, typename Context, Scheduler Sched, perf::Profiler Prof>
    requires (!Pipeline<Stage>)
void execute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched,
    Prof& profiler
) {
    profiler.start();
    detail::execute_stage(stage, contexts, sched);
    profiler.record(stage_name<Stage>(0));
}

} // namespace parallel
} // namespace mist

#include <concepts>
#include <stdexcept>

namespace mist {

// =============================================================================
// Runge-Kutta Concept
// =============================================================================

template<typename P>
concept RungeKutta = requires(
    typename P::state_t s,
    const typename P::exec_context_t& ctx,
    double dt,
    double alpha) {
    { copy(s, s) } -> std::same_as<void>;
    { rk_step(s, s, dt, alpha, ctx) } -> std::same_as<void>;
};

// =============================================================================
// Runge-Kutta Time Integration
// =============================================================================

template<RungeKutta P>
void rk1_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
}

template<RungeKutta P>
void rk2_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
    rk_step(state, temp, dt, 0.5, ctx);
}

template<RungeKutta P>
void rk3_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
    rk_step(state, temp, dt, 0.25, ctx);
    rk_step(state, temp, dt, 2.0 / 3.0, ctx);
}

template<RungeKutta P>
void rk_advance(
    int order,
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    switch (order) {
        case 1: rk1_step<P>(state, temp, dt, ctx); break;
        case 2: rk2_step<P>(state, temp, dt, ctx); break;
        case 3: rk3_step<P>(state, temp, dt, ctx); break;
        default:
            throw std::runtime_error("rk_order must be 1, 2, or 3");
    }
}

} // namespace mist

