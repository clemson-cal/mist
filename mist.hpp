#pragma once

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
