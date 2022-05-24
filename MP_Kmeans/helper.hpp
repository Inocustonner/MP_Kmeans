#pragma once
#include <array>
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#include <chrono>
#include <algorithm>
#include <type_traits>


template<class ArrayT, size_t ...I>
void scalar_add_eq(ArrayT& a1, const ArrayT& a2, std::index_sequence<I...>) {
    ((a1[I] += a2[I]), ...);
}

template<typename T, size_t N>
void scalar_add_eq(std::array<T, N>& a1, const std::array<T, N>& a2) {
    scalar_add_eq(a1, a2, std::make_index_sequence<N>{});
}

template<typename T, size_t N>
void scalar_add_eq(std::array<T, N>* a1, std::array<T, N>* a2) {
    scalar_add_eq(*a1, *a2);
}

template<class ArrayT, typename T, size_t ...I>
void scalar_div_eq(ArrayT& a1, const ArrayT& a2, T div_by, std::index_sequence<I...>) {
    ((a1[I] = a2[I] / div_by), ...);
}

template<typename T, size_t N>
void scalar_div_eq(std::array<T, N>& dest_a, const std::array<T, N>& src_a, T div_by) {
    scalar_div_eq(dest_a, src_a, div_by, std::make_index_sequence<N>{});
}


#if _DEBUG
template<size_t N>
void print_point(const std::array<float, N>& p) {
    printf("%f", p[0]);
    for (size_t i = 1; i < N; ++i) {
        printf(" %f", p[i]);
    }
    putchar('\n');
}
#endif


struct benchmark_info_t {
    double average;
};

template <typename F, typename FBefore>
[[nodiscard]] auto benchmark(F f, FBefore fb, int iterations) {
    using namespace std::chrono;
    double aggregate = 0;
    for (int i = 0; i < iterations; ++i) {
        if constexpr (std::is_invocable_v<FBefore>) {
            fb();
        }
        auto start = high_resolution_clock::now();
        f();
        auto end = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(end - start).count() / 1000.;
        aggregate += ms;
    }
    benchmark_info_t info;
    info.average = aggregate / iterations;
    return info;
}

template <typename F>
[[nodiscard]] auto benchmark(F f, int iterations) {
    return benchmark(std::forward<F>(f), nullptr, iterations);
}