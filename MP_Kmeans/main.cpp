#include <cstdio>
#include <string_view>
#include <omp.h>
#include "KMeansBase.hpp"
#include "KMeansOMP.hpp"
#include "KMeansTP.hpp"

#pragma comment(lib, "libomp.lib")

// void test_vec_scalar_reduction_omp(int threads) {
//     using ArrT = std::array<float, 3>;
//     ArrT v;
//     v.fill(0);
//     #pragma omp declare reduction \
//         (scalar_add:ArrT:scalar_add_eq(omp_out, omp_in)) initializer(omp_priv=omp_orig)

//     #pragma omp parallel for num_threads(threads) reduction(scalar_add: v)
//     for (int i = 0; i < std::size(v); ++i) {
//         v[i] = 1;
//     }

//     print_point(v);
// }

int main(int argc, const char* argv[]) {
    // for (int i = 0; i < argc; ++i) {
    //     printf("%d %s\n", i, argv[i]);
    // }

    // i=1 threads count
    // i=2 centroids
    // i=3 generations
    // i=4 input file
    // i=5 output file
    
    // test_vec_scalar_reduction_omp(2);
    // return 0;
    if (argc != 6) {
        printf("Invalid number of arguments\n", argc);
    }

    int num_threads = atoi(argv[1]);
    int centroids = atoi(argv[2]);
    int generations = atoi(argv[3]);
    auto filepath = std::string_view(argv[4]);
    auto outpath = std::string_view(argv[5]);

    constexpr size_t dim = 2;

    constexpr int iterations = 10;

    auto print_bench_results = [iterations](benchmark_info_t& info) {
        printf("avg = %f out of %d iterations\n", info.average, iterations);
    };

    auto bench_single_kmeans = [=]() {
        printf("Single thread\n");

        auto calc_kmeans = [generations, &filepath, centroids]() {
            KMeansBase<float, dim> kmeans(filepath, centroids);
            kmeans.calculate(generations);
        };
        
        auto binfo = benchmark(calc_kmeans, iterations);
        print_bench_results(binfo);
        putchar('\n');
    };

    auto bench_omp_kmeans = [=]() {
        printf("OMP\n");

        auto calc_kmeans = [generations, &filepath, centroids, num_threads]() {
            KMeansOMP<float, dim> kmeans(filepath, centroids, num_threads);
            kmeans.calculate(generations);
        };
        
        auto binfo = benchmark(calc_kmeans, iterations);
        print_bench_results(binfo);
    };

    auto bench_tp_kmeans = [=]() {
        printf("TP\n");

        auto calc_kmeans = [generations, &filepath, centroids, num_threads]() {
            KMeansTP<float, dim> kmeans(filepath, centroids, num_threads);
            kmeans.calculate(generations);
        };
        
        auto binfo = benchmark(calc_kmeans, iterations);
        print_bench_results(binfo);
    };

    bench_single_kmeans();
    bench_omp_kmeans();
    bench_tp_kmeans();
    // KMeansOMP<float, dim> kmeans(filepath, clusters, num_threads);
    // KMeansTP<float, dim> kmeans(filepath, centroids, num_threads);
    // auto result = kmeans.calculate(generations);
    // printf("result = %f\n", result);
    // kmeans.export_clusters(outpath);
}