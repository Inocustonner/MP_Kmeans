#pragma once
#include <array>
#include <random>
#include <string_view>
#include <tuple>
#include <vector>

#include "csv.hpp"
#include "enumerate.hpp"
#include "helper.hpp"

constexpr size_t POINTS_DEFAULT_RESERVE = 100;

template <typename UnitT, size_t Dim = 2>
class KMeansBase {
  protected:
    using Point = std::array<UnitT, Dim>;

    struct PointInfo {
        size_t centroid = (size_t)(-1);
        Point point;
        UnitT sqr_dist = (UnitT)-1;
    };

    struct CentroidInfo {
        size_t points = 0;
        Point sum_coords;
    };

    std::vector<PointInfo> points;
    std::vector<Point> centroids;
    std::vector<CentroidInfo> c_infos; // Linked to centroids, but changes every time.

  public:
    KMeansBase(std::string_view points_filepath, size_t centroids_amount) {
        // read the file in
        csv::CSVReader reader(points_filepath);
        points.reserve(POINTS_DEFAULT_RESERVE);

        centroids.reserve(centroids_amount);
        // for (size_t i = 0; i < cluster_amount; i++)
        //     centroids[i] = Point{};

        for (auto &row : reader) {
            PointInfo pi;
            for (auto [i, cell] : enumerate(row))
                pi.point[i] = cell.get<UnitT>();
            points.push_back(std::move(pi));
        }

        init_centroids(centroids_amount);
    }

    UnitT calculate(int generations) {
        UnitT sum = 0;
        for (int gen = 0; gen < generations; ++gen) {
            sum = update_points_and_sum();
            for (size_t i = 0; i < std::size(centroids); ++i) {
                scalar_div_eq(centroids[i], c_infos[i].sum_coords, (UnitT)c_infos[i].points);
            }
#if _DEBUG
            printf("result = %f\n", sum);
#endif
        }
        return sum;
    }

    void export_clusters(std::string_view filepath) {
        std::ofstream out(filepath.data());
        // NOTE: LETS ASSERT THAT out always opens
        csv::CSVWriter<std::ofstream> writer(out);

        for (const auto &point_info : points) {
            std::array<UnitT, Dim + 1> row;
            std::copy(std::cbegin(point_info.point), std::cend(point_info.point), std::begin(row));
            row[Dim] = (UnitT)point_info.centroid;
            writer << row;
        }
    }

  protected:
    void init_centroids(size_t centroids_amount) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<UnitT> real_distrib(0., 1.);
        std::uniform_int_distribution<size_t> int_distrib(0, std::size(points));

        // initialize entire c_infos
        c_infos = std::vector<CentroidInfo>(centroids_amount, CentroidInfo());

        // set first centroid to random point
        centroids.push_back(points[int_distrib(gen)].point);

#if _DEBUG
        print_point(centroids[0]);
#endif
        for (size_t centr_id = 1; centr_id < centroids_amount; ++centr_id) {
            auto sum = update_points_and_sum();
            sum *= real_distrib(gen);
            for (const auto &point_info : points) {
                sum -= point_info.sqr_dist;
                if (sum <= 0) {
#if _DEBUG
                    print_point(point_info.point);
#endif
                    centroids.push_back(point_info.point);
                    break;
                }
            }
        }
    }

    std::tuple<size_t, UnitT> get_nearest_centroid(const Point &p) const noexcept {
        UnitT match_sqr_dist = std::numeric_limits<UnitT>::max();
        size_t centr_id = (size_t)-1;
        for (size_t i = 0; i < std::size(centroids); i++) {
            UnitT sqr_dist = 0;
            for (size_t d = 0; d < Dim; d++)
                sqr_dist += (p[d] - centroids[i][d]) * (p[d] - centroids[i][d]);

            if (sqr_dist < match_sqr_dist) {
                centr_id = i;
                match_sqr_dist = sqr_dist;
            }
        }
        return {centr_id, match_sqr_dist};
    }

    virtual UnitT update_points_and_sum() {
        UnitT sum = 0;

        // zero c_infos
        for (auto &c_info : c_infos) {
            c_info.sum_coords.fill(0);
            c_info.points = 0;
        }

        for (size_t point_i = 0; point_i < std::size(points); ++point_i) {
            auto &pi = points[point_i];
            std::tie(pi.centroid, pi.sqr_dist) = get_nearest_centroid(pi.point);

            auto &c_info = c_infos[pi.centroid];
            scalar_add_eq(c_info.sum_coords, pi.point);
            c_info.points++;

            sum += pi.sqr_dist;
        }
        return sum;
    }
};