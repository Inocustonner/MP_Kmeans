#pragma once

#pragma once
#include "KMeansBase.hpp"
#include "thread_pool.hpp"

template <typename UnitT, size_t Dim = 2>
class KMeansTP : public KMeansBase<UnitT, Dim> {
  private:
    thread_pool pool;
    std::mutex upd_c_infos;

  public:
    KMeansTP(std::string_view points_filepath, size_t centroids_amount, int num_threads)
        : KMeansBase<UnitT, Dim>(points_filepath, centroids_amount), pool(num_threads) {}

    UnitT update_points_and_sum() override {
        UnitT sum = 0;

        // zero c_infos
        for (auto &c_info : this->c_infos) {
            c_info.sum_coords.fill(0);
            c_info.points = 0;
        }

        auto loop = [this, &sum](const size_t& point_i, const size_t& last_point_i) {
            auto local_c_infos = this->c_infos;
            decltype(sum) local_sum = 0;

            for (size_t point_i = 0; point_i < std::size(this->points); ++point_i) {
                auto &pi = this->points[point_i];
                std::tie(pi.centroid, pi.sqr_dist) = this->get_nearest_centroid(pi.point);

                auto &c_info = local_c_infos[pi.centroid];
                scalar_add_eq(c_info.sum_coords, pi.point);
                c_info.points++;

                local_sum += pi.sqr_dist;
            }

            const std::scoped_lock lock(this->upd_c_infos);
            sum += local_sum;

            size_t i = 0;
            for (auto local_c_info = std::cbegin(local_c_infos); local_c_info != std::cend(local_c_infos); ++local_c_info, ++i) {
                scalar_add_eq(this->c_infos[i].sum_coords, local_c_info->sum_coords);
                this->c_infos[i].points += local_c_info->points;
            }
        };

        pool.parallelize_loop(0, std::size(this->points), loop);
        pool.wait_for_tasks();
        return sum;
    }
};