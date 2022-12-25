#pragma once

#include <vector>

#include "../core/mod.cuh"

namespace zero
{
    using namespace core;

    namespace cluster
    {
        class KMeans
        {
        private:
            int cluster_cnt_;
            int feature_cnt_;
            Tensor *clusters_;

        public:
            KMeans();
            KMeans(int cluster_cnt, int feature_cnt);
            KMeans(const KMeans &src);
            ~KMeans();

            void load(const char *path);
            void save(const char *path);

            void initialize(Tensor *x);
            float train(Tensor *x);
            Tensor *predict(Tensor *x);

            void validate_input(Tensor *x);

            Tensor *clusters();

            static KMeans *save_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path);
            static void elbow_analysis(Tensor *x, int cluster_cnt_lower, int cluster_cnt_upper,
                                       int iter_cnt, const char *csv_path);
        };
    }
}