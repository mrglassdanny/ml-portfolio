#pragma once

#include <iostream>
#include <random>

namespace schneizel
{
    class StopWatch
    {
    private:
        clock_t beg_;
        clock_t end_;

    public:
        StopWatch();
        ~StopWatch();

        void start();
        void stop();

        double get_elapsed_seconds();
        void print_elapsed_seconds();
    };
}