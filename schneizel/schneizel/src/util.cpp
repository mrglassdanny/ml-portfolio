#include "util.h"

namespace schneizel
{
    StopWatch::StopWatch()
    {
        this->beg_ = 0;
        this->end_ = 0;
    }

    StopWatch::~StopWatch()
    {
    }

    void StopWatch::start()
    {
        this->beg_ = clock();
        this->end_ = this->beg_;
    }

    void StopWatch::stop()
    {
        this->end_ = clock();
    }

    double StopWatch::get_elapsed_seconds()
    {
        return ((double)(this->end_ - this->beg_)) / CLOCKS_PER_SEC;
    }

    void StopWatch::print_elapsed_seconds()
    {
        printf("ELAPSED SECONDS: %f\n", this->get_elapsed_seconds());
    }
}