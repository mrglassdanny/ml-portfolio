#include "util.h"

namespace tallgeese
{
    namespace core
    {
        CpuStopWatch::CpuStopWatch()
        {
            this->beg_ = 0;
            this->end_ = 0;
        }

        CpuStopWatch::~CpuStopWatch()
        {
        }

        void CpuStopWatch::start()
        {
            this->beg_ = clock();
            this->end_ = this->beg_;
        }

        void CpuStopWatch::stop()
        {
            this->end_ = clock();
        }

        double CpuStopWatch::get_elapsed_seconds()
        {
            return ((double)(this->end_ - this->beg_)) / CLOCKS_PER_SEC;
        }

        void CpuStopWatch::print_elapsed_seconds()
        {
            printf("ELAPSED SECONDS: %f\n", this->get_elapsed_seconds());
        }
     }
}
