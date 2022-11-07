#include "util.cuh"

using namespace zero::core;

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

CudaStopWatch::CudaStopWatch()
{
    cudaEventCreate(&this->beg_);
    cudaEventCreate(&this->end_);
}

CudaStopWatch::~CudaStopWatch()
{
}

void CudaStopWatch::start()
{
    cudaEventRecord(this->beg_, 0);
}

void CudaStopWatch::stop()
{
    cudaThreadSynchronize();
    cudaEventRecord(this->end_, 0);
    cudaEventSynchronize(this->end_);
}

double CudaStopWatch::get_elapsed_seconds()
{
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, this->beg_, this->end_);

    return ((double)elapsed_ms / 1000.0);
}

void CudaStopWatch::print_elapsed_seconds()
{
    printf("ELAPSED SECONDS: %f\n", this->get_elapsed_seconds());
}

long long FileUtils::get_file_size(const char *name)
{

    HANDLE hFile = CreateFile((LPCSTR)name, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return -1; // error condition, could call GetLastError to find out more

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        return -1; // error condition, could call GetLastError to find out more
    }

    CloseHandle(hFile);
    return size.QuadPart;
}