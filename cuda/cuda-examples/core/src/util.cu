#include "util.cuh"

StopWatch::StopWatch()
{
	this->beg = 0;
	this->end = 0;
}

StopWatch::~StopWatch()
{

}

void StopWatch::start()
{
	this->beg = clock();
	this->end = this->beg;
}

void StopWatch::stop()
{
	this->end = clock();
}

double StopWatch::get_elapsed_seconds()
{
	return ((double)(this->end - this->beg)) / CLOCKS_PER_SEC;
}

void StopWatch::print_elapsed_seconds()
{
	printf("ELAPSED SECONDS: %f\n", this->get_elapsed_seconds());
}