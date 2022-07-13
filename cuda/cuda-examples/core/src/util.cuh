#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random>


class StopWatch
{
private:
	clock_t beg;
	clock_t end;
public:
	StopWatch();
	~StopWatch();

	void start();
	void stop();

	double get_elapsed_seconds();
	void print_elapsed_seconds();
};