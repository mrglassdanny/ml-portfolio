#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <vector>
#include <random>

namespace tallgeese
{
	namespace core
	{
		class StopWatch
		{
		public:
			virtual void start() = 0;
			virtual void stop() = 0;

			virtual double get_elapsed_seconds() = 0;
			virtual void print_elapsed_seconds() = 0;
		};

		class CpuStopWatch : public StopWatch
		{
		private:
			clock_t beg_;
			clock_t end_;

		public:
			CpuStopWatch();
			~CpuStopWatch();

			virtual void start();
			virtual void stop();

			virtual double get_elapsed_seconds();
			virtual void print_elapsed_seconds();
		};

		class FileUtils
		{
		public:
			static long long get_file_size(const char* name);
		};
	}
}
