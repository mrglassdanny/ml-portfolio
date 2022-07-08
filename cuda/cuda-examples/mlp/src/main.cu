#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <conio.h>
#include <ctype.h>
#include <random>
#include <windows.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>




class Matrix {
private:
	bool cuda_flg;
	int rows;
	int cols;
	float* data;

public:
	Matrix(bool cuda_flg, int rows, int cols)
	{
		this->cuda_flg = cuda_flg;
		this->rows = rows;
		this->cols = cols;
		
		if (cuda_flg) {
			cudaMalloc(&this->data, sizeof(float) * (rows * cols));
		} else {
			this->data = (float*)malloc(sizeof(float) * (rows * cols));
		}
	}

	~Matrix()
	{
		if (this->cuda_flg) {
			cudaFree(this->data);
		} else {
			free(this->data);
		}
	}

	void to_cpu()
	{
		if (this->cuda_flg) {
			float* dst = (float*)malloc(sizeof(float) * (this->rows * this->cols));
			cudaMemcpy(dst, this->data, sizeof(float) * (this->rows * this->cols), cudaMemcpyDeviceToHost);
			cudaFree(this->data);
			this->data = dst;
			this->cuda_flg = false;
		}
	}

	void to_cuda()
	{
		if (!this->cuda_flg) {
			float* dst;
			cudaMalloc(&dst, sizeof(float) * (this->rows * this->cols));
			cudaMemcpy(dst, this->data, sizeof(float) * (this->rows * this->cols), cudaMemcpyHostToDevice);
			free(this->data);
			this->data = dst;
			this->cuda_flg = true;
		}
	}

	float get_val(int row, int col)
	{
		float val;
		cudaMemcpy(&val, &this->data[row * this->cols + col], sizeof(float), cudaMemcpyDefault);
		return val;
	}

	void set_val(int row, int col, float val)
	{
		cudaMemcpy(&this->data[row * this->cols + col], &val, sizeof(float), cudaMemcpyDefault);
	}

	void zeros()
	{
		if (this->cuda_flg) {
			cudaMemset(this->data, 0, sizeof(float) * (this->rows * this->cols));
		} else {
			memset(this->data, 0, sizeof(float) * (this->rows * this->cols));
		}
	}

	void rands(float mean, float stddev)
	{
		bool orig_cuda_flg = this->cuda_flg;

		this->to_cpu();

		{
			std::random_device rd;
			std::mt19937 gen(rd());

			for (int i = 0; i < (this->rows * this->cols); i++)
			{
				std::normal_distribution<float> d(mean, stddev);
				this->data[i] = d(gen);
			}
		}

		if (orig_cuda_flg) {
			this->to_cuda();
		}
	}

	void print()
	{
		bool orig_cuda_flg = this->cuda_flg;

		this->to_cpu();

		printf("[");

		for (int i = 0; i < this->rows; i++) {

			if (i == 0) {
				printf(" [ \t");
			} else {
				printf("  [ \t");
			}

			for (int j = 0; j < this->cols; j++) {

				float val = this->data[i * this->cols + j];

				if (val >= 0.0f) {
					printf(" %f\t", val);
				} else {
					printf("%f\t", val);
				}
			}

			if (i == this->rows - 1) {
				printf(" ] ]");
			} else {
				printf(" ],\n");
			}
		}

		printf("\n");

		if (orig_cuda_flg) {
			this->to_cuda();
		}
	}
};



int main(int argc, char** argv)
{
	Matrix mtx(true, 5, 3);

	mtx.rands(0.0f, 1.0f);

	mtx.print();

	printf("%f\n", mtx.get_val(0, 2));

	mtx.set_val(0, 1, 10000.0f);
	mtx.set_val(0, 2, 10000.0f);

	printf("%f\n", mtx.get_val(0, 2));

	mtx.print();

	return 0;
}