#include <stdio.h>

#include <nn/mod.cuh>

struct Dataset
{
	NdArray *x;
	NdArray *y;
};

Dataset get_train_dataset()
{
	int img_row_cnt = 28;
	int img_col_cnt = 28;
	int img_area = img_row_cnt * img_col_cnt;
	int img_cnt = 60000;

	FILE *img_file = fopen("data/train-images.idx3-ubyte", "rb");
	FILE *lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");

	fseek(img_file, sizeof(int) * 4, 0);
	unsigned char *img_buf = (unsigned char *)malloc((sizeof(unsigned char) * img_area * img_cnt));
	fread(img_buf, 1, (sizeof(unsigned char) * img_area * img_cnt), img_file);

	fseek(lbl_file, sizeof(int) * 2, 0);
	unsigned char *lbl_buf = (unsigned char *)malloc(sizeof(unsigned char) * img_cnt);
	fread(lbl_buf, 1, (sizeof(unsigned char) * img_cnt), lbl_file);

	fclose(img_file);
	fclose(lbl_file);

	float *img_flt_buf = (float *)malloc(sizeof(float) * (img_area * img_cnt));
	for (int i = 0; i < (img_area * img_cnt); i++)
	{
		img_flt_buf[i] = ((float)img_buf[i] / (255.0));
	}

	float *lbl_flt_buf = (float *)malloc(sizeof(float) * (img_cnt));
	for (int i = 0; i < (img_cnt); i++)
	{
		lbl_flt_buf[i] = ((float)lbl_buf[i]);
	}

	free(img_buf);
	free(lbl_buf);

	auto x = NdArray::from_data(Shape(img_cnt, 1, img_row_cnt, img_col_cnt), img_flt_buf);
	auto y = NdArray::from_data(Shape(img_cnt, 1), lbl_flt_buf);

	free(lbl_flt_buf);
	free(img_flt_buf);

	return Dataset{x, y};
}

Dataset get_test_dataset()
{
	int img_row_cnt = 28;
	int img_col_cnt = 28;
	int img_area = img_row_cnt * img_col_cnt;
	int img_cnt = 10000;

	FILE *img_file = fopen("data/t10k-images.idx3-ubyte", "rb");
	FILE *lbl_file = fopen("data/t10k-labels.idx1-ubyte", "rb");

	fseek(img_file, sizeof(int) * 4, 0);
	unsigned char *img_buf = (unsigned char *)malloc((sizeof(unsigned char) * img_area * img_cnt));
	fread(img_buf, 1, (sizeof(unsigned char) * img_area * img_cnt), img_file);

	fseek(lbl_file, sizeof(int) * 2, 0);
	unsigned char *lbl_buf = (unsigned char *)malloc(sizeof(unsigned char) * img_cnt);
	fread(lbl_buf, 1, (sizeof(unsigned char) * img_cnt), lbl_file);

	fclose(img_file);
	fclose(lbl_file);

	float *img_flt_buf = (float *)malloc(sizeof(float) * (img_area * img_cnt));
	for (int i = 0; i < (img_area * img_cnt); i++)
	{
		img_flt_buf[i] = ((float)img_buf[i] / (255.0));
	}

	float *lbl_flt_buf = (float *)malloc(sizeof(float) * (img_cnt));
	for (int i = 0; i < (img_cnt); i++)
	{
		lbl_flt_buf[i] = ((float)lbl_buf[i]);
	}

	free(img_buf);
	free(lbl_buf);

	auto x = NdArray::from_data(Shape(img_cnt, 1, img_row_cnt, img_col_cnt), img_flt_buf);
	auto y = NdArray::from_data(Shape(img_cnt, 1), lbl_flt_buf);

	free(lbl_flt_buf);
	free(img_flt_buf);

	return Dataset{x, y};
}

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");

	auto x = NdArray::random(true, Shape(1, 2, 16, 16), 0.0f, 1.0f);
	auto y = NdArray::zeros(true, Shape(1, 3));

	auto model = new nn::Model();

	model->conv2d(x->shape(), Shape(2, 2, 2, 2), nn::layer::Padding{2, 2}, nn::layer::Stride{2, 2});
	model->sigmoid();
	model->conv2d(Shape(2, 2, 2, 2), nn::layer::Stride{1, 1});
	model->sigmoid();
	model->linear(32);
	model->tanh();
	model->linear(y->shape());
	model->tanh();

	model->set_loss(new nn::loss::MSE());

	model->summarize();

	model->validate_gradients(x, y, true);

	return 0;
}