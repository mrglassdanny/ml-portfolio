#include <stdio.h>

#include <nn/mod.cuh>

struct Batch
{
	NdArray *x;
	NdArray *y;
};

std::vector<Batch> get_train_dataset(int batch_size)
{
	int img_row_cnt = 28;
	int img_col_cnt = 28;
	int img_area = img_row_cnt * img_col_cnt;
	int img_cnt = 60000;

	std::vector<Batch> batches;

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

	for (int i = 0; i < img_cnt / batch_size; i++)
	{
		auto x = NdArray::from_data(Shape(batch_size, 1, img_row_cnt, img_col_cnt), &img_flt_buf[i * batch_size * img_area]);
		auto y = NdArray::from_data(Shape(batch_size, 1), &lbl_flt_buf[i * batch_size]);
		auto oh_y = NdArray::one_hot(y, 9);
		delete y;

		batches.push_back({x, oh_y});
	}

	free(lbl_flt_buf);
	free(img_flt_buf);

	return batches;
}

std::vector<Batch> get_test_dataset(int batch_size)
{
	int img_row_cnt = 28;
	int img_col_cnt = 28;
	int img_area = img_row_cnt * img_col_cnt;
	int img_cnt = 10000;

	std::vector<Batch> batches;

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

	for (int i = 0; i < img_cnt / batch_size; i++)
	{
		auto x = NdArray::from_data(Shape(batch_size, 1, img_row_cnt, img_col_cnt), &img_flt_buf[i * batch_size * img_area]);
		auto y = NdArray::from_data(Shape(batch_size, 1), &lbl_flt_buf[i * batch_size]);
		auto oh_y = NdArray::one_hot(y, 9);
		delete y;

		batches.push_back({x, oh_y});
	}

	free(lbl_flt_buf);
	free(img_flt_buf);

	return batches;
}

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");
	srand(time(NULL));

	int batch_size = 2;

	auto train_ds = get_train_dataset(batch_size);

	auto model = new nn::Model();

	model->linear(Shape(batch_size, 1, 28, 28), 512);
	model->sigmoid();
	model->linear(512);
	model->sigmoid();
	model->linear(512);
	model->sigmoid();
	model->linear(256);
	model->sigmoid();
	model->linear(Shape(batch_size, 10));
	model->sigmoid();

	model->set_loss(new nn::loss::MSE());
	model->set_optimizer(new nn::optim::SGD(model->parameters(), 0.0001f));

	model->summarize();

	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < train_ds.size(); j++)
		{	
			auto batch = &train_ds[j];

			auto x = batch->x;
			auto y = batch->y;

			auto p = model->forward(x);

			if (j % 20 == 0)
			{
				auto l = model->loss(p, y);
				printf("EPOCH: %d\tLOSS: %f\tACCURACY: %f%%\n", i + 1, l, model->accuracy(p, y) * 100.0f);
			}

			model->backward(p, y);
			delete p;
			model->step();
		}
	}

	/*auto batch = &train_ds[0];
	auto x = batch->x;
	auto y = batch->y;

	model->validate_gradients(x, y, true);*/

	return 0;
}