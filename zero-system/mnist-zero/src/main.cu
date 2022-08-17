#include <stdio.h>
#include <conio.h>

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

Batch get_test_batch()
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
	auto oh_y = NdArray::one_hot(y, 9);
	delete y;

	free(lbl_flt_buf);
	free(img_flt_buf);

	return {x, oh_y};
}

void train_mnist(nn::Model *model, int batch_size)
{
	auto train_ds = get_train_dataset(batch_size);

	float validation_loss = 0.0f;
	int train_batch_cnt = train_ds.size();
	int validation_batch_cnt = (int)(train_batch_cnt * 0.10f);

	std::vector<int> validation_batch_idxs;
	for (int v = 0; v < validation_batch_cnt; v++)
	{
		validation_batch_idxs.push_back(rand() % train_batch_cnt);
	}

	for (int i = 0;; i++)
	{
		for (int j = 0; j < train_batch_cnt; j++)
		{
			auto batch = &train_ds[j];
			auto x = batch->x;
			auto y = batch->y;

			bool validation_batch_flg = false;
			for (int v = 0; v < validation_batch_cnt; v++)
			{
				if (validation_batch_idxs[v] == j)
				{
					validation_batch_flg = true;
					break;
				}
			}

			if (!validation_batch_flg)
			{
				auto p = model->forward(x);
				model->backward(p, y);
				model->step();
				delete p;
			}
			else
			{
				auto p = model->forward(x);
				validation_loss += model->loss(p, y);
				delete p;
			}
		}

		printf("EPOCH: %d\tVALIDATION LOSS: %f\n", i + 1, (validation_loss / (float)validation_batch_cnt));
		validation_loss = 0.0f;

		if (_kbhit())
		{
			if (_getch() == 'q')
			{
				break;
			}
		}
	}
}

void test_mnist(nn::Model *model)
{
	auto test_batch = get_test_batch();
	auto x = test_batch.x;
	auto y = test_batch.y;

	model->change_batch_size(x->shape()[0]);

	auto p = model->forward(x);
	auto l = model->loss(p, y);
	printf("TEST LOSS: %f\tACCURACY: %f%%\n", l, model->accuracy(p, y) * 100.0f);

	delete p;

	delete x;
	delete y;
}

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");
	srand(time(NULL));

	auto model = new nn::Model();
	int batch_size = 1;

	model->linear(Shape(batch_size, 1, 28, 28), 16);
	model->sigmoid();
	model->linear(16);
	model->sigmoid();
	model->linear(Shape(batch_size, 10));
	model->sigmoid();

	model->set_loss(new nn::loss::MSE());
	model->set_optimizer(new nn::optim::Adam(model->parameters(), 0.01f, BETA_1, BETA_2));

	model->summarize();

	train_mnist(model, batch_size);
	test_mnist(model);

	return 0;
}