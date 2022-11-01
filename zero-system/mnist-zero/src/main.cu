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

float test_mnist(nn::Model *model, int epoch, bool train, bool csv);

void train_mnist(nn::Model *model, int batch_size, int epochs, bool test_each_epoch)
{
	auto train_ds = get_train_dataset(batch_size);

	int train_batch_cnt = train_ds.size();

	bool quit = false;

	int epoch;

	for (epoch = 0; epoch < epochs; epoch++)
	{
		for (int j = 0; j < train_batch_cnt; j++)
		{
			auto batch = &train_ds[j];
			auto x = batch->x;
			auto y = batch->y;

			auto p = model->forward(x);

			model->backward(p, y);
			model->step();

			delete p;

			if (_kbhit())
			{
				if (_getch() == 'q')
				{
					quit = true;
					break;
				}
			}
		}

		if (test_each_epoch)
		{
			float test_acc_pct = test_mnist(model, epoch, false, true);
			if (test_acc_pct >= 99.0f)
			{
				model->optimizer()->scale_learning_rate(0.1f);
			}
		}

		if (quit)
		{
			break;
		}
	}

	for (auto batch : train_ds)
	{
		delete batch.x;
		delete batch.y;
	}
}

float test_mnist(nn::Model *model, int epoch, bool train, bool csv)
{
	float test_acc_pct = 0.0f;
	float train_acc_pct = 0.0f;

	// TEST:
	{
		auto ds = get_test_dataset(model->batch_size());

		int batch_cnt = ds.size();

		float loss = 0.0f;
		float acc = 0.0f;

		for (int j = 0; j < batch_cnt; j++)
		{
			auto batch = &ds[j];
			auto x = batch->x;
			auto y = batch->y;

			auto p = model->forward(x);
			loss += model->loss(p, y);
			acc += model->accuracy(p, y);
			delete p;
		}

		test_acc_pct = (acc / (float)batch_cnt) * 100.0f;

		if (csv)
		{
			FILE *f = fopen("temp/test.csv", "a");
			fprintf(f, "EPOCH: %d\tTEST LOSS: %f\tTEST ACCURACY: %f%%\n",
					epoch,
					(loss / (float)batch_cnt),
					test_acc_pct);
			fclose(f);
		}
		else
		{
			printf("EPOCH: %d\tTEST LOSS: %f\tTEST ACCURACY: %f%%\n",
				   epoch,
				   (loss / (float)batch_cnt),
				   test_acc_pct);
		}

		for (auto batch : ds)
		{
			delete batch.x;
			delete batch.y;
		}
	}

	// TRAIN
	if (train)
	{
		auto ds = get_train_dataset(model->batch_size());

		int batch_cnt = ds.size();

		float loss = 0.0f;
		float acc = 0.0f;

		for (int j = 0; j < batch_cnt; j++)
		{
			auto batch = &ds[j];
			auto x = batch->x;
			auto y = batch->y;

			auto p = model->forward(x);
			loss += model->loss(p, y);
			acc += model->accuracy(p, y);
			delete p;
		}

		train_acc_pct = (acc / (float)batch_cnt) * 100.0f;

		if (csv)
		{
			FILE *f = fopen("temp/test.csv", "a");
			fprintf(f, "EPOCH: %d\tTRAIN LOSS: %f\tTRAIN ACCURACY: %f%%\n",
					epoch,
					(loss / (float)batch_cnt),
					train_acc_pct);
			fclose(f);
		}
		else
		{
			printf("EPOCH: %d\tTRAIN LOSS: %f\tTRAIN ACCURACY: %f%%\n",
				   epoch,
				   (loss / (float)batch_cnt),
				   train_acc_pct);
		}

		for (auto batch : ds)
		{
			delete batch.x;
			delete batch.y;
		}
	}

	return test_acc_pct;
}

void grad_tests()
{
	auto m1 = new nn::Model();
	auto m2 = new nn::Model();
	auto m3 = new nn::Model();
	auto m4 = new nn::Model();
	auto m5 = new nn::Model();
	auto m6 = new nn::Model();
	auto m7 = new nn::Model();

	int batch_size = 1;

	// m1
	{
		auto x = NdArray::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = NdArray::ones(true, Shape(batch_size, 1));

		m1->linear(x->shape(), 16, nn::layer::ActivationType::Sigmoid);
		m1->linear(16, nn::layer::ActivationType::Tanh);
		m1->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m1->set_loss(new nn::loss::MSE());
		m1->set_optimizer(new nn::optim::SGD(m1->parameters(), 0.01f));

		m1->summarize();
		m1->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m2
	{
		auto x = NdArray::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = NdArray::zeros(true, Shape(batch_size, 10));
		y->set_val(3, 1.0f);

		m2->linear(x->shape(), 16, nn::layer::ActivationType::Tanh);
		m2->linear(16, nn::layer::ActivationType::Sigmoid);
		m2->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m2->set_loss(new nn::loss::CrossEntropy());
		m2->set_optimizer(new nn::optim::SGD(m2->parameters(), 0.01f));

		m2->summarize();
		m2->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m3
	{
		auto x = NdArray::random(true, Shape(batch_size, 2, 21, 14), 0.0f, 1.0f);
		auto y = NdArray::zeros(true, Shape(batch_size, 4));
		y->set_val(3, 1.0f);

		m3->conv2d(x->shape(), Shape(4, 2, 3, 2), nn::layer::Stride{3, 2}, nn::layer::ActivationType::Sigmoid);
		m3->conv2d(Shape(4, 4, 2, 2), nn::layer::Stride{1, 1}, nn::layer::ActivationType::Sigmoid);
		m3->linear(16, nn::layer::ActivationType::Sigmoid);
		m3->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m3->set_loss(new nn::loss::MSE());
		m3->set_optimizer(new nn::optim::SGD(m3->parameters(), 0.01f));

		m3->summarize();
		m3->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m4
	{
		auto x = NdArray::random(true, Shape(batch_size, 1, 12, 12), 0.0f, 1.0f);
		auto y = NdArray::zeros(true, Shape(batch_size, 4));
		y->set_val(3, 1.0f);

		m4->conv2d(x->shape(), Shape(4, 1, 3, 3), nn::layer::Stride{1, 1}, nn::layer::ActivationType::Sigmoid);
		m4->conv2d(Shape(4, 4, 4, 4), nn::layer::Stride{1, 1}, nn::layer::ActivationType::Tanh);
		m4->conv2d(Shape(4, 4, 2, 2), nn::layer::Stride{1, 1}, nn::layer::ActivationType::None);
		m4->linear(16, nn::layer::ActivationType::Sigmoid);
		m4->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m4->set_loss(new nn::loss::CrossEntropy());
		m4->set_optimizer(new nn::optim::SGD(m4->parameters(), 0.01f));

		m4->summarize();
		m4->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m5
	{
		auto x = NdArray::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = NdArray::ones(true, Shape(batch_size, 1));

		m5->full_residual(x->shape(), 16, nn::layer::ActivationType::Tanh);
		m5->full_residual(16, nn::layer::ActivationType::Sigmoid);
		m5->full_residual(16, nn::layer::ActivationType::Sigmoid);
		m5->full_residual(y->shape(), nn::layer::ActivationType::Sigmoid);

		m5->set_loss(new nn::loss::MSE());
		m5->set_optimizer(new nn::optim::SGD(m5->parameters(), 0.01f));

		m5->summarize();
		m5->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m6
	{
		auto x = NdArray::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = NdArray::zeros(true, Shape(batch_size, 10));
		y->set_val(3, 1.0f);

		m6->full_residual(x->shape(), 16, nn::layer::ActivationType::Tanh);
		m6->linear(16, nn::layer::ActivationType::Sigmoid);
		m6->full_residual(16, nn::layer::ActivationType::Sigmoid);
		m6->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m6->set_loss(new nn::loss::CrossEntropy());
		m6->set_optimizer(new nn::optim::SGD(m6->parameters(), 0.01f));

		m6->summarize();
		m6->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m7
	{
		auto x = NdArray::random(true, Shape(batch_size, 1, 12, 12), 0.0f, 1.0f);
		auto y = NdArray::zeros(true, Shape(batch_size, 4));
		y->set_val(3, 1.0f);

		m7->conv2d(x->shape(), Shape(4, 1, 3, 3), nn::layer::Stride{1, 1}, nn::layer::ActivationType::Sigmoid);
		m7->conv2d(Shape(4, 4, 4, 4), nn::layer::Stride{1, 1}, nn::layer::ActivationType::Tanh);
		m7->conv2d(Shape(4, 4, 2, 2), nn::layer::Stride{1, 1}, nn::layer::ActivationType::None);
		m7->full_residual(16, nn::layer::ActivationType::Sigmoid);
		m7->full_residual(16, nn::layer::ActivationType::Sigmoid);
		m7->linear(y->shape(), nn::layer::ActivationType::Sigmoid);

		m7->set_loss(new nn::loss::CrossEntropy());
		m7->set_optimizer(new nn::optim::SGD(m7->parameters(), 0.01f));

		m7->summarize();
		m7->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	delete m1;
	delete m2;
	delete m3;
	delete m4;
	delete m5;
	delete m6;
	delete m7;
}

void mnist_conv(int batch_size, int epochs)
{
	Shape input_shape = Shape(batch_size, 1, 28, 28);
	Shape output_shape = Shape(batch_size, 10);

	auto model = new nn::Model();

	model->conv2d(input_shape, Shape(64, 1, 5, 5), nn::layer::Stride{1, 1}, nn::layer::ActivationType::ReLU);
	model->conv2d(Shape(64, 64, 3, 3), nn::layer::Stride{3, 3}, nn::layer::ActivationType::ReLU);
	model->conv2d(Shape(64, 64, 3, 3), nn::layer::Stride{1, 1}, nn::layer::ActivationType::ReLU);
	model->linear(512, nn::layer::ActivationType::ReLU);
	model->linear(256, nn::layer::ActivationType::ReLU);
	model->linear(128, nn::layer::ActivationType::ReLU);
	model->linear(output_shape, nn::layer::ActivationType::Sigmoid);

	model->set_loss(new nn::loss::CrossEntropy());
	model->set_optimizer(new nn::optim::SGDMomentum(model->parameters(), 0.2f, BETA_1));

	model->summarize();

	train_mnist(model, batch_size, epochs, true);
	test_mnist(model, epochs, true, false);

	delete model;
}

void mnist_conv_fr(int batch_size, int epochs)
{
	Shape input_shape = Shape(batch_size, 1, 28, 28);
	Shape output_shape = Shape(batch_size, 10);

	auto model = new nn::Model();

	model->conv2d(input_shape, Shape(64, 1, 5, 5), nn::layer::Stride{1, 1}, nn::layer::ActivationType::ReLU);
	model->conv2d(Shape(64, 64, 3, 3), nn::layer::Stride{3, 3}, nn::layer::ActivationType::ReLU);
	model->conv2d(Shape(64, 64, 3, 3), nn::layer::Stride{1, 1}, nn::layer::ActivationType::ReLU);
	model->full_residual(512, nn::layer::ActivationType::ReLU);
	model->full_residual(256, nn::layer::ActivationType::ReLU);
	model->full_residual(128, nn::layer::ActivationType::ReLU);
	model->linear(output_shape, nn::layer::ActivationType::Sigmoid);

	model->set_loss(new nn::loss::CrossEntropy());
	model->set_optimizer(new nn::optim::SGDMomentum(model->parameters(), 0.01f, BETA_1));

	model->summarize();

	train_mnist(model, batch_size, epochs, true);
	test_mnist(model, epochs, true, false);

	delete model;
}

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");
	srand(time(NULL));

	mnist_conv(50, 20);
	// mnist_conv_fr(50, 20);

	return 0;
}