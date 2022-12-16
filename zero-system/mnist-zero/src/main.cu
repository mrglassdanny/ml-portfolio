#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

using namespace zero::core;
using namespace zero::nn;

struct Batch
{
	Tensor *x;
	Tensor *y;
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
		auto x = Tensor::from_data(Shape(batch_size, 1, img_row_cnt, img_col_cnt), &img_flt_buf[i * batch_size * img_area]);
		auto y = Tensor::from_data(Shape(batch_size, 1), &lbl_flt_buf[i * batch_size]);
		auto oh_y = Tensor::one_hot(y, 9);
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
		auto x = Tensor::from_data(Shape(batch_size, 1, img_row_cnt, img_col_cnt), &img_flt_buf[i * batch_size * img_area]);
		auto y = Tensor::from_data(Shape(batch_size, 1), &lbl_flt_buf[i * batch_size]);
		auto oh_y = Tensor::one_hot(y, 9);
		delete y;

		batches.push_back({x, oh_y});
	}

	free(lbl_flt_buf);
	free(img_flt_buf);

	return batches;
}

float test_mnist(Model *model, int epoch, bool train, bool to_file);

void train_mnist(Model *model, int batch_size, int epochs, bool test_each_epoch)
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

float test_mnist(Model *model, int epoch, bool train, bool to_file)
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
			acc += model->accuracy(p, y, Model::classification_accuracy_fn);
			delete p;
		}

		test_acc_pct = (acc / (float)batch_cnt) * 100.0f;

		if (to_file)
		{
			FILE *f = fopen("temp/test.txt", "a");
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
			acc += model->accuracy(p, y, Model::classification_accuracy_fn);
			delete p;
		}

		train_acc_pct = (acc / (float)batch_cnt) * 100.0f;

		if (to_file)
		{
			FILE *f = fopen("temp/test.txt", "a");
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
	auto m1 = new Model();
	auto m2 = new Model();
	auto m3 = new Model();
	auto m4 = new Model();

	int batch_size = 1;

	// m1
	{
		auto x = Tensor::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = Tensor::ones(true, Shape(batch_size, 1));

		m1->set_initializer(new XavierInitializer());

		m1->linear(x->shape(), 16, new Sigmoid());
		m1->linear(16, new Tanh());
		m1->linear(y->shape(), new Sigmoid());

		m1->set_loss(new loss::MSE());
		m1->set_optimizer(new optim::SGD(m1->parameters(), 0.01f));

		m1->summarize();
		m1->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m2
	{
		auto x = Tensor::random(true, Shape(batch_size, 64), 0.0f, 1.0f);
		auto y = Tensor::zeros(true, Shape(batch_size, 10));
		y->set_val(3, 1.0f);

		m2->set_initializer(new XavierInitializer());

		m2->linear(x->shape(), 16, new Tanh());
		m2->linear(16, new Sigmoid());
		m2->linear(y->shape(), new Sigmoid());

		m2->set_loss(new loss::CrossEntropy());
		m2->set_optimizer(new optim::SGD(m2->parameters(), 0.01f));

		m2->summarize();
		m2->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m3
	{
		auto x = Tensor::random(true, Shape(batch_size, 2, 21, 14), 0.0f, 1.0f);
		auto y = Tensor::zeros(true, Shape(batch_size, 4));
		y->set_val(3, 1.0f);

		m3->set_initializer(new XavierInitializer());

		m3->conv2d(x->shape(), Shape(4, 2, 3, 2), layer::Stride{3, 2}, new Sigmoid());
		m3->conv2d(Shape(4, 4, 2, 2), layer::Stride{1, 1}, new Sigmoid());
		m3->linear(16, new Sigmoid());
		m3->linear(y->shape(), new Sigmoid());

		m3->set_loss(new loss::MSE());
		m3->set_optimizer(new optim::SGD(m3->parameters(), 0.01f));

		m3->summarize();
		m3->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	// m4
	{
		auto x = Tensor::random(true, Shape(batch_size, 1, 12, 12), 0.0f, 1.0f);
		auto y = Tensor::zeros(true, Shape(batch_size, 4));
		y->set_val(3, 1.0f);

		m4->set_initializer(new XavierInitializer());

		m4->conv2d(x->shape(), Shape(4, 1, 3, 3), layer::Stride{1, 1}, new Sigmoid());
		m4->conv2d(Shape(4, 4, 4, 4), layer::Stride{1, 1}, new Tanh());
		m4->conv2d(Shape(4, 4, 2, 2), layer::Stride{1, 1}, nullptr);
		m4->linear(16, new Sigmoid());
		m4->linear(y->shape(), new Sigmoid());

		m4->set_loss(new loss::CrossEntropy());
		m4->set_optimizer(new optim::SGD(m4->parameters(), 0.01f));

		m4->summarize();
		m4->validate_gradients(x, y, false);

		delete x;
		delete y;
	}

	delete m1;
	delete m2;
	delete m3;
	delete m4;
}

void mnist_conv(int batch_size, int epochs)
{
	auto input_shape = Shape(batch_size, 1, 28, 28);
	auto output_shape = Shape(batch_size, 10);

	auto model = new Model();

	model->set_initializer(new XavierInitializer());

	model->conv2d(input_shape, Shape(64, 1, 5, 5), layer::Stride{1, 1}, new ReLU());
	model->conv2d(Shape(64, 64, 3, 3), layer::Stride{3, 3}, new ReLU());
	model->conv2d(Shape(64, 64, 3, 3), layer::Stride{1, 1}, new ReLU());
	model->linear(512, new ReLU());
	model->linear(256, new ReLU());
	model->linear(128, new ReLU());
	model->linear(output_shape, new Sigmoid());

	model->set_loss(new loss::CrossEntropy());
	model->set_optimizer(new optim::SGDMomentum(model->parameters(), 0.15f, ZERO_NN_BETA_1));

	model->summarize();

	train_mnist(model, batch_size, epochs, true);
	test_mnist(model, epochs, true, false);

	delete model;
}

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");
	srand(time(NULL));

	// grad_tests();

	mnist_conv(50, 30);

	return 0;
}