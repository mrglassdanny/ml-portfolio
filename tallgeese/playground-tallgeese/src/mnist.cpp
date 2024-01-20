
#include "mnist.h"

namespace mnist
{
	std::vector<Batch> get_train_dataset(int batch_size)
	{
		int img_row_cnt = 28;
		int img_col_cnt = 28;
		int img_area = img_row_cnt * img_col_cnt;
		int img_cnt = 60000;

		std::vector<Batch> batches;

		FILE *img_file = fopen("data/mnist/train-images.idx3-ubyte", "rb");
		FILE *lbl_file = fopen("data/mnist/train-labels.idx1-ubyte", "rb");

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
			auto x = Tensor::from_data({batch_size, 1, img_row_cnt, img_col_cnt}, &img_flt_buf[i * batch_size * img_area]);
			auto y = Tensor::from_data({batch_size, 1}, &lbl_flt_buf[i * batch_size]);
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

		FILE *img_file = fopen("data/mnist/t10k-images.idx3-ubyte", "rb");
		FILE *lbl_file = fopen("data/mnist/t10k-labels.idx1-ubyte", "rb");

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
			auto x = Tensor::from_data({batch_size, 1, img_row_cnt, img_col_cnt}, &img_flt_buf[i * batch_size * img_area]);
			auto y = Tensor::from_data({batch_size, 1}, &lbl_flt_buf[i * batch_size]);
			auto oh_y = Tensor::one_hot(y, 9);
			delete y;

			batches.push_back({x, oh_y});
		}

		free(lbl_flt_buf);
		free(img_flt_buf);

		return batches;
	}

	void train_mnist(Model *model, int batch_size, int epochs)
	{
		auto train_ds = get_train_dataset(batch_size);

		int batch_cnt = train_ds.size();

		bool quit = false;

		int epoch;

		for (epoch = 0; epoch < epochs; epoch++)
		{
			for (int j = 0; j < batch_cnt; j++)
			{
				auto batch = &train_ds[j];
				auto x = batch->x;
				auto y = batch->y;

				auto p = model->forward(x);
				model->loss(p, y);
				model->backward();
				model->step();
				model->reset();

				if (j % 100 == 0)
				{
					system("cls");
					int progress = (int)((j + 1) * 1.0f / batch_cnt * 100.0f);
					printf("TRAINING\n");
					printf("EPOCH: %d\tPROGRESS: %d%%\n", epoch + 1, progress);
					for (int i = 0; i < 50; i++)
					{
						if (i * 2 < progress)
						{
							printf("=");
						}
						else
						{
							printf(".");
						}
					}
				}

				if (_kbhit())
				{
					if (_getch() == 'q')
					{
						quit = true;
						break;
					}
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

	float test_mnist(Model *model)
	{
		float test_acc_pct = 0.0f;
		float train_acc_pct = 0.0f;

		auto ds = get_test_dataset(model->get_batch_size());

		int batch_cnt = ds.size();

		float loss = 0.0f;
		float acc = 0.0f;

		for (int j = 0; j < batch_cnt; j++)
		{
			auto batch = &ds[j];
			auto x = batch->x;
			auto y = batch->y;

			auto p = model->forward(x);
			loss += model->loss(p, y).v;
			acc += model->accuracy(p, y, Model::classification_accuracy_fn);
			model->reset();

			if (j % 100 == 0)
			{
				system("cls");
				int progress = (int)((j + 1) * 1.0f / batch_cnt * 100.0f);
				printf("TESTING\n");
				printf("PROGRESS: %d%%\n", progress);
				for (int i = 0; i < 50; i++)
				{
					if (i < progress / 2)
					{
						printf("=");
					}
					else
					{
						printf(".");
					}
				}
			}
		}

		test_acc_pct = (acc / (float)batch_cnt) * 100.0f;

		printf("\n\nTEST LOSS: %f\tTEST ACCURACY: %f%%\n",
			   (loss / (float)batch_cnt),
			   test_acc_pct);

		for (auto batch : ds)
		{
			delete batch.x;
			delete batch.y;
		}

		return test_acc_pct;
	}
}
