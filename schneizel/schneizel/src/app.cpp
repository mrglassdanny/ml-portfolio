#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

int main()
{
	Board board;

	at::Tensor a = at::ones({ 2, 2 }, at::kInt);
	at::Tensor b = at::randn({ 2, 2 });
	auto c = a + b.to(at::kInt);

	return 0;
}