#pragma once

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace chess;

#define CHANNEL_CNT 6

class PosEvalModel
{
public:
    Tensor *w1;
    Tensor *w2;

    PosEvalModel(int w1_filter_cnt, int w2_filter_cnt);
    ~PosEvalModel();

    Tensor *forward(Tensor *x);
    void backward(Tensor *p, Tensor *y);
};