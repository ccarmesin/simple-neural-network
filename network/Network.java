package com.camillo.network;

import com.camillo.Matrix;

public interface Network {

    Matrix test(Matrix inputs);

    void train(Matrix inputs, Matrix labels);

}
