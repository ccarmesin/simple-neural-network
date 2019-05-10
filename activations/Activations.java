package com.camillo.activations;

public interface Activations {

    Sigmoid Sigmoid = new Sigmoid();
    ReLU ReLU = new ReLU();
    TanH TanH = new TanH();

    double activation(double x);
    double derivative(double x);

}
