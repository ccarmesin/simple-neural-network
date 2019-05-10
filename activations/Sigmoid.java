package com.camillo.activations;

public class Sigmoid implements Activations {

    @Override
    public double activation(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }

}
