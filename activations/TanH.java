package com.camillo.activations;

public class TanH implements Activations {


    @Override
    public double activation(double x) {
        return (2 / (1 + Math.pow(Math.E, -2*x))) -1;
    }

    @Override
    public double derivative(double x) {
        return 1 / Math.pow(Math.cosh(x), 2);
    }
}
