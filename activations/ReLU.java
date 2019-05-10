package com.camillo.activations;

public class ReLU implements Activations {

    @Override
    public double activation(double x) {
        if(x >= 0)
            return x;
        else
            return 0;
    }

    @Override
    public double derivative(double x) {
        if(x >= 0)
            return 1;
        else
            return 0;
    }

}
