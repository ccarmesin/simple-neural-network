package com.camillo.datasets;

import com.camillo.Matrix;

public class Dataholder {

    private double[] inputs, labels;

    Dataholder(double[] inputs, double[] labels) {
        this.inputs = inputs;
        this.labels = labels;
    }

    public Matrix getLabel() {

        Matrix m1 = new Matrix(new double[][]{labels});
        return m1.transpose();

    }

    public Matrix getInput() {

        Matrix m1 = new Matrix(new double[][]{inputs});
        return m1.transpose();

    }

}
