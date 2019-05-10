package com.camillo.datasets;

import com.camillo.Matrix;

public interface Dataset {

    XOR XOR = new XOR();

    void load();

    Matrix getLabelAt(int index);

    Matrix getInputAt(int index);

    Dataholder getRandomDataholder();

}
