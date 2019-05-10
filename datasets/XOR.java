package com.camillo.datasets;

import com.camillo.Matrix;

import java.util.ArrayList;

public class XOR implements Dataset {

    private ArrayList<Dataholder> trainingData = new ArrayList<>();

    @Override
    public void load() {

        Dataholder sample1 = new Dataholder(new double[]{1,0}, new double[]{1});
        Dataholder sample2 = new Dataholder(new double[]{0,1}, new double[]{1});
        Dataholder sample3 = new Dataholder(new double[]{1,1}, new double[]{0});
        Dataholder sample4 = new Dataholder(new double[]{0,0}, new double[]{0});

        trainingData.add(sample1);
        trainingData.add(sample2);
        trainingData.add(sample3);
        trainingData.add(sample4);

    }

    @Override
    public Matrix getLabelAt(int index) {

        if(index > this.trainingData.size()) {

            System.out.println("Given index is larger than dataset");
            return null;

        }

        return trainingData.get(index).getLabel();

    }

    @Override
    public Matrix getInputAt(int index) {

        if(index > this.trainingData.size()) {

            System.out.println("Given index is larger than dataset");
            return null;

        }

        return trainingData.get(index).getInput();

    }

    @Override
    public Dataholder getRandomDataholder() {

        int randomIndex = (int) Math.floor(Math.random() * trainingData.size());
        return trainingData.get(randomIndex);

    }

}
