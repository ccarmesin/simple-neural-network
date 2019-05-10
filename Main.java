package com.camillo;

import com.camillo.activations.Activations;
import com.camillo.datasets.Dataholder;
import com.camillo.datasets.Dataset;
import com.camillo.network.DeepNetwork;
import com.camillo.network.Network;

public class Main {

    public static void main(String[] args) {

        // Dataset to train the network on
        Dataset dataset = Dataset.XOR;

        // Network type (DeepNetwork with multiple hidden layers, NeuralNetwork with just one hidden layer)
        Network network = new DeepNetwork(Activations.Sigmoid, 2,1, 8, 8);

        // Train the network on multiple epochs(batchsize is always one)
        train(network, dataset, 1000);

        evaluate(network, dataset);

    }

    /**
     * Train the created neural network
     *
     * @param network to train
     * @param dataset to train the network on
     * @param epochs training iterations
     */
    static void train(Network network, Dataset dataset, int epochs) {

        dataset.load();

        for(int i = 0; i < epochs; i++) {

            Dataholder dataholder = dataset.getRandomDataholder();
            network.train(dataholder.getInput(), dataholder.getLabel());

        }

        network.test(dataset.getInputAt(0)).print();
        network.test(dataset.getInputAt(1)).print();
        network.test(dataset.getInputAt(2)).print();
        network.test(dataset.getInputAt(3)).print();

    }

    static void evaluate(Network network, Dataset dataset) {

        network.test(dataset.getInputAt(0)).print();
        network.test(dataset.getInputAt(1)).print();
        network.test(dataset.getInputAt(2)).print();
        network.test(dataset.getInputAt(3)).print();

    }
}
