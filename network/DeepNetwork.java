package com.camillo.network;

import com.camillo.Matrix;
import com.camillo.activations.Activations;

import java.util.ArrayList;

public class DeepNetwork implements Network {

    private ArrayList<Matrix> weights = new ArrayList<>();

    private ArrayList<Matrix> momentums = new ArrayList<>();

    private double learning_rate;

    private Activations activation;

    private double momentum_correction;


    public DeepNetwork(Activations activation, int input_neurons, int output_neurons, int... hidden_neurons) {

        this.activation = activation;

        for(int i = -1; i < hidden_neurons.length; i++) {

            Matrix layer;
            Matrix momentum;
            if(i == -1) {

                layer = new Matrix(hidden_neurons[0], input_neurons);

                momentum = new Matrix(hidden_neurons[0], input_neurons);

            } else if(i == hidden_neurons.length - 1) {

                layer = new Matrix(output_neurons, hidden_neurons[hidden_neurons.length - 1]);

                momentum = new Matrix(output_neurons, hidden_neurons[hidden_neurons.length - 1]);

            } else {

                layer = new Matrix(hidden_neurons[i], hidden_neurons[i + 1]);

                momentum = new Matrix(hidden_neurons[i], hidden_neurons[i + 1]);

            }

            layer.randomize();
            this.weights.add(layer);

            momentum.fillNull();
            this.momentums.add(momentum);

        }

        this.learning_rate = 0.1;
        this.momentum_correction = 0.9;

    }

    public Matrix test(Matrix inputs) {

        // Declare matrix that update each layer and contains the output of the previous layer
        Matrix layer_outputs = inputs;

        // Loop each layer
        for (Matrix weight : this.weights) {

            // The input to the current layer is weights(me --> me + 1) times the layer_outputs
            Matrix layer_inputs = Matrix.multiply(weight, layer_outputs);
            // The inputs of the layer pass through sigmoid activation function to get the outputs
            assert layer_inputs != null;
            layer_outputs = Matrix.map(layer_inputs, true, this.activation);

        }

        // Return the result as Matrix
        return layer_outputs;

    }

    private Matrix[] feedforwardForBackProp(Matrix inputs) {

        // Declare matrix that update each layer and contains the output of the previous layer
        Matrix[] outputs = new Matrix[this.weights.size() + 1];
        outputs[0] = inputs;

        // Loop each layer
        for(int i = 0; i < this.weights.size(); i++) {

            // The input to the current layer is weights(me --> me + 1) times the layer_outputs
            Matrix layer_inputs = Matrix.multiply(this.weights.get(i), outputs[i]);
            // The inputs of the layer pass through sigmoid activation function to get the outputs
            assert layer_inputs != null;
            outputs[i + 1] = Matrix.map(layer_inputs, true, this.activation);

        }

        // Return the result as Matrix
        return outputs;

    }

    public void train(Matrix inputs, Matrix targets) {

        // Store outputs for each layer in allOutputs
        Matrix[] allOutputs = feedforwardForBackProp(inputs);

        // get the total result of the NN(the last output)
        Matrix finalOutput = allOutputs[allOutputs.length - 1];

        // Set the error of the current layer first to target - finalOutput to get the error of the output layer
        // Error is TARGET - OUTPUT
        Matrix layer_error = Matrix.subtract(targets, finalOutput);


        // Now we are starting back propagation!

        for(int i = weights.size(); i > 0; i--) {

            // Calculate the gradient by mapping the output of this layer
            Matrix gradient = Matrix.map(allOutputs[i], false, this.activation);

            // Weight by errors and learning rate
            if(i != weights.size()) {

                // Transpose me(i) - 1 <-> me weights(not sure)
                Matrix layer_weight_T = this.weights.get(i).transpose();

                // Hidden errors is output error multiplied by weights for current layer
                assert layer_error != null;
                layer_error = Matrix.multiply(layer_weight_T, layer_error);

            }

            // Multiply the error of this layer and the learning rate with the output
            gradient.multiply(layer_error);
            gradient.multiply(this.learning_rate);

            // Change in weights from me(i) - 1 --> me(i)
            Matrix hidden_outputs_T = allOutputs[i - 1].transpose();
            Matrix deltaW_output = Matrix.multiply(gradient, hidden_outputs_T);
            this.weights.get(i - 1).add(deltaW_output);

            // Add the momentum
            momentums.get(i - 1).add(deltaW_output);
            // Multiply momentum with momentum correction
            momentums.get(i - 1).multiply(this.momentum_correction);
            // Add momentum to all weights
            this.weights.get(i - 1).add(momentums.get(i - 1));

        }


    }

    void printHiddenLayers() {

        for (Matrix weight : weights) {

            weight.printForm();

        }

    }

}
