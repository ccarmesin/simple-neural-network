package com.camillo.network;

import com.camillo.Matrix;
import com.camillo.activations.Activations;

public class NeuralNetwork implements Network {

    private Matrix weights_ih;
    private Matrix weights_ho;

    private Activations activation;

    private double learning_rate;

    public NeuralNetwork(Activations activation, int input_neurons, int hidden_neurons, int output_neurons) {

        this.activation = activation;

        this.weights_ih = new Matrix(hidden_neurons, input_neurons);
        this.weights_ho = new Matrix(output_neurons, hidden_neurons);

        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.learning_rate = 0.1;

    }

    @Override
    public Matrix test(Matrix inputs) {

        // The input to the hidden layer is the weights (wih) multiplied by inputs
        Matrix hidden_inputs = Matrix.multiply(this.weights_ih, inputs);
        // The outputs of the hidden layer pass through sigmoid activation function
        assert hidden_inputs != null;
        Matrix hidden_outputs = Matrix.map(hidden_inputs, true, this.activation);

        // The input to the output layer is the weights (who) multiplied by hidden layer
        Matrix output_inputs = Matrix.multiply(this.weights_ho, hidden_outputs);

        // The output of the network passes through sigmoid activation function
        // Return the result as an array
        assert output_inputs != null;
        return Matrix.map(output_inputs, true, this.activation);

    }

    @Override
    public void train(Matrix inputs, Matrix labels) {

        // The input to the hidden layer is the weights (wih) multiplied by inputs
        Matrix hidden_inputs = Matrix.multiply(this.weights_ih, inputs);
        // The outputs of the hidden layer pass through sigmoid activation function
        assert hidden_inputs != null;
        Matrix hidden_outputs = Matrix.map(hidden_inputs, true, this.activation);

        // The input to the output layer is the weights (who) multiplied by hidden layer
        Matrix output_inputs = Matrix.multiply(this.weights_ho, hidden_outputs);

        // The output of the network passes through sigmoid activation function
        assert output_inputs != null;
        Matrix outputs = Matrix.map(output_inputs, true, this.activation);

        // Error is LABEL - OUTPUT
        Matrix output_errors = Matrix.subtract(labels, outputs);

        // Now we are starting back propogation!

        // Transpose hidden <-> output weights
        Matrix whoT = this.weights_ho.transpose();

        // Hidden errors is output error multiplied by weights (who)
        Matrix hidden_errors = Matrix.multiply(whoT, output_errors);

        // Calculate the gradient
        Matrix gradient_output = Matrix.map(outputs, false, this.activation);

        // Weight by errors and learning rate
        gradient_output.multiply(output_errors);
        gradient_output.multiply(this.learning_rate);

        // Change in weights from HIDDEN --> OUTPUT
        Matrix hidden_outputs_T = hidden_outputs.transpose();
        Matrix deltaW_output = Matrix.multiply(gradient_output, hidden_outputs_T);
        this.weights_ho.add(deltaW_output);

        // Gradients for next layer, more back propagation!

        // Calculate the gradient
        Matrix gradient_hidden = Matrix.map(hidden_outputs, false, this.activation);

        // Weight by errors and learning rate
        gradient_hidden.multiply(hidden_errors);
        gradient_hidden.multiply(this.learning_rate);

        // Change in weights from INPUT --> HIDDEN
        Matrix inputs_T = inputs.transpose();
        Matrix deltaW_hidden = Matrix.multiply(gradient_hidden, inputs_T);
        this.weights_ih.add(deltaW_hidden);

    }
}
