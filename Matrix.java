package com.camillo;

import com.camillo.activations.Activations;
import com.camillo.network.*;

public class Matrix {

    // Dimensions of matrix
    private int rows, columns;

    // Value inside the matrix
    private double[][] data;

    public Matrix(int rows, int columns) {

        this.rows = rows;
        this.columns = columns;
        this.data = new double[rows][columns];

    }

    public Matrix(double[][] data) {

        this.rows = data.length;
        this.columns = data[0].length;
        this.data = data;

    }

    /**
     * Transpose this matrix
     *
     * @return Transposed matrix
     */
    public Matrix transpose() {

        Matrix result = new Matrix(this.columns, this.rows);

        for (int i = 0; i < result.rows; i++) {

            for (int j = 0; j < result.columns; j++) {

                result.data[i][j] = this.data[j][i];

            }

        }

        return result;

    }

    /**
     * Simple matrix, matrix multiplication
     *
     * @param m1 Matrix that will be right multiplied with this matrix
     */
    public void multiply(Matrix m1) {

        if((this.rows != m1.rows) || (this.columns != m1.columns)) {
            System.out.println("Matrix has the wrong size");
            return;
        }

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                this.data[i][j] *= m1.data[i][j];
            }
        }

    }

    /**
     * Simple scalar, matrix multiplication
     *
     * @param n Scalar to multiply the matrix with
     */
    public void multiply(double n) {

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < this.columns; j++) {

                // New value
                this.data[i][j] *= n;

            }

        }

    }

    /**
     * Adds matrix with same dimensions to this matrix
     *
     * @param m1 Matrix to add
     */
    public void add(Matrix m1) {

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < this.columns; j++) {

                this.data[i][j] += m1.data[i][j];

            }

        }

    }

    /**
     * Adds a scalar to each value of the matrix
     *
     * @param n scalar to add
     */
    void add(double n) {

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < this.columns; j++) {

                this.data[i][j] += n;

            }

        }

    }

    /**
     * Randomize matrix values
     */
    public void randomize() {

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < this.columns; j++) {

                this.data[i][j] = Math.random();

            }

        }

    }

    /**
     * Fill each value with 0
     */
    public void fillNull() {

        for (int i = 0; i < this.rows; i++) {

            for (int j = 0; j < this.columns; j++) {

                this.data[i][j] = 0;

            }

        }

    }

    /**
     * Print the matrix(value)
     */
    void print() {

        for (int i = 0; i < rows; i++) {

            String row = "";

            for (int j = 0; j < columns; j++) {

                row += "   " + this.data[i][j];

            }

            System.out.println(row);

        }

        System.out.println("-------");

    }

    /**
     * Print the matrix(dimensions)
     */
    public void printForm() {

        System.out.println(this.rows + "x" + this.columns);

    }


    /**
     * Map the matrix with given activation function
     *
     * @param m1 input matrix to map
     * @param type of mapping true for forward pass, false for backward pass
     * @param activation selected activation function
     * @return Mapped matrix
     */
    public static Matrix map(Matrix m1, boolean type, Activations activation) {

        Matrix result = new Matrix(m1.rows, m1.columns);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.columns; j++) {
                result.data[i][j] = getNewValue(m1.data[i][j], type, activation);
            }
        }
        return result;

    }

    /**
     * Multiply two matrices
     *
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Result matrix or null if not the same dimensions
     */
    public static Matrix multiply(Matrix m1, Matrix m2) {

        // Won't work if columns of A don't equal columns of B
        if (m1.columns != m2.rows) {

            System.out.println("Incompatible matrix sizes!");
            return null;

        }
        // Make a new matrix
        Matrix result = new Matrix(m1.rows, m2.columns);

        for (int i = 0; i < m1.rows; i++) {

            for (int j = 0; j < m2.columns; j++) {

                for (int k = 0; k < m1.columns; k++) {

                    result.data[i][j] += m1.data[i][k] * m2.data[k][j];

                }

            }

        }

        return result;

    }

    /**
     * Subtract second matrix from the first
     *
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Result matrix or null if not the same dimensions
     */
    public static Matrix subtract(Matrix m1, Matrix m2) {

        Matrix result = new Matrix(m1.rows, m1.columns);

        for (int i = 0; i < result.rows; i++) {

            for (int j = 0; j < result.columns; j++) {

                result.data[i][j] = m1.data[i][j] - m2.data[i][j];

            }

        }

        return result;

    }

    /**
     * Map a single value
     *
     * @param oldValue current value to apply the map (activation) function to
     * @param type of mapping true for forward pass, false for backward pass
     * @param activation selected activation function
     * @return new (mapped) value
     */
    private static double getNewValue(double oldValue, boolean type, Activations activation) {

        if(type)
            return activation.activation(oldValue);
        else
            return activation.derivative(oldValue);

    }

}
