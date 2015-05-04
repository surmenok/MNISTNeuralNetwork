package com.surmenok.pavel.mnist;

public interface INeuralNetworkError {
    double getError(double[] outputData, double[] idealData);
}
