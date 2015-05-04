package com.surmenok.pavel.mnist;

public class ClassificationError implements INeuralNetworkError {
    @Override
    public double getError(double[] outputData, double[] idealData) {
        int maxIndex = 0;
        double maxValue = outputData[0];

        for(int j = 1; j < outputData.length; j++) {
            if(maxValue < outputData[j]) {
                maxIndex = j;
                maxValue = outputData[j];
            }
        }

        if(idealData[maxIndex] != 1) {
            return 1;
        }

        return 0;
    }
}
