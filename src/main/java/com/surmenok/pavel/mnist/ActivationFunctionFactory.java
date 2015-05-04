package com.surmenok.pavel.mnist;

import org.encog.engine.network.activation.*;

public class ActivationFunctionFactory {
    public ActivationFunction create(String activationType) {
        final ActivationFunction activationFunction;

        if (activationType.equals("BiPolar")) {
            activationFunction = new ActivationBiPolar();
        }
        else if (activationType.equals("BipolarSteepenedSigmoid")) {
            activationFunction = new ActivationBipolarSteepenedSigmoid();
        }
        else if (activationType.equals("ClippedLinear")) {
            activationFunction = new ActivationClippedLinear();
        }
        else if (activationType.equals("Competitive")) {
            activationFunction = new ActivationCompetitive();
        }
        else if (activationType.equals("Elliott")) {
            activationFunction = new ActivationElliott();
        }
        else if (activationType.equals("ElliottSymmetric")) {
            activationFunction = new ActivationElliottSymmetric();
        }
        else if (activationType.equals("Gaussian")) {
            activationFunction = new ActivationGaussian();
        }
        else if (activationType.equals("Linear")) {
            activationFunction = new ActivationLinear();
        }
        else if (activationType.equals("LOG")) {
            activationFunction = new ActivationLOG();
        }
        else if (activationType.equals("Ramp")) {
            activationFunction = new ActivationRamp();
        }
        else if (activationType.equals("Sigmoid")) {
            activationFunction = new ActivationSigmoid();
        }
        else if (activationType.equals("SIN")) {
            activationFunction = new ActivationSIN();
        }
        else if (activationType.equals("SoftMax")) {
            activationFunction = new ActivationSoftMax();
        }
        else if (activationType.equals("SteepenedSigmoid")) {
            activationFunction = new ActivationSteepenedSigmoid();
        }
        else if (activationType.equals("Step")) {
            activationFunction = new ActivationStep();
        }
        else if (activationType.equals("TANH")) {
            activationFunction = new ActivationTANH();
        }
        else {
            activationFunction = null;
        }
        return activationFunction;
    }
}
