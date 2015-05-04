package com.surmenok.pavel.mnist;

import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;

public class PropagationFactory {
    public Propagation create(String trainingType, BasicNetwork network, MLDataSet trainingSet) {
        final Propagation train;
        if (trainingType.equals("ResilientPropagation")) {
            train = new ResilientPropagation(network, trainingSet);
        }
        else if (trainingType.equals("Backpropagation")) {
            train = new Backpropagation(network, trainingSet);
        }
        else if (trainingType.equals("QuickPropagation")) {
            train = new QuickPropagation(network, trainingSet);
        }
        else if (trainingType.equals("ScaledConjugateGradient")) {
            train = new ScaledConjugateGradient(network, trainingSet);
        }
        else if (trainingType.equals("ManhattanPropagation")) {
            train = new ManhattanPropagation(network, trainingSet, 0.1);
        }
        else {
            train = null;
        }
        return train;
    }
}
