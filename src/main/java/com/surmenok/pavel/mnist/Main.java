package com.surmenok.pavel.mnist;

import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

import java.io.IOException;

public class Main {
    public static void main(final String args[]) throws IOException, InvalidFileFormatException {
        NeuralNetworkTrainer trainer = createNeuralNetworkTrainer();

        MLDataSet trainingSet = getDataSet("c:\\Stuff\\MNIST\\train-labels.idx1-ubyte", "c:\\Stuff\\MNIST\\train-images.idx3-ubyte");
        MLDataSet validationSet = getDataSet("c:\\Stuff\\MNIST\\t10k-labels.idx1-ubyte", "c:\\Stuff\\MNIST\\t10k-images.idx3-ubyte");

        trainer.setTrainingSet(trainingSet);
        trainer.setValidationSet(validationSet);

        trainer.train();

        // Use network to make predictions, or save weights somewhere
        // BasicNetwork bestNetwork = trainer.getBestNetwork();

        Encog.getInstance().shutdown();
    }

    private static NeuralNetworkTrainer createNeuralNetworkTrainer() {
        int iterationsCount = 3;
        int epochsCount = 200;

        int minHiddenLayerNeuronCount = 100;
        int maxHiddenLayerNeuronCount = 200;
        int hiddenLayerNeuronCountStep = 100;

        int minHiddenLayerCount = 1;
        int maxHiddenLayerCount = 2;

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(new ClassificationError());

        trainer.setIterationsCount(iterationsCount);
        trainer.setEpochsCount(epochsCount);
        trainer.setMinHiddenLayerNeuronCount(minHiddenLayerNeuronCount);
        trainer.setMaxHiddenLayerNeuronCount(maxHiddenLayerNeuronCount);
        trainer.setHiddenLayerNeuronCountStep(hiddenLayerNeuronCountStep);
        trainer.setMinHiddenLayerCount(minHiddenLayerCount);
        trainer.setMaxHiddenLayerCount(maxHiddenLayerCount);

        return trainer;
    }

    private static MLDataSet getDataSet(String labelsFileName, String imagesFileName) throws IOException, InvalidFileFormatException {
        MNISTReader mnistReader = new MNISTReader();

        double[][] images = mnistReader.readImages(imagesFileName);
        double[][] labels = mnistReader.readLabels(labelsFileName);

        MLDataSet dataSet = new BasicMLDataSet(images, labels);

        return dataSet;
    }
}
