package com.surmenok.pavel.mnist;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTReader {
    public double[][] readImages(String fileName) throws IOException, InvalidFileFormatException {
        try(DataInputStream stream = new DataInputStream(new FileInputStream(fileName))) {
            int magicNumber = stream.readInt();
            if (magicNumber != 2051) {
                throw new InvalidFileFormatException("Wrong magic number: " + magicNumber + "; expected: 2051");
            }

            int imageCount = stream.readInt();
            int rowCount = stream.readInt();
            int columnCount = stream.readInt();

            double[][] allImages = new double[imageCount][rowCount * columnCount];

            for (int i = 0; i < imageCount; i++) {
                if (stream.available() <= 0) {
                    throw new InvalidFileFormatException("The file contains less than " + imageCount + " images");
                }

                for (int column = 0; column < columnCount; column++) {
                    for (int row = 0; row < rowCount; row++) {
                        allImages[i][column * rowCount + row] = stream.readUnsignedByte();
                    }
                }
            }

            return allImages;
        }
    }

    public double[][] readLabels(String fileName) throws IOException, InvalidFileFormatException {
        try(DataInputStream stream = new DataInputStream(new FileInputStream(fileName))) {

            int magicNumber = stream.readInt();
            if (magicNumber != 2049) {
                throw new InvalidFileFormatException("Wrong magic number: " + magicNumber + "; expected: 2049");
            }

            int labelCount = stream.readInt();

            double[][] allLabels = new double[labelCount][10];

            for (int i = 0; i < labelCount; i++) {
                if (stream.available() <= 0) {
                    throw new InvalidFileFormatException("The file contains less than " + labelCount + " images");
                }

                byte label = stream.readByte();
                allLabels[i][label % 10] = 1;
            }

            return allLabels;
        }
    }
}
