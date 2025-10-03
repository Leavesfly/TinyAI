package io.leavesfly.tinyai.ml.dataset.simple;

import io.leavesfly.tinyai.util.Utils;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Random;

public class SpiralDateSet extends ArrayDataset {
    public SpiralDateSet(int batchSize) {
        super(batchSize);
    }

    public SpiralDateSet(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        super(batchSize);
        xs = _xs;
        ys = _ys;
    }

    @Override
    public void doPrepare() {

        int num_data = 100;
        int num_class = 3;
        int input_dim = 2;
        int data_size = num_class * num_data;

        xs = new NdArray[data_size];
        ys = new NdArray[data_size];
        Random random = new Random(0);

        for (int j = 0; j < num_class; j++) {
            for (int i = 0; i < num_data; i++) {
                float rate = i / (float) num_data;
                float radius = (float) (1.0 * rate);
                float theta = (float) (j * 4.0 + 4.0 * rate + random.nextGaussian() * 0.2);

                int index = j * num_data + i;
                float[] x = new float[input_dim];
                x[0] = (float) (radius * Math.sin(theta));
                x[1] = (float) (radius * Math.cos(theta));
                xs[index] = NdArray.of(x);
                ys[index] = NdArray.of((float) j);
            }
        }

        DataSet trainDataset = build(batchSize, xs, ys);
        splitDatasetMap.put(Usage.TRAIN.name(), trainDataset);

        DataSet testDataset = build(batchSize, xs, ys);
        splitDatasetMap.put(Usage.TEST.name(), testDataset);
    }


    @Override
    protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        ArrayDataset dataSet = new SpiralDateSet(batchSize);
        dataSet.setXs(_xs);
        dataSet.setYs(_ys);
        return dataSet;
    }

    public static SpiralDateSet toSpiralDateSet(Variable x, Variable y) {
        int size = x.getValue().getShape().getRow();
        NdArray[] xs = new NdArray[size];
        NdArray[] ys = new NdArray[size];

        float[][] x_mat = x.getValue().getMatrix();
        float[][] y_mat = y.getValue().getMatrix();

        for (int i = 0; i < size; i++) {
            xs[i] = NdArray.of(x_mat[i]);
            ys[i] = NdArray.of(Utils.argMax(y_mat[i]));
        }
        return new SpiralDateSet(100, xs, ys);
    }
}
