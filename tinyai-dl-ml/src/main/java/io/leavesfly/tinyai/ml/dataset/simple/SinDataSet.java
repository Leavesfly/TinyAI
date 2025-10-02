package io.leavesfly.tinyai.ml.dataset.simple;

import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;

public class SinDataSet extends ArrayDataset {
    public SinDataSet(int batchSize) {
        super(batchSize);
    }

    @Override
    protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
        ArrayDataset dataSet = new SinDataSet(batchSize);
        dataSet.setXs(_xs);
        dataSet.setYs(_ys);
        return dataSet;
    }

    @Override
    public void doPrepare() {
        int size = 1000;
        int tmpSize = size + 1;
        NdArray tmp = NdArray.linSpace(0f, (float) (Math.PI * 4), tmpSize);

        //训练数据构造
        NdArray[] tmpArray = new NdArray[tmpSize];
        for (int i = 0; i < tmpSize; i++) {
            tmpArray[i] = NdArray.of((float) Math.sin(tmp.getMatrix()[0][i]));
        }
        NdArray[] _xs = Arrays.copyOfRange(tmpArray, 0, size);
        NdArray[] _ys = Arrays.copyOfRange(tmpArray, 1, size + 1);
        DataSet trainDataset = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TRAIN.name(), trainDataset);

        //测试数据构造
        tmpArray = new NdArray[tmpSize];
        for (int i = 0; i < tmpSize; i++) {
            tmpArray[i] = NdArray.of((float) Math.cos(tmp.getMatrix()[0][i]));
        }
        _xs = Arrays.copyOfRange(tmpArray, 0, size);
        _ys = Arrays.copyOfRange(tmpArray, 1, size + 1);
        DataSet testDataset = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TEST.name(), testDataset);
    }
}