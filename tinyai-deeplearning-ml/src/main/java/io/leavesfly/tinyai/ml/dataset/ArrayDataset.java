package io.leavesfly.tinyai.ml.dataset;

import io.leavesfly.tinyai.util.Utils;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * 能全量装载到内存的DataSet的简单实现
 * <p>
 * 该类是DataSet的抽象实现，适用于能够完全加载到内存中的数据集。
 * 提供了批次处理、数据分割和数据打乱等基本功能。
 *
 * @author TinyDL
 * @version 1.0
 */
public abstract class ArrayDataset extends DataSet {

    protected NdArray[] xs;
    protected NdArray[] ys;

    /**
     * 构造函数
     *
     * @param batchSize 批次大小
     */
    public ArrayDataset(int batchSize) {
        super(batchSize);
    }


    @Override
    public List<Batch> getBatches() {

        int size = xs.length;
        NdArray[][] ndArrayXs = new NdArray[size / batchSize][batchSize];
        NdArray[][] ndArrayYs = new NdArray[size / batchSize][batchSize];

        List<Batch> batches = new ArrayList<>();
        for (int i = 0; i < ndArrayXs.length; i++) {
            for (int j = 0; j < ndArrayXs[0].length; j++) {
                ndArrayXs[i][j] = xs[i * batchSize + j];
                ndArrayYs[i][j] = ys[i * batchSize + j];
            }
        }
        for (int i = 0; i < size / batchSize; i++) {
            batches.add(new Batch(ndArrayXs[i], ndArrayYs[i]));
        }
        return batches;
    }

    @Override
    public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation) {

        if (trainRatio + testRatio + validaRation != 1.0f) {
            throw new RuntimeException("splitDataset parameters error! ");
        }

        if (!Objects.isNull(splitDatasetMap) && !splitDatasetMap.isEmpty()) {
            return splitDatasetMap;
        }

        int start = 0;
        int end = (int) (xs.length * trainRatio);
        NdArray[] _xs = Arrays.copyOfRange(xs, start, end);
        NdArray[] _ys = Arrays.copyOfRange(ys, start, end);
        DataSet dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TRAIN.name(), dataSet);

        start = end;
        end = start + (int) (xs.length * testRatio);
        _xs = Arrays.copyOfRange(xs, start, end);
        _ys = Arrays.copyOfRange(ys, start, end);
        dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.TEST.name(), dataSet);

        start = end;
        end = xs.length; // 使用剩余的所有数据作为验证集
        _xs = Arrays.copyOfRange(xs, start, end);
        _ys = Arrays.copyOfRange(ys, start, end);
        dataSet = build(batchSize, _xs, _ys);
        splitDatasetMap.put(Usage.VALIDATION.name(), dataSet);

        return splitDatasetMap;
    }

    @Override
    public void shuffle() {
        int size = xs.length;
        Integer[] shuffleIndex = Utils.getSeqIndex(size);
        Collections.shuffle(Arrays.asList(shuffleIndex));

        NdArray[] _xs = new NdArray[size];
        NdArray[] _ys = new NdArray[size];
        for (int i = 0; i < size; i++) {
            _xs[i] = xs[shuffleIndex[i]];
            _ys[i] = ys[shuffleIndex[i]];
        }
        xs = _xs;
        ys = _ys;
    }

    /**
     * 构建数据集
     *
     * @param batchSize 批次大小
     * @param xs        输入数据数组
     * @param ys        输出数据数组
     * @return 数据集实例
     */
    protected abstract DataSet build(int batchSize, NdArray[] xs, NdArray[] ys);

    @Override
    public int getSize() {
        return xs.length;
    }


    /**
     * 获取输入数据数组
     *
     * @return 输入数据数组
     */
    public NdArray[] getXs() {
        return xs;
    }

    /**
     * 获取输出数据数组
     *
     * @return 输出数据数组
     */
    public NdArray[] getYs() {
        return ys;
    }

    /**
     * 设置输入数据数组
     *
     * @param xs 输入数据数组
     */
    public void setXs(NdArray[] xs) {
        this.xs = xs;
    }

    /**
     * 设置输出数据数组
     *
     * @param ys 输出数据数组
     */
    public void setYs(NdArray[] ys) {
        this.ys = ys;
    }
}