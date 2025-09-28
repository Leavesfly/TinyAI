package io.leavesfly.tinyai.ml.dataset;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 表示机器学习中的用于训练、测试或验证的数据集抽象类
 * 
 * 该类是所有数据集实现的基类，定义了数据集的基本操作接口：
 * 1. 批次数据的获取
 * 2. 数据集的准备和预处理
 * 3. 数据的随机打乱
 * 4. 数据集的分割（训练/测试/验证）
 * 
 * @author TinyDL
 * @version 1.0
 */
public abstract class DataSet {

    protected int batchSize;

    protected Map<String, DataSet> splitDatasetMap = new HashMap<>();

    private boolean hadPrepared = false;

    /**
     * 构造函数
     * @param batchSize 批次大小
     */
    public DataSet(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * 获取批次数据列表
     * @return 批次数据列表
     */
    public abstract List<Batch> getBatches();


    /**
     * 准备数据集
     * 该方法确保数据集只被准备一次
     */
    public void prepare() {
        if (!hadPrepared) {
            doPrepare();
        }
        hadPrepared = true;
    }

    /**
     * 执行数据集准备操作
     * 子类需要实现具体的数据准备逻辑
     */
    public abstract void doPrepare();

    /**
     * 将数据随机打散
     */
    public abstract void shuffle();

    /**
     * 将数据随机拆分成训练、测试和验证集三部分
     * @param trainRatio 训练集比例
     * @param testRatio 测试集比例
     * @param validaRation 验证集比例
     * @return 分割后的数据集映射
     */
    public abstract Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validaRation);

    /**
     * 获取训练数据集
     * @return 训练数据集
     */
    public DataSet getTrainDataSet() {
        return splitDatasetMap.get(Usage.TRAIN.name());
    }

    /**
     * 获取测试数据集
     * @return 测试数据集
     */
    public DataSet getTestDataSet() {
        return splitDatasetMap.get(Usage.TEST.name());

    }

    /**
     * 获取验证数据集
     * @return 验证数据集
     */
    public DataSet getValidationDataSet() {
        return splitDatasetMap.get(Usage.VALIDATION.name());
    }

    /**
     * 获取数据集大小
     * @return 数据集大小
     */
    public abstract int getSize();

    /**
     * 数据集用途枚举
     */
    public static enum Usage {
        /**
         * 训练集
         */
        TRAIN, 
        
        /**
         * 测试集
         */
        TEST, 
        
        /**
         * 验证集
         */
        VALIDATION
    }
}