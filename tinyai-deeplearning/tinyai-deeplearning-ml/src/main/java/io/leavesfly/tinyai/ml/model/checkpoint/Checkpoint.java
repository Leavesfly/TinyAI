package io.leavesfly.tinyai.ml.model.checkpoint;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * 训练检查点类，用于保存和加载完整训练状态，支持断点续训。
 * 类似 PyTorch 的 checkpoint 机制，可保存：
 * - 模型参数
 * - 优化器状态
 * - 训练epoch
 * - 学习率
 * - 损失值
 * - 其他自定义元数据
 *
 * @author TinyDL
 * @version 2.0
 */
public class Checkpoint implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 模型状态字典
     */
    private Map<String, NdArray> modelStateDict;

    /**
     * 优化器状态字典
     */
    private Map<String, Object> optimizerStateDict;

    /**
     * 当前epoch
     */
    private int epoch;

    /**
     * 当前学习率
     */
    private float learningRate;

    /**
     * 最佳损失值
     */
    private float bestLoss;

    /**
     * 其他元数据
     */
    private Map<String, Object> metadata;

    /**
     * 构造函数
     */
    public Checkpoint() {
        this.modelStateDict = new HashMap<>();
        this.optimizerStateDict = new HashMap<>();
        this.metadata = new HashMap<>();
        this.epoch = 0;
        this.learningRate = 0.0f;
        this.bestLoss = Float.MAX_VALUE;
    }

    /**
     * 构建器类 - 支持流畅的API调用
     */
    public static class Builder {
        private final Checkpoint checkpoint;

        public Builder() {
            this.checkpoint = new Checkpoint();
        }

        /**
         * 设置模型状态
         *
         * @param model 模型对象
         * @return Builder实例
         */
        public Builder model(Model model) {
            if (model != null && model.getModule() != null) {
                checkpoint.modelStateDict = model.getModule().stateDict();
            }
            return this;
        }

        /**
         * 设置模型状态字典
         *
         * @param stateDict 状态字典
         * @return Builder实例
         */
        public Builder modelStateDict(Map<String, NdArray> stateDict) {
            checkpoint.modelStateDict = stateDict;
            return this;
        }

        /**
         * 设置优化器状态
         *
         * @param optimizer 优化器对象
         * @return Builder实例
         */
        public Builder optimizer(Optimizer optimizer) {
            if (optimizer != null) {
                checkpoint.optimizerStateDict = optimizer.state_dict();
            }
            return this;
        }

        /**
         * 设置优化器状态字典
         *
         * @param stateDict 状态字典
         * @return Builder实例
         */
        public Builder optimizerStateDict(Map<String, Object> stateDict) {
            checkpoint.optimizerStateDict = stateDict;
            return this;
        }

        /**
         * 设置epoch
         *
         * @param epoch 当前epoch
         * @return Builder实例
         */
        public Builder epoch(int epoch) {
            checkpoint.epoch = epoch;
            return this;
        }

        /**
         * 设置学习率
         *
         * @param lr 学习率
         * @return Builder实例
         */
        public Builder learningRate(float lr) {
            checkpoint.learningRate = lr;
            return this;
        }

        /**
         * 设置最佳损失
         *
         * @param loss 损失值
         * @return Builder实例
         */
        public Builder bestLoss(float loss) {
            checkpoint.bestLoss = loss;
            return this;
        }

        /**
         * 添加元数据
         *
         * @param key   键
         * @param value 值
         * @return Builder实例
         */
        public Builder metadata(String key, Object value) {
            checkpoint.metadata.put(key, value);
            return this;
        }

        /**
         * 构建Checkpoint对象
         *
         * @return Checkpoint实例
         */
        public Checkpoint build() {
            return checkpoint;
        }
    }

    /**
     * 保存检查点到文件
     *
     * @param filePath 文件路径
     * @throws IOException 保存失败时抛出
     */
    public void save(String filePath) throws IOException {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(this);
        }
    }

    /**
     * 从文件加载检查点
     *
     * @param filePath 文件路径
     * @return Checkpoint实例
     * @throws IOException            读取失败时抛出
     * @throws ClassNotFoundException 反序列化失败时抛出
     */
    public static Checkpoint load(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (Checkpoint) ois.readObject();
        }
    }

    /**
     * 恢复模型状态
     *
     * @param model  模型对象
     * @param strict 是否严格匹配
     */
    public void restoreModel(Model model, boolean strict) {
        if (modelStateDict != null && !modelStateDict.isEmpty() && model.getModule() != null) {
            model.getModule().loadStateDict(modelStateDict, strict);
        }
    }

    /**
     * 恢复模型状态(默认严格匹配)
     *
     * @param model 模型对象
     */
    public void restoreModel(Model model) {
        restoreModel(model, true);
    }

    /**
     * 恢复优化器状态
     *
     * @param optimizer 优化器对象
     */
    public void restoreOptimizer(Optimizer optimizer) {
        if (optimizerStateDict != null && !optimizerStateDict.isEmpty()) {
            optimizer.load_state_dict(optimizerStateDict);
        }
    }

    // ==================== Getter和Setter ====================

    public Map<String, NdArray> getModelStateDict() {
        return modelStateDict;
    }

    public void setModelStateDict(Map<String, NdArray> modelStateDict) {
        this.modelStateDict = modelStateDict;
    }

    public Map<String, Object> getOptimizerStateDict() {
        return optimizerStateDict;
    }

    public void setOptimizerStateDict(Map<String, Object> optimizerStateDict) {
        this.optimizerStateDict = optimizerStateDict;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getBestLoss() {
        return bestLoss;
    }

    public void setBestLoss(float bestLoss) {
        this.bestLoss = bestLoss;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    /**
     * 获取元数据
     *
     * @param key 键
     * @return 值
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }

    /**
     * 设置元数据
     *
     * @param key   键
     * @param value 值
     */
    public void putMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    @Override
    public String toString() {
        return "Checkpoint{" +
                "epoch=" + epoch +
                ", learningRate=" + learningRate +
                ", bestLoss=" + bestLoss +
                ", modelParams=" + (modelStateDict != null ? modelStateDict.size() : 0) +
                ", optimizerState=" + (optimizerStateDict != null ? "present" : "null") +
                ", metadata=" + metadata.keySet() +
                '}';
    }
}
