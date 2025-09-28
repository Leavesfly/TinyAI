package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.ndarr.Shape;

import java.io.Serializable;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * 模型元数据信息类
 * <p>
 * 该类用于存储和管理模型的完整元数据信息，包括：
 * 1. 基本信息：模型名称、版本、创建时间等
 * 2. 架构信息：输入输出形状、参数数量、层类型统计等
 * 3. 训练信息：训练轮次、损失值、优化器配置等
 * 4. 性能信息：评估指标、训练时间等
 * 5. 自定义属性：用户自定义的扩展信息
 *
 * @author TinyDL
 * @version 1.0
 */
public class ModelInfo implements Serializable {

    private static final long serialVersionUID = 1L;

    // 基本信息
    private String modelName;
    private String modelVersion;
    private String frameworkVersion;
    private Date createdTime;
    private Date lastModifiedTime;
    private String description;

    // 架构信息
    private String architectureType;  // 如 "MLP", "CNN", "RNN", "Transformer"
    private Shape inputShape;
    private Shape outputShape;
    private int totalLayers;
    private long totalParameters;
    private Map<String, Integer> layerCounts;  // 各类型层的数量统计

    // 训练信息
    private int trainedEpochs;
    private double finalLoss;
    private double bestLoss;
    private String optimizerType;
    private double learningRate;
    private int batchSize;
    private String lossFunction;

    // 性能信息
    private Map<String, Double> metrics;  // 如 accuracy, precision, recall等
    private long trainingTimeMs;
    private String hardwareInfo;

    // 自定义属性
    private Map<String, Object> customProperties;

    /**
     * 默认构造函数
     */
    public ModelInfo() {
        this.createdTime = new Date();
        this.lastModifiedTime = new Date();
        this.frameworkVersion = "TinyDL-0.01";
        this.layerCounts = new HashMap<>();
        this.metrics = new HashMap<>();
        this.customProperties = new HashMap<>();
    }

    /**
     * 带模型名称的构造函数
     *
     * @param modelName 模型名称
     */
    public ModelInfo(String modelName) {
        this();
        this.modelName = modelName;
    }

    // Getter and Setter methods

    /**
     * 获取模型名称
     *
     * @return 模型名称
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * 设置模型名称
     *
     * @param modelName 模型名称
     */
    public void setModelName(String modelName) {
        this.modelName = modelName;
        updateLastModifiedTime();
    }

    /**
     * 获取模型版本
     *
     * @return 模型版本
     */
    public String getModelVersion() {
        return modelVersion;
    }

    /**
     * 设置模型版本
     *
     * @param modelVersion 模型版本
     */
    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
        updateLastModifiedTime();
    }

    /**
     * 获取框架版本
     *
     * @return 框架版本
     */
    public String getFrameworkVersion() {
        return frameworkVersion;
    }

    /**
     * 设置框架版本
     *
     * @param frameworkVersion 框架版本
     */
    public void setFrameworkVersion(String frameworkVersion) {
        this.frameworkVersion = frameworkVersion;
        updateLastModifiedTime();
    }

    /**
     * 获取创建时间
     *
     * @return 创建时间
     */
    public Date getCreatedTime() {
        return createdTime;
    }

    /**
     * 设置创建时间
     *
     * @param createdTime 创建时间
     */
    public void setCreatedTime(Date createdTime) {
        this.createdTime = createdTime;
    }

    /**
     * 获取最后修改时间
     *
     * @return 最后修改时间
     */
    public Date getLastModifiedTime() {
        return lastModifiedTime;
    }

    /**
     * 设置最后修改时间
     *
     * @param lastModifiedTime 最后修改时间
     */
    public void setLastModifiedTime(Date lastModifiedTime) {
        this.lastModifiedTime = lastModifiedTime;
    }

    /**
     * 获取模型描述
     *
     * @return 模型描述
     */
    public String getDescription() {
        return description;
    }

    /**
     * 设置模型描述
     *
     * @param description 模型描述
     */
    public void setDescription(String description) {
        this.description = description;
        updateLastModifiedTime();
    }

    /**
     * 获取架构类型
     *
     * @return 架构类型
     */
    public String getArchitectureType() {
        return architectureType;
    }

    /**
     * 设置架构类型
     *
     * @param architectureType 架构类型
     */
    public void setArchitectureType(String architectureType) {
        this.architectureType = architectureType;
        updateLastModifiedTime();
    }

    /**
     * 获取输入形状
     *
     * @return 输入形状
     */
    public Shape getInputShape() {
        return inputShape;
    }

    /**
     * 设置输入形状
     *
     * @param inputShape 输入形状
     */
    public void setInputShape(Shape inputShape) {
        this.inputShape = inputShape;
        updateLastModifiedTime();
    }

    /**
     * 获取输出形状
     *
     * @return 输出形状
     */
    public Shape getOutputShape() {
        return outputShape;
    }

    /**
     * 设置输出形状
     *
     * @param outputShape 输出形状
     */
    public void setOutputShape(Shape outputShape) {
        this.outputShape = outputShape;
        updateLastModifiedTime();
    }

    /**
     * 获取总层数
     *
     * @return 总层数
     */
    public int getTotalLayers() {
        return totalLayers;
    }

    /**
     * 设置总层数
     *
     * @param totalLayers 总层数
     */
    public void setTotalLayers(int totalLayers) {
        this.totalLayers = totalLayers;
        updateLastModifiedTime();
    }

    /**
     * 获取总参数数量
     *
     * @return 总参数数量
     */
    public long getTotalParameters() {
        return totalParameters;
    }

    /**
     * 设置总参数数量
     *
     * @param totalParameters 总参数数量
     */
    public void setTotalParameters(long totalParameters) {
        this.totalParameters = totalParameters;
        updateLastModifiedTime();
    }

    /**
     * 获取层类型统计
     *
     * @return 层类型统计映射
     */
    public Map<String, Integer> getLayerCounts() {
        return layerCounts;
    }

    /**
     * 设置层类型统计
     *
     * @param layerCounts 层类型统计映射
     */
    public void setLayerCounts(Map<String, Integer> layerCounts) {
        this.layerCounts = layerCounts;
        updateLastModifiedTime();
    }

    /**
     * 添加层类型统计
     *
     * @param layerType 层类型
     * @param count     数量
     */
    public void addLayerCount(String layerType, int count) {
        this.layerCounts.put(layerType, count);
        updateLastModifiedTime();
    }

    /**
     * 获取训练轮次
     *
     * @return 训练轮次
     */
    public int getTrainedEpochs() {
        return trainedEpochs;
    }

    /**
     * 设置训练轮次
     *
     * @param trainedEpochs 训练轮次
     */
    public void setTrainedEpochs(int trainedEpochs) {
        this.trainedEpochs = trainedEpochs;
        updateLastModifiedTime();
    }

    /**
     * 获取最终损失值
     *
     * @return 最终损失值
     */
    public double getFinalLoss() {
        return finalLoss;
    }

    /**
     * 设置最终损失值
     *
     * @param finalLoss 最终损失值
     */
    public void setFinalLoss(double finalLoss) {
        this.finalLoss = finalLoss;
        updateLastModifiedTime();
    }

    /**
     * 获取最佳损失值
     *
     * @return 最佳损失值
     */
    public double getBestLoss() {
        return bestLoss;
    }

    /**
     * 设置最佳损失值
     *
     * @param bestLoss 最佳损失值
     */
    public void setBestLoss(double bestLoss) {
        this.bestLoss = bestLoss;
        updateLastModifiedTime();
    }

    /**
     * 获取优化器类型
     *
     * @return 优化器类型
     */
    public String getOptimizerType() {
        return optimizerType;
    }

    /**
     * 设置优化器类型
     *
     * @param optimizerType 优化器类型
     */
    public void setOptimizerType(String optimizerType) {
        this.optimizerType = optimizerType;
        updateLastModifiedTime();
    }

    /**
     * 获取学习率
     *
     * @return 学习率
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * 设置学习率
     *
     * @param learningRate 学习率
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        updateLastModifiedTime();
    }

    /**
     * 获取批次大小
     *
     * @return 批次大小
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * 设置批次大小
     *
     * @param batchSize 批次大小
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        updateLastModifiedTime();
    }

    /**
     * 获取损失函数
     *
     * @return 损失函数
     */
    public String getLossFunction() {
        return lossFunction;
    }

    /**
     * 设置损失函数
     *
     * @param lossFunction 损失函数
     */
    public void setLossFunction(String lossFunction) {
        this.lossFunction = lossFunction;
        updateLastModifiedTime();
    }

    /**
     * 获取性能指标
     *
     * @return 性能指标映射
     */
    public Map<String, Double> getMetrics() {
        return metrics;
    }

    /**
     * 设置性能指标
     *
     * @param metrics 性能指标映射
     */
    public void setMetrics(Map<String, Double> metrics) {
        this.metrics = metrics;
        updateLastModifiedTime();
    }

    /**
     * 添加性能指标
     *
     * @param metricName 指标名称
     * @param value      指标值
     */
    public void addMetric(String metricName, double value) {
        this.metrics.put(metricName, value);
        updateLastModifiedTime();
    }

    /**
     * 获取训练时间（毫秒）
     *
     * @return 训练时间（毫秒）
     */
    public long getTrainingTimeMs() {
        return trainingTimeMs;
    }

    /**
     * 设置训练时间（毫秒）
     *
     * @param trainingTimeMs 训练时间（毫秒）
     */
    public void setTrainingTimeMs(long trainingTimeMs) {
        this.trainingTimeMs = trainingTimeMs;
        updateLastModifiedTime();
    }

    /**
     * 获取硬件信息
     *
     * @return 硬件信息
     */
    public String getHardwareInfo() {
        return hardwareInfo;
    }

    /**
     * 设置硬件信息
     *
     * @param hardwareInfo 硬件信息
     */
    public void setHardwareInfo(String hardwareInfo) {
        this.hardwareInfo = hardwareInfo;
        updateLastModifiedTime();
    }

    /**
     * 获取自定义属性
     *
     * @return 自定义属性映射
     */
    public Map<String, Object> getCustomProperties() {
        return customProperties;
    }

    /**
     * 设置自定义属性
     *
     * @param customProperties 自定义属性映射
     */
    public void setCustomProperties(Map<String, Object> customProperties) {
        this.customProperties = customProperties;
        updateLastModifiedTime();
    }

    /**
     * 添加自定义属性
     *
     * @param key   属性键
     * @param value 属性值
     */
    public void addCustomProperty(String key, Object value) {
        this.customProperties.put(key, value);
        updateLastModifiedTime();
    }

    // 工具方法

    /**
     * 更新最后修改时间
     */
    private void updateLastModifiedTime() {
        this.lastModifiedTime = new Date();
    }

    /**
     * 获取模型的简要信息
     *
     * @return 简要信息字符串
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Model: ").append(modelName != null ? modelName : "Unknown").append("\n");
        sb.append("Architecture: ").append(architectureType != null ? architectureType : "Unknown").append("\n");
        sb.append("Input Shape: ").append(inputShape != null ? inputShape.toString() : "Unknown").append("\n");
        sb.append("Output Shape: ").append(outputShape != null ? outputShape.toString() : "Unknown").append("\n");
        sb.append("Total Layers: ").append(totalLayers).append("\n");
        sb.append("Total Parameters: ").append(totalParameters).append("\n");
        if (trainedEpochs > 0) {
            sb.append("Trained Epochs: ").append(trainedEpochs).append("\n");
            sb.append("Final Loss: ").append(String.format("%.6f", finalLoss)).append("\n");
        }
        return sb.toString();
    }

    /**
     * 获取详细信息
     *
     * @return 详细信息字符串
     */
    public String getDetailedInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 模型详细信息 ===\n");
        sb.append("名称: ").append(modelName != null ? modelName : "未知").append("\n");
        sb.append("版本: ").append(modelVersion != null ? modelVersion : "未知").append("\n");
        sb.append("框架版本: ").append(frameworkVersion).append("\n");
        sb.append("创建时间: ").append(createdTime).append("\n");
        sb.append("最后修改: ").append(lastModifiedTime).append("\n");
        if (description != null) {
            sb.append("描述: ").append(description).append("\n");
        }

        sb.append("\n=== 架构信息 ===\n");
        sb.append("架构类型: ").append(architectureType != null ? architectureType : "未知").append("\n");
        sb.append("输入形状: ").append(inputShape != null ? inputShape.toString() : "未知").append("\n");
        sb.append("输出形状: ").append(outputShape != null ? outputShape.toString() : "未知").append("\n");
        sb.append("总层数: ").append(totalLayers).append("\n");
        sb.append("总参数量: ").append(totalParameters).append("\n");

        if (!layerCounts.isEmpty()) {
            sb.append("层类型统计:\n");
            for (Map.Entry<String, Integer> entry : layerCounts.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }

        if (trainedEpochs > 0) {
            sb.append("\n=== 训练信息 ===\n");
            sb.append("训练轮次: ").append(trainedEpochs).append("\n");
            sb.append("最终损失: ").append(String.format("%.6f", finalLoss)).append("\n");
            sb.append("最佳损失: ").append(String.format("%.6f", bestLoss)).append("\n");
            if (optimizerType != null) {
                sb.append("优化器: ").append(optimizerType).append("\n");
            }
            if (learningRate > 0) {
                sb.append("学习率: ").append(learningRate).append("\n");
            }
            if (batchSize > 0) {
                sb.append("批次大小: ").append(batchSize).append("\n");
            }
            if (lossFunction != null) {
                sb.append("损失函数: ").append(lossFunction).append("\n");
            }
            if (trainingTimeMs > 0) {
                sb.append("训练时间: ").append(trainingTimeMs / 1000.0).append(" 秒\n");
            }
        }

        if (!metrics.isEmpty()) {
            sb.append("\n=== 性能指标 ===\n");
            for (Map.Entry<String, Double> entry : metrics.entrySet()) {
                sb.append(entry.getKey()).append(": ").append(String.format("%.4f", entry.getValue())).append("\n");
            }
        }

        if (!customProperties.isEmpty()) {
            sb.append("\n=== 自定义属性 ===\n");
            for (Map.Entry<String, Object> entry : customProperties.entrySet()) {
                sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return getSummary();
    }
}