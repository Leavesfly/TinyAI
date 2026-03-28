package io.leavesfly.tinyai.deepseek.base;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek 训练器抽象基类
 *
 * 参考 MiniMind 的 BaseTrainer 设计，提取 V3 和 R1 各训练器的公共逻辑：
 * 1. 训练配置管理（epoch、梯度裁剪阈值、日志间隔、检查点目录）
 * 2. 训练状态跟踪（当前 epoch、全局步数、损失历史）
 * 3. 梯度裁剪（全局梯度范数裁剪，防止梯度爆炸）
 * 4. 检查点保存与目录管理
 * 5. 损失历史统计工具方法
 *
 * 子类需实现具体的训练流程（{@link #train()}）和训练器标识（{@link #getTrainerName()}、{@link #getCheckpointPrefix()}）。
 *
 * @author leavesfly
 * @version 1.0
 */
public abstract class DeepSeekTrainerBase {

    protected final DeepSeekModelBase model;

    // ==================== 训练配置 ====================
    protected int maxEpochs;
    protected float maxGradNorm;
    protected int logInterval;
    protected String checkpointDir;

    // ==================== 训练状态 ====================
    protected int currentEpoch;
    protected int globalStep;
    protected List<Float> lossHistory;

    /**
     * 构造函数
     *
     * @param model          DeepSeek 模型（V3 或 R1）
     * @param maxEpochs      最大训练轮数
     * @param maxGradNorm    梯度裁剪阈值
     * @param logInterval    日志输出间隔（步数）
     * @param checkpointDir  检查点保存目录
     */
    protected DeepSeekTrainerBase(DeepSeekModelBase model,
                                  int maxEpochs,
                                  float maxGradNorm,
                                  int logInterval,
                                  String checkpointDir) {
        this.model = model;
        this.maxEpochs = maxEpochs;
        this.maxGradNorm = maxGradNorm;
        this.logInterval = logInterval;
        this.checkpointDir = checkpointDir;

        this.currentEpoch = 0;
        this.globalStep = 0;
        this.lossHistory = new ArrayList<>();
    }

    // ==================== 公共方法 ====================

    /**
     * 梯度裁剪（全局梯度范数裁剪）
     *
     * 计算所有可训练参数的梯度 L2 范数，
     * 若超过 maxGradNorm 阈值则等比缩放所有梯度，防止梯度爆炸。
     */
    protected void clipGradients() {
        double totalNorm = 0.0;
        Map<String, Parameter> params = model.getModule().namedParameters("", true);

        for (Parameter param : params.values()) {
            if (param.requiresGrad() && param.grad() != null) {
                double norm = param.grad().mul(param.grad()).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }

        totalNorm = Math.sqrt(totalNorm);

        if (totalNorm > 0.0 && totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.requiresGrad() && param.grad() != null) {
                    NdArray clippedGrad = param.grad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }

    /**
     * 保存检查点
     *
     * 使用 {@link #getCheckpointPrefix()} 作为文件名前缀，
     * 保存格式为 "{prefix}_{suffix}.model"。
     *
     * @param suffix 检查点后缀（如 "final"、"step_1000"）
     */
    protected void saveCheckpoint(String suffix) {
        try {
            String filepath = checkpointDir + File.separator
                    + String.format("%s_%s.model", getCheckpointPrefix(), suffix);
            model.saveModel(filepath);
            System.out.println(getTrainerName() + " 检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println(getTrainerName() + " 保存检查点失败: " + e.getMessage());
        }
    }

    /**
     * 创建检查点目录（若不存在则递归创建）
     */
    protected void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }

    /**
     * 计算列表中最近 N 个元素的平均值
     *
     * @param values 值列表
     * @param last   取最近多少个元素
     * @return 平均值
     */
    protected float getAverage(List<Float> values, int last) {
        if (values == null || values.isEmpty()) {
            return 0.0f;
        }
        int start = Math.max(0, values.size() - last);
        float sum = 0.0f;
        for (int i = start; i < values.size(); i++) {
            sum += values.get(i);
        }
        return sum / (values.size() - start);
    }

    /**
     * 计算 float 数组的平均值
     *
     * @param values 值数组
     * @return 平均值
     */
    protected float calculateAverage(float[] values) {
        if (values == null || values.length == 0) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (float value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    /**
     * 计算 List<Float> 的平均值
     *
     * @param values 值列表
     * @return 平均值
     */
    protected float calculateAverage(List<Float> values) {
        if (values == null || values.isEmpty()) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (float value : values) {
            sum += value;
        }
        return sum / values.size();
    }

    /**
     * 获取损失历史
     *
     * @return 损失历史列表（防御性拷贝）
     */
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }

    // ==================== 抽象方法 ====================

    /**
     * 开始训练（由子类实现具体的训练流程）
     */
    public abstract void train();

    /**
     * 获取训练器名称（用于日志输出）
     *
     * @return 训练器名称，如 "DeepSeek-V3 Pretrain"
     */
    protected abstract String getTrainerName();

    /**
     * 获取检查点文件名前缀
     *
     * @return 检查点前缀，如 "deepseek_v3_pretrain"
     */
    protected abstract String getCheckpointPrefix();
}
