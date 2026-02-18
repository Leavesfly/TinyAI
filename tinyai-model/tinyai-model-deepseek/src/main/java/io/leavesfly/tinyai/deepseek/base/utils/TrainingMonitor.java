package io.leavesfly.tinyai.deepseek.base.utils;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek 系列训练监控工具
 * 
 * 提供统一的：
 * - 日志格式化输出
 * - 训练指标计算（移动平均、最佳值追踪）
 * - 进度展示
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TrainingMonitor {
    
    // ========== 日志格式常量 ==========
    private static final String SEPARATOR = "=".repeat(80);
    private static final String THIN_SEPARATOR = "-".repeat(80);
    
    // ========== 指标追踪 ==========
    private final List<Double> lossHistory;
    private double bestLoss;
    private int bestEpoch;
    
    public TrainingMonitor() {
        this.lossHistory = new ArrayList<>();
        this.bestLoss = Double.MAX_VALUE;
        this.bestEpoch = 0;
    }
    
    // ========== 日志输出方法 ==========
    
    /**
     * 打印训练开始横幅
     */
    public void printTrainingStart(String modelName, TrainingConfig config) {
        System.out.println(SEPARATOR);
        System.out.println("🚀 DeepSeek 模型训练");
        System.out.println(SEPARATOR);
        System.out.println("模型信息:");
        System.out.println("  模型: " + modelName);
        System.out.println("  参数量: " + config.totalParams);
        if (config.activeParams != null) {
            System.out.println("  激活参数: " + config.activeParams + 
                             String.format(" (%.1f%%)", config.activationRatio));
        }
        System.out.println();
        System.out.println("训练配置:");
        System.out.println("  训练样本: " + config.sampleCount);
        System.out.println("  批次大小: " + config.batchSize);
        System.out.println("  最大轮次: " + config.maxEpochs);
        System.out.println("  初始学习率: " + config.initialLearningRate);
        System.out.println("  Warmup步数: " + config.warmupSteps);
        if (config.moeEnabled) {
            System.out.println("  MoE专家数: " + config.numExperts);
            System.out.println("  Top-K选择: " + config.topK);
            System.out.println("  负载均衡权重: " + config.loadBalanceWeight);
        }
        System.out.println(SEPARATOR);
    }
    
    /**
     * 打印轮次开始信息
     */
    public void printEpochStart(int epoch, int totalEpochs) {
        System.out.println();
        System.out.println(THIN_SEPARATOR);
        System.out.println(String.format("📊 Epoch %d/%d", epoch, totalEpochs));
        System.out.println(THIN_SEPARATOR);
    }
    
    /**
     * 打印训练步骤信息
     */
    public void printStep(int epoch, int step, double loss, double learningRate) {
        System.out.printf("  [Epoch %d | Step %d] Loss: %.6f | LR: %.8f%n",
                         epoch, step, loss, learningRate);
    }
    
    /**
     * 打印训练步骤信息（带MoE损失）
     */
    public void printStepWithMoE(int epoch, int step, double lmLoss, double moeLoss, 
                                 double totalLoss, double learningRate) {
        System.out.printf("  [Epoch %d | Step %d] LM: %.6f | MoE: %.6f | Total: %.6f | LR: %.8f%n",
                         epoch, step, lmLoss, moeLoss, totalLoss, learningRate);
    }
    
    /**
     * 打印轮次结束信息
     */
    public void printEpochEnd(int epoch, double avgLoss, long timeMs) {
        lossHistory.add(avgLoss);
        
        // 更新最佳记录
        boolean isBest = avgLoss < bestLoss;
        if (isBest) {
            bestLoss = avgLoss;
            bestEpoch = epoch;
        }
        
        System.out.println(THIN_SEPARATOR);
        System.out.printf("✅ Epoch %d 完成 | 平均Loss: %.6f | 用时: %.2fs%s%n",
                         epoch, avgLoss, timeMs / 1000.0, isBest ? " 🌟 (最佳)" : "");
        
        // 显示趋势
        if (lossHistory.size() >= 2) {
            double prevLoss = lossHistory.get(lossHistory.size() - 2);
            double improvement = prevLoss - avgLoss;
            String trend = improvement > 0 ? "↓" : (improvement < 0 ? "↑" : "→");
            System.out.printf("  Loss变化: %s %.6f%n", trend, Math.abs(improvement));
        }
    }
    
    /**
     * 打印训练完成信息
     */
    public void printTrainingEnd(int totalEpochs, long totalTimeMs) {
        System.out.println();
        System.out.println(SEPARATOR);
        System.out.println("🎉 训练完成!");
        System.out.println(SEPARATOR);
        System.out.println("训练统计:");
        System.out.println("  总轮次: " + totalEpochs);
        System.out.println("  总用时: " + formatTime(totalTimeMs));
        System.out.println("  最佳Loss: " + String.format("%.6f", bestLoss) + 
                         " (Epoch " + bestEpoch + ")");
        System.out.println("  最终Loss: " + String.format("%.6f", lossHistory.get(lossHistory.size() - 1)));
        
        // 收敛性分析
        if (lossHistory.size() >= 3) {
            double recentImprovement = lossHistory.get(lossHistory.size() - 3) - 
                                      lossHistory.get(lossHistory.size() - 1);
            if (Math.abs(recentImprovement) < 0.001) {
                System.out.println("  状态: ✅ 已收敛");
            } else if (recentImprovement > 0) {
                System.out.println("  状态: 📈 持续下降");
            } else {
                System.out.println("  状态: ⚠️ 可能过拟合");
            }
        }
        System.out.println(SEPARATOR);
    }
    
    /**
     * 打印检查点保存信息
     */
    public void printCheckpointSaved(String filename, double loss) {
        System.out.println("  💾 检查点已保存: " + filename + 
                         String.format(" (Loss: %.6f)", loss));
    }
    
    // ========== 指标计算方法 ==========
    
    /**
     * 计算移动平均损失
     */
    public double getMovingAverage(int window) {
        if (lossHistory.isEmpty()) return 0.0;
        
        int start = Math.max(0, lossHistory.size() - window);
        double sum = 0.0;
        for (int i = start; i < lossHistory.size(); i++) {
            sum += lossHistory.get(i);
        }
        return sum / (lossHistory.size() - start);
    }
    
    /**
     * 获取最佳损失
     */
    public double getBestLoss() {
        return bestLoss;
    }
    
    /**
     * 获取最佳轮次
     */
    public int getBestEpoch() {
        return bestEpoch;
    }
    
    /**
     * 获取损失历史
     */
    public List<Double> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    // ========== 工具方法 ==========
    
    /**
     * 格式化参数数量
     */
    public static String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.valueOf(count);
        }
    }
    
    /**
     * 格式化时间
     */
    public static String formatTime(long milliseconds) {
        long seconds = milliseconds / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        
        if (hours > 0) {
            return String.format("%dh %dm %ds", hours, minutes % 60, seconds % 60);
        } else if (minutes > 0) {
            return String.format("%dm %ds", minutes, seconds % 60);
        } else {
            return String.format("%.2fs", milliseconds / 1000.0);
        }
    }
    
    // ========== 配置类 ==========
    
    /**
     * 训练配置信息（用于日志展示）
     */
    public static class TrainingConfig {
        public String totalParams;
        public String activeParams;
        public double activationRatio;
        public int sampleCount;
        public int batchSize;
        public int maxEpochs;
        public double initialLearningRate;
        public int warmupSteps;
        public boolean moeEnabled;
        public int numExperts;
        public int topK;
        public double loadBalanceWeight;
        
        public TrainingConfig(String totalParams, int sampleCount, int batchSize,
                            int maxEpochs, double initialLearningRate, int warmupSteps) {
            this.totalParams = totalParams;
            this.sampleCount = sampleCount;
            this.batchSize = batchSize;
            this.maxEpochs = maxEpochs;
            this.initialLearningRate = initialLearningRate;
            this.warmupSteps = warmupSteps;
            this.moeEnabled = false;
        }
        
        public TrainingConfig withMoE(String activeParams, double activationRatio,
                                     int numExperts, int topK, double loadBalanceWeight) {
            this.moeEnabled = true;
            this.activeParams = activeParams;
            this.activationRatio = activationRatio;
            this.numExperts = numExperts;
            this.topK = topK;
            this.loadBalanceWeight = loadBalanceWeight;
            return this;
        }
    }
}
