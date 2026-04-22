package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 训练器抽象基类
 * 
 * 提供训练循环框架、梯度裁剪、检查点保存等公共功能
 * 子类需实现具体的训练步骤逻辑
 * 
 * @author leavesfly
 * @since 2024
 */
public abstract class BaseTrainer {
    
    protected final MiniMindModel model;
    
    // 训练配置
    protected int maxEpochs;
    protected float maxGradNorm;
    protected int logInterval;
    protected int saveInterval;
    protected String checkpointDir;
    
    // 训练状态
    protected int currentEpoch;
    protected int currentStep;
    protected List<Float> lossHistory;
    
    /**
     * 构造函数
     * 
     * @param model 模型
     */
    public BaseTrainer(MiniMindModel model) {
        this.model = model;
        this.maxEpochs = 10;
        this.maxGradNorm = 1.0f;
        this.logInterval = 100;
        this.saveInterval = 1000;
        this.checkpointDir = "./checkpoints";
        
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 开始训练
     */
    public void train() {
        printTrainingInfo();
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println(getTrainerName() + "训练完成!");
    }
    
    /**
     * 训练一个epoch
     * <p>
     * 健壮性说明：
     * 当 {@link #trainStep(Object)} 返回 NaN/Inf（例如 SFTTrainer 在检测到损失溢出时
     * 会返回 {@code Float.NaN} 以跳过本 batch 的参数更新），本方法需要：
     *   1) 不将该值累加到 epochLoss（否则平均损失被污染为 NaN）；
     *   2) 不计入 lossHistory（否则后续统计/日志打印会被污染）；
     *   3) 不增加 batchCount（用于平均损失分母），但 currentStep 仍自增，
     *      以保持全局步数与日志/checkpoint 节奏一致；
     *   4) 打印警告便于定位训练异常。
     */
    protected void trainOneEpoch() {
        prepareDataset();
        model.setTraining(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        int skippedCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (hasNextBatch()) {
            Object batch = getNextBatch();
            
            // 训练一步
            float stepLoss = trainStep(batch);
            
            currentStep++;
            
            // 跳过 NaN/Inf 损失，避免污染统计
            if (Float.isNaN(stepLoss) || Float.isInfinite(stepLoss)) {
                skippedCount++;
                System.err.printf("警告: Step %d 损失异常(%s)，已跳过累加与历史记录%n",
                    currentStep, Float.isNaN(stepLoss) ? "NaN" : "Inf");
            } else {
                epochLoss += stepLoss;
                batchCount++;
                lossHistory.add(stepLoss);
            }
            
            // 打印日志
            if (currentStep % logInterval == 0) {
                printTrainingLog();
            }
            
            // 保存检查点
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        if (skippedCount > 0) {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 跳过批次: %d | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, skippedCount, epochEndTime - epochStartTime);
        } else {
            System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
                currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        }
        
        resetDataset();
    }
    
    /**
     * 梯度裁剪
     */
    protected void clipGradients() {
        // 计算梯度范数
        double totalNorm = 0.0;
        
        for (var param : model.getAllParams().values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 如果超过阈值,进行裁剪
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var param : model.getAllParams().values()) {
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    protected void saveCheckpoint() {
        String filename = String.format("%s_checkpoint_epoch%d_step%d.model", 
            getCheckpointPrefix(), currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
            System.out.println(getTrainerName() + "检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    protected void createCheckpointDir() {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 打印训练日志
     */
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
            .skip(Math.max(0, lossHistory.size() - logInterval))
            .mapToDouble(Float::doubleValue)
            .average()
            .orElse(0.0);
        
        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f%n",
            currentEpoch + 1, maxEpochs, currentStep, avgLoss);
    }
    
    /**
     * 获取损失历史
     * 
     * @return 损失历史列表
     */
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    // ==================== 抽象方法 ====================
    
    /**
     * 训练一步
     * 
     * @param batch 批次数据
     * @return 损失值
     */
    protected abstract float trainStep(Object batch);
    
    /**
     * 获取训练器名称
     * 
     * @return 训练器名称
     */
    protected abstract String getTrainerName();
    
    /**
     * 打印训练信息
     */
    protected abstract void printTrainingInfo();
    
    /**
     * 准备数据集（如打乱数据）
     */
    protected abstract void prepareDataset();
    
    /**
     * 检查是否还有下一个批次
     * 
     * @return 是否有下一个批次
     */
    protected abstract boolean hasNextBatch();
    
    /**
     * 获取下一个批次
     * 
     * @return 批次数据
     */
    protected abstract Object getNextBatch();
    
    /**
     * 重置数据集
     */
    protected abstract void resetDataset();
    
    /**
     * 获取检查点文件名前缀
     * 
     * @return 检查点前缀
     */
    protected abstract String getCheckpointPrefix();
}