package io.leavesfly.tinyai.banana.training;

import io.leavesfly.tinyai.banana.block.BananaBlock;
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.banana.training.dataset.BananaDataset;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Banana 多模态预训练器。
 *
 * <p>基于图像重建目标进行跨模态对齐的预训练器：使用文本编码器与图像编码器分别获得特征，
 * 再通过 {@link io.leavesfly.tinyai.banana.fusion.MultiModalFusion#forwardBoth} 得到
 * 融合后的文本/图像特征，并以融合图像特征与原始图像特征之间的均方误差作为损失，
 * 鼓励跨模态注意力保留视觉信息。</p>
 *
 * <p>主要特性：</p>
 * <ul>
 *   <li>学习率调度：线性预热 + 余弦退火，当 {@code warmupSteps=0} 时跳过预热阶段；</li>
 *   <li>梯度裁剪：基于全局 L2 范数，阈值由 {@code maxGradNorm} 控制；</li>
 *   <li>检查点保存：按固定步数间隔落盘，文件名包含 epoch 和 step；</li>
 *   <li>训练状态通过 {@link io.leavesfly.tinyai.util.Config#train} 切换，结束后恢复原值。</li>
 * </ul>
 *
 * <p>注意：为了避免 {@code warmupSteps=0}、{@code batchCount=0} 等边界导致除零，
 * {@link #updateLearningRate()} 对所有除数做了防御。</p>
 *
 * @author TinyAI
 * @since 2024
 */
public class PretrainTrainer {

    private final BananaModel model;
    private final BananaBlock bananaBlock;
    private final BananaConfig config;
    private final BananaDataset dataset;
    private final Adam optimizer;
    
    // 训练配置
    private int maxEpochs;
    private float initialLearningRate;
    private float maxGradNorm;       // 梯度裁剪阈值
    private int warmupSteps;         // 学习率预热步数
    private int logInterval;         // 日志打印间隔
    private int saveInterval;        // 检查点保存间隔
    private String checkpointDir;    // 检查点目录
    
    // 训练状态
    private int currentEpoch;
    private int currentStep;
    private float currentLearningRate;
    private List<Float> lossHistory;
    
    /**
     * 构造函数
     * 
     * @param model 模型
     * @param dataset 预训练数据集
     */
    public PretrainTrainer(BananaModel model, BananaDataset dataset) {
        if (model == null) {
            throw new IllegalArgumentException("model 不能为 null");
        }
        if (dataset == null) {
            throw new IllegalArgumentException("dataset 不能为 null");
        }
        this.model = model;
        this.bananaBlock = model.getBananaBlock();
        this.config = model.getConfig();
        this.dataset = dataset;

        // 默认配置
        this.maxEpochs = 10;
        this.initialLearningRate = 1e-4f;
        this.maxGradNorm = 1.0f;
        this.warmupSteps = 1000;
        this.logInterval = 100;
        this.saveInterval = 1000;
        this.checkpointDir = "./checkpoints/banana/pretrain";
        
        // 创建优化器 (Adam；TinyAI 当前未提供 AdamW/decoupled weight decay，如需 L2 正则需自行扩展)
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮次
     * @param learningRate 学习率
     * @param warmupSteps 预热步数
     * @param maxGradNorm 梯度裁剪阈值
     * @return this
     */
    public PretrainTrainer configure(int maxEpochs, float learningRate, 
                                      int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        return this;
    }
    
    /**
     * 设置检查点配置
     * 
     * @param checkpointDir 检查点目录
     * @param saveInterval 保存间隔(步数)
     * @return this
     */
    public PretrainTrainer setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("开始Banana多模态预训练");
        System.out.println("=".repeat(60));
        System.out.println("模型配置: Banana " + config.getClass().getSimpleName());
        long totalParams = calculateTotalParams();
        System.out.println("总参数量: " + totalParams + " (" + String.format("%.2fM", totalParams / 1_000_000.0) + ")");
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("初始学习率: " + initialLearningRate);
        System.out.println("图像大小: " + config.getImageSize() + "x" + config.getImageSize());
        System.out.println("=".repeat(60));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("训练完成!");
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);  // 打乱数据
        
        // 设置训练模式(通过Config.train控制)
        boolean prevTrain = io.leavesfly.tinyai.util.Config.train;
        io.leavesfly.tinyai.util.Config.train = true;
        
        try {
            double epochLoss = 0.0;
            int batchCount = 0;
            
            long epochStartTime = System.currentTimeMillis();
            
            while (dataset.hasNextBatch()) {
                BananaDataset.Batch batch = dataset.getNextBatch();
                
                // 训练一步
                float stepLoss = trainStep(batch);
                
                epochLoss += stepLoss;
                batchCount++;
                currentStep++;
                
                // 记录损失
                lossHistory.add(stepLoss);
                
                // 打印日志
                if (currentStep % logInterval == 0) {
                    double avgLoss = lossHistory.stream()
                        .skip(Math.max(0, lossHistory.size() - logInterval))
                        .mapToDouble(Float::doubleValue)
                        .average()
                        .orElse(0.0);
                    
                    System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | LR: %.6f%n",
                        currentEpoch + 1, maxEpochs, currentStep, avgLoss, currentLearningRate);
                }
                
                // 保存检查点
                if (currentStep % saveInterval == 0) {
                    saveCheckpoint();
                }
            }
            
            long epochEndTime = System.currentTimeMillis();
            double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
            
            System.out.println(String.format(
                "Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms",
                currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime
            ));
        } finally {
            io.leavesfly.tinyai.util.Config.train = prevTrain;
        }
        
        dataset.reset();
    }
    
    /**
     * 训练一步。
     *
     * <p>预训练目标：<b>CLIP 风格的对称对比对齐</b>（InfoNCE）。</p>
     * <p>流程：</p>
     * <ol>
     *   <li>编码文本 / 图像得到序列特征；</li>
     *   <li>序列维度均值池化为 {@code [batch, hidden]}；</li>
     *   <li>L2 归一化；</li>
     *   <li>计算相似度矩阵 {@code S = T · I^T * logitScale}（形状 {@code [B, B]}）；</li>
     *   <li>对 S 的行 softmax → 对角为正样本概率；对 S 的列 softmax → 同理；</li>
     *   <li>损失 = 两方向 InfoNCE 的平均值 = {@code -mean(log(diag))}。</li>
     * </ol>
     *
     * @param batch 批次数据
     * @return 损失值
     */
    private float trainStep(BananaDataset.Batch batch) {
        // 更新学习率
        updateLearningRate();

        // 输入封装
        Variable textVar = new Variable(batch.getTextInput());
        Variable imageVar = new Variable(batch.getImageInput());

        // 编码 + 可选双向融合 + 池化 + L2 归一化（统一流水线，见 BaseBananaTrainer）
        Variable[] pair = BaseBananaTrainer.encodeAndPoolPair(model, bananaBlock, textVar, imageVar);
        Variable textEmb = pair[0];
        Variable imageEmb = pair[1];

        // CLIP 风格 InfoNCE 对称对比损失
        Variable loss = BaseBananaTrainer.computeContrastiveLoss(
                textEmb, imageEmb, BaseBananaTrainer.DEFAULT_LOGIT_SCALE);

        float lossValue = loss.getValue().getNumber().floatValue();

        // 梯度更新
        model.clearGrads();
        loss.backward();
        clipGradients();
        optimizer.update();
        loss.unChainBackward();

        return lossValue;
    }
    
    /**
     * 更新学习率（线性预热 + 余弦退火）。
     *
     * <p>当 {@code warmupSteps <= 0} 时跳过预热阶段，直接进入余弦退火；
     * 当 {@code decaySteps <= 0}（如总步数小于等于预热步数）时保持初始学习率，
     * 避免出现除零导致的 NaN/Infinity。</p>
     */
    private void updateLearningRate() {
        if (warmupSteps > 0 && currentStep < warmupSteps) {
            // 线性预热：第 0 步也给予一个极小学习率，避免首步完全不更新
            float ratio = (float) (currentStep + 1) / warmupSteps;
            currentLearningRate = initialLearningRate * ratio;
        } else {
            // 余弦退火：对 decaySteps<=0 与 batchCount=0 做防御
            int totalSteps = maxEpochs * Math.max(1, dataset.getBatchCount());
            int decaySteps = Math.max(1, totalSteps - Math.max(0, warmupSteps));
            int currentDecayStep = Math.max(0, currentStep - Math.max(0, warmupSteps));
            if (currentDecayStep > decaySteps) {
                currentDecayStep = decaySteps;
            }
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / (double) decaySteps));
            currentLearningRate = initialLearningRate * (float) cosineDecay;
        }

        // 更新优化器学习率
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        BaseBananaTrainer.clipGradients(model, maxGradNorm);
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint() {
        String filename = String.format("checkpoint_epoch%d_step%d.model", 
            currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        BaseBananaTrainer.createCheckpointDir(checkpointDir);
    }
    
    /**
     * 计算总参数量
     */
    private long calculateTotalParams() {
        return BaseBananaTrainer.calculateTotalParams(model);
    }
    
    /**
     * 获取损失历史
     * 
     * @return 损失历史列表
     */
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    /**
     * 设置日志间隔
     * 
     * @param logInterval 日志打印间隔
     */
    public void setLogInterval(int logInterval) {
        this.logInterval = logInterval;
    }
}