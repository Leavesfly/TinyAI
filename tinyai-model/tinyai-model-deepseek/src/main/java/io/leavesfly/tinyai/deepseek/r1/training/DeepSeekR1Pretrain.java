package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ml.training.parallel.ParallelTrainingUtils;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * DeepSeek-R1预训练器
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练,
 * 同时训练推理和反思能力
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Pretrain extends DeepSeekTrainerBase {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1Config config;
    private final DeepSeekR1Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final SGD optimizer;
    
    // 训练超参数
    private float initialLearningRate;
    private float minLearningRate;
    private int warmupSteps;
    private int saveInterval;
    
    // 训练状态
    private float currentLearningRate;
    private List<Float> reasoningConfidenceHistory;
    
    // 并行训练配置
    private boolean enableParallel = true;
    private int parallelThreads = Runtime.getRuntime().availableProcessors();
    private ExecutorService executorService;
    private boolean parallelTrainingAvailable = false;
    
    /**
     * 构造函数
     */
    public DeepSeekR1Pretrain(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
        super(model, 10, 1.0f, 100, "./checkpoints/deepseek_r1/pretrain");
        
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认超参数
        this.initialLearningRate = 2.5e-4f;
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.saveInterval = 5000;
        
        // 创建优化器
        // 使用SGD替代Adam，减少临时NdArray对象创建，降低内存占用
        this.optimizer = new SGD(model, initialLearningRate);
        
        // 初始化状态
        this.currentLearningRate = 0.0f;
        this.reasoningConfidenceHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DeepSeekR1Pretrain configure(int maxEpochs, float learningRate,
                                         int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        // 同步学习率到优化器
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 设置检查点配置
     */
    public DeepSeekR1Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 设置日志输出间隔
     */
    public DeepSeekR1Pretrain setLogInterval(int logInterval) {
        this.logInterval = logInterval;
        return this;
    }
    
    /**
     * 配置并行训练
     * @param enable 是否启用并行训练
     * @param threads 并行线程数（0表示自动）
     */
    public DeepSeekR1Pretrain configureParallel(boolean enable, int threads) {
        this.enableParallel = enable;
        this.parallelThreads = threads > 0 ? threads : Runtime.getRuntime().availableProcessors();
        return this;
    }
    
    /**
     * 初始化并行训练环境
     */
    private void initParallelTraining() {
        if (!enableParallel) {
            parallelTrainingAvailable = false;
            return;
        }
        
        // 检查模型是否支持并行训练（可序列化）
        if (!ParallelTrainingUtils.isModelParallelizable(model)) {
            System.out.println("⚠️ 模型不支持序列化，回退到串行模式");
            parallelTrainingAvailable = false;
            return;
        }
        
        // 测试深拷贝是否能正确创建DeepSeekR1Model（而不是普通Model）
        try {
            Model copiedModel = ParallelTrainingUtils.deepCopyModel(model);
            if (!(copiedModel instanceof DeepSeekR1Model)) {
                System.out.println("⚠️ 模型深拷贝类型不匹配，回退到串行模式");
                System.out.println("  - 原始类型: " + model.getClass().getName());
                System.out.println("  - 拷贝类型: " + copiedModel.getClass().getName());
                parallelTrainingAvailable = false;
                return;
            }
        } catch (Exception e) {
            System.out.println("⚠️ 模型深拷贝测试失败: " + e.getMessage() + "，回退到串行模式");
            e.printStackTrace();
            parallelTrainingAvailable = false;
            return;
        }
        
        // 创建线程池
        executorService = Executors.newFixedThreadPool(parallelThreads);
        parallelTrainingAvailable = true;
    }
    
    /**
     * 关闭并行训练环境
     */
    private void shutdownParallelTraining() {
        if (executorService != null) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
            }
        }
    }
    
    /**
     * 开始训练
     */
    @Override
    public void train() {
        DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String startTime = LocalDateTime.now().format(timeFormatter);
        
        // 初始化并行训练环境
        initParallelTraining();
        
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 预训练");
        System.out.println("=".repeat(70));
        System.out.println("模型参数:");
        System.out.println("  - 嵌入维度: " + config.getNEmbd());
        System.out.println("  - Transformer层数: " + config.getNLayer());
        System.out.println("  - 注意力头数: " + config.getNHead());
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getTopK());
        System.out.println("  - 架构: Pre-LayerNorm + MoE");
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("并行训练:");
        if (parallelTrainingAvailable) {
            System.out.println("  - 状态: ✅ 已启用(真正并行)");
            System.out.println("  - 线程数: " + parallelThreads);
        } else if (enableParallel) {
            System.out.println("  - 状态: ⚠️ 已回退到串行模式");
            System.out.println("  - 原因: 模型不支持序列化");
        } else {
            System.out.println("  - 状态: ❌ 未启用");
        }
        System.out.println("=".repeat(70));
        System.out.println("[" + startTime + "] 训练开始...");
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        // 关闭并行训练环境
        shutdownParallelTraining();
        
        System.out.println("\n训练完成!");
        System.out.println("最终损失: " + getAverageLoss(lossHistory, 100));
        System.out.println("平均推理置信度: " + getAverageLoss(reasoningConfidenceHistory, 100));
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String epochStartTime = LocalDateTime.now().format(timeFormatter);
        
        System.out.println("\n[" + epochStartTime + "] Epoch " + (currentEpoch + 1) + "/" + maxEpochs + " 开始" + 
            (parallelTrainingAvailable ? " (并行模式-" + parallelThreads + "线程)" : " (串行模式)"));
        
        dataset.prepare(true);
        
        double epochLoss = 0.0;
        double epochConfidence = 0.0;
        int batchCount = 0;
        
        long epochStartMs = System.currentTimeMillis();
        
        // 收集所有batch
        List<DeepSeekR1Dataset.Batch> batches = new ArrayList<>();
        while (dataset.hasNext()) {
            batches.add(dataset.nextBatch());
        }
        
        if (parallelTrainingAvailable && batches.size() > 1) {
            // 真正的并行训练
            EpochResult parallelResult = trainBatchesParallel(batches);
            epochLoss = parallelResult.totalLoss;
            epochConfidence = parallelResult.totalConfidence;
            batchCount = batches.size();
            // globalStep 已在 trainBatchesParallel 内部更新
        } else {
            // 顺序训练
            for (DeepSeekR1Dataset.Batch batch : batches) {
                StepResult stepResult = trainStep(batch);
                
                epochLoss += stepResult.loss;
                epochConfidence += stepResult.confidence;
                batchCount++;
                globalStep++;
                
                lossHistory.add(stepResult.loss);
                reasoningConfidenceHistory.add(stepResult.confidence);
                
                if (globalStep % logInterval == 0) {
                    float avgLoss = getAverageLoss(lossHistory, logInterval);
                    float avgConf = getAverageLoss(reasoningConfidenceHistory, logInterval);
                    System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Conf: %.4f | LR: %.6f%n",
                        currentEpoch + 1, maxEpochs, globalStep, avgLoss, avgConf, currentLearningRate);
                }
                
                if (globalStep % saveInterval == 0) {
                    saveCheckpoint("step_" + globalStep);
                }
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochConf = batchCount > 0 ? epochConfidence / batchCount : 0.0;
        
        String epochEndTimeStr = LocalDateTime.now().format(timeFormatter);
        System.out.printf("[%s] Epoch %d 完成 | 平均损失: %.4f | 置信度: %.4f | 耗时: %dms%n",
            epochEndTimeStr, currentEpoch + 1, avgEpochLoss, avgEpochConf, epochEndTime - epochStartMs);
        
        dataset.reset();
        System.gc();
    }
    
    /**
     * Epoch结果（并行训练用）
     */
    private static class EpochResult {
        final double totalLoss;
        final double totalConfidence;
        
        EpochResult(double totalLoss, double totalConfidence) {
            this.totalLoss = totalLoss;
            this.totalConfidence = totalConfidence;
        }
    }
    
    /**
     * 并行训练批次 - 使用多线程并行处理
     */
    private EpochResult trainBatchesParallel(List<DeepSeekR1Dataset.Batch> batches) {
        long startTime = System.currentTimeMillis();
        DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        
        int batchCount = batches.size();
        int logInterval = Math.max(1, batchCount / 5);  // 每20%输出一次
        
        AtomicInteger processedCount = new AtomicInteger(0);
        double[] totalLoss = {0.0};
        double[] totalConfidence = {0.0};
        
        // 按线程数分组处理批次
        for (int i = 0; i < batchCount; i += parallelThreads) {
            // 更新学习率
            updateLearningRate();
            
            int endIndex = Math.min(i + parallelThreads, batchCount);
            List<DeepSeekR1Dataset.Batch> batchGroup = batches.subList(i, endIndex);
            
            // 为每个批次创建并行任务
            List<Future<BatchResult>> futures = new ArrayList<>();
            
            for (DeepSeekR1Dataset.Batch batch : batchGroup) {
                // 创建模型副本进行并行处理
                Future<BatchResult> future = executorService.submit(() -> {
                    try {
                        // 深拷贝模型
                        DeepSeekR1Model modelCopy = (DeepSeekR1Model) ParallelTrainingUtils.deepCopyModel(model);
                        
                        // 前向传播
                        NdArray inputIds = batch.getInputIds();
                        NdArray targetIds = batch.getTargetIds();
                        Variable inputVar = new Variable(inputIds);
                        
                        DeepSeekR1Model.ReasoningResult result = modelCopy.performReasoning(inputVar);
                        Variable logits = result.logits;
                        
                        // 计算损失
                        int[] logitsShape = logits.getValue().getShape().getShapeDims();
                        int bs = logitsShape[0];
                        int seqLen = logitsShape[1];
                        int vocabSize = logitsShape[2];
                        
                        Variable logits2D = logits.reshape(Shape.of(bs * seqLen, vocabSize));
                        Variable targetVar = new Variable(targetIds.reshape(Shape.of(bs * seqLen, 1)));
                        
                        Variable loss = lossFunction.loss(targetVar, logits2D);
                        
                        float lossValue = loss.getValue().getNumber().floatValue();
                        float moeLoss = (float) result.moeLoss;
                        
                        // 反向传播（在副本上）
                        modelCopy.clearGrads();
                        loss.backward();
                        
                        // 收集梯度
                        Map<String, Parameter> params = modelCopy.getModule().namedParameters("", true);
                        Map<String, NdArray> gradients = new java.util.HashMap<>();
                        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
                            if (entry.getValue().grad() != null) {
                                gradients.put(entry.getKey(), entry.getValue().grad());
                            }
                        }
                        
                        // 释放计算图
                        loss.unChainBackward();
                        logits.unChainBackward();
                        inputVar.unChainBackward();
                        
                        return new BatchResult(true, lossValue, moeLoss, gradients);
                    } catch (Exception e) {
                        System.err.println("⚠️ 并行批次处理失败: " + e.getMessage());
                        return new BatchResult(false, 0, 0, null);
                    }
                });
                futures.add(future);
            }
            
            // 收集结果并聚合梯度
            Map<String, NdArray> aggregatedGradients = new java.util.HashMap<>();
            int successCount = 0;
            
            for (Future<BatchResult> future : futures) {
                try {
                    BatchResult result = future.get();
                    if (result.success) {
                        totalLoss[0] += result.loss;
                        totalConfidence[0] += result.confidence;
                        successCount++;
                        
                        // 聚合梯度
                        for (Map.Entry<String, NdArray> entry : result.gradients.entrySet()) {
                            String key = entry.getKey();
                            NdArray grad = entry.getValue();
                            if (aggregatedGradients.containsKey(key)) {
                                aggregatedGradients.put(key, aggregatedGradients.get(key).add(grad));
                            } else {
                                aggregatedGradients.put(key, grad);
                            }
                        }
                        
                        lossHistory.add(result.loss);
                        reasoningConfidenceHistory.add(result.confidence);
                    }
                } catch (ExecutionException e) {
                    // 解包 ExecutionException 获取原始异常
                    Throwable cause = e.getCause();
                    System.err.println("⚠️ 并行批次执行异常: " + (cause != null ? cause.getMessage() : e.getMessage()));
                    if (cause != null) {
                        cause.printStackTrace();
                    }
                } catch (InterruptedException e) {
                    // 处理线程中断
                    System.err.println("⚠️ 并行批次处理被中断: " + e.getMessage());
                    Thread.currentThread().interrupt();
                }
            }
            
            // 平均梯度并应用到主模型
            if (successCount > 0) {
                final int sc = successCount;
                aggregatedGradients.replaceAll((k, v) -> v.divNum(sc));
                
                // 应用梯度到主模型
                Map<String, Parameter> mainParams = model.getModule().namedParameters("", true);
                for (Map.Entry<String, NdArray> entry : aggregatedGradients.entrySet()) {
                    Parameter param = mainParams.get(entry.getKey());
                    if (param != null) {
                        param.setGrad(entry.getValue());
                    }
                }
                
                // 梯度裁剪和参数更新
                clipGradients();
                optimizer.update();
                model.clearGrads();
            }
            
            processedCount.addAndGet(batchGroup.size());
            globalStep += batchGroup.size();  // 更新全局步数以支持学习率调度
            
            // 定期输出进度
            int processed = processedCount.get();
            if (processed % logInterval == 0 || processed == batchCount) {
                String timeNow = LocalDateTime.now().format(timeFormatter);
                int progress = (int)(processed * 100.0 / batchCount);
                long elapsed = System.currentTimeMillis() - startTime;
                float avgLoss = (float)(totalLoss[0] / processed);
                System.out.printf("  [%s] 进度: %d/%d (%d%%) | Loss: %.4f | 耗时: %dms%n", 
                    timeNow, processed, batchCount, progress, avgLoss, elapsed);
            }
        }
        
        return new EpochResult(totalLoss[0], totalConfidence[0]);
    }
    
    /**
     * 批次处理结果
     */
    private static class BatchResult {
        final boolean success;
        final float loss;
        final float confidence;
        final Map<String, NdArray> gradients;
        
        BatchResult(boolean success, float loss, float confidence, Map<String, NdArray> gradients) {
            this.success = success;
            this.loss = loss;
            this.confidence = confidence;
            this.gradients = gradients;
        }
    }
    
    /**
     * 训练单步
     */
    private StepResult trainStep(DeepSeekR1Dataset.Batch batch) {
        long stepStartTime = System.currentTimeMillis();
        
        // 更新学习率
        updateLearningRate();
        
        // 准备输入
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播(带推理和反思)
        long forwardStart = System.currentTimeMillis();
        DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
        Variable logits = result.logits;
        long forwardTime = System.currentTimeMillis() - forwardStart;
        
        // 计算损失 - SoftmaxCE只支持2D输入，需要reshape
        // logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        // targets: [batch_size, seq_len] -> [batch_size * seq_len, 1]
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
        
        Variable loss = lossFunction.loss(targetVar, logits2D);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        float moeLoss = (float) result.moeLoss;  // 使用 MoE 损失
        
        // 清空梯度
        model.clearGrads();
        
        // 反向传播
        long backwardStart = System.currentTimeMillis();
        loss.backward();
        long backwardTime = System.currentTimeMillis() - backwardStart;
        
        // 梯度裁剪
        clipGradients();
        
        // 参数更新
        optimizer.update();
        
        // 彻底断开计算图，释放内存
        loss.unChainBackward();
        logits.unChainBackward();
        inputVar.unChainBackward();
        
        // 输出单步耗时（每10步输出一次）
        if (globalStep % 10 == 0 || globalStep <= 3) {
            long stepTime = System.currentTimeMillis() - stepStartTime;
            System.out.printf("  Step %d: Loss=%.4f | Forward=%dms | Backward=%dms | Total=%dms%n", 
                globalStep, lossValue, forwardTime, backwardTime, stepTime);
        }
        
        return new StepResult(lossValue, moeLoss);
    }
    
    /**
     * 更新学习率(warmup + cosine衰减)
     */
    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            // 线性warmup
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            // 余弦退火
            int totalSteps = maxEpochs * dataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = globalStep - warmupSteps;
            
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            float decayedLR = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }
        // 同步学习率到优化器
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 获取平均损失
     */
    private float getAverageLoss(List<Float> history, int n) {
        if (history.isEmpty()) {
            return 0.0f;
        }
        
        int start = Math.max(0, history.size() - n);
        float sum = 0.0f;
        for (int i = start; i < history.size(); i++) {
            sum += history.get(i);
        }
        return sum / (history.size() - start);
    }
    
    /**
     * 获取训练器名称
     */
    @Override
    public String getTrainerName() {
        return "DeepSeek-R1 Pretrain";
    }
    
    /**
     * 获取检查点前缀
     */
    @Override
    public String getCheckpointPrefix() {
        return "deepseek_r1_pretrain";
    }
    
    /**
     * 单步训练结果
     */
    private static class StepResult {
        final float loss;
        final float confidence;
        
        StepResult(float loss, float confidence) {
            this.loss = loss;
            this.confidence = confidence;
        }
    }
    
    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        return new TrainingStats(
            currentEpoch,
            globalStep,
            currentLearningRate,
            lossHistory.isEmpty() ? 0.0f : lossHistory.get(lossHistory.size() - 1),
            getAverageLoss(lossHistory, 100),
            getAverageLoss(reasoningConfidenceHistory, 100)
        );
    }
    
    /**
     * 训练统计信息
     */
    public static class TrainingStats {
        public final int epoch;
        public final int step;
        public final float learningRate;
        public final float currentLoss;
        public final float avgLoss;
        public final float avgConfidence;
        
        public TrainingStats(int epoch, int step, float learningRate,
                           float currentLoss, float avgLoss, float avgConfidence) {
            this.epoch = epoch;
            this.step = step;
            this.learningRate = learningRate;
            this.currentLoss = currentLoss;
            this.avgLoss = avgLoss;
            this.avgConfidence = avgConfidence;
        }
        
        @Override
        public String toString() {
            return String.format(
                "TrainingStats{epoch=%d, step=%d, lr=%.6f, loss=%.4f, avgLoss=%.4f, avgConf=%.4f}",
                epoch, step, learningRate, currentLoss, avgLoss, avgConfidence
            );
        }
    }
}
