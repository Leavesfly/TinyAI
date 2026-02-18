package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1RLVRDataset;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.*;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * DeepSeek-R1强化学习训练器 (RLVR - Reinforcement Learning from Verifiable Rewards)
 * 
 * RLVR vs RLHF:
 * 
 * | 维度 | RLHF | RLVR |
 * |------|------|------|
 * | 奖励来源 | 人类主观反馈 | 可验证的客观标准 |
 * | 奖励类型 | 连续值(0-1) | 二值(0或1) |
 * | 验证方式 | 奖励模型近似 | 规则/测试用例验证 |
 * | 适用场景 | 开放性任务 | 可验证任务(数学、代码) |
 * | 训练速度 | 慢(需人工标注) | 快(自动验证) |
 * | 抗奖励欺骗 | 弱 | 强 |
 * 
 * 训练流程:
 * 1. 模型生成推理输出
 * 2. 验证器执行可验证奖励计算
 * 3. 基于二值奖励进行策略优化
 * 4. 更新模型参数
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1RLVRTrainer {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1RLVRDataset dataset;
    private final SGD optimizer;
    
    // 验证器映射
    private final Map<String, Verifier> verifiers;
    
    // 训练参数
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    
    // 奖励权重
    private float correctnessWeight;     // 正确性权重 (主)
    private float reasoningQualityWeight; // 推理质量权重 (辅)
    private float verificationWeight;     // 验证完整性权重 (辅)
    
    private int logInterval;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    
    // 训练统计
    private List<Float> correctnessHistory;
    private List<Float> rewardHistory;
    private List<Float> qualityHistory;
    
    /**
     * 构造函数
     * 
     * @param model DeepSeek-R1模型
     * @param dataset RLVR数据集
     */
    public DeepSeekR1RLVRTrainer(DeepSeekR1Model model, DeepSeekR1RLVRDataset dataset) {
        this.model = model;
        this.dataset = dataset;
        
        // 初始化验证器
        this.verifiers = new HashMap<>();
        this.verifiers.put("math", new MathVerifier());
        this.verifiers.put("code", new CodeVerifier());
        this.verifiers.put("logic", new LogicVerifier());
        
        // RLVR训练参数（与RLHF类似但更激进）
        this.maxEpochs = 5;
        this.learningRate = 5e-5f;  // RLVR可以使用稍大的学习率
        this.maxGradNorm = 1.0f;
        
        // 奖励权重配置
        this.correctnessWeight = 0.7f;      // 正确性最重要
        this.reasoningQualityWeight = 0.2f;  // 推理质量
        this.verificationWeight = 0.1f;      // 验证完整性
        
        this.logInterval = 10;
        this.checkpointDir = "./checkpoints/deepseek_r1_rlvr";
        
        // 使用SGD优化器
        this.optimizer = new SGD(model, learningRate);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.correctnessHistory = new ArrayList<>();
        this.rewardHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮数
     * @param learningRate 学习率
     * @param correctnessWeight 正确性权重
     * @param reasoningQualityWeight 推理质量权重
     * @param verificationWeight 验证完整性权重
     * @return 训练器自身
     */
    public DeepSeekR1RLVRTrainer configure(int maxEpochs, float learningRate,
                                           float correctnessWeight, 
                                           float reasoningQualityWeight,
                                           float verificationWeight) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.correctnessWeight = correctnessWeight;
        this.reasoningQualityWeight = reasoningQualityWeight;
        this.verificationWeight = verificationWeight;
        
        // 同步学习率到优化器
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 强化学习训练 (RLVR)");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("奖励权重配置:");
        System.out.println("  - 正确性权重: " + correctnessWeight);
        System.out.println("  - 推理质量权重: " + reasoningQualityWeight);
        System.out.println("  - 验证完整性权重: " + verificationWeight);
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        saveCheckpoint("final");
        printTrainingSummary();
        System.out.println("\nRLVR训练完成!");
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochCorrectness = 0.0;
        double epochReward = 0.0;
        double epochQuality = 0.0;
        int count = 0;
        
        while (dataset.hasNext()) {
            DeepSeekR1RLVRDataset.Batch batch = dataset.nextBatch();
            
            // 前向传播获取推理结果
            Variable inputVar = new Variable(batch.getInputIds());
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
            
            // 计算可验证奖励
            float batchCorrectness = 0.0f;
            String[] groundTruths = batch.getGroundTruths();
            String[] verifierTypes = batch.getVerifierTypes();
            
            // 为每个样本计算奖励
            for (int i = 0; i < batch.getBatchSize(); i++) {
                // 从模型logits解码生成输出
                String modelOutput = generateOutputFromLogits(result.logits, i);
                
                // 选择验证器
                Verifier verifier = verifiers.get(verifierTypes[i]);
                if (verifier == null) {
                    verifier = verifiers.get("math"); // 默认使用数学验证器
                }
                
                // 执行验证
                VerificationResult verification = verifier.verify(modelOutput, groundTruths[i]);
                
                // 累计正确性
                batchCorrectness += verification.getReward();
            }
            
            // 平均正确性奖励
            float avgCorrectness = batchCorrectness / batch.getBatchSize();
            
            // 推理质量评分（基于MoE损失）
            float qualityReward = (float) (1.0 - result.moeLoss);  // MoE损失越小，质量越高
            
            // 验证完整性评分（简化，不再依赖推理步数）
            float verificationScore = avgCorrectness;  // 直接使用正确性作为验证分数
            
            // 综合奖励（用于监控）
            float totalReward = correctnessWeight * avgCorrectness +
                               reasoningQualityWeight * qualityReward +
                               verificationWeight * verificationScore;
            
            // ========== RLVR核心改进: 基于目标答案的监督损失 ==========
            // 原问题: 仅用奖励值做反向传播，没有与模型输出建立计算图连接
            // 解决方案: 使用交叉熵损失引导模型学习正确答案的token分布
            Variable loss = computeRLVRLoss(result.logits, groundTruths, batch.getBatchSize(), avgCorrectness);
            
            // 调试输出：每10步打印损失值和预测信息
            if (globalStep % 10 == 0) {
                float lossVal = loss.getValue().getNumber().floatValue();
                int predictedAnswer = decodeNumberFromLogits(result.logits.getValue(), 0, 
                    result.logits.getValue().getShape().getShapeDims()[result.logits.getValue().getShape().getShapeDims().length - 1]);
                int targetAnswer = -1;
                try { targetAnswer = parseTargetAnswer(groundTruths[0]); } catch (Exception e) {}
                System.out.printf("  [DEBUG] Loss: %.4f | Predict: %d | Target: %d%n", lossVal, predictedAnswer, targetAnswer);
            }
            
            // 反向传播
            model.clearGrads();
            loss.backward();
            
            // 调试：检查参数梯度是否存在
            if (globalStep % 50 == 0) {
                int paramsWithGrad = 0;
                int paramsWithoutGrad = 0;
                StringBuilder noGradParams = new StringBuilder();
                for (var entry : model.getAllParams().entrySet()) {
                    if (entry.getValue().getGrad() != null) {
                        paramsWithGrad++;
                    } else {
                        paramsWithoutGrad++;
                        if (noGradParams.length() < 100) {
                            noGradParams.append(entry.getKey()).append(", ");
                        }
                    }
                }
                System.out.printf("  [DEBUG] Params with grad: %d | without: %d%n", paramsWithGrad, paramsWithoutGrad);
                
                // 打印logits的前几个值
                float[] logitsArr = result.logits.getValue().getArray();
                int[] logitsShape = result.logits.getValue().getShape().getShapeDims();
                int vocabSizeDebug = logitsShape[logitsShape.length - 1];
                int printLen = Math.min(5, logitsArr.length);
                StringBuilder sb = new StringBuilder("  [DEBUG] Logits[0:5]: ");
                for (int i = 0; i < printLen; i++) {
                    sb.append(String.format("%.2f ", logitsArr[i]));
                }
                sb.append(String.format("| vocabSize=%d", vocabSizeDebug));
                System.out.println(sb);
            }
            
            clipGradients();
            optimizer.update();
            
            // 记录统计
            correctnessHistory.add(avgCorrectness);
            rewardHistory.add(totalReward);
            qualityHistory.add(qualityReward);
            
            epochCorrectness += avgCorrectness;
            epochReward += totalReward;
            epochQuality += qualityReward;
            count++;
            globalStep++;
            
            // 日志输出
            if (globalStep % logInterval == 0) {
                System.out.printf(
                    "Epoch %d | Step %d | Correctness: %.4f | Reward: %.4f | Quality: %.4f%n",
                    currentEpoch + 1, globalStep, avgCorrectness, totalReward, qualityReward
                );
            }
        }
        
        // Epoch总结
        System.out.printf(
            "Epoch %d 完成 | 平均正确性: %.4f | 平均奖励: %.4f | 平均质量: %.4f%n",
            currentEpoch + 1, 
            epochCorrectness / count, 
            epochReward / count, 
            epochQuality / count
        );
        
        dataset.reset();
        
        // 保存检查点
        if ((currentEpoch + 1) % 2 == 0) {
            saveCheckpoint("epoch_" + (currentEpoch + 1));
        }
    }
    
    /**
     * 从logits生成输出文本
     * 
     * 基于模型logits解码生成答案，而不是随机生成
     * 这对于RLVR训练至关重要，因为需要验证模型实际输出的正确性
     */
    private String generateOutputFromLogits(Variable logits, int batchIdx) {
        NdArray logitsArray = logits.getValue();
        int[] shape = logitsArray.getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];
        
        // 从logits中解码出数字答案
        // 策略：提取最后几个位置的logits，解码为数字token
        int decodedNumber = decodeNumberFromLogits(logitsArray, batchIdx, vocabSize);
        
        // 使用答案模板
        String[] templates = {
            "Let me solve this step by step. The answer is %d.",
            "After careful reasoning, the result is %d.",
            "Through logical deduction, I conclude the answer is %d."
        };
        
        int templateIdx = globalStep % templates.length;
        return String.format(templates[templateIdx], decodedNumber);
    }
    
    /**
     * 从logits中解码数字
     * 
     * 对整个词表进行argmax，找到概率最高的位置作为答案
     * 这与computeRLVRLoss中的目标设置保持一致：
     * - 训练时在答案位置（如42）设置one-hot
     * - 解码时找到最高概率的位置作为预测答案
     */
    private int decodeNumberFromLogits(NdArray logitsArray, int batchIdx, int vocabSize) {
        int[] shape = logitsArray.getShape().getShapeDims();
        int seqLen = shape.length >= 2 ? shape[1] : 1;
        
        // 对最后一个位置的整个词表进行argmax
        double maxLogit = Double.NEGATIVE_INFINITY;
        int maxIdx = 0;
        
        // 遍历整个词表，找到概率最高的位置
        for (int v = 0; v < vocabSize; v++) {
            double logitVal;
            if (shape.length == 3) {
                // [batch, seq, vocab]
                logitVal = (double) logitsArray.get(batchIdx, seqLen - 1, v);
            } else if (shape.length == 2) {
                // [seq, vocab]
                logitVal = (double) logitsArray.get(seqLen - 1, v);
            } else {
                logitVal = (double) logitsArray.get(v);
            }
            
            if (logitVal > maxLogit) {
                maxLogit = logitVal;
                maxIdx = v;
            }
        }
        
        // 返回最高概率位置作为预测答案
        // 这与损失函数中的目标设置一致：答案数字直接作为词表索引
        return maxIdx;
    }
    
    /**
     * 计算RLVR损失
     * 
     * 使用Variable的softmaxCrossEntropy方法，正确建立计算图
     * 这是RLVR的核心：让模型学习在正确答案的token位置输出高概率
     * 
     * @param logits 模型输出的logits
     * @param groundTruths 标准答案数组
     * @param batchSize 批次大小
     * @param currentCorrectness 当前正确率（用于奖励缩放）
     * @return 损失变量
     */
    private Variable computeRLVRLoss(Variable logits, String[] groundTruths, int batchSize, float currentCorrectness) {
        // 获取形状信息
        int[] shape = logits.getValue().getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];
        int seqLen = shape.length >= 2 ? shape[1] : 1;
        int actualBatchSize = shape.length >= 1 ? shape[0] : 1;
        
        // 解析目标答案，使用-1标记无效样本
        float[] targetIndices = new float[actualBatchSize];
        boolean[] validMask = new boolean[actualBatchSize];
        int validSamples = 0;
        for (int b = 0; b < Math.min(batchSize, actualBatchSize); b++) {
            try {
                int targetAnswer = parseTargetAnswer(groundTruths[b]);
                targetIndices[b] = Math.min(Math.max(0, targetAnswer), vocabSize - 1);
                validMask[b] = true;
                validSamples++;
            } catch (NumberFormatException e) {
                // 无效样本：设为0但标记为无效
                targetIndices[b] = 0;
                validMask[b] = false;
            }
        }
        
        if (validSamples == 0) {
            return logits.sum().mul(new Variable(NdArray.of(0.0001f)));
        }
        
        // ========== 简化方案：使用sliceRange提取最后位置 ==========
        // logits形状: [batch, seq, vocab]
        // 提取最后一个位置: [batch, 1, vocab]
        Variable lastPosLogits;
        if (shape.length == 3) {
            lastPosLogits = logits.sliceRange(1, seqLen - 1, seqLen);
            // squeeze去掉中间维度: [batch, vocab]
            lastPosLogits = lastPosLogits.reshape(Shape.of(actualBatchSize, vocabSize));
        } else if (shape.length == 2) {
            // 已经是[seq, vocab]，取最后一行
            lastPosLogits = logits.sliceRange(0, seqLen - 1, seqLen);
        } else {
            lastPosLogits = logits;
        }
        
        // 创建目标Variable
        Variable target = new Variable(NdArray.of(targetIndices).reshape(Shape.of(actualBatchSize, 1)));
        
        // 计算softmax交叉熵损失
        Variable loss = lastPosLogits.softmaxCrossEntropy(target);
        
        // 根据当前正确率动态调整损失权重
        float rewardScale = 1.0f + (1.0f - currentCorrectness) * 2.0f;
        Variable scaleVar = new Variable(NdArray.of(rewardScale));
        
        return loss.mul(scaleVar);
    }

    /**
     * 解析目标答案为整数
     */
    private int parseTargetAnswer(String groundTruth) throws NumberFormatException {
        if (groundTruth == null || groundTruth.trim().isEmpty()) {
            throw new NumberFormatException("答案为空");
        }
        
        String answer = groundTruth.trim().toLowerCase();
        
        // 支持布尔值
        if (answer.equals("true") || answer.equals("yes")) {
            return 1;
        }
        if (answer.equals("false") || answer.equals("no")) {
            return 0;
        }
        
        // 提取第一个数字
        String cleaned = groundTruth.trim().replaceAll("[^0-9.-]", " ").trim();
        String[] parts = cleaned.split("\\s+");
        if (parts.length > 0 && !parts[0].isEmpty()) {
            return (int) Double.parseDouble(parts[0]);
        }
        throw new NumberFormatException("无法解析: " + groundTruth);
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        
        for (Parameter param : params.values()) {
            if (param.grad() != null) {
                double norm = param.grad().mul(param.grad()).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.grad() != null) {
                    param.setGrad(param.grad().mulNum(scale));
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filepath = checkpointDir + File.separator +
                            String.format("deepseek_r1_rlvr_%s.model", suffix);
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (Exception e) {
            System.err.println("创建目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 打印训练总结
     */
    private void printTrainingSummary() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("RLVR训练总结");
        System.out.println("=".repeat(70));
        
        if (!correctnessHistory.isEmpty()) {
            float avgCorrectness = calculateAverage(correctnessHistory);
            float avgReward = calculateAverage(rewardHistory);
            float avgQuality = calculateAverage(qualityHistory);
            
            System.out.printf("总训练步数: %d\n", globalStep);
            System.out.printf("平均正确率: %.4f\n", avgCorrectness);
            System.out.printf("平均综合奖励: %.4f\n", avgReward);
            System.out.printf("平均推理质量: %.4f\n", avgQuality);
            
            // 计算趋势
            if (correctnessHistory.size() >= 10) {
                float earlyCorrectness = calculateAverage(
                    correctnessHistory.subList(0, 10)
                );
                float lateCorrectness = calculateAverage(
                    correctnessHistory.subList(
                        correctnessHistory.size() - 10, 
                        correctnessHistory.size()
                    )
                );
                float improvement = lateCorrectness - earlyCorrectness;
                System.out.printf("正确率提升: %.4f\n", improvement);
            }
        }
        
        System.out.println("=".repeat(70));
    }
    
    /**
     * 计算平均值
     */
    private float calculateAverage(List<Float> values) {
        if (values == null || values.isEmpty()) return 0.0f;
        float sum = 0.0f;
        for (float v : values) sum += v;
        return sum / values.size();
    }
    
    /**
     * 获取训练统计
     */
    public Map<String, Object> getTrainingStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_steps", globalStep);
        stats.put("avg_correctness", calculateAverage(correctnessHistory));
        stats.put("avg_reward", calculateAverage(rewardHistory));
        stats.put("avg_quality", calculateAverage(qualityHistory));
        return stats;
    }
}
