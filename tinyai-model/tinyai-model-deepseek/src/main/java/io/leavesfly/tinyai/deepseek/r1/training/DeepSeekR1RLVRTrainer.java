package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1RLVRDataset;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.*;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * DeepSeek-R1强化学习训练器 (GRPO - Group Relative Policy Optimization)
 * 
 * 对标 arXiv:2501.12948 DeepSeek-R1论文中的GRPO算法。
 * 
 * GRPO vs 简单策略梯度：
 * 
 * | 维度         | 简单策略梯度       | GRPO                          |
 * |--------------|-----------------|-------------------------------|
 * | 采样策略     | 每问题1个输出     | 每问题G个输出（组采样）       |
 * | 基线         | 固定基线或无     | 组内均值奖励（相对优势）     |
 * | 优势函数     | r 或 r-b         | (r_i-mean)/std（组内归一化）    |
 * | 策略更新     | 无约束           | PPO-clip防止策略劇变           |
 * | 值函数网络   | 需要             | 不需要（GRPO的核心优势）      |
 * 
 * GRPO训练流程（对标论文第4节）：
 * 1. 对每个问题 q，从当前策略采样 G 个输出 {o1,...,oG}
 * 2. 验证器为每个输出计算奖励 {r1,...,rG}
 * 3. 组内相对优势： A_i = (r_i - mean(r)) / std(r)
 * 4. PPO clip目标： L = clip(A_g, -ε, +ε) * CE(logits, o_g)
 * 5. 反向传播更新参数
 * 
 * RLVR 可验证奖励类型：
 * - 数学验证器 (MathVerifier)
 * - 代码验证器 (CodeVerifier)
 * - 逻辑验证器 (LogicVerifier)
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1RLVRTrainer extends DeepSeekTrainerBase {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1RLVRDataset dataset;
    private final SGD optimizer;
    
    // 验证器映射
    private final Map<String, Verifier> verifiers;
    
    // 训练参数
    private float learningRate;
    
    // 奖励权重
    private float correctnessWeight;     // 正确性权重 (主)
    private float reasoningQualityWeight; // 推理质量权重 (辅)
    private float verificationWeight;     // 验证完整性权重 (辅)
    
    // 训练统计
    private List<Float> correctnessHistory;
    private List<Float> qualityHistory;
    
    // ========== GRPO参数 ==========
    /** 组采样大小G：每个问题采样的输出数量（论文默认16，教学简化为4） */
    private int groupSize = 4;
    /** PPO clip范围ε：限制优势幅度，防止策略更新过激 */
    private float clipEps = 0.2f;
    /** 优势归一化稳定项：防止std为0时除以零 */
    private float advantageEps = 1e-8f;
    /** 采样温度：>1.0 输出更多样。=1.0 正常采样 */
    private float temperature = 1.0f;
    
    /**
     * 构造函数
     * 
     * @param model DeepSeek-R1模型
     * @param dataset RLVR数据集
     */
    public DeepSeekR1RLVRTrainer(DeepSeekR1Model model, DeepSeekR1RLVRDataset dataset) {
        super(model, 5, 1.0f, 10, "./checkpoints/deepseek_r1/rlvr");
        
        this.model = model;
        this.dataset = dataset;
        
        // 初始化验证器
        this.verifiers = new HashMap<>();
        this.verifiers.put("math", new MathVerifier());
        this.verifiers.put("code", new CodeVerifier());
        this.verifiers.put("logic", new LogicVerifier());
        
        // RLVR训练参数（与RLHF类似但更激进）
        this.learningRate = 5e-5f;  // RLVR可以使用稍大的学习率
        
        // 奖励权重配置
        this.correctnessWeight = 0.7f;      // 正确性最重要
        this.reasoningQualityWeight = 0.2f;  // 推理质量
        this.verificationWeight = 0.1f;      // 验证完整性
        
        // 使用SGD优化器
        this.optimizer = new SGD(model, learningRate);
        
        // 初始化状态
        this.correctnessHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮数
     * @param learningRate 学习率
     * @param groupSize GRPO组大小G（每个问题采样数）
     * @param clipEps GRPO PPO clip范围
     * @param temperature 采样温度
     * @return 训练器自身
     */
    public DeepSeekR1RLVRTrainer configure(int maxEpochs, float learningRate,
                                           int groupSize, float clipEps, float temperature) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.groupSize = groupSize;
        this.clipEps = clipEps;
        this.temperature = temperature;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     */
    @Override
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 强化学习训练 (GRPO - Group Relative Policy Optimization)");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("GRPO参数:");
        System.out.println("  - 组采样大小 G: " + groupSize);
        System.out.println("  - PPO clip ε: " + clipEps);
        System.out.println("  - 采样温度: " + temperature);
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        saveCheckpoint("final");
        printTrainingSummary();
        System.out.println("\nGRPO训练完成!");
    }
    
    /**
     * 训练一个epoch（GRPO组采样算法）
     * 
     * 对每个batch中的每个样本，进行以下操作：
     * 1. 前向传播获取当前策略logits
     * 2. 组采样：从 logits 中采样 G 个不同的输出
     * 3. 验证器打分：计算每个输出的奖励
     * 4. 组内相对优势：A_i = (r_i - mean) / std
     * 5. GRPO损失：clip(A_g, -ε, +ε) * CE(logits, o_g)
     * 6. 反向传播 + 参数更新
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
    
        double epochAvgReward = 0.0;
        double epochAvgLoss   = 0.0;
        int count = 0;
        int epochGrpoTotal = 0;
        int epochFallbackTotal = 0;
    
        while (dataset.hasNext()) {
            DeepSeekR1RLVRDataset.Batch batch = dataset.nextBatch();
    
            // 前向传播（获取当前策略logits）
            Variable inputVar = new Variable(batch.getInputIds());
            DeepSeekR1Model.ReasoningResult result = model.performReasoning(inputVar);
    
            BatchResult br = processBatch(batch, result);
    
            if (br.totalLoss == null || br.validSamples == 0) {
                // 即使跳过更新，也需要释放计算图，防止内存泄漏
                result.logits.unChainBackward();
                inputVar.unChainBackward();
                globalStep++;
                continue;
            }
    
            // 对batch求平均 + 反向传播更新
            Variable avgLoss = br.totalLoss.mul(new Variable(NdArray.of(1.0f / br.validSamples)));
            float batchLossValue = avgLoss.getValue().getNumber().floatValue();
            float batchAvgReward = br.totalReward / br.validSamples;
    
            model.clearGrads();
            avgLoss.backward();
            clipGradients();
            optimizer.update();
    
            // 彻底断开计算图，释放内存（与 SFTrainer/Pretrain 保持一致）
            avgLoss.unChainBackward();
            result.logits.unChainBackward();
            inputVar.unChainBackward();
    
            // 记录统计
            correctnessHistory.add(batchAvgReward);
            qualityHistory.add((float) result.moeLoss);
            lossHistory.add(batchLossValue);
            
            epochAvgReward += batchAvgReward;
            epochAvgLoss   += batchLossValue;
            count++;
            globalStep++;
    
            if (globalStep % logInterval == 0) {
                System.out.printf(
                    "Epoch %d | Step %d | Loss: %.4f | GRPO: %d | Fallback: %d | Reward: %.4f%n",
                    currentEpoch + 1, globalStep, batchLossValue, br.grpoSamples, br.fallbackSamples, batchAvgReward
                );
            }
            epochGrpoTotal += br.grpoSamples;
            epochFallbackTotal += br.fallbackSamples;
        }
    
        System.out.printf(
            "Epoch %d 完成 | GRPO样本: %d | Fallback样本: %d | 平均Loss: %.4f | 平均奖励: %.4f%n",
            currentEpoch + 1, epochGrpoTotal, epochFallbackTotal,
            count > 0 ? epochAvgLoss / count : 0.0,
            count > 0 ? epochAvgReward / count : 0.0
        );
    
        dataset.reset();
        if ((currentEpoch + 1) % 10 == 0) {
            saveCheckpoint("epoch_" + (currentEpoch + 1));
        }
    }
    
    /** 单个 batch GRPO处理结果 */
    private static class BatchResult {
        Variable totalLoss;
        float totalReward;
        int validSamples;
        int grpoSamples;
        int fallbackSamples;
    }
    
    /**
     * 对单个 batch 执行 GRPO 组采样并汇总损失
     *
     * @param batch  当前批次数据
     * @param result 前向传播结果
     * @return 汇总损失及统计信息
     */
    private BatchResult processBatch(DeepSeekR1RLVRDataset.Batch batch,
                                     DeepSeekR1Model.ReasoningResult result) {
        BatchResult br = new BatchResult();
        String[] groundTruths  = batch.getGroundTruths();

        for (int i = 0; i < batch.getBatchSize(); i++) {
            // Step1: 组采样（G 个输出）
            double targetValue = parseGroundTruthAsDouble(groundTruths[i]);
            float[] groupRewards = new float[groupSize];
            int[]   groupTokens  = new int[groupSize];
            for (int g = 0; g < groupSize; g++) {
                groupTokens[g]  = sampleTokenFromLogits(result.logits, i, temperature);
                groupRewards[g] = computeProximityReward(groupTokens[g], targetValue);
            }
    
            // Step2: 组内相对优势 A_i = (r_i - mean) / std
            float[] advantages = computeGroupAdvantages(groupRewards);
    
            // 检查是否有有效学习信号
            boolean hasSignal = false;
            for (float a : advantages) {
                if (Math.abs(a) > 1e-6f) { hasSignal = true; break; }
            }
    
            // Step3: 计算损失
            Variable sampleLoss;
            float meanReward = calculateAverage(groupRewards);
            if (!hasSignal) {
                // Fallback：组内无差异信号，全错时用监督学习引导
                if (meanReward < 0.5f) {
                    sampleLoss = computeFallbackCELoss(result.logits, i, groundTruths[i]);
                    if (sampleLoss != null) br.fallbackSamples++;
                } else {
                    continue; // 全对：无需更新
                }
            } else {
                // GRPO正常路径
                sampleLoss = computeGRPOLoss(result.logits, i, groupTokens, advantages);
                if (sampleLoss != null) br.grpoSamples++;
            }
            if (sampleLoss == null) continue;
    
            br.totalLoss   = (br.totalLoss == null) ? sampleLoss : br.totalLoss.add(sampleLoss);
            br.totalReward += meanReward;
            br.validSamples++;
        }
        return br;
    }
    
    /**
     * 提取当前样本最后位置的 logits → [1, vocabSize]
     * 
     * 统一处理 3D [batchSize, seqLen, vocabSize]、2D [seqLen, vocabSize]、1D [vocabSize] 三种形状，
     * 供 computeGRPOLoss、computeFallbackCELoss、sampleTokenFromLogits 共用。
     * 
     * @param logits    模型输出 logits
     * @param sampleIdx 当前样本在 batch 中的索引（仅 3D 时使用）
     * @return shape [1, vocabSize] 的 logits
     */
    private Variable extractSampleLogits(Variable logits, int sampleIdx) {
        int[] shape = logits.getValue().getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];
        int seqLen = shape.length >= 2 ? shape[shape.length - 2] : 1;

        if (shape.length == 3) {
            // [batchSize, seqLen, vocabSize] → 取 sampleIdx 行，再取最后时间步
            Variable sample = logits.sliceRange(0, sampleIdx, sampleIdx + 1); // [1, seqLen, vocabSize]
            Variable last   = sample.sliceRange(1, seqLen - 1, seqLen);       // [1, 1, vocabSize]
            return last.reshape(Shape.of(1, vocabSize));                       // [1, vocabSize]
        } else if (shape.length == 2) {
            return logits.sliceRange(0, seqLen - 1, seqLen);                  // [1, vocabSize]
        } else {
            return logits;                                                     // [vocabSize]
        }
    }

    /**
     * 计算 GRPO 损失（对标论文公式 (4)）
     * 
     * L_GRPO = -1/G * Σ_g [ clip(A_g, -ε, +ε) * log π(o_g | q) ]
     * 
     * 其中 log π(o_g | q) 用 softmax cross-entropy 近似：
     * 对每个采样 token o_g，计算 -log p(o_g | logits) 即 CE 损失。
     * clip(A_g, -ε, +ε) 作为该 CE 损失的加权系数（相当于PPO的保守裁剪）。
     * 
     * @param logits      模型输出 logits
     * @param sampleIdx   当前样本在 batch 中的索引
     * @param groupTokens 组内 G 个采样 token
     * @param advantages  组内归一化优势 A_g
     * @return GRPO 损失（标量 Variable）
     */
    private Variable computeGRPOLoss(Variable logits, int sampleIdx,
                                     int[] groupTokens, float[] advantages) {
        Variable sampleLogits = extractSampleLogits(logits, sampleIdx);

        // 对 G 个采样 token 分别计算 clip(A_g) * CE
        Variable grpoLoss = null;
        for (int g = 0; g < groupTokens.length; g++) {
            // PPO clip：将优势幅度限制在 [-clipEps, +clipEps]
            float clippedAdv = Math.max(-clipEps, Math.min(clipEps, advantages[g]));
            if (Math.abs(clippedAdv) < 1e-9f) continue;

            // 目标 token（采样到的那个词）作为 CE 的 label
            float[] targetArr = {groupTokens[g]};
            Variable target = new Variable(NdArray.of(targetArr).reshape(Shape.of(1, 1)));

            // CE(logits, token_g) = -log p(token_g)
            // GRPO：最大化 clip(A_g) * log π(o_g) ↔ 最小化 clip(A_g) * CE
            // （A_g 正 → 鼓励该输出，A_g 负 → 抑制该输出）
            Variable weighted = sampleLogits.softmaxCrossEntropy(target)
                    .mul(new Variable(NdArray.of(clippedAdv)));
            grpoLoss = (grpoLoss == null) ? weighted : grpoLoss.add(weighted);
        }

        if (grpoLoss == null) return null;

        // 对组内有效样本求平均
        return grpoLoss.mul(new Variable(NdArray.of(1.0f / groupTokens.length)));
    }

    /**
     * 计算组内相对优势（对标论文 A_i = (r_i - mean) / std）
     * 
     * 这是 GRPO 相对于简单策略梯度的核心创新：
     * - 不需要独立的值函数网络
     * - 用同组其他样本奖励的均值作为基线
     * - std 归一化控制优势幅度
     * 
     * @param rewards G 个输出的奖励数组
     * @return 归一化后的优势数组
     */
    private float[] computeGroupAdvantages(float[] rewards) {
        int G = rewards.length;
        // 计算组内均值
        float mean = 0.0f;
        for (float r : rewards) mean += r;
        mean /= G;

        // 计算组内标准差
        float variance = 0.0f;
        for (float r : rewards) variance += (r - mean) * (r - mean);
        variance /= G;
        float std = (float) Math.sqrt(variance + advantageEps);

        // 归一化优势 A_i = (r_i - mean) / std
        // 当 std 过小（组内所有奖励相同）时，使用均匀优势（所有优势设为0）
        float[] advantages = new float[G];
        if (std < advantageEps) {
            // 标准差过小，使用均匀优势
            for (int i = 0; i < G; i++) {
                advantages[i] = 0.0f;
            }
        } else {
            for (int i = 0; i < G; i++) {
                advantages[i] = (rewards[i] - mean) / std;
            }
        }
        return advantages;
    }

    /**
     * Fallback CE 损失：当组内无差异信号且全错时，用监督学习引导模型学习正确答案
     * 
     * 使用 groundTruth 的哈希值映射到合法 token 范围，确保始终能生成有效损失。
     * 
     * @param logits      模型输出 logits
     * @param sampleIdx   当前样本在 batch 中的索引
     * @param groundTruth 正确答案字符串
     * @return CE 损失
     */
    private Variable computeFallbackCELoss(Variable logits, int sampleIdx, String groundTruth) {
        int vocabSize = logits.getValue().getShape().getShapeDims()[logits.getValue().getShape().getShapeDims().length - 1];
        int targetToken = mapGroundTruthToToken(groundTruth, vocabSize);

        Variable sampleLogits = extractSampleLogits(logits, sampleIdx);

        float[] targetArr = {targetToken};
        Variable target = new Variable(NdArray.of(targetArr).reshape(Shape.of(1, 1)));
        return sampleLogits.softmaxCrossEntropy(target);
    }
    
    /**
     * 将 groundTruth 字符串映射到合法 token 范围 [0, vocabSize)
     * 
     * 优先尝试解析为数值并取模，失败时使用哈希值取模，确保始终返回合法 token。
     */
    private int mapGroundTruthToToken(String groundTruth, int vocabSize) {
        if (groundTruth == null || groundTruth.trim().isEmpty()) {
            return 0;
        }
        try {
            int parsed = (int) Double.parseDouble(groundTruth.trim());
            return Math.abs(parsed) % vocabSize;
        } catch (NumberFormatException ignored) {
        }
        // 布尔值
        String lower = groundTruth.trim().toLowerCase();
        if (lower.equals("true") || lower.equals("yes")) return 1 % vocabSize;
        if (lower.equals("false") || lower.equals("no")) return 0;
        // 兜底：哈希映射
        return Math.abs(groundTruth.hashCode()) % vocabSize;
    }
    
    /**
     * 解析 groundTruth 为 double 数值
     * 
     * 支持纯数字、布尔值，解析失败时使用哈希值作为替代。
     */
    private double parseGroundTruthAsDouble(String groundTruth) {
        if (groundTruth == null || groundTruth.trim().isEmpty()) return 0.0;
        String trimmed = groundTruth.trim().toLowerCase();
        if (trimmed.equals("true") || trimmed.equals("yes")) return 1.0;
        if (trimmed.equals("false") || trimmed.equals("no")) return 0.0;
        try {
            return Double.parseDouble(trimmed);
        } catch (NumberFormatException e) {
            return Math.abs(groundTruth.hashCode()) % 1000.0;
        }
    }
    
    /**
     * 基于采样 token 与正确答案的距离计算连续奖励
     * 
     * 使用高斯核函数：reward = exp(-distance² / (2σ²))
     * 距离越近奖励越高（趋近1.0），距离越远奖励越低（趋近0.0）。
     * 不同采样 token 与目标的距离不同，自然产生差异化的组内奖励信号。
     * 
     * @param sampledToken 采样到的 token id
     * @param targetValue  正确答案的数值
     * @return 连续奖励值 [0, 1]
     */
    private float computeProximityReward(int sampledToken, double targetValue) {
        double distance = sampledToken - targetValue;
        // σ 控制奖励衰减速度，较小的 σ 使奖励更集中在正确答案附近
        double sigma = Math.max(10.0, Math.abs(targetValue) * 0.3);
        return (float) Math.exp(-(distance * distance) / (2.0 * sigma * sigma));
    }

    /**
     * 温度采样：从 logits 中以温度 τ 按概率采样一个 token
     * 
     * softmax(logits / τ) 后多项式采样，τ>1 更随机，τ→0 趋向 argmax
     * 
     * @param logits    模型输出 logits
     * @param sampleIdx 当前样本在 batch 中的索引
     * @param temp      采样温度
     * @return 采样到的 token id
     */
    private int sampleTokenFromLogits(Variable logits, int sampleIdx, float temp) {
        // 复用 extractSampleLogits 统一切片，直接读取其底层数组
        Variable sliced = extractSampleLogits(logits, sampleIdx);
        NdArray logitsArr = sliced.getValue();
        int[] shape = logitsArr.getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];

        // 提取最后位置 logits
        float[] rawLogits = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            rawLogits[v] = shape.length >= 2 ? logitsArr.get(0, v) : logitsArr.get(v);
        }

        // 温度缩放并 softmax
        float maxVal = rawLogits[0];
        for (float x : rawLogits) if (x > maxVal) maxVal = x;

        float[] probs = new float[vocabSize];
        float sumExp = 0.0f;
        for (int v = 0; v < vocabSize; v++) {
            probs[v] = (float) Math.exp((rawLogits[v] - maxVal) / temp);
            sumExp += probs[v];
        }
        for (int v = 0; v < vocabSize; v++) probs[v] /= sumExp;

        // 多项式采样（CDF 逆变换）
        float rand = (float) Math.random();
        float cumProb = 0.0f;
        for (int v = 0; v < vocabSize; v++) {
            cumProb += probs[v];
            if (rand <= cumProb) return v;
        }
        return vocabSize - 1;
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
            float avgReward = calculateAverage(correctnessHistory);
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
     * 获取训练统计
     */
    public Map<String, Object> getTrainingStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_steps", globalStep);
        stats.put("avg_correctness", calculateAverage(correctnessHistory));
        stats.put("avg_reward", calculateAverage(correctnessHistory));
        stats.put("avg_quality", calculateAverage(qualityHistory));
        return stats;
    }
    
    /**
     * 获取训练器名称
     */
    @Override
    public String getTrainerName() {
        return "DeepSeek-R1 RLVR";
    }
    
    /**
     * 获取检查点前缀
     */
    @Override
    public String getCheckpointPrefix() {
        return "deepseek_r1_rlvr";
    }
}
