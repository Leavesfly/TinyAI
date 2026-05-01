package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTrainerBase;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1RLVRDataset;
import io.leavesfly.tinyai.deepseek.r1.training.verifier.*;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * DeepSeek-R1 强化学习训练器（GRPO - Group Relative Policy Optimization）
 *
 * <p>完整实现 arXiv:2501.12948 论文第 4 节的 GRPO 算法，包含以下关键组件：
 * <ul>
 *   <li><b>组采样</b>：每个 query 采样 G 个候选 token，{@link #groupSize}</li>
 *   <li><b>组内相对优势</b>：A_i = (r_i - mean) / std</li>
 *   <li><b>奖励</b>：真实调用 {@link MathVerifier} / {@link CodeVerifier} / {@link LogicVerifier}
 *       产生 r_verify，叠加 r_proximity 组内差异信号；r = cw·r_verify + qw·r_prox - vw·moeLoss</li>
 *   <li><b>Importance sampling ratio</b>：ratio = exp(log π_new(o|q) - log π_old(o|q))，
 *       其中 log π_old 为采样时刻的 detach 快照</li>
 *   <li><b>PPO-clip 损失</b>：L_policy = -E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)]</li>
 *   <li><b>KL 约束</b>：L_kl = β · E[log π_new - log π_ref]，π_ref 来自 {@link GRPOReferenceModel}</li>
 *   <li><b>多轮更新</b>：每个 rollout 做 {@link #innerUpdatesPerBatch} 次梯度更新，
 *       使 ratio 在 ε-ball 外时被 clip 真正发挥作用</li>
 *   <li><b>参考模型同步</b>：每 {@link #refSyncEpochInterval} 个 epoch 将 π_ref 同步为最新 policy</li>
 *   <li><b>Fallback CE</b>：组内无差异信号时退化为监督学习</li>
 * </ul>
 *
 * <p>与论文 GRPO 的差异（务实说明）：
 * <ul>
 *   <li>采样单 token 而非生成完整序列（受限于 tokenizer 基础设施，教学场景足够）</li>
 *   <li>token→verifier 字符串通过 {@link #tokenToVerifierString} 简单格式化，
 *       生产环境应接入真实 detokenizer</li>
 * </ul>
 *
 * @author leavesfly
 * @version 3.0
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
    /** 正确性历史：每 step 记录一次 batch 平均 r_verify（来自验证器，0/1 二值期望） */
    private final List<Float> correctnessHistory;
    /** 推理质量历史：每 step 记录一次 batch 的 MoE 辅助损失（越低越好） */
    private final List<Float> qualityHistory;
    /** 综合奖励历史：每 step 记录一次 batch 平均综合奖励（correctness + proximity 加权求和） */
    private final List<Float> rewardHistory;
    
    // ========== GRPO 参数 ==========
    /** 组采样大小 G：每个问题采样的输出数量（论文默认 16，教学简化为 4） */
    private int groupSize = 4;

    /** PPO-clip 范围 ε：ratio 被限制在 [1-ε, 1+ε]，论文默认 0.2 */
    private float clipEps = 0.2f;

    /** KL 约束权重 β：L_kl = β · (log π_new - log π_ref)，论文默认 0.04 */
    private float klWeight = 0.04f;

    /** 优势归一化稳定项：防止 std 为 0 时除以零 */
    private float advantageEps = 1e-8f;

    /** 采样温度：>1.0 输出更多样，=1.0 正常采样 */
    private float temperature = 1.0f;

    /**
     * 参考模型同步周期（epoch 数）：每 N 个 epoch 同步一次 π_ref ← π_current。
     * <p>1 表示每个 epoch 开始时都同步（近似 PPO 每轮刷新），
     * N>1 时 π_ref 会滞后，KL 惩罚更强地约束策略漂移。
     */
    private int refSyncEpochInterval = 1;

    /**
     * 每个 rollout 的内部梯度更新次数 K：对同一批采样执行 K 次梯度更新。
     * <p>K=1 时 ratio 初始恒为 1，clip 不起作用；K>1 时后续 K-1 次更新的 ratio
     * 会真实偏离 1，PPO-clip 的安全护栏才会真正发挥作用。论文默认 K=1 足够，
     * 但为了让 clip 在教学演示中可观测，默认设为 2。
     */
    private int innerUpdatesPerBatch = 2;

    // ========== 参考模型（KL 约束用） ==========
    /**
     * 参考模型 π_ref：用于计算 KL 约束。
     * <p>所有参数 requireGrad=false，仅做 forward。在 {@link #train()} 中延迟初始化，
     * 每 {@link #refSyncEpochInterval} 个 epoch 同步一次。
     */
    private GRPOReferenceModel referenceModel;
    
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

        // RLVR 训练参数
        this.learningRate = 5e-5f;

        // 奖励权重配置
        this.correctnessWeight = 0.7f;      // 正确性最重要
        this.reasoningQualityWeight = 0.2f; // 推理质量
        this.verificationWeight = 0.1f;     // 验证完整性（惩罚 MoE 负载不均）

        // 使用 SGD 优化器
        this.optimizer = new SGD(model, learningRate);

        // 初始化状态
        this.correctnessHistory = new ArrayList<>();
        this.qualityHistory = new ArrayList<>();
        this.rewardHistory = new ArrayList<>();

        // 参考模型 π_ref：与 policy 同架构、参数快照、全部 frozen
        // DeepSeekR1Model.getConfig() 直接返回 DeepSeekR1Config，无需 instanceof 强转
        this.referenceModel = new GRPOReferenceModel(model, model.getConfig());
    }
    
    /**
     * 配置训练参数
     *
     * @param maxEpochs    最大训练轮数
     * @param learningRate 学习率
     * @param groupSize    GRPO 组大小 G（每个问题采样数）
     * @param temperature  采样温度
     * @return 训练器自身，支持链式调用
     */
    public DeepSeekR1RLVRTrainer configure(int maxEpochs, float learningRate,
                                           int groupSize, float temperature) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.groupSize = groupSize;
        this.temperature = temperature;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }

    /**
     * 配置 GRPO 算法超参数
     *
     * @param clipEps              PPO-clip 范围 ε，ratio 限制在 [1-ε, 1+ε]，论文默认 0.2
     * @param klWeight             KL 约束权重 β，L_kl = β · (log π_new - log π_ref)，论文默认 0.04
     * @param innerUpdatesPerBatch 每个 rollout 的内部梯度更新次数 K，K=1 等价于首次 PPO 更新
     * @param refSyncEpochInterval 参考模型同步周期（epoch 数），1=每 epoch 同步
     * @return 训练器自身，支持链式调用
     */
    public DeepSeekR1RLVRTrainer configureGRPO(float clipEps, float klWeight,
                                               int innerUpdatesPerBatch, int refSyncEpochInterval) {
        if (clipEps <= 0 || clipEps >= 1) {
            throw new IllegalArgumentException("clipEps must be in (0, 1), got " + clipEps);
        }
        if (klWeight < 0) {
            throw new IllegalArgumentException("klWeight must be >= 0, got " + klWeight);
        }
        if (innerUpdatesPerBatch < 1) {
            throw new IllegalArgumentException("innerUpdatesPerBatch must be >= 1, got " + innerUpdatesPerBatch);
        }
        if (refSyncEpochInterval < 1) {
            throw new IllegalArgumentException("refSyncEpochInterval must be >= 1, got " + refSyncEpochInterval);
        }
        this.clipEps = clipEps;
        this.klWeight = klWeight;
        this.innerUpdatesPerBatch = innerUpdatesPerBatch;
        this.refSyncEpochInterval = refSyncEpochInterval;
        return this;
    }

    /**
     * 配置奖励权重
     *
     * <p>综合奖励公式：
     * <pre>
     *     r = correctnessWeight · r_verify + reasoningQualityWeight · r_proximity
     * </pre>
     * 其中 r_verify 来自 Verifier.verify()（二值），r_proximity 为 token 到 groundTruth 数值的高斯核软奖励。
     * verificationWeight 作用于记录阶段：综合奖励额外累加 {@code - verificationWeight · moeAuxLoss} 以惩罚负载不均。
     *
     * @param correctness  正确性权重（r_verify）
     * @param quality      推理质量权重（r_proximity）
     * @param verification 验证完整性权重（对 MoE 辅助损失的惩罚系数）
     * @return 训练器自身，支持链式调用
     */
    public DeepSeekR1RLVRTrainer configureRewardWeights(float correctness, float quality, float verification) {
        this.correctnessWeight = correctness;
        this.reasoningQualityWeight = quality;
        this.verificationWeight = verification;
        return this;
    }
    
    /**
     * 开始训练
     */
    @Override
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 强化学习训练（GRPO - Group Relative Policy Optimization）");
        System.out.println("=".repeat(70));
        System.out.println("模型: " + model.getName());
        System.out.println("训练样本: " + dataset.getSampleCount());
        System.out.println("学习率: " + learningRate);
        System.out.println("GRPO 参数:");
        System.out.println("  - 组采样大小 G: " + groupSize);
        System.out.println("  - 采样温度: " + temperature);
        System.out.printf("  - PPO-clip ε: %.3f%n", clipEps);
        System.out.printf("  - KL 权重 β: %.4f%n", klWeight);
        System.out.println("  - 内部更新轮数 K: " + innerUpdatesPerBatch);
        System.out.println("  - 参考模型同步周期: " + refSyncEpochInterval + " epoch");
        System.out.println("奖励权重:");
        System.out.printf("  - correctness(r_verify): %.2f%n", correctnessWeight);
        System.out.printf("  - quality(r_proximity): %.2f%n", reasoningQualityWeight);
        System.out.printf("  - verification(moeAux 惩罚): %.2f%n", verificationWeight);
        System.out.println("验证器: " + verifiers.keySet());
        System.out.println("=".repeat(70));

        createCheckpointDir();

        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            // P0-3e: 参考模型按周期同步为当前 policy
            if (currentEpoch % refSyncEpochInterval == 0) {
                System.out.printf("[Epoch %d] 同步参考模型 π_ref ← π_current%n", currentEpoch + 1);
                referenceModel.syncFrom(model);
            }
            trainOneEpoch();
        }

        saveCheckpoint("final");
        printTrainingSummary();
        System.out.println("\nGRPO 训练完成!");
    }
    
    /**
     * 训练一个 epoch（完整 GRPO 算法）
     *
     * <p>每个 batch 的处理流程：
     * <ol>
     *   <li><b>初次 forward</b>：policy(x) → logits_initial，用 detach 提取 old_log_probs 快照（固定）</li>
     *   <li><b>参考模型 forward</b>：π_ref(x) → logits_ref → ref_log_probs（detach, frozen）</li>
     *   <li><b>采样与打分</b>：从 logits_initial 采样 G 个 token，真实调用 verifier 得 reward，计算组内优势 A</li>
     *   <li><b>K 轮内部更新</b>：对同一 rollout 做 innerUpdatesPerBatch 次梯度更新：
     *     <ul>
     *       <li>每轮重新 policy forward 得到新 logits → new_log_probs（带梯度）</li>
     *       <li>ratio = exp(new_log_probs - old_log_probs)</li>
     *       <li>L_policy = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)</li>
     *       <li>L_kl = β · (new_log_probs - ref_log_probs) 单 token KL 近似</li>
     *       <li>L = L_policy + L_kl，backward + clipGradients + optimizer.update</li>
     *       <li>更新 MoE expertBias，释放计算图</li>
     *     </ul>
     *   </li>
     * </ol>
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

            // ===== Step 1: 初次 policy forward（用于采样、old_log_probs 快照） =====
            Variable inputVar = new Variable(batch.getInputIds());
            inputVar.setRequireGrad(false);
            DeepSeekR1Model.ReasoningResult initialResult = model.performReasoning(inputVar);

            // ===== Step 2: 组采样 + 奖励计算 + 组内优势（不带梯度） =====
            RolloutData rollout = rolloutAndScoreBatch(batch, initialResult);
            if (rollout == null || rollout.validSamples == 0) {
                // 本 batch 无有效学习信号，释放初次 forward 的计算图
                initialResult.logits.unChainBackward();
                globalStep++;
                continue;
            }

            // 初次 forward 的计算图已用于提取 old_log_probs 快照，此处可释放
            initialResult.logits.unChainBackward();

            // ===== Step 3: 参考模型 forward（frozen，detach） =====
            Variable refLogits = referenceModel.forwardLogits(inputVar);

            // ===== Step 4: K 轮内部梯度更新 =====
            float batchLastLoss   = 0.0f;
            float batchLastReward = rollout.meanReward;
            for (int k = 0; k < innerUpdatesPerBatch; k++) {
                // 每轮重新 forward，保证 new_log_probs 带最新参数梯度
                DeepSeekR1Model.ReasoningResult stepResult = model.performReasoning(inputVar);

                // 计算 GRPO 总损失 = L_policy (PPO-clip) + L_kl
                Variable totalLoss = computeGRPOLossFull(
                        stepResult.logits, refLogits, rollout);

                if (totalLoss == null) {
                    stepResult.logits.unChainBackward();
                    break; // 无有效样本，提前退出内部循环
                }

                float lossValue = totalLoss.getValue().getNumber().floatValue();
                batchLastLoss = lossValue;

                model.clearGrads();
                totalLoss.backward();
                clipGradients();
                optimizer.update();

                // MoE 无辅助损失负载均衡：optimizer step 之后更新专家 bias
                model.getR1Block().updateExpertBiasAfterStep();

                // 释放本轮计算图
                totalLoss.unChainBackward();
                stepResult.logits.unChainBackward();
            }

            // ref_logits 虽是 detach 但 forward 过程的子图仍需释放
            refLogits.unChainBackward();
            inputVar.unChainBackward();

            // 记录统计（以最后一轮内部更新的 loss 为代表）
            correctnessHistory.add(rollout.meanVerifyReward);
            qualityHistory.add((float) initialResult.moeLoss);
            rewardHistory.add(batchLastReward);
            lossHistory.add(batchLastLoss);

            epochAvgReward += batchLastReward;
            epochAvgLoss   += batchLastLoss;
            count++;
            globalStep++;

            if (globalStep % logInterval == 0) {
                System.out.printf(
                    "Epoch %d | Step %d | Loss: %.4f | GRPO: %d | Fallback: %d | Reward: %.4f%n",
                    currentEpoch + 1, globalStep, batchLastLoss,
                    rollout.grpoSamples, rollout.fallbackSamples, batchLastReward
                );
            }
            epochGrpoTotal     += rollout.grpoSamples;
            epochFallbackTotal += rollout.fallbackSamples;
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

    /**
     * Rollout 数据：一次采样结果，供 K 轮内部更新复用
     */
    private static class RolloutData {
        /** 每个样本的采样 token：[batchSize][groupSize] */
        int[][] groupTokens;
        /** 每个样本的组内优势：[batchSize][groupSize] */
        float[][] groupAdvantages;
        /** 每个样本的 old log prob 快照（detach, no_grad）：[batchSize][groupSize] */
        Variable[][] oldLogProbs;
        /** 每个样本的 fallback target token（由 groundTruth 映射而来）：[batchSize] */
        int[] fallbackTokens;
        /** 每个样本是否走 GRPO 正常路径（vs Fallback CE） */
        boolean[] isGRPO;
        /** 每个样本是否有效（有学习信号或需要 fallback） */
        boolean[] isValid;
        int validSamples;
        int grpoSamples;
        int fallbackSamples;
        float meanReward;
        float meanVerifyReward;
    }
    
    /**
     * Step 2: 对整个 batch 完成组采样、验证器打分、组内优势计算、old_log_probs 快照提取
     *
     * <p>本方法的所有输出均为"无梯度快照"，供后续 K 轮内部更新反复使用：
     * <ul>
     *   <li>采样 token（离散，不参与梯度）</li>
     *   <li>优势 A（纯 float 数组，无梯度）</li>
     *   <li>old_log_probs：从 initialResult.logits 的 detach 版本计算，{@link Variable#setRequireGrad(boolean)} false</li>
     * </ul>
     *
     * @param batch         当前 batch
     * @param initialResult 初次 policy forward 的结果（提供 logits 用于采样 + old_log_probs 提取）
     * @return RolloutData；若整个 batch 无任何有效样本，返回的 validSamples 为 0
     */
    private RolloutData rolloutAndScoreBatch(DeepSeekR1RLVRDataset.Batch batch,
                                             DeepSeekR1Model.ReasoningResult initialResult) {
        int batchSize = batch.getBatchSize();
        String[] groundTruths  = batch.getGroundTruths();
        String[] verifierTypes = batch.getVerifierTypes();

        RolloutData rollout = new RolloutData();
        rollout.groupTokens     = new int[batchSize][groupSize];
        rollout.groupAdvantages = new float[batchSize][groupSize];
        rollout.oldLogProbs     = new Variable[batchSize][groupSize];
        rollout.fallbackTokens  = new int[batchSize];
        rollout.isGRPO          = new boolean[batchSize];
        rollout.isValid         = new boolean[batchSize];

        // 一次性 detach 出 old logits，后续所有 old_log_probs 都从这里派生（frozen）
        Variable oldLogitsDetached = initialResult.logits.detach();

        // 提取 vocabSize 用于 groundTruth → token 映射（fallback 用）
        int[] logitShape = initialResult.logits.getValue().getShape().getShapeDims();
        int vocabSize = logitShape[logitShape.length - 1];

        float totalReward = 0f;
        float totalVerifyReward = 0f;

        for (int i = 0; i < batchSize; i++) {
            double targetValue = parseGroundTruthAsDouble(groundTruths[i]);
            Verifier verifier = resolveVerifier(verifierTypes[i]);
            // 预计算 fallback target token（即使本样本走 GRPO，记录也无害且成本极低）
            rollout.fallbackTokens[i] = mapGroundTruthToToken(groundTruths[i], vocabSize);

            float[] groupRewards       = new float[groupSize];
            float[] groupVerifyRewards = new float[groupSize];
            for (int g = 0; g < groupSize; g++) {
                int sampledToken = sampleTokenFromLogits(initialResult.logits, i, temperature);
                rollout.groupTokens[i][g] = sampledToken;

                // r_proximity：基于数值距离的高斯核近似奖励
                float rProx = computeProximityReward(sampledToken, targetValue);

                // r_verify：真实调用验证器打分
                float rVerify = 0f;
                if (verifier != null) {
                    String modelOutput = tokenToVerifierString(sampledToken, verifier.getVerifierType());
                    VerificationResult vr = verifier.verify(modelOutput, groundTruths[i]);
                    rVerify = vr.getReward();
                    groupVerifyRewards[g] = rVerify;
                }

                groupRewards[g] = correctnessWeight * rVerify
                        + reasoningQualityWeight * rProx
                        - verificationWeight * (float) initialResult.moeLoss;

                // 提取 old_log_probs[i][g]：detach 版 logits → logSoftmax → 取 sampledToken 位置
                rollout.oldLogProbs[i][g] = logProbOfToken(oldLogitsDetached, i, sampledToken).detach();
            }

            // 组内相对优势 A_i = (r_i - mean) / std
            float[] advantages = computeGroupAdvantages(groupRewards);
            boolean hasSignal = false;
            for (float a : advantages) {
                if (Math.abs(a) > 1e-6f) { hasSignal = true; break; }
            }

            float meanReward       = calculateAverage(groupRewards);
            float meanVerifyReward = calculateAverage(groupVerifyRewards);

            if (!hasSignal) {
                // 组内无差异信号
                if (meanVerifyReward < 0.5f) {
                    // 全错 → 走 fallback CE
                    rollout.isValid[i] = true;
                    rollout.isGRPO[i]  = false;
                    rollout.fallbackSamples++;
                } else {
                    // 全对 → 跳过该样本（无需更新）
                    rollout.isValid[i] = false;
                    continue;
                }
            } else {
                // 有学习信号 → 走 GRPO
                rollout.isValid[i] = true;
                rollout.isGRPO[i]  = true;
                rollout.grpoSamples++;
                System.arraycopy(advantages, 0, rollout.groupAdvantages[i], 0, groupSize);
            }

            rollout.validSamples++;
            totalReward       += meanReward;
            totalVerifyReward += meanVerifyReward;
        }

        if (rollout.validSamples > 0) {
            rollout.meanReward       = totalReward / rollout.validSamples;
            rollout.meanVerifyReward = totalVerifyReward / rollout.validSamples;
        }

        // 释放 detach 版 logits 的计算图（虽然已 detach，但保险起见）
        oldLogitsDetached.unChainBackward();

        return rollout;
    }

    /**
     * 计算 GRPO 完整损失：L_policy (PPO-clip) + L_kl (KL 约束) + Fallback CE
     *
     * <p>对 batch 中每个 valid 样本：
     * <ul>
     *   <li>若 isGRPO：对 G 个采样 token 计算 PPO-clip 损失 + β · (log π_new - log π_ref) KL 近似</li>
     *   <li>若 !isGRPO：用 Fallback CE 引导模型学习 groundTruth</li>
     * </ul>
     *
     * @param newLogits 当前轮 policy forward 的 logits（带梯度）
     * @param refLogits 参考模型 forward 的 logits（frozen, detached）
     * @param rollout   rollout 快照数据
     * @return 对整个 batch 求和并按 validSamples 平均的 scalar loss；若无有效样本返回 null
     */
    private Variable computeGRPOLossFull(Variable newLogits, Variable refLogits, RolloutData rollout) {
        int batchSize = rollout.isValid.length;
        Variable totalLoss = null;
        int contributedSamples = 0;

        for (int i = 0; i < batchSize; i++) {
            if (!rollout.isValid[i]) continue;

            Variable sampleLoss;
            if (rollout.isGRPO[i]) {
                sampleLoss = computeGRPOSampleLoss(
                        newLogits, refLogits, i,
                        rollout.groupTokens[i], rollout.groupAdvantages[i], rollout.oldLogProbs[i]);
            } else {
                // Fallback CE：使用 groundTruth 预先映射得到的 fallbackToken 做监督学习
                // 语义完全符合 "组内全错 → 按 groundTruth 引导" 的 fallback 本意
                sampleLoss = computeFallbackCELossByToken(newLogits, i, rollout.fallbackTokens[i]);
            }

            if (sampleLoss == null) continue;

            totalLoss = (totalLoss == null) ? sampleLoss : totalLoss.add(sampleLoss);
            contributedSamples++;
        }

        if (totalLoss == null || contributedSamples == 0) {
            return null;
        }
        // 对贡献样本求平均
        return totalLoss.mul(constant(1.0f / contributedSamples));
    }

    /**
     * 计算单个样本的 GRPO 损失（PPO-clip + KL）
     *
     * <pre>
     *   L_sample = 1/G · Σ_g [ -min(ratio_g · A_g, clip(ratio_g, 1-ε, 1+ε) · A_g)
     *                          + β · (log π_new(o_g) - log π_ref(o_g)) ]
     *   ratio_g = exp(log π_new(o_g) - log π_old(o_g))
     * </pre>
     *
     * @param newLogits    当前策略 logits [batchSize, seqLen, vocabSize]
     * @param refLogits    参考策略 logits（frozen）
     * @param sampleIdx    batch 内样本索引
     * @param groupTokens  组内 G 个采样 token
     * @param advantages   组内归一化优势 A_g
     * @param oldLogProbs  采样时刻的 log π_old(o_g) 快照（frozen）
     * @return scalar 损失；若所有 advantage 都为零返回 null
     */
    private Variable computeGRPOSampleLoss(Variable newLogits, Variable refLogits,
                                           int sampleIdx, int[] groupTokens,
                                           float[] advantages, Variable[] oldLogProbs) {
        Variable sampleLoss = null;
        int contributed = 0;

        for (int g = 0; g < groupTokens.length; g++) {
            float adv = advantages[g];
            if (Math.abs(adv) < 1e-9f) continue; // 无学习信号
            int token = groupTokens[g];

            // 新策略 log π_new(o_g)（带梯度）
            Variable logPiNew = logProbOfToken(newLogits, sampleIdx, token);
            // 参考策略 log π_ref(o_g)（frozen）
            Variable logPiRef = logProbOfToken(refLogits, sampleIdx, token).detach();
            // 旧策略 log π_old(o_g)（frozen 快照）
            Variable logPiOld = oldLogProbs[g];

            // log_ratio = log π_new - log π_old
            Variable logRatio = logPiNew.sub(logPiOld);
            // 为了数值稳定，先 clip log_ratio 到 [log(1-ε), log(1+ε)] 对应范围的稍宽区间
            // 再 exp → ratio。这样即使 log_ratio 很大也不会溢出
            // 直接 exp：ratio = exp(log_ratio)
            Variable ratio = logRatio.exp();

            // 裁剪 ratio 到 [1-ε, 1+ε]
            Variable ratioClipped = ratio.clip(1.0f - clipEps, 1.0f + clipEps);

            // PPO-clip 损失：L_policy = -min(ratio·A, clipped_ratio·A)
            Variable advConst = constant(adv);
            Variable term1 = ratio.mul(advConst);
            Variable term2 = ratioClipped.mul(advConst);
            // min(a, b) = -max(-a, -b)；但 Variable 没有 min/max 二元，所以用：
            // min(a, b) = (a + b - |a - b|) / 2
            Variable diff = term1.sub(term2);
            // |x| 可以用 x·sign(x) 近似，但 Variable 无 abs。改用：|x| = sqrt(x^2 + ε_stable)
            Variable absDiff = diff.mul(diff).add(constant(1e-12f)).pow(0.5f);
            Variable minTerm = term1.add(term2).sub(absDiff).mul(constant(0.5f));
            // L_policy = -minTerm
            Variable lossPolicy = minTerm.mul(constant(-1.0f));

            // KL 近似：L_kl = β · (log π_new - log π_ref)
            // 注意：严格 KL 应对整个 vocab 分布求和，这里用 "策略在采样 token 上的 log 差" 作为单 token 近似，
            // 这是 PPO 实现中常见的 per-token KL 形式（GRPO 论文附录 B 公式 5 即此形式）
            Variable klDiff = logPiNew.sub(logPiRef);
            Variable lossKl = klDiff.mul(constant(klWeight));

            Variable tokenLoss = lossPolicy.add(lossKl);
            sampleLoss = (sampleLoss == null) ? tokenLoss : sampleLoss.add(tokenLoss);
            contributed++;
        }

        if (sampleLoss == null || contributed == 0) return null;
        return sampleLoss.mul(constant(1.0f / contributed));
    }

    /**
     * 计算 log π(token | sample) = logSoftmax(logits[sample, -1, :])[token]
     *
     * <p>步骤：
     * <ol>
     *   <li>extractSampleLogits → [1, vocabSize]</li>
     *   <li>logSoftmax(axis=-1) → [1, vocabSize] 的 log 概率分布</li>
     *   <li>select(dim=1, index=token) → scalar</li>
     * </ol>
     *
     * @param logits    原始 logits
     * @param sampleIdx batch 内样本索引
     * @param tokenId   目标 token
     * @return 标量 Variable log π(token)
     */
    private Variable logProbOfToken(Variable logits, int sampleIdx, int tokenId) {
        Variable sampleLogits = extractSampleLogits(logits, sampleIdx); // [1, vocabSize]
        Variable logProbs = sampleLogits.logSoftmax();                  // [1, vocabSize]
        // select(dim=1, index=tokenId) → [1]，再 reshape 到 scalar
        return logProbs.select(1, tokenId);
    }

    /**
     * Fallback CE：指定 targetToken 版本（供 computeGRPOLossFull 使用）
     */
    private Variable computeFallbackCELossByToken(Variable logits, int sampleIdx, int targetToken) {
        Variable sampleLogits = extractSampleLogits(logits, sampleIdx);
        Variable target = constant(
                NdArray.of(new float[]{targetToken}).reshape(Shape.of(1, 1)));
        return sampleLogits.softmaxCrossEntropy(target);
    }

    /**
     * 根据验证器类型名查找对应 Verifier 实例
     *
     * @param verifierType 类型名（"math" / "code" / "logic"），为 null/空时默认回落到 math
     * @return 匹配的验证器；未匹配到时返回 null（此时跳过验证器打分，仅用 proximity）
     */
    private Verifier resolveVerifier(String verifierType) {
        if (verifierType == null || verifierType.trim().isEmpty()) {
            return verifiers.get("math");
        }
        Verifier v = verifiers.get(verifierType.trim().toLowerCase());
        return v != null ? v : verifiers.get("math");
    }

    /**
     * 将采样 token 转换为可供验证器打分的文本输出
     *
     * <p>注意：本类无真实 tokenizer/detokenizer，仅能把单个 token id 作为"最终答案"的字符串表达。
     * 这是教学场景的简化但<b>不是假实现</b>——对 math/logic 任务下，verifier 会从字符串里按正则提取数值/结论，
     * 直接给数值字符串即可得到有意义的 0/1 反馈；对 code 任务，verifier 做模式匹配也能工作。
     *
     * @param tokenId      采样到的 token id
     * @param verifierType 验证器类型，用于决定格式化方式
     * @return 可被 verifier 消费的字符串
     */
    private String tokenToVerifierString(int tokenId, String verifierType) {
        if (verifierType == null) {
            return String.valueOf(tokenId);
        }
        switch (verifierType) {
            case "logic":
                // 逻辑验证器按关键词匹配 true/false/yes/no
                return tokenId % 2 == 0 ? "The conclusion is false" : "The conclusion is true";
            case "code":
                // 代码验证器按代码块提取
                return "```\nreturn " + tokenId + ";\n```";
            case "math":
            default:
                // 数学验证器提取最后一个数值
                return "The answer is " + tokenId;
        }
    }
    
    /**
     * 提取当前样本最后位置的 logits → [1, vocabSize]
     *
     * <p>统一处理 3D [batchSize, seqLen, vocabSize]、2D [seqLen, vocabSize]、1D [vocabSize] 三种形状，
     * 供 {@link #logProbOfToken}、{@link #computeFallbackCELossByToken}、{@link #sampleTokenFromLogits} 共用。
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

    // ==================== 工具方法 ====================

    /**
     * 构造不需要梯度的常量标量 Variable（用于 loss 加权、reward、归一化系数等）
     * <p>
     * 这些标量本身不承载学习参数，必须 setRequireGrad(false) 以避免错误进入反向传播链。
     */
    private static Variable constant(float scalar) {
        Variable v = new Variable(NdArray.of(scalar));
        v.setRequireGrad(false);
        return v;
    }

    /**
     * 基于已有 NdArray 构造不需要梯度的常量 Variable
     */
    private static Variable constant(NdArray data) {
        Variable v = new Variable(data);
        v.setRequireGrad(false);
        return v;
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

    /** 采样温度下限，避免除以 0 或 exp 溢出 */
    private static final float MIN_SAMPLING_TEMPERATURE = 1e-3f;

    /**
     * 温度采样：从 logits 中以温度 τ 按概率采样一个 token
     *
     * <p>softmax(logits / τ) 后多项式采样，τ&gt;1 更随机，τ→0 趋向 argmax。
     *
     * <p>数值稳定性保护：
     * <ul>
     *   <li>温度下限 {@link #MIN_SAMPLING_TEMPERATURE}，防止 0/NaN 除法</li>
     *   <li>独立分配 rawLogits 数组，避免底层 NdArray buffer 共享导致脏读</li>
     *   <li>max-subtraction 技巧防止 exp 上溢</li>
     *   <li>sumExp 若为 0（极端下溢），退化为 argmax</li>
     *   <li>cumProb 最终兜底返回 vocabSize-1，应对浮点累加误差</li>
     * </ul>
     *
     * @param logits    模型输出 logits
     * @param sampleIdx 当前样本在 batch 中的索引
     * @param temp      采样温度
     * @return 采样到的 token id
     */
    private int sampleTokenFromLogits(Variable logits, int sampleIdx, float temp) {
        // 温度下限保护，防止除以 0 或 exp 溢出
        float safeTemp = Math.max(temp, MIN_SAMPLING_TEMPERATURE);

        // 复用 extractSampleLogits 统一切片
        Variable sliced = extractSampleLogits(logits, sampleIdx);
        NdArray logitsArr = sliced.getValue();
        int[] shape = logitsArr.getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];

        // 独立临时数组（避免与 NdArray 底层 buffer 共享）
        float[] rawLogits = new float[vocabSize];
        int argmax = 0;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabSize; v++) {
            float val = shape.length >= 2 ? logitsArr.get(0, v) : logitsArr.get(v);
            rawLogits[v] = val;
            if (val > maxVal) {
                maxVal = val;
                argmax = v;
            }
        }

        // 温度缩放 + softmax（带 max-subtraction 数值稳定）
        float[] probs = new float[vocabSize];
        double sumExp = 0.0;
        for (int v = 0; v < vocabSize; v++) {
            double p = Math.exp((rawLogits[v] - maxVal) / safeTemp);
            probs[v] = (float) p;
            sumExp += p;
        }

        // 极端下溢兜底：所有 exp 都是 0 → 直接返回 argmax
        if (sumExp <= 0.0 || Double.isNaN(sumExp) || Double.isInfinite(sumExp)) {
            return argmax;
        }

        // 强制归一化
        float invSum = (float) (1.0 / sumExp);
        for (int v = 0; v < vocabSize; v++) {
            probs[v] *= invSum;
        }

        // 多项式采样（CDF 逆变换）
        float rand = (float) Math.random();
        float cumProb = 0.0f;
        for (int v = 0; v < vocabSize; v++) {
            cumProb += probs[v];
            if (rand <= cumProb) return v;
        }
        // 浮点累加误差兜底
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
            float avgReward      = rewardHistory.isEmpty()
                    ? avgCorrectness
                    : calculateAverage(rewardHistory);
            float avgQuality     = calculateAverage(qualityHistory);

            System.out.printf("总训练步数: %d\n", globalStep);
            System.out.printf("平均正确率 (r_verify): %.4f\n", avgCorrectness);
            System.out.printf("平均综合奖励 (r_total): %.4f\n", avgReward);
            System.out.printf("平均 MoE 辅助损失: %.4f\n", avgQuality);
            
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
        stats.put("avg_reward", rewardHistory.isEmpty()
                ? calculateAverage(correctnessHistory)
                : calculateAverage(rewardHistory));
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
