package io.leavesfly.tinyai.minimind.training.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.training.rlaif.BaseRLTrainer;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.*;

/**
 * Agent 强化学习训练器
 * <p>
 * 对标 Python minimind3 train_agent.py rl_train_epoch (L414-L577)。
 * <p>
 * 训练流程：
 * 1. 从 AgentDataset 加载 batch（prompt + tools + gt）
 * 2. 使用 AgentRolloutEngine 生成多轮 rollout（每个 prompt 生成 numGenerations 个候选）
 * 3. 使用 AgentRewardCalculator 计算多维度奖励
 * 4. 计算组相对优势（GRPO）
 * 5. PPO Clipped Surrogate + 可微 KL 散度约束更新策略
 * 6. 余弦退火学习率 + 梯度裁剪 + 梯度累积
 * <p>
 * 核心公式（对标 Python L521-L531）：
 * - kl_div = ref_logps - per_token_logps
 * - per_token_kl = exp(kl_div) - kl_div - 1     （k3 估计器，非负、低方差）
 * - ratio = exp(new_logps - old_logps)
 * - clipped_ratio = clip(ratio, 1-eps, 1+eps)
 * - loss = -(min(ratio*adv, clipped_ratio*adv) - beta*per_token_kl)
 * <p>
 * 三个关键实现约束：
 * 1. <b>序列必须带 prompt 条件</b>。训练输入是 {@code prompt + completion}，标签只在
 *    completion 段有效、prompt 段填 ignore_index。若只编码 completion，算出来的是
 *    无条件概率 log π(y)，与 rollout 时"看着 prompt 生成 y"的分布不是同一个量，
 *    优势与 logProb 相乘得到的梯度方向是错误的。
 * 2. <b>KL 必须可微</b>。把 KL 先算成 float 再包成 requireGrad=false 的常量，等于给损失
 *    加了一个与参数无关的偏移——前向数值好看，反向对 KL 约束零贡献，模型可以无限偏离参考策略。
 * 3. <b>old_logps 必须是更新前的快照</b>。用当前 new_logps 的取值充当 old_logps 会让
 *    ratio 恒等于 1，clipped surrogate 退化成普通策略梯度，信任区约束完全失效。
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentTrainer extends BaseRLTrainer {

    private final MiniMindModel policyModel;       // 策略模型（训练）
    private final MiniMindModel refModel;           // 参考模型（冻结）
    private final AgentDataset dataset;
    private final AgentConfig config;
    private final AgentRolloutEngine rolloutEngine;
    private final MiniMindTokenizer tokenizer;
    private final Adam optimizer;

    // 训练状态
    private float currentLearningRate;
    private int accumulationCounter = 0;
    /** tokenizer 截断 prompt 只告警一次 */
    private boolean tokenizerTruncationWarned;
    /** 组内优势全为零（无学习信号）只告警一次 */
    private boolean noSignalWarned;

    /**
     * 同一批 rollout 的重复利用轮数（PPO inner epochs）
     * <p>
     * 大于 1 时，第 2 轮起的 new_logps 已经与 old_logps 快照产生真实偏移，
     * ratio ≠ 1，clipped surrogate 的信任区约束才会真正生效。
     */
    private int ppoEpochs = 1;

    // 统计
    private final List<Float> rewardHistory;
    private final List<Float> klHistory;

    /**
     * 构造函数
     *
     * @param policyModel 策略模型（将被训练）
     * @param refModel    参考模型（冻结，用于 KL 约束）
     * @param dataset     Agent 数据集
     * @param config      Agent RL 配置
     * @param tokenizer   分词器
     */
    public AgentTrainer(MiniMindModel policyModel, MiniMindModel refModel,
                         AgentDataset dataset, AgentConfig config,
                         MiniMindTokenizer tokenizer) {
        super(policyModel);
        this.policyModel = policyModel;
        this.refModel = refModel;
        this.dataset = dataset;
        this.config = config;
        this.tokenizer = tokenizer;
        this.rolloutEngine = new AgentRolloutEngine(policyModel, tokenizer);

        // 验证配置
        config.validate();

        // 冻结参考模型：既切 dropout，也关掉 requireGrad，避免任何路径把梯度写进参考模型
        freezeReferenceModel();

        // 训练参数
        this.maxEpochs = config.getMaxEpochs();
        this.maxGradNorm = config.getGradClip();
        this.logInterval = config.getLogInterval();
        this.saveInterval = config.getSaveInterval();
        this.checkpointDir = "./checkpoints/minimind/agent";
        this.currentLearningRate = config.getLearningRate();

        // 优化器
        this.optimizer = new Adam(policyModel, config.getLearningRate(),
                0.9f, 0.999f, 1e-8f);

        // 统计
        this.rewardHistory = new ArrayList<>();
        this.klHistory = new ArrayList<>();
    }

    /**
     * 设置同一批 rollout 的重复利用轮数
     */
    public AgentTrainer setPpoEpochs(int ppoEpochs) {
        this.ppoEpochs = Math.max(1, ppoEpochs);
        return this;
    }

    /**
     * 冻结参考模型
     */
    private void freezeReferenceModel() {
        refModel.setTraining(false);
        for (Parameter param : refModel.getAllParams().values()) {
            param.setRequiresGrad(false);
        }
    }

    // ==================== 核心训练逻辑 ====================

    @Override
    protected float trainStep(Object batch) {
        // 更新学习率
        updateLearningRate();

        AgentDataset.Batch agentBatch = (AgentDataset.Batch) batch;
        List<List<Map<String, String>>> messagesBatch = agentBatch.getMessagesBatch();
        List<List<String>> toolsBatch = agentBatch.getToolsBatch();
        List<List<String>> gtBatch = agentBatch.getGtBatch();
        int numGen = Math.max(1, config.getNumGenerations());

        // ========== 1. Rollout 生成（eval 模式，与推理时分布一致） ==========
        policyModel.setTraining(false);
        List<AgentRolloutEngine.RolloutResult> rolloutResults = rolloutEngine.rolloutBatch(
                messagesBatch, toolsBatch, numGen,
                config.getMaxTurns(), config.getMaxGenLen(), config.getTemperature());

        // 收集生成结果
        List<String> completions = new ArrayList<>();
        List<List<String>> turnOutputsBatch = new ArrayList<>();
        List<Boolean> unfinishedBatch = new ArrayList<>();

        for (AgentRolloutEngine.RolloutResult r : rolloutResults) {
            completions.add(r.getFinalOutput());
            turnOutputsBatch.add(r.getTurnOutputs());
            unfinishedBatch.add(r.isUnfinished());
        }

        // ========== 2. 计算奖励 ==========
        float[] rewards = AgentRewardCalculator.calculateRewards(
                completions, gtBatch, toolsBatch, numGen,
                turnOutputsBatch, unfinishedBatch);

        rewardHistory.add(average(rewards));

        // ========== 3. 计算组相对优势（对标 Python L516-L519） ==========
        float[] advantages = computeGroupAdvantages(rewards, numGen);
        warnIfNoLearningSignal(advantages, rewards, numGen);

        // ========== 4. 编码 prompt + completion（prompt 段标签置 ignore_index） ==========
        List<EncodedSequence> sequences = encodeRollouts(rolloutResults);
        if (sequences.isEmpty()) {
            System.err.println("警告: Agent RL 本批次没有可用的 rollout 序列，跳过更新");
            return Float.NaN;
        }

        // ========== 5. 更新前快照：old_logps（策略）与 ref_logps（参考模型） ==========
        float[] oldLogProbs = snapshotLogProbs(policyModel, sequences);
        float[] refLogProbs = snapshotLogProbs(refModel, sequences);

        // ========== 6. 策略更新（可复用同一批 rollout 多轮） ==========
        int innerEpochs = Math.max(1, ppoEpochs);
        float totalLoss = 0.0f;
        int validCount = 0;

        for (int epoch = 0; epoch < innerEpochs; epoch++) {
            policyModel.setTraining(true);

            for (int idx = 0; idx < sequences.size(); idx++) {
                EncodedSequence seq = sequences.get(idx);

                Variable inputVar = new Variable(seq.input);
                Variable labelVar = new Variable(seq.labels);
                labelVar.setRequireGrad(false);

                // 新策略前向传播
                Variable newLogits = policyModel.predict(inputVar);

                // 逐样本 logProb（此处 batch=1，形状 [1]），计算图连通
                Variable newLogProb = computePerSampleLogProbs(newLogits, labelVar);

                // 可微 KL（k3 估计器）：kl = exp(ref-new) - (ref-new) - 1
                Variable klDiv = constant(refLogProbs[idx]).sub(newLogProb);
                Variable perTokenKl = klDiv.exp().sub(klDiv).sub(constant(1.0f));

                // 优势取自原始 rollout 下标：sequences 会丢弃无法编码的样本，
                // 直接用循环下标会与 rewards/advantages 错位
                float advantage = seq.rolloutIndex < advantages.length
                        ? advantages[seq.rolloutIndex] : 0.0f;

                // PPO Clipped Surrogate（与 GRPO / PPO 共用同一份权威实现）
                Variable surrogate = clippedSurrogateLoss(newLogProb,
                        new float[]{oldLogProbs[idx]}, new float[]{advantage},
                        config.getEpsilon());

                // loss = -(min(...) - beta*kl) = surrogate + beta*kl
                Variable candidateLoss = surrogate.add(perTokenKl.mul(constant(config.getBeta())));

                float lossVal = candidateLoss.getValue().getNumber().floatValue();
                float klVal = perTokenKl.getValue().getNumber().floatValue();
                if (!Float.isFinite(lossVal)) {
                    System.err.printf("警告: Agent RL 样本 %d 损失异常(%s)，已跳过%n",
                            idx, Float.isNaN(lossVal) ? "NaN" : "Inf");
                    candidateLoss.unChainBackward();
                    continue;
                }

                // 梯度累积缩放
                int accumSteps = Math.max(1, config.getAccumulationSteps());
                if (accumSteps > 1) {
                    candidateLoss = candidateLoss.mul(constant(1.0f / accumSteps));
                }

                candidateLoss.backward();
                candidateLoss.unChainBackward();

                totalLoss += lossVal;
                klHistory.add(klVal);
                validCount++;
            }

            // 梯度累积更新
            accumulationCounter++;
            if (accumulationCounter % Math.max(1, config.getAccumulationSteps()) == 0) {
                flushAccumulatedGradients();
            }
        }

        return validCount > 0 ? totalLoss / validCount : Float.NaN;
    }

    /**
     * 裁剪并应用已累积的梯度
     */
    private void flushAccumulatedGradients() {
        clipGradients(policyModel, config.getGradClip());
        optimizer.update();
        policyModel.clearGrads();
        accumulationCounter = 0;
    }

    /**
     * epoch 收尾：把不足 accumulationSteps 的残留梯度刷掉
     * <p>
     * 否则这些梯度会被带进下一个 epoch 的首次 update，使那一步的尺度偏大，
     * 且对应的是上一个 epoch 的数据。
     */
    @Override
    protected void onEpochEnd() {
        if (accumulationCounter > 0) {
            flushAccumulatedGradients();
        }
        super.onEpochEnd();
    }

    // ==================== 序列编码 ====================

    /**
     * 一条编码后的训练序列
     * <p>
     * {@code input} = prompt + completion 的 token；{@code labels} 在 prompt 段、
     * 序列末位（没有下一个 token 可监督）以及被截断掉的位置均为 {@link #IGNORE_INDEX}。
     * <p>
     * {@code rolloutIndex} 是该序列在 rollout 结果（也就是 rewards / advantages）中的原始下标：
     * 编码阶段会丢弃过短、为空或 prompt 已占满 maxSeqLen 的样本，列表下标因此不再与奖励对齐。
     */
    private static final class EncodedSequence {
        final NdArray input;
        final NdArray labels;
        final int rolloutIndex;

        EncodedSequence(NdArray input, NdArray labels, int rolloutIndex) {
            this.input = input;
            this.labels = labels;
            this.rolloutIndex = rolloutIndex;
        }
    }

    /**
     * 把 rollout 结果编码成"带 prompt 条件"的训练序列
     * <p>
     * 完整上下文由 {@code AgentRolloutEngine} 以 {@code context + generated} 的形式给出，
     * 因此 prompt 文本可由 fullContext 去掉尾部 completion 得到。
     * <p>
     * prompt 与 completion <b>分开编码后拼接</b>，而不是对 fullContext 整体编码：
     * <ol>
     *   <li>{@code tokenizer.encode(text)} 等价于 {@code encode(text, true, true)}，会补上 BOS/EOS。
     *       用它算 prompt 长度会多算一个尾部 EOS，使边界右移，第一个 completion token
     *       反而不被监督；</li>
     *   <li>tokenizer 自己也会按 {@code tokenizer.maxSeqLen} 截断。completion 位于上下文尾部，
     *       整体编码时会被<b>整段切掉</b>，导致该 rollout 看起来有内容却没有任何可监督位置。</li>
     * </ol>
     * 分开编码后边界由构造保证精确：{@code promptLen == promptTokens.size()}。
     * <p>
     * 拼接后若仍超出模型 {@code maxSeqLen}，采用<b>左截断</b>（丢弃靠前的 prompt token）：
     * Agent RL 要优化的是 completion 部分的策略，右截断会把 completion 整段切掉。
     *
     * @return 可用序列列表（过短、全被截断或没有任何有效标签的样本会被丢弃）
     */
    private List<EncodedSequence> encodeRollouts(List<AgentRolloutEngine.RolloutResult> rolloutResults) {
        int maxSeqLen = policyModel.getConfig().getMaxSeqLen();
        List<EncodedSequence> sequences = new ArrayList<>();
        int leftTruncated = 0;
        int dropped = 0;

        for (int rolloutIndex = 0; rolloutIndex < rolloutResults.size(); rolloutIndex++) {
            AgentRolloutEngine.RolloutResult result = rolloutResults.get(rolloutIndex);
            String completion = result.getFinalOutput();
            String fullContext = result.getFullContext();
            if (completion == null || completion.isEmpty()
                    || fullContext == null || fullContext.isEmpty()) {
                dropped++;
                continue;
            }

            // 还原 prompt：fullContext = prompt + completion
            String promptText = fullContext.endsWith(completion)
                    ? fullContext.substring(0, fullContext.length() - completion.length())
                    : "";

            // 优先用 rollout 引擎记录的"模型真正看到的 token"：
            // 对 fullContext 重新编码会受 tokenizer.maxSeqLen 截断（completion 在尾部会被整段切掉）、
            // 单参 encode 补 BOS/EOS（promptLen 会多算 1）、BPE 跨边界合并三重影响
            List<Integer> promptTokens;
            List<Integer> completionTokens;
            if (result.hasTokenTrace()) {
                promptTokens = result.getFinalPromptTokenIds();
                completionTokens = result.getFinalCompletionTokenIds();
            } else {
                // 回退路径：从文本重建（不带 BOS/EOS）
                promptTokens = promptText.isEmpty()
                        ? Collections.emptyList()
                        : tokenizer.encode(promptText, false, false);
                completionTokens = tokenizer.encode(completion, false, false);
                if (promptTokens.size() >= tokenizer.getMaxSeqLen() && !tokenizerTruncationWarned) {
                    tokenizerTruncationWarned = true;
                    System.err.println("⚠️ AgentRL: prompt 已被 tokenizer.maxSeqLen="
                            + tokenizer.getMaxSeqLen() + " 截断，训练看到的 prompt 与 rollout 时的不完全一致");
                }
            }
            if (completionTokens.isEmpty()) {
                dropped++;
                continue;
            }

            List<Integer> tokens = new ArrayList<>(promptTokens.size() + completionTokens.size());
            tokens.addAll(promptTokens);
            tokens.addAll(completionTokens);
            if (tokens.size() < 2) {
                dropped++;
                continue;
            }

            int promptLen = promptTokens.size();

            int start = 0;
            int seqLen = tokens.size();
            if (seqLen > maxSeqLen) {
                start = seqLen - maxSeqLen;
                seqLen = maxSeqLen;
                promptLen = Math.max(0, promptLen - start);
                leftTruncated++;
            }

            float[] inputData = new float[seqLen];
            float[] labelData = new float[seqLen];

            boolean hasSupervision = false;
            for (int i = 0; i < seqLen; i++) {
                inputData[i] = tokens.get(start + i);

                if (i < seqLen - 1) {
                    int target = tokens.get(start + i + 1);
                    // 位置 i 的标签是下一个 token；第一个 completion token 在（截断后）下标
                    // promptLen 处，对应的位置是 i = promptLen - 1
                    boolean supervised = (i >= promptLen - 1);
                    labelData[i] = supervised ? target : IGNORE_INDEX;
                    hasSupervision |= supervised;
                } else {
                    // 末位没有下一个 token 可监督；若填 tokens[seqLen-1] 就等于让模型
                    // 预测自己刚看到的 token，形成标签泄漏
                    labelData[i] = IGNORE_INDEX;
                }
            }

            if (!hasSupervision) {
                // 左截断后窗口内只剩 1 个 token，无法构成有效的策略梯度信号
                dropped++;
                continue;
            }

            sequences.add(new EncodedSequence(
                    NdArray.of(inputData, Shape.of(1, seqLen)),
                    NdArray.of(labelData, Shape.of(1, seqLen)),
                    rolloutIndex));
        }

        if (leftTruncated > 0) {
            System.err.println("⚠️ AgentRL: " + leftTruncated + " 条 rollout 超出 maxSeqLen="
                    + maxSeqLen + "，已左截断 prompt 以保留 completion");
        }
        if (dropped > 0) {
            System.err.println("⚠️ AgentRL: " + dropped + " 条 rollout 被丢弃（为空、过短或无可监督位置）");
        }

        return sequences;
    }

    /**
     * 在 eval 模式下采集每条序列的逐样本 logProb（detach，只保留数值）
     * <p>
     * 用于 old_logps（更新前的策略快照）与 ref_logps（冻结参考模型）。
     * 两者都必须是"不随本次反向传播变化"的常量，否则 ratio / KL 失去参照意义。
     * <p>
     * 本方法会把 target 置为 eval 模式并且<b>不负责恢复</b>：调用方（trainStep）
     * 在快照之后会显式重新 {@code setTraining(true)}。若在快照阶段开着 dropout，
     * old_logps 就成了一个带噪声的随机参照，ratio 的偏移里混进了 dropout 噪声而非策略漂移。
     *
     * @param target    目标模型
     * @param sequences 编码后的序列
     * @return 每条序列的平均 logProb
     */
    private float[] snapshotLogProbs(MiniMindModel target, List<EncodedSequence> sequences) {
        target.setTraining(false);

        float[] logProbs = new float[sequences.size()];
        for (int i = 0; i < sequences.size(); i++) {
            EncodedSequence seq = sequences.get(i);
            Variable inputVar = new Variable(seq.input);
            Variable labelVar = new Variable(seq.labels);
            labelVar.setRequireGrad(false);

            Variable logits = target.predict(inputVar).detach();
            Variable logProb = computePerSampleLogProbs(logits, labelVar);
            logProbs[i] = logProb.getValue().getArray()[0];
            logProb.unChainBackward();
        }

        return logProbs;
    }

    // ==================== 优势与学习率 ====================

    /**
     * 组内优势全为零时告警
     * <p>
     * 组相对优势 {@code A = (R - mean_group) / std_group}，当同一 prompt 的 numGen 个候选
     * 拿到<b>完全相同的奖励</b>时，分子恒为 0 → 所有优势为 0 → surrogate 损失与它的梯度
     * 都恒为 0，本步完全空转。又因为参数没有变化，后续 ppo epoch 的 ratio 仍为 1，
     * 整个 batch 都不会产生任何更新——现象上就是"损失恒为 0.0、参数一动不动"。
     * <p>
     * 这在数学上是正确的（没有偏好差异就没有学习信号），但必须让使用者看见：
     * 常见成因是奖励函数区分度不足、或模型太弱使所有候选生成几乎相同的输出。
     */
    private void warnIfNoLearningSignal(float[] advantages, float[] rewards, int numGen) {
        if (advantages.length == 0) {
            return;
        }
        boolean allZero = true;
        for (float a : advantages) {
            if (Math.abs(a) > 1e-8f) {
                allZero = false;
                break;
            }
        }
        if (allZero && !noSignalWarned) {
            noSignalWarned = true;
            System.err.println("⚠️ AgentRL: 组内优势全为 0（奖励="
                    + java.util.Arrays.toString(rewards) + ", numGen=" + numGen
                    + "），本步没有学习信号，参数不会更新。"
                    + "常见原因：同一 prompt 的各候选拿到完全相同的奖励（奖励函数区分度不足，"
                    + "或模型输出高度同质）。后续同类告警不再重复打印。");
        }
    }

    /**
     * 计算组相对优势（对标 Python L516-L519）
     * <p>
     * advantages = (rewards - group_mean) / (group_std + 1e-4)
     */
    private float[] computeGroupAdvantages(float[] rewards, int numGen) {
        float[] advantages = new float[rewards.length];
        if (rewards.length == 0 || numGen <= 0) {
            return advantages;
        }
        int numGroups = (int) Math.ceil((double) rewards.length / numGen);

        for (int g = 0; g < numGroups; g++) {
            int start = g * numGen;
            int end = Math.min(start + numGen, rewards.length);
            int groupSize = end - start;

            // 组内均值
            float mean = 0.0f;
            for (int i = start; i < end; i++) {
                mean += rewards[i];
            }
            mean /= groupSize;

            // 组内标准差
            float variance = 0.0f;
            for (int i = start; i < end; i++) {
                variance += (rewards[i] - mean) * (rewards[i] - mean);
            }
            float std = (float) Math.sqrt(variance / groupSize);

            // 归一化优势
            for (int i = start; i < end; i++) {
                advantages[i] = (rewards[i] - mean) / (std + 1e-4f);
            }
        }

        return advantages;
    }

    /**
     * 更新学习率（余弦退火 with 10% floor，对标 Python CosineAnnealingLR）
     */
    private void updateLearningRate() {
        int totalSteps = Math.max(1, maxEpochs * dataset.getBatchCount());
        currentLearningRate = computeScheduledLearningRate(
                config.getLearningRate(), currentStep, totalSteps, 0);
        optimizer.setLearningRate(currentLearningRate);
    }

    private static float average(float[] values) {
        if (values == null || values.length == 0) {
            return 0.0f;
        }
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    // ==================== 训练入口 ====================

    /**
     * 训练入口
     * <p>
     * 复用基类实现（检查点目录创建、训练结束后无论成功失败都复位为 eval 模式），
     * 只在结束时追加统计输出。
     */
    @Override
    public void train() {
        super.train();
        printAgentStats();
    }

    /**
     * 打印 Agent 训练统计
     */
    private void printAgentStats() {
        System.out.println("=".repeat(70));
        System.out.println("Agent RL 训练统计");
        System.out.println("=".repeat(70));

        if (!lossHistory.isEmpty()) {
            double avgLoss = lossHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均损失: %.4f%n", avgLoss);
            System.out.printf("  最终损失: %.4f%n", lossHistory.get(lossHistory.size() - 1));
        }

        if (!rewardHistory.isEmpty()) {
            double avgReward = rewardHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均奖励: %.4f%n", avgReward);
            System.out.printf("  最终奖励: %.4f%n", rewardHistory.get(rewardHistory.size() - 1));
        }

        if (!klHistory.isEmpty()) {
            double avgKl = klHistory.stream().mapToDouble(Float::doubleValue).average().orElse(0.0);
            System.out.printf("  平均 KL: %.6f%n", avgKl);
        }

        System.out.printf("  总训练步数: %d%n", currentStep);
        System.out.printf("  配置: %s%n", config);
        System.out.println("=".repeat(70));
    }

    // ==================== BaseTrainer 抽象方法实现 ====================

    @Override
    protected String getTrainerName() {
        return "Agent RL";
    }

    @Override
    protected void printTrainingInfo() {
        System.out.println("=".repeat(70));
        System.out.println("开始 Agent RL 训练 (工具调用强化学习)");
        System.out.println("=".repeat(70));
        System.out.println("  配置: " + config);
        System.out.printf("  训练样本数: %d%n", dataset.getSampleCount());
        System.out.printf("  批次数量: %d%n", dataset.getBatchCount());
        System.out.printf("  每 prompt 候选数: %d%n", config.getNumGenerations());
        System.out.printf("  最大工具交互轮数: %d%n", config.getMaxTurns());
        System.out.printf("  学习率: %.2e%n", config.getLearningRate());
        System.out.printf("  Beta (KL): %.2f, Epsilon (clip): %.2f%n",
                config.getBeta(), config.getEpsilon());
        System.out.println("=".repeat(70));
    }

    @Override
    protected void printTrainingLog() {
        double avgLoss = lossHistory.stream()
                .skip(Math.max(0, lossHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        double avgReward = rewardHistory.stream()
                .skip(Math.max(0, rewardHistory.size() - logInterval))
                .mapToDouble(Float::doubleValue)
                .average()
                .orElse(0.0);

        System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Reward: %.4f | LR: %.6f%n",
                currentEpoch + 1, maxEpochs, currentStep, avgLoss, avgReward, currentLearningRate);
    }

    @Override
    protected void prepareDataset() {
        dataset.prepare(true);
    }

    @Override
    protected boolean hasNextBatch() {
        return dataset.hasNext();
    }

    @Override
    protected Object getNextBatch() {
        return dataset.nextBatch();
    }

    @Override
    protected void resetDataset() {
        dataset.reset();
    }

    @Override
    protected String getCheckpointPrefix() {
        return "agent";
    }

    // ==================== Getter ====================

    public List<Float> getRewardHistory() {
        return new ArrayList<>(rewardHistory);
    }

    public List<Float> getKlHistory() {
        return new ArrayList<>(klHistory);
    }
}
