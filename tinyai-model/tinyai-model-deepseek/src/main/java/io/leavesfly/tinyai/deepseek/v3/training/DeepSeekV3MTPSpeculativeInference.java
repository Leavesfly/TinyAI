package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.inference.DeepSeekBaseInference;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3MTPHead;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于 MTP（Multi-Token Prediction）的推测解码推理引擎
 * <p>
 * 核心思想：利用训练阶段学到的 MTP 头作为轻量级 draft model，
 * 一次前向传播猜测多个 token，再用主模型验证，从而减少解码步数。
 * <p>
 * 算法流程（每轮迭代）：
 * 1. Draft 阶段：主模型前向传播一次，同时用 MTP 头猜测后续 D 个 token
 * 2. Verify 阶段：将猜测的 D 个 token 拼入序列，主模型再前向传播一次验证
 * 3. Accept/Reject：逐个比对 draft 和 verify 的结果，
 *    接受连续匹配的 token，从第一个不匹配处用 verify 的分布重新采样
 * <p>
 * 优势：
 * - MTP 头参数量远小于主模型（仅投影层 + RMSNorm），draft 几乎零额外开销
 * - 与外部 draft model 不同，MTP 头与主模型共享嵌入和输出投影，一致性更高
 * - 在 acceptance rate 较高时，可实现接近 D+1 倍的吞吐量提升
 *
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3MTPSpeculativeInference extends DeepSeekBaseInference {

    private final DeepSeekV3Model model;
    private final DeepSeekV3MTPHead mtpHead;
    private final int mtpDepth;
    private final int maxSeqLen;
    private final int eosTokenId;
    private final int vocabSize;

    /** 推测解码统计信息 */
    private long totalDraftTokens;
    private long totalAcceptedTokens;
    private long totalVerifyRounds;

    /**
     * 构造函数
     *
     * @param model DeepSeek-V3 模型（必须启用 MTP）
     * @throws IllegalArgumentException 如果模型未启用 MTP
     */
    public DeepSeekV3MTPSpeculativeInference(DeepSeekV3Model model) {
        super();
        this.model = model;
        this.mtpHead = model.getV3Block().getMtpHead();
        if (this.mtpHead == null) {
            throw new IllegalArgumentException(
                    "模型未启用 MTP（mtpDepth=0），无法使用推测解码推理");
        }
        this.mtpDepth = mtpHead.getMtpDepth();
        this.maxSeqLen = model.getConfig().getNPositions();
        this.vocabSize = model.getConfig().getVocabSize();
        this.eosTokenId = vocabSize - 1;
        resetStatistics();
    }

    /**
     * 使用推测解码进行贪婪生成
     *
     * @param promptIds    提示词 token 序列
     * @param maxNewTokens 最大生成 token 数
     * @return 生成结果（含推测解码统计）
     */
    public SpeculativeResult generateGreedy(int[] promptIds, int maxNewTokens) {
        return generateSpeculative(promptIds, maxNewTokens, VerifyMode.GREEDY, 1.0f);
    }

    /**
     * 使用推测解码进行 Temperature 采样生成
     *
     * @param promptIds    提示词 token 序列
     * @param maxNewTokens 最大生成 token 数
     * @param temperature  温度参数
     * @return 生成结果
     */
    public SpeculativeResult generateWithTemperature(int[] promptIds, int maxNewTokens,
                                                     float temperature) {
        return generateSpeculative(promptIds, maxNewTokens, VerifyMode.TEMPERATURE, temperature);
    }

    /**
     * 推测解码核心循环
     * <p>
     * 每轮循环：
     * 1. 对当前序列做一次主模型前向传播，得到 logits + hiddenStates
     * 2. 从主模型 logits 最后一个位置采样第一个新 token
     * 3. 用 MTP 头从 hiddenStates 猜测后续 D 个 draft token
     * 4. 将 (第一个新 token + D 个 draft token) 拼入序列，做一次验证前向传播
     * 5. 逐个比对验证结果，确定接受数量
     */
    private SpeculativeResult generateSpeculative(int[] promptIds, int maxNewTokens,
                                                  VerifyMode mode, float temperature) {
        resetStatistics();
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }

        int newTokenCount = 0;
        boolean hitEos = false;

        while (newTokenCount < maxNewTokens && !hitEos) {
            if (generated.size() >= maxSeqLen) {
                break;
            }

            // ====== Step 1: 主模型前向传播（Draft 起点）======
            int[] currentSeq = toIntArray(generated);
            Variable inputVar = new Variable(createInputArray(currentSeq));
            DeepSeekV3Block.DetailedForwardResult draftResult =
                    model.predictWithDetails(inputVar);

            NdArray draftLogits = draftResult.logits.getValue();
            int lastPos = currentSeq.length - 1;

            // 从主模型采样第一个新 token（这个 token 一定会被接受）
            int firstNewToken = sampleFromLogits(draftLogits, lastPos, mode, temperature);
            if (firstNewToken == eosTokenId) {
                generated.add(firstNewToken);
                newTokenCount++;
                hitEos = true;
                draftResult.logits.unChainBackward();
                inputVar.unChainBackward();
                break;
            }

            // ====== Step 2: 用 MTP 头猜测后续 D 个 draft token ======
            int[] draftTokens = draftWithMTP(
                    draftResult.hiddenStates, inputVar, draftResult.logits, lastPos, mode, temperature);

            // 释放 draft 阶段的计算图
            draftResult.logits.unChainBackward();
            inputVar.unChainBackward();

            // 构造候选序列：当前序列 + firstNewToken + draftTokens
            List<Integer> candidateTokens = new ArrayList<>();
            candidateTokens.add(firstNewToken);
            for (int dt : draftTokens) {
                candidateTokens.add(dt);
            }

            totalDraftTokens += draftTokens.length;

            // 检查 draft token 中是否包含 EOS
            int draftLenBeforeEos = draftTokens.length;
            for (int i = 0; i < draftTokens.length; i++) {
                if (draftTokens[i] == eosTokenId) {
                    draftLenBeforeEos = i + 1;
                    break;
                }
            }

            // ====== Step 3: 验证（将候选 token 拼入序列，主模型再次前向传播）======
            // 构建验证序列
            List<Integer> verifySeq = new ArrayList<>(generated);
            for (int i = 0; i < 1 + draftLenBeforeEos; i++) {
                verifySeq.add(candidateTokens.get(i));
            }

            // 限制序列长度
            if (verifySeq.size() > maxSeqLen) {
                verifySeq = new ArrayList<>(verifySeq.subList(0, maxSeqLen));
            }

            int[] verifyArr = toIntArray(verifySeq);
            Variable verifyInput = new Variable(createInputArray(verifyArr));
            DeepSeekV3Block.DetailedForwardResult verifyResult =
                    model.predictWithDetails(verifyInput);
            NdArray verifyLogits = verifyResult.logits.getValue();

            totalVerifyRounds++;

            // ====== Step 4: 逐个比对，确定接受数量 ======
            int accepted = verifyAndAccept(
                    verifyLogits, candidateTokens, draftLenBeforeEos,
                    currentSeq.length, mode, temperature);

            // 释放验证计算图
            verifyResult.logits.unChainBackward();
            verifyInput.unChainBackward();

            // 统计被接受的 draft token 数：accepted 包含 firstNewToken(1) + 匹配的draft + 可能的bonus(1)
            // 真正被接受的 draft token = min(accepted - 1, draftLenBeforeEos)，排除 firstNewToken 和 bonus
            int acceptedDraftCount = Math.min(Math.max(0, accepted - 1), draftLenBeforeEos);
            totalAcceptedTokens += acceptedDraftCount;

            // ====== Step 5: 将接受的 token 加入生成序列 ======
            for (int i = 0; i < accepted; i++) {
                int token = candidateTokens.get(i);
                generated.add(token);
                newTokenCount++;
                if (token == eosTokenId) {
                    hitEos = true;
                    break;
                }
                if (newTokenCount >= maxNewTokens || generated.size() >= maxSeqLen) {
                    break;
                }
            }
        }

        return new SpeculativeResult(
                toIntArray(generated),
                totalDraftTokens,
                totalAcceptedTokens,
                totalVerifyRounds
        );
    }

    /**
     * 使用 MTP 头生成 draft token
     * <p>
     * MTP 第 k 层预测的是位置 i+k+1 的 token，因此：
     * - depth=0 预测 i+2（即 firstNewToken 之后的 token）
     * - depth=1 预测 i+3
     * - ...
     * <p>
     * 这里用 computeMTPLogits 对每个深度分别预测。
     */
    private int[] draftWithMTP(Variable hiddenStates, Variable inputVar,
                               Variable mainLogits, int lastPos,
                               VerifyMode mode, float temperature) {
        int[] draftTokens = new int[mtpDepth];
        NdArray inputData = inputVar.getValue();

        for (int depth = 0; depth < mtpDepth; depth++) {
            try {
                // MTP 头计算该深度的 logits
                Variable mtpLogits = mtpHead.computeMTPLogits(hiddenStates, inputVar, depth);
                NdArray mtpLogitsData = mtpLogits.getValue();

                // MTP logits 形状 [1, validLen, vocabSize]，取最后一个位置
                int validLen = mtpLogitsData.getShape().getDimension(1);
                if (validLen <= 0) {
                    // 序列太短，无法预测，用 PAD 填充
                    draftTokens[depth] = PAD_TOKEN_ID;
                    continue;
                }
                int mtpLastPos = validLen - 1;
                draftTokens[depth] = sampleFromLogits(mtpLogitsData, mtpLastPos, mode, temperature);
            } catch (Exception e) {
                // MTP 预测失败时，回退到 PAD
                draftTokens[depth] = PAD_TOKEN_ID;
            }
        }
        return draftTokens;
    }

    /**
     * 验证 draft token 并确定接受数量
     * <p>
     * 验证逻辑：
     * - candidateTokens[0] = firstNewToken（由主模型直接采样，一定接受）
     * - candidateTokens[1..n] = MTP draft token
     * - 对于每个 draft token，比较验证模型在对应位置的采样结果：
     *   - 如果一致 → 接受，继续检查下一个
     *   - 如果不一致 → 拒绝，用验证模型的结果替换，停止
     *
     * @return 接受的 token 数量（包含 firstNewToken 和可能的额外验证 token）
     */
    private int verifyAndAccept(NdArray verifyLogits, List<Integer> candidateTokens,
                                int draftLen, int prefixLen,
                                VerifyMode mode, float temperature) {
        // candidateTokens[0] = firstNewToken，总是接受
        int accepted = 1;

        // 验证每个 draft token
        for (int i = 0; i < draftLen; i++) {
            // 验证模型在位置 prefixLen + i 的输出对应预测 candidateTokens[i+1]（如果存在）
            int verifyPos = prefixLen + i;
            if (verifyPos >= verifyLogits.getShape().getDimension(1)) {
                break;
            }

            int verifyToken = sampleFromLogits(verifyLogits, verifyPos, mode, temperature);
            int draftToken = candidateTokens.get(i + 1);

            if (verifyToken == draftToken) {
                // draft token 被接受
                accepted++;
            } else {
                // draft token 被拒绝，用验证模型的 token 替换
                candidateTokens.set(i + 1, verifyToken);
                accepted++;
                break;
            }
        }

        // 如果所有 draft token 都被接受，还可以从验证模型最后位置再采样一个 bonus token
        if (accepted == 1 + draftLen) {
            int bonusPos = prefixLen + draftLen;
            if (bonusPos < verifyLogits.getShape().getDimension(1)) {
                int bonusToken = sampleFromLogits(verifyLogits, bonusPos, mode, temperature);
                candidateTokens.add(bonusToken);
                accepted++;
            }
        }

        return accepted;
    }

    /**
     * 根据验证模式从 logits 中采样 token
     */
    private int sampleFromLogits(NdArray logits, int pos, VerifyMode mode, float temperature) {
        if (mode == VerifyMode.GREEDY) {
            return argmax(logits, 0, pos);
        }
        float[] probs = applySoftmax(logits, 0, pos, temperature);
        return sample(probs);
    }

    /**
     * 重置统计信息
     */
    public void resetStatistics() {
        totalDraftTokens = 0;
        totalAcceptedTokens = 0;
        totalVerifyRounds = 0;
    }

    /**
     * 获取 draft token 接受率
     */
    public double getAcceptanceRate() {
        return totalDraftTokens == 0 ? 0.0 : (double) totalAcceptedTokens / totalDraftTokens;
    }

    /**
     * 获取平均每轮接受的 token 数（含第一个确定 token）
     */
    public double getAvgTokensPerRound() {
        return totalVerifyRounds == 0 ? 0.0 :
                (double) (totalVerifyRounds + totalAcceptedTokens) / totalVerifyRounds;
    }

    /**
     * 打印推测解码统计信息
     */
    public void printStatistics() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("📊 推测解码统计");
        System.out.println("=".repeat(60));
        System.out.printf("  MTP 深度: %d%n", mtpDepth);
        System.out.printf("  验证轮数: %d%n", totalVerifyRounds);
        System.out.printf("  Draft token 总数: %d%n", totalDraftTokens);
        System.out.printf("  接受 token 数: %d%n", totalAcceptedTokens);
        System.out.printf("  接受率: %.2f%%%n", getAcceptanceRate() * 100);
        System.out.printf("  平均每轮生成 token: %.2f%n", getAvgTokensPerRound());
        System.out.printf("  理论最大加速比: %.1fx%n", (double) (1 + mtpDepth));
        System.out.println("=".repeat(60));
    }

    // ==================== 内部类 ====================

    /** 验证模式枚举 */
    private enum VerifyMode {
        GREEDY,
        TEMPERATURE
    }

    /**
     * 推测解码生成结果
     */
    public static class SpeculativeResult {
        /** 生成的完整 token 序列（包含 prompt） */
        public final int[] tokens;
        /** draft token 总数 */
        public final long draftTokens;
        /** 被接受的 draft token 数 */
        public final long acceptedTokens;
        /** 验证轮数 */
        public final long verifyRounds;

        public SpeculativeResult(int[] tokens, long draftTokens,
                                 long acceptedTokens, long verifyRounds) {
            this.tokens = tokens;
            this.draftTokens = draftTokens;
            this.acceptedTokens = acceptedTokens;
            this.verifyRounds = verifyRounds;
        }

        /**
         * 获取接受率
         */
        public double getAcceptanceRate() {
            return draftTokens == 0 ? 0.0 : (double) acceptedTokens / draftTokens;
        }

        @Override
        public String toString() {
            return String.format(
                    "SpeculativeResult{tokens=%d, draftTokens=%d, accepted=%d, " +
                            "acceptanceRate=%.2f%%, verifyRounds=%d}",
                    tokens.length, draftTokens, acceptedTokens,
                    getAcceptanceRate() * 100, verifyRounds);
        }
    }
}
