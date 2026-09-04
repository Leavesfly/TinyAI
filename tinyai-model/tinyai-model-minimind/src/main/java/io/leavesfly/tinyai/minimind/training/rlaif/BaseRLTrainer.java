package io.leavesfly.tinyai.minimind.training.rlaif;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.BaseTrainer;
import io.leavesfly.tinyai.minimind.training.rlaif.ppo.ValueNetwork;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.Map;

/**
 * 强化学习训练器基类
 * <p>
 * 提供 RL 训练器共有的工具方法：
 * - computePerSampleLogProbs: 逐样本计算序列对数概率（保持 [batch] 维度）
 * - computeScalarEntropy: 计算标量平均熵（用于熵正则）
 * - clipGradients: 梯度裁剪（支持 MiniMindModel 与 ValueNetwork）
 * <p>
 * 为什么必须保留 batch 维：
 * RL 的优势 A[i][k] 是"第 i 个 prompt 的第 k 个候选"的信号，必须与该样本自己的
 * log π(y_i|x_i) 相乘。若把整个 batch 的 logProb 归约成一个标量再乘以各样本的优势，
 * 梯度会退化成 (Σ_i A[i][k]) · ∂f(batch级logprob)/∂θ —— 等价于把整个 batch 当成
 * 一个样本训练，per-sample 的 credit assignment 完全丢失。
 *
 * @author leavesfly
 * @since 2024
 */
public abstract class BaseRLTrainer extends BaseTrainer {

    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     */
    protected static final int IGNORE_INDEX = -100;

    /**
     * 构造函数
     *
     * @param model 模型
     */
    public BaseRLTrainer(MiniMindModel model) {
        super(model);
    }

    /**
     * 逐样本计算序列对数概率 log π(y_i|x_i)
     * <p>
     * 实现方式：逐样本用 {@code sliceRange} 切出 logits，reshape 成 2D 后交给
     * {@code softmaxCrossEntropy}（内部完成 logSoftmax → gather(label) → 对有效 token
     * 求平均 NLL，且天然支持 ignore_index(-100)：前向跳过、反向该行梯度置零），
     * 取负即为该样本的平均对数概率，最后 {@code cat} 回 [batch]。
     * <p>
     * 两个刻意的取舍：
     * 1. 不用 {@code split} 切片。split 是多输出函数，当多个分片汇入同一个损失时，
     *    先被访问分片的梯度会被后一个分片的 multi-output backward 重复计入
     *    （其 .grad 未被清理），造成梯度翻倍。sliceRange 是单输出函数，无此问题。
     * 2. 不构造 [tokens, vocab] 的 one-hot 稠密矩阵做 gather，避免大词表下的内存爆炸。
     * <p>
     * 注意：返回的是"每 token 平均"对数概率（长度归一化）。这能避免长回答因 token 多
     * 而获得更大权重；若需要序列求和口径，请乘以各自的有效 token 数。
     *
     * @param logits 模型输出 [batch, seq_len, vocab_size]
     * @param labels 标签 Variable [batch, seq_len]，忽略位应为负数
     * @return 逐样本对数概率 [batch]（保持计算图）
     */
    protected Variable computePerSampleLogProbs(Variable logits, Variable labels) {
        return perSampleLogProbs(logits, labels);
    }

    /**
     * {@link #computePerSampleLogProbs(Variable, Variable)} 的静态版本
     * <p>
     * 开放给不属于本类子类的损失对象（如 {@code SPOLoss}）复用，保证 GRPO / PPO / SPO /
     * Agent RL 四者使用完全相同的 logProb 口径（长度归一 + ignore_index 跳过 padding）。
     *
     * @param logits 模型输出 [batch, seq_len, vocab_size]
     * @param labels 标签 Variable [batch, seq_len]，忽略位应为负数
     * @return 逐样本对数概率 [batch]（保持计算图）
     */
    public static Variable perSampleLogProbs(Variable logits, Variable labels) {
        int[] shape = logits.getValue().getShape().getShapeDims();
        if (shape.length != 3) {
            throw new IllegalArgumentException(
                "perSampleLogProbs 需要 3D logits [batch, seq, vocab]，实际维度: " + shape.length);
        }
        int batchSize = shape[0];
        int seqLen = shape[1];
        int vocabSize = shape[2];

        float[] labelData = labels.getValue().getArray();

        Variable[] perSample = new Variable[batchSize];
        for (int b = 0; b < batchSize; b++) {
            // 取出该样本的标签列，越界标签改写为 ignore_index
            float[] labelColumn = new float[seqLen];
            for (int s = 0; s < seqLen; s++) {
                int labelIdx = (int) labelData[b * seqLen + s];
                labelColumn[s] = (labelIdx >= 0 && labelIdx < vocabSize) ? labelIdx : IGNORE_INDEX;
            }

            Variable sampleLogits = logits.sliceRange(0, b, b + 1)
                .reshape(Shape.of(seqLen, vocabSize));
            Variable sampleLabels = new Variable(NdArray.of(labelColumn, Shape.of(seqLen, 1)));
            sampleLabels.setRequireGrad(false);

            // softmaxCrossEntropy 返回有效 token 的平均 NLL，取负即为平均对数概率。
            // 该算子的标量输出形状是 [1,1]，必须 reshape 成 [1] 后再 cat：
            // 否则 cat(dim=0) 得到的是 [batch, 1] 而不是 [batch]，
            // 与形状为 [batch] 的优势/旧 logProb 常量相减时会因广播不兼容直接报错。
            perSample[b] = sampleLogits.softmaxCrossEntropy(sampleLabels).neg()
                .reshape(Shape.of(1));
        }

        return Variable.cat(perSample, 0);
    }

    /**
     * 逐样本计算序列对数概率（数值版，不构建计算图）
     * <p>
     * 用于旧策略 / 参考模型的 logProb 采集：这些值只作为常量参与后续计算。
     * 传入的 logits 应当已经 detach，否则前向仍会白建一份计算图。
     *
     * @param logits 模型输出 [batch, seq_len, vocab_size]
     * @param labels 标签 Variable [batch, seq_len]
     * @return 逐样本对数概率 [batch]
     */
    protected float[] computePerSampleLogProbValues(Variable logits, Variable labels) {
        return computePerSampleLogProbs(logits, labels).getValue().getArray().clone();
    }

    /**
     * 计算标量平均熵（用于熵正则化）
     * <p>
     * H = -Σ_v p·log p，先按 token 归约再对全部 token 求平均，返回形状为 [1] 的标量。
     * <p>
     * 必须返回标量：若沿用 {@code mean(-1, true)}，该实现会把结果广播回原形状 [B,L,V]，
     * 于是"标量策略损失 - 系数 * 熵"得到的是 [B,L,V] 张量而非标量。backward() 对非标量
     * 损失用 ones(shape) 初始化梯度，等价于对所有元素求和，熵项的等效权重会被放大 B*L 倍；
     * 同时日志里的 getNumber() 只取 buffer[0]，打印出的损失是单个元素值而非平均值。
     * <p>
     * 概率项 detach，使梯度只通过 logSoftmax 一条路径回传，避免双重计数。
     *
     * @param logits 模型输出 [batch, seq_len, vocab] 或 [n, vocab]
     * @return 标量平均熵（形状 [1]）
     */
    protected Variable computeScalarEntropy(Variable logits) {
        return scalarEntropy(logits);
    }

    /**
     * {@link #computeScalarEntropy(Variable)} 的静态版本，供各损失类复用
     *
     * @param logits 模型输出 [batch, seq_len, vocab] 或 [n, vocab]
     * @return 标量平均熵（形状 [1]）
     */
    public static Variable scalarEntropy(Variable logits) {
        int[] dims = logits.getValue().getShape().getShapeDims();

        // p * log p（p 已 detach，梯度只经由 logProbs 回传）
        Variable logProbs = logits.logSoftmax();
        Variable probs = logits.softMax().detach();
        Variable pLogP = probs.mul(logProbs);

        // 沿 vocab 维求和得到逐 token 的负熵，用 sumTo 避免构造额外的大张量
        long tokenCount;
        Variable perTokenEntropy;
        if (dims.length == 3) {
            perTokenEntropy = pLogP.sumTo(Shape.of(dims[0], dims[1], 1)).neg();
            tokenCount = (long) dims[0] * dims[1];
        } else if (dims.length == 2) {
            perTokenEntropy = pLogP.sumTo(Shape.of(dims[0], 1)).neg();
            tokenCount = dims[0];
        } else {
            throw new IllegalArgumentException(
                "scalarEntropy 需要 2D/3D logits，实际维度: " + dims.length);
        }

        // 对全部 token 求平均 → 标量（归一到形状 [1]）
        float invTokenCount = tokenCount > 0 ? 1.0f / tokenCount : 0.0f;
        return toScalar(perTokenEntropy.sum().mul(constant(invTokenCount)));
    }

    /**
     * PPO / GRPO 共用的 Clipped Surrogate 损失（唯一权威实现）
     * <p>
     * L^{CLIP} = -mean_i( min(r_i·A_i, clip(r_i, 1-ε, 1+ε)·A_i) )
     * 其中 r_i = exp(logπ_new_i - logπ_old_i)，均为逐样本量。
     * <p>
     * 实现要点：
     * 1. 优势与旧策略 logProb 作为外部信号，以 requireGrad=false 的常量参与，不回传梯度
     * 2. 用 {@code clip} 而非手写 min/max，Clip 的反向对越界位置返回 0 梯度，
     *    正是 PPO 期望的"超出信任区就停止更新"行为
     * 3. 最终归约到标量（sum 后乘 1/batch），不能用 mean(axis, keepdims)：
     *    后者会把结果广播回原形状，使损失非标量、backward 时按元素求和而放大梯度
     *
     * @param newLogProbs  新策略逐样本对数概率 [batchSize]（保持计算图）
     * @param oldLogProbs  旧策略逐样本对数概率 [batchSize]（常量）
     * @param advantages   逐样本优势 [batchSize]（常量）
     * @param clipEpsilon  裁剪范围 ε
     * @return 标量损失 Variable
     */
    public static Variable clippedSurrogateLoss(Variable newLogProbs, float[] oldLogProbs,
                                                float[] advantages, float clipEpsilon) {
        int batchSize = newLogProbs.getValue().getShape().getDimension(0);

        Variable logRatio = newLogProbs.sub(constant(oldLogProbs));
        Variable ratio = logRatio.exp();

        Variable advVar = constant(advantages);
        Variable surrogate1 = ratio.mul(advVar);
        Variable surrogate2 = ratio.clip(1.0f - clipEpsilon, 1.0f + clipEpsilon).mul(advVar);

        Variable condition = surrogate1.lt(surrogate2);
        Variable minSurrogate = Variable.where(condition, surrogate1, surrogate2);

        float invBatchSize = batchSize > 0 ? 1.0f / batchSize : 0.0f;
        return toScalar(minSurrogate.neg().sum().mul(constant(invBatchSize)));
    }
    
    /**
     * 梯度裁剪（支持 MiniMindModel 与 ValueNetwork）
     * <p>
     * 只通过 NdArray 公开接口读写梯度，不假设具体后端实现。
     *
     * @param target  模型对象（MiniMindModel 或 ValueNetwork）
     * @param maxNorm 最大梯度范数，<=0 表示不裁剪
     */
    protected void clipGradients(Object target, float maxNorm) {
        if (maxNorm <= 0) {
            return;
        }

        Map<String, Parameter> params;
        if (target instanceof MiniMindModel) {
            params = ((MiniMindModel) target).getAllParams();
        } else if (target instanceof ValueNetwork) {
            params = ((ValueNetwork) target).getAllParams();
        } else {
            return;
        }

        double normSquare = 0.0;
        for (Parameter param : params.values()) {
            NdArray grad = param.getGrad();
            if (grad == null) {
                continue;
            }
            for (float g : grad.getArray()) {
                normSquare += (double) g * g;
            }
        }

        double totalNorm = Math.sqrt(normSquare);
        // 写成 !(x > y) 以便 NaN 范数也被拦住，避免用 NaN 系数污染全部梯度
        if (!(totalNorm > maxNorm)) {
            return;
        }

        float scale = (float) (maxNorm / (totalNorm + 1e-6));
        for (Parameter param : params.values()) {
            NdArray grad = param.getGrad();
            if (grad == null) {
                continue;
            }
            param.setGrad(grad.mulNum(scale));
        }
    }
}
