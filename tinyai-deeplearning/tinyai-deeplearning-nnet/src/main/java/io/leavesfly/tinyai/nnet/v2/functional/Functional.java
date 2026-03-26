package io.leavesfly.tinyai.nnet.v2.functional;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.util.Config;

import java.util.Objects;

/**
 * A lightweight, stateless functional API mirroring常用层的前向计算。
 * <p>
 * 设计目标：
 * <ul>
 *   <li>与PyTorch的 {@code torch.nn.functional} 类似，提供直接的函数式调用</li>
 *   <li>不持有内部状态，参数/缓冲由调用方显式传入</li>
 *   <li>复用已有的自动微分体系（Variable/Function），梯度可自动回传到传入的参数</li>
 * </ul>
 *
 * 注意：本类仅封装核心高频算子，更多算子可按需补充。
 */
public final class Functional {

    private Functional() {
        // 工具类不允许实例化
    }

    /* --------------------------------- 激活函数 --------------------------------- */

    public static Variable relu(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.relu();
    }

    public static Variable sigmoid(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.sigmoid();
    }

    public static Variable tanh(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.tanh();
    }

    public static Variable softmax(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.softMax();
    }

    /**
     * GELU激活函数
     * GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     */
    public static Variable gelu(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.gelu();
    }

    /**
     * SiLU激活函数（Swish）
     * SiLU(x) = x * sigmoid(x)
     */
    public static Variable silu(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.silu();
    }

    /**
     * LeakyReLU激活函数
     * LeakyReLU(x) = max(negative_slope * x, x)
     */
    public static Variable leakyRelu(Variable input, float negativeSlope) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.leakyRelu(negativeSlope);
    }

    public static Variable leakyRelu(Variable input) {
        return leakyRelu(input, 0.01f);
    }

    /**
     * ELU激活函数
     * ELU(x) = x if x >= 0, else alpha * (exp(x) - 1)
     */
    public static Variable elu(Variable input, float alpha) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.elu(alpha);
    }

    public static Variable elu(Variable input) {
        return elu(input, 1.0f);
    }

    /**
     * LogSoftmax激活函数
     * LogSoftmax(x) = log(softmax(x))
     */
    public static Variable logSoftmax(Variable input, int axis) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.logSoftmax(axis);
    }

    public static Variable logSoftmax(Variable input) {
        return logSoftmax(input, -1);
    }

    /* --------------------------------- 线性与正则 --------------------------------- */

    public static Variable linear(Variable input, Variable weight) {
        return linear(input, weight, null);
    }

    /**
     * 线性映射：y = x W^T + b
     */
    public static Variable linear(Variable input, Variable weight, Variable bias) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(weight, "weight cannot be null");
        return input.linear(weight, bias);
    }

    /**
     * Dropout（功能等价于Dropout Module）。
     *
     * @param input    输入
     * @param p        丢弃概率，取值 [0,1)
     * @param training 是否处于训练模式（推理模式将直接返回输入）
     */
    public static Variable dropout(Variable input, float p, boolean training) {
        Objects.requireNonNull(input, "input cannot be null");
        Dropout dropout = new Dropout("functional_dropout", p);
        dropout.train(training);
        return dropout.forward(input);
    }

    /**
     * Dropout，训练模式默认取决于全局Config.train。
     */
    public static Variable dropout(Variable input, float p) {
        return dropout(input, p, Boolean.TRUE.equals(Config.train));
    }

    /* --------------------------------- 归一化 --------------------------------- */

    /**
     * LayerNorm 前向：y = gamma * (x - mean) / sqrt(var + eps) + beta
     * gamma/beta 由调用方提供，可传入可学习的Parameter或常量Variable。
     */
    public static Variable layerNorm(Variable input, Variable gamma, Variable beta, float eps) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(gamma, "gamma cannot be null");
        Objects.requireNonNull(beta, "beta cannot be null");

        Variable mean = input.mean(-1, true);
        Variable variance = input.var(-1, true);
        Variable normalized = input.sub(mean).div(variance.add(new Variable(eps)).sqrt());
        return normalized.mul(gamma).add(beta);
    }

    public static Variable layerNorm(Variable input, Variable gamma, Variable beta) {
        return layerNorm(input, gamma, beta, 1e-5f);
    }

    /**
     * RMSNorm 前向：y = x / RMS(x) * weight
     * RMS(x) = sqrt(mean(x^2) + eps)
     *
     * @param input  输入张量
     * @param weight 缩放参数
     * @param eps    数值稳定项
     */
    public static Variable rmsNorm(Variable input, Variable weight, float eps) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(weight, "weight cannot be null");

        // x^2
        Variable xSquared = input.mul(input);
        // mean(x^2)
        Variable meanSquared = xSquared.mean(-1, true);
        // RMS = sqrt(mean(x^2) + eps)
        Variable rms = meanSquared.add(new Variable(eps)).sqrt();
        // x / RMS * weight
        return input.div(rms).mul(weight);
    }

    public static Variable rmsNorm(Variable input, Variable weight) {
        return rmsNorm(input, weight, 1e-6f);
    }

    /**
     * BatchNorm1d 前向。支持训练/推理，两种模式下的行为与 BatchNorm1d Module 保持一致。
     *
     * @param input       输入 (batch, features[, length])
     * @param gamma       缩放参数，可为空（等价于不使用affine）
     * @param beta        平移参数，可为空（等价于不使用affine）
     * @param runningMean 运行时均值缓冲，可为空（不跟踪统计量时）
     * @param runningVar  运行时方差缓冲，可为空（不跟踪统计量时）
     * @param training    是否训练模式
     * @param momentum    运行时统计量动量
     * @param eps         数值稳定项
     */
    public static Variable batchNorm1d(Variable input,
                                       Variable gamma,
                                       Variable beta,
                                       NdArray runningMean,
                                       NdArray runningVar,
                                       boolean training,
                                       float momentum,
                                       float eps) {
        Objects.requireNonNull(input, "input cannot be null");

        int[] dims = input.getValue().getShape().getShapeDims();
        if (dims.length < 2) {
            throw new IllegalArgumentException("BatchNorm1d expects input with at least 2 dimensions");
        }

        Variable mean;
        Variable var;

        if (training) {
            mean = input.mean(0, false);
            var = input.var(0, false);

            if (runningMean != null && runningVar != null) {
                float[] rm = runningMean.getArray();
                float[] rv = runningVar.getArray();
                float[] bm = mean.getValue().getArray();
                float[] bv = var.getValue().getArray();

                if (rm.length != bm.length || rv.length != bv.length) {
                    throw new IllegalArgumentException("runningMean/runningVar size mismatch with input features");
                }

                for (int i = 0; i < rm.length; i++) {
                    rm[i] = (1 - momentum) * rm[i] + momentum * bm[i];
                    rv[i] = (1 - momentum) * rv[i] + momentum * bv[i];
                }
            }
        } else {
            if (runningMean == null || runningVar == null) {
                throw new IllegalArgumentException("runningMean and runningVar are required in eval mode");
            }
            mean = new Variable(runningMean);
            var = new Variable(runningVar);
        }

        Variable normalized = input.sub(mean).div(var.add(new Variable(eps)).sqrt());
        if (gamma != null) {
            normalized = normalized.mul(gamma);
        }
        if (beta != null) {
            normalized = normalized.add(beta);
        }
        return normalized;
    }

    public static Variable batchNorm1d(Variable input,
                                       Variable gamma,
                                       Variable beta,
                                       NdArray runningMean,
                                       NdArray runningVar,
                                       boolean training) {
        return batchNorm1d(input, gamma, beta, runningMean, runningVar, training, 0.1f, 1e-5f);
    }

    /* --------------------------------- 嵌入 --------------------------------- */

    /**
     * 词嵌入查找，等价于 Embedding Module 的前向。
     *
     * @param indices 索引张量，支持1D或2D
     * @param weight  形状为 (num_embeddings, embedding_dim) 的嵌入矩阵
     */
    public static Variable embedding(Variable indices, Variable weight) {
        Objects.requireNonNull(indices, "indices cannot be null");
        Objects.requireNonNull(weight, "weight cannot be null");

        NdArray idxValue = indices.getValue();
        int dim = idxValue.getShape().getDimNum();
        NdArray weightData = weight.getValue();
        int embeddingDim = weightData.getShape().getColumn();

        if (dim == 1) {
            int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[0]);
            return weight.getItem(slices, null);
        } else if (dim == 2) {
            int batchSize = idxValue.getShape().getRow();
            int seqLen = idxValue.getShape().getColumn();

            NdArray result = NdArray.zeros(Shape.of(batchSize, seqLen, embeddingDim));
            for (int i = 0; i < batchSize; i++) {
                int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[i]);
                Variable embRow = weight.getItem(slices, null);
                NdArray embVal = embRow.getValue();
                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        result.set(embVal.get(j, k), i, j, k);
                    }
                }
            }

            if (seqLen == 1) {
                result = result.reshape(Shape.of(batchSize, embeddingDim));
            }
            return new Variable(result);
        } else {
            throw new IllegalArgumentException("Embedding only supports 1D or 2D index tensors, got shape: " + idxValue.getShape());
        }
    }

    /* --------------------------------- 注意力 --------------------------------- */

    /**
     * 缩放点积注意力
     * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
     *
     * @param query    查询张量 (batch, seq_len, d_k) 或 (batch, heads, seq_len, d_k)
     * @param key      键张量
     * @param value    值张量
     * @param attnMask 注意力掩码（可选，被屏蔽位置应为-inf或很大的负数）
     * @param dropout  dropout比率
     * @param training 是否训练模式
     */
    public static Variable scaledDotProductAttention(Variable query, Variable key, Variable value,
                                                     Variable attnMask, float dropout, boolean training) {
        Objects.requireNonNull(query, "query cannot be null");
        Objects.requireNonNull(key, "key cannot be null");
        Objects.requireNonNull(value, "value cannot be null");

        // 获取d_k用于缩放
        int[] qShape = query.getValue().getShape().getShapeDims();
        int dK = qShape[qShape.length - 1];

        // Q * K^T
        Variable scores = query.matMul(key.transpose());

        // 缩放
        double scale = Math.sqrt(dK);
        Variable scaledScores = scores.div(new Variable((float) scale));

        // 应用掩码
        if (attnMask != null) {
            scaledScores = scaledScores.add(attnMask);
        }

        // Softmax（添加数值稳定性处理：先减去最大值）
        // 注意：Max 算子不支持负数轴，需要转换为正数轴
        int lastAxis = scaledScores.getShape().getDimNum() - 1;
        Variable maxScores = scaledScores.max(lastAxis, true);
        Variable stableScores = scaledScores.sub(maxScores);
        Variable attnWeights = stableScores.softMax();

        // Dropout
        if (training && dropout > 0) {
            attnWeights = dropout(attnWeights, dropout, true);
        }

        // 加权求和
        return attnWeights.matMul(value);
    }

    public static Variable scaledDotProductAttention(Variable query, Variable key, Variable value) {
        return scaledDotProductAttention(query, key, value, null, 0f, false);
    }

    public static Variable scaledDotProductAttention(Variable query, Variable key, Variable value, Variable attnMask) {
        return scaledDotProductAttention(query, key, value, attnMask, 0f, false);
    }

    /* --------------------------------- 损失函数 --------------------------------- */

    /**
     * 交叉熵损失（带LogSoftmax）
     * CrossEntropy(input, target) = NLLLoss(LogSoftmax(input), target)
     *
     * @param input  模型输出 (batch, num_classes)
     * @param target 目标标签 (batch,) 或 (batch, num_classes) one-hot
     */
    public static Variable crossEntropyLoss(Variable input, Variable target) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(target, "target cannot be null");

        // LogSoftmax
        Variable logProbs = input.logSoftmax(-1);

        // 负对数似然
        return nllLoss(logProbs, target);
    }

    /**
     * 负对数似然损失
     *
     * @param input  对数概率 (batch, num_classes)
     * @param target 目标标签 (batch,) 整数索引
     */
    public static Variable nllLoss(Variable input, Variable target) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(target, "target cannot be null");

        NdArray inputData = input.getValue();
        NdArray targetData = target.getValue();

        int batchSize = inputData.getShape().getShapeDims()[0];
        int numClasses = inputData.getShape().getShapeDims()[1];

        float[] losses = new float[1];
        float totalLoss = 0;

        float[] inputArr = inputData.getArray();
        float[] targetArr = targetData.getArray();

        for (int i = 0; i < batchSize; i++) {
            int targetIdx = (int) targetArr[i];
            if (targetIdx >= 0 && targetIdx < numClasses) {
                totalLoss -= inputArr[i * numClasses + targetIdx];
            }
        }

        losses[0] = totalLoss / batchSize;
        return new Variable(NdArray.of(losses, Shape.of(1)));
    }

    /**
     * 均方误差损失
     * MSE(input, target) = mean((input - target)^2)
     */
    public static Variable mseLoss(Variable input, Variable target) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(target, "target cannot be null");

        Variable diff = input.sub(target);
        Variable squared = diff.mul(diff);
        // 计算所有元素的均值
        NdArray squaredData = squared.getValue();
        float sum = 0;
        float[] arr = squaredData.getArray();
        for (float v : arr) {
            sum += v;
        }
        return new Variable(NdArray.of(new float[]{sum / arr.length}, Shape.of(1)));
    }

    /**
     * 二元交叉熵损失（需要输入已经过sigmoid）
     * BCE(input, target) = -mean(target * log(input) + (1 - target) * log(1 - input))
     */
    public static Variable binaryCrossEntropyLoss(Variable input, Variable target) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(target, "target cannot be null");

        // 添加小常数避免log(0)
        float eps = 1e-7f;
        Variable inputClipped = input.clip(eps, 1 - eps);

        Variable logInput = inputClipped.log();
        Variable logOneMinusInput = inputClipped.neg().add(new Variable(1f)).log();

        Variable term1 = target.mul(logInput);
        Variable term2 = target.neg().add(new Variable(1f)).mul(logOneMinusInput);

        Variable loss = term1.add(term2).neg();
        // 计算均值
        NdArray lossData = loss.getValue();
        float sum = 0;
        float[] arr = lossData.getArray();
        for (float v : arr) {
            sum += v;
        }
        return new Variable(NdArray.of(new float[]{sum / arr.length}, Shape.of(1)));
    }

    /**
     * 带Logits的二元交叉熵损失
     * BCEWithLogits(input, target) = BCE(sigmoid(input), target)
     * 使用更数值稳定的计算
     */
    public static Variable binaryCrossEntropyWithLogitsLoss(Variable input, Variable target) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(target, "target cannot be null");

        // max(input, 0) - input * target + log(1 + exp(-abs(input)))
        Variable maxInputZero = input.clip(0, Float.MAX_VALUE);
        Variable term1 = maxInputZero.sub(input.mul(target));

        // log(1 + exp(-abs(input)))
        Variable absInput = input.clip(0, Float.MAX_VALUE).sub(input.clip(Float.MIN_VALUE, 0).neg());
        Variable term2 = absInput.neg().exp().add(new Variable(1f)).log();

        Variable loss = term1.add(term2);
        // 计算均值
        NdArray lossData = loss.getValue();
        float sum = 0;
        float[] arr = lossData.getArray();
        for (float v : arr) {
            sum += v;
        }
        return new Variable(NdArray.of(new float[]{sum / arr.length}, Shape.of(1)));
    }
}