package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 掩码 Softmax 交叉熵损失函数。
 *
 * 用于处理序列模型中的掩码交叉熵损失计算，特别适用于变长序列。
 * 较短的序列会被填充到固定长度，掩码用于忽略填充部分的损失计算。
 *
 * 主要应用场景：
 * 1. 机器翻译中的变长目标序列
 * 2. 文本生成任务中的填充部分
 * 3. 序列到序列模型的训练
 *
 * 损失计算公式：Loss = -Σ(mask_i * y_i * log(softmax(pred_i))) / Σ(mask_i)
 * 其中 mask_i 为掩码值（0 或 1），用于指示有效的序列位置。
 *
 * @author TinyDL
 * @version 1.0
 */
public class MaskedSoftmaxCELoss extends Loss {
    
    /**
     * 默认的填充标记ID，通常用0表示填充位置
     */
    private static final int DEFAULT_PAD_TOKEN = 0;
    
    /**
     * 计算带掩码的Softmax交叉熵损失
     * 
     * @param y       真实标签，形状为(batch_size, seq_len)
     * @param predict 预测值，形状为(batch_size, seq_len, vocab_size)
     * @return 掩码后的平均损失值
     */
    @Override
    public Variable loss(Variable y, Variable predict) {
        return maskedSoftmaxCrossEntropy(y, predict, DEFAULT_PAD_TOKEN);
    }
    
    /**
     * 计算带掩码的Softmax交叉熵损失（指定填充标记）
     * 
     * @param y        真实标签，形状为(batch_size, seq_len)
     * @param predict  预测值，形状为(batch_size, seq_len, vocab_size)
     * @param padToken 填充标记ID
     * @return 掩码后的平均损失值
     */
    public Variable maskedSoftmaxCrossEntropy(Variable y, Variable predict, int padToken) {
        validateInputs(y, predict);

        // 先创建掩码，再将标签中填充位修正到合法下标，避免 -1 等非法索引导致 getItem 抛错
        Variable mask = createMask(y, padToken);
        int vocabSize = predict.getValue().getShape().getDimension(2);
        Variable sanitizedLabels = sanitizeLabels(y, padToken, vocabSize);

        // 计算常规的softmax交叉熵损失
        Variable loss = computeSoftmaxCrossEntropy(sanitizedLabels, predict);
        
        // 应用掩码
        Variable maskedLoss = applyMask(loss, mask);
        
        return maskedLoss;
    }
    
    /**
     * 计算带掩码的Softmax交叉熵损失（使用自定义掩码）
     * 
     * @param y       真实标签，形状为(batch_size, seq_len)
     * @param predict 预测值，形状为(batch_size, seq_len, vocab_size)
     * @param mask    自定义掩码，形状为(batch_size, seq_len)，1表示有效位置，0表示忽略位置
     * @return 掩码后的平均损失值
     */
    public Variable maskedSoftmaxCrossEntropyWithMask(Variable y, Variable predict, Variable mask) {
        validateInputs(y, predict);
        // 计算常规的softmax交叉熵损失
        Variable loss = computeSoftmaxCrossEntropy(y, predict);
        
        // 应用自定义掩码
        Variable maskedLoss = applyMask(loss, mask);
        
        return maskedLoss;
    }
    
    /**
     * 根据填充标记创建掩码
     * 
     * @param labels   标签数组
     * @param padToken 填充标记ID
     * @return 掩码变量，1表示有效位置，0表示填充位置
     */
    private Variable createMask(Variable labels, int padToken) {
        NdArray labelsArray = labels.getValue();
        Shape shape = labelsArray.getShape();
        
        // 创建掩码数组
        float[][] mask = new float[shape.getRow()][shape.getColumn()];
        float[][] labelsMatrix = labelsArray.getMatrix();
        
        for (int i = 0; i < shape.getRow(); i++) {
            for (int j = 0; j < shape.getColumn(); j++) {
                // 非填充位置标记为1，填充位置标记为0
                mask[i][j] = (labelsMatrix[i][j] != padToken) ? 1.0f : 0.0f;
            }
        }
        
        return new Variable(NdArray.of(mask));
    }

    /**
     * 将标签中的填充或非法索引修正到合法范围，防止后续索引越界
     */
    private Variable sanitizeLabels(Variable labels, int padToken, int vocabSize) {
        NdArray labelsArray = labels.getValue();
        float[][] src = labelsArray.getMatrix();
        float[][] sanitized = new float[src.length][src[0].length];

        for (int i = 0; i < src.length; i++) {
            for (int j = 0; j < src[i].length; j++) {
                int idx = Math.round(src[i][j]);
                if (idx == padToken) {
                    idx = 0; // 任意合法类别，后续会被mask掉
                } else if (idx < 0) {
                    idx = 0;
                } else if (idx >= vocabSize) {
                    idx = vocabSize - 1;
                }
                sanitized[i][j] = idx;
            }
        }

        return new Variable(NdArray.of(sanitized));
    }
    
    /**
     * 计算标准的Softmax交叉熵损失（每个位置的损失）
     * 
     * 使用 logSoftmax + one-hot gather 的方式逐位置计算交叉熵，
     * 全程保持计算图连通，确保梯度能正确回传。
     * 
     * @param y       真实标签 (batch_size, seq_len)
     * @param predict 预测值 (batch_size, seq_len, vocab_size)
     * @return 每个位置的损失值，保持在计算图中
     */
    private Variable computeSoftmaxCrossEntropy(Variable y, Variable predict) {
        int[] predictShape = predict.getValue().getShape().getShapeDims();
        int batchSize = predictShape[0];
        int seqLen = predictShape[1];
        int vocabSize = predictShape[2];
        int totalTokens = batchSize * seqLen;
        
        // 1. reshape logits 为 (totalTokens, vocabSize)，保持计算图
        Variable logitsFlat = predict.reshape(Shape.of(totalTokens, vocabSize));
        
        // 2. 计算 logSoftmax（在计算图中，支持反向传播）
        Variable logProbs = logitsFlat.logSoftmax(-1);
        
        // 3. 构造 one-hot 矩阵来 gather 目标 token 的 log prob
        float[] labelsData = y.getValue().getArray();
        float[] oneHotData = new float[totalTokens * vocabSize];
        for (int i = 0; i < totalTokens; i++) {
            int labelIdx = Math.round(labelsData[i]);
            if (labelIdx >= 0 && labelIdx < vocabSize) {
                oneHotData[i * vocabSize + labelIdx] = 1.0f;
            }
        }
        
        // 4. logProbs * oneHot 只保留目标 token 位置的值
        Variable oneHotVar = new Variable(NdArray.of(oneHotData, Shape.of(totalTokens, vocabSize)));
        Variable selectedLogProbs = logProbs.mul(oneHotVar);
        
        // 5. 对 vocabSize 维度求和，得到每个 token 的 log prob -> (totalTokens, 1)
        Variable tokenLogProbs = selectedLogProbs.sumTo(Shape.of(totalTokens, 1));
        
        // 6. 取负值得到交叉熵损失，reshape 为 (batchSize, seqLen)
        Variable negTokenLogProbs = tokenLogProbs.mul(new Variable(NdArray.of(-1.0f)));
        return negTokenLogProbs.reshape(Shape.of(batchSize, seqLen));
    }
    
    /**
     * 应用掩码到损失值
     * 
     * @param loss 原始损失
     * @param mask 掩码
     * @return 掩码后的平均损失
     */
    private Variable applyMask(Variable loss, Variable mask) {
        // 将损失与掩码相乘，忽略填充位置的损失
        Variable maskedLoss = loss.mul(mask);
        
        // 计算有效位置的总损失
        Variable totalLoss = maskedLoss.sum();
        
        // 计算有效位置的数量
        Variable validCount = mask.sum();
        
        // 返回平均损失（总损失/有效位置数量）
        return totalLoss.div(validCount);
    }
    
    /**
     * 计算序列级别的掩码损失统计信息
     * 
     * @param y       真实标签
     * @param predict 预测值
     * @param padToken 填充标记
     * @return 损失统计信息
     */
    public LossStats computeLossStats(Variable y, Variable predict, int padToken) {
        Variable mask = createMask(y, padToken);
        Variable loss = computeSoftmaxCrossEntropy(y, predict);
        Variable maskedLoss = loss.mul(mask);
        
        float totalLoss = maskedLoss.sum().getValue().getNumber().floatValue();
        float validTokens = mask.sum().getValue().getNumber().floatValue();
        float avgLoss = validTokens > 0 ? totalLoss / validTokens : 0.0f;
        
        return new LossStats(totalLoss, avgLoss, (int) validTokens);
    }
    
    /**
     * 损失统计信息类
     */
    public static class LossStats {
        public final float totalLoss;      // 总损失
        public final float averageLoss;    // 平均损失
        public final int validTokens;      // 有效token数量
        
        public LossStats(float totalLoss, float averageLoss, int validTokens) {
            this.totalLoss = totalLoss;
            this.averageLoss = averageLoss;
            this.validTokens = validTokens;
        }
        
        @Override
        public String toString() {
            return String.format("LossStats{总损失=%.4f, 平均损失=%.4f, 有效tokens=%d}", 
                               totalLoss, averageLoss, validTokens);
        }
    }
    
    /**
     * 验证输入参数的有效性
     * 
     * @param y       真实标签
     * @param predict 预测值
     * @throws IllegalArgumentException 当输入参数无效时抛出
     */
    private void validateInputs(Variable y, Variable predict) {
        if (y == null || predict == null) {
            throw new IllegalArgumentException("输入变量不能为null");
        }
        
        Shape yShape = y.getValue().getShape();
        Shape predictShape = predict.getValue().getShape();
        
        if (predictShape.getDimNum() != 3) {
            throw new IllegalArgumentException("预测值必须是3维数组 (batch_size, seq_len, vocab_size)");
        }
        
        if (yShape.getDimNum() != 2) {
            throw new IllegalArgumentException("标签必须是2维数组 (batch_size, seq_len)");
        }
        
        if (yShape.getRow() != predictShape.getDimension(0) || 
            yShape.getColumn() != predictShape.getDimension(1)) {
            throw new IllegalArgumentException("标签和预测值的批次大小和序列长度必须匹配");
        }
    }
}