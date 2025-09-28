package io.leavesfly.tinyai.mlearning.loss;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 掩码Softmax交叉熵损失函数
 * <p>
 * 用于处理序列模型中的掩码交叉熵损失计算，特别适用于处理变长序列。
 * 在序列处理中，较短的序列会被填充到固定长度，掩码用于忽略填充部分的损失计算。
 *
 * @author TinyDL
 * @version 1.0
 * todo
 */
public class MaskedSoftmaxCELoss extends SoftmaxCrossEntropy {

}