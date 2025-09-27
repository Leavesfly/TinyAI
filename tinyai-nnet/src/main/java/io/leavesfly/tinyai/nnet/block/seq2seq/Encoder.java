package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;

/**
 * 序列到序列模型编码器抽象基类
 * <p>
 * 编码器负责将输入序列编码为一个固定长度的上下文向量或状态表示，
 * 该表示包含了输入序列的所有关键信息，用于指导解码器生成目标序列。
 * <p>
 * 主要职责：
 * 1. 序列编码：将变长输入序列转换为固定维度的表示
 * 2. 信息压缩：提取输入序列的关键特征和语义信息
 * 3. 状态传递：为解码器提供初始化状态
 * <p>
 * 典型的编码器架构包括：
 * - RNN-based编码器（LSTM、GRU等）
 * - CNN-based编码器
 * - Transformer编码器（基于自注意力机制）
 * <p>
 * 使用示例：
 * ```java
 * Encoder encoder = new MySeq2SeqEncoder("encoder", inputShape, outputShape);
 * Variable encodedState = encoder.layerForward(inputSequence);
 * ```
 *
 * @author TinyDL
 * @version 0.01
 * @since 2025-01-01
 */
public abstract class Encoder extends Block {

    /**
     * 构造编码器实例
     *
     * @param _name         编码器名称，用于标识和调试
     * @param _xInputShape  输入序列的形状，通常为 [batch_size, seq_length, input_dim]
     * @param _yOutputShape 输出状态的形状，通常为 [batch_size, hidden_dim] 或 [batch_size, seq_length, hidden_dim]
     * @throws IllegalArgumentException 当输入参数无效时抛出
     */
    public Encoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);

        // 参数验证
        if (_name == null || _name.trim().isEmpty()) {
            throw new IllegalArgumentException("编码器名称不能为空");
        }
        if (_xInputShape == null || _yOutputShape == null) {
            throw new IllegalArgumentException("输入和输出形状不能为null");
        }
    }

    /**
     * 获取编码器的最终隐藏状态
     * <p>
     * 某些编码器（如RNN-based）需要提供最终的隐藏状态给解码器使用。
     * 默认实现返回null，子类可以根据需要重写此方法。
     *
     * @return 最终隐藏状态，如果不适用则返回null
     */
    public NdArray getFinalHiddenState() {
        return null;
    }

    /**
     * 重置编码器的内部状态
     * <p>
     * 对于有状态的编码器（如RNN），在处理新序列前应重置状态。
     * 默认实现为空，子类可以根据需要重写。
     */
    public void resetState() {
        // 默认实现：无状态编码器不需要重置
        super.resetState();
    }

    /**
     * 获取编码后的序列长度
     *
     * @return 编码后的序列长度，如果不适用则返回输入序列长度
     */
    public int getEncodedSequenceLength() {
        if (outputShape != null && outputShape.getDimNum() >= 2) {
            return outputShape.getDimension(1);
        }
        return inputShape != null && inputShape.getDimNum() >= 2 ?
                inputShape.getDimension(1) : -1;
    }
}
