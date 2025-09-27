package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;

/**
 * 序列到序列模型解码器抽象基类
 * <p>
 * 解码器负责接收编码器的输出状态，并根据该状态和目标序列的部分输入，
 * 逐步生成目标序列。解码过程可以是自回归的（传统解码）或并行的（Transformer）。
 * <p>
 * 主要职责：
 * 1. 状态初始化：接收编码器的输出状态并初始化内部状态
 * 2. 序列生成：根据上下文信息逐步生成目标序列
 * 3. 注意力机制：（可选）在生成过程中关注输入序列的不同部分
 * 4. 输出映射：将隐藏状态映射到目标词汇表空间
 * <p>
 * 典型的解码器架构包括：
 * - RNN-based解码器（LSTM、GRU等）
 * - 注意力解码器（Attention-based）
 * - Transformer解码器（基于自注意力和交叉注意力）
 * <p>
 * 使用示例：
 * ```java
 * Decoder decoder = new MySeq2SeqDecoder("decoder", inputShape, outputShape);
 * decoder.initState(encoderState);
 * Variable output = decoder.layerForward(targetSequence);
 * ```
 *
 * @author TinyDL
 * @version 0.01
 * @since 2025-01-01
 */
public abstract class Decoder extends Block {

    /**
     * 编码器传递的初始状态
     */
    protected NdArray encoderState;

    /**
     * 标记状态是否已初始化
     */
    protected boolean stateInitialized = false;

    /**
     * 构造解码器实例
     *
     * @param _name         解码器名称，用于标识和调试
     * @param _xInputShape  输入序列的形状，通常为 [batch_size, target_seq_length, input_dim]
     * @param _yOutputShape 输出序列的形状，通常为 [batch_size, target_seq_length, vocab_size]
     * @throws IllegalArgumentException 当输入参数无效时抛出
     */
    public Decoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);

        // 参数验证
        if (_name == null || _name.trim().isEmpty()) {
            throw new IllegalArgumentException("解码器名称不能为空");
        }
        if (_xInputShape == null || _yOutputShape == null) {
            throw new IllegalArgumentException("输入和输出形状不能为null");
        }
    }

    /**
     * 初始化解码器的内部状态
     * <p>
     * 该方法必须在执行layerForward之前调用，用于接收编码器的输出状态。
     * 不同类型的解码器可能需要不同的初始化策略。
     *
     * @param encoderOutput 编码器的输出状态，不能为null
     * @throws IllegalArgumentException 当encoderOutput为null时抛出
     * @throws IllegalStateException    当解码器尚未初始化时抛出
     */
    public abstract void initState(NdArray encoderOutput);

    /**
     * 检查解码器状态是否已初始化
     *
     * @return 如果状态已初始化则返回true，否则返回false
     */
    public boolean isStateInitialized() {
        return stateInitialized;
    }

    /**
     * 获取编码器状态
     *
     * @return 编码器传递的初始状态，如果未初始化则返回null
     */
    public NdArray getEncoderState() {
        return encoderState;
    }

    /**
     * 重置解码器的内部状态
     * <p>
     * 清除所有内部状态，为处理新序列做准备。
     * 子类应该重写此方法以改复位其特定的状态。
     */
    public void resetState() {
        this.encoderState = null;
        this.stateInitialized = false;
        super.resetState();
    }

    /**
     * 获取当前解码器的隐藏状态
     * <p>
     * 某些解码器（如RNN-based）可能需要提供当前的隐藏状态。
     * 默认实现返回null，子类可以根据需要重写。
     *
     * @return 当前隐藏状态，如果不适用则返回null
     */
    public NdArray getCurrentHiddenState() {
        return null;
    }

    /**
     * 设置解码器的隐藏状态
     * <p>
     * 对于支持状态设置的解码器，允许外部直接设置内部状态。
     * 默认实现为空，子类可以根据需要重写。
     *
     * @param hiddenState 要设置的隐藏状态
     */
    public void setCurrentHiddenState(NdArray hiddenState) {
        // 默认实现：无状态解码器不需要设置
    }

    /**
     * 检查解码器前向传播的前置条件
     * <p>
     * 在执行layerForward之前调用，检查是否满足执行条件。
     *
     * @throws IllegalStateException 当前置条件不满足时抛出
     */
    protected void validateForwardPreconditions() {
        if (!stateInitialized) {
            throw new IllegalStateException(
                    "解码器状态尚未初始化，请先调用initState()方法");
        }
    }

    /**
     * {@inheritDoc}
     * <p>
     * 解码器的前向传播实现应该在调用父类方法之前检查前置条件。
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        validateForwardPreconditions();
        return super.layerForward(inputs);
    }
}
