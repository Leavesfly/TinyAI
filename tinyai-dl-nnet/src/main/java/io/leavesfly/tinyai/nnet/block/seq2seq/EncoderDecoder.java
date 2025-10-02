package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;

/**
 * 序列到序列模型的编码解码器组合类
 * <p>
 * 这个类将编码器和解码器组合成一个完整的序列到序列模型。
 * 它协调编码器和解码器的工作流程，实现完整的序列转换功能。
 * <p>
 * 工作流程：
 * 1. 编码阶段：使用编码器将输入序列编码为上下文状态
 * 2. 状态传递：将编码器的输出状态传递给解码器
 * 3. 解码阶段：使用解码器根据上下文状态生成目标序列
 * <p>
 * 适用场景：
 * - 智能翻译（机器翻译）
 * - 文本摘要
 * - 对话系统
 * - 问答系统
 * - 代码生成
 * <p>
 * 使用示例：
 * ```java
 * // 创建编码器和解码器
 * Encoder encoder = new Seq2SeqEncoder("encoder", encoderInputShape, encoderOutputShape);
 * Decoder decoder = new Seq2SeqDecoder("decoder", decoderInputShape, decoderOutputShape);
 * <p>
 * // 创建完整模型
 * EncoderDecoder model = new EncoderDecoder("seq2seq_model", encoder, decoder);
 * <p>
 * // 执行前向传播
 * Variable output = model.layerForward(sourceSequence, targetSequence);
 * ```
 *
 * @author TinyDL
 * @version 0.01
 * @since 2025-01-01
 */
public class EncoderDecoder extends Block {

    /**
     * 编码器实例
     */
    private final Encoder encoder;

    /**
     * 解码器实例
     */
    private final Decoder decoder;

    /**
     * 标记模型是否已初始化
     */
    private boolean modelInitialized = false;

    /**
     * 构造编码解码器组合模型
     *
     * @param _name   模型名称，用于标识和调试
     * @param encoder 编码器实例，不能为null
     * @param decoder 解码器实例，不能为null
     * @throws IllegalArgumentException 当输入参数无效时抛出
     */
    public EncoderDecoder(String _name, Encoder encoder, Decoder decoder) {
        super(_name, validateAndGetInputShape(encoder), validateAndGetOutputShape(decoder));

        // 参数验证
        if (_name == null || _name.trim().isEmpty()) {
            throw new IllegalArgumentException("模型名称不能为空");
        }
        if (encoder == null) {
            throw new IllegalArgumentException("编码器不能为null");
        }
        if (decoder == null) {
            throw new IllegalArgumentException("解码器不能为null");
        }

        this.encoder = encoder;
        this.decoder = decoder;

        // 将编码器和解码器添加为子层
        addLayer(encoder);
        addLayer(decoder);
    }

    /**
     * 验证并获取编码器的输入形状
     */
    private static Shape validateAndGetInputShape(Encoder encoder) {
        if (encoder == null) {
            throw new IllegalArgumentException("编码器不能为null");
        }
        if (encoder.getInputShape() == null) {
            throw new IllegalArgumentException("编码器的输入形状不能为null");
        }
        return encoder.getInputShape();
    }

    /**
     * 验证并获取解码器的输出形状
     */
    private static Shape validateAndGetOutputShape(Decoder decoder) {
        if (decoder == null) {
            throw new IllegalArgumentException("解码器不能为null");
        }
        if (decoder.getOutputShape() == null) {
            throw new IllegalArgumentException("解码器的输出形状不能为null");
        }
        return decoder.getOutputShape();
    }

    /**
     * 初始化模型
     * <p>
     * 初始化编码器和解码器的参数。该方法在模型第一次使用前会自动调用。
     */
    @Override
    public void init() {
        if (!modelInitialized) {
            try {
                // 初始化编码器和解码器
                encoder.init();
                decoder.init();

                modelInitialized = true;

                System.out.println(String.format(
                        "EncoderDecoder '%s' 初始化成功 - 编码器: %s, 解码器: %s",
                        name, encoder.getName(), decoder.getName()
                ));
            } catch (Exception e) {
                throw new RuntimeException(String.format(
                        "EncoderDecoder '%s' 初始化失败: %s", name, e.getMessage()), e);
            }
        }
    }

    /**
     * 执行前向传播
     * <p>
     * 该方法实现了完整的序列到序列转换流程。
     *
     * @param inputs 输入参数，期望包含两个参数：
     *               inputs[0] - 编码器输入序列（源序列）
     *               inputs[1] - 解码器输入序列（目标序列的前缀）
     * @return 解码器输出的目标序列
     * @throws IllegalArgumentException 当输入参数不正确时抛出
     * @throws IllegalStateException    当模型尚未初始化时抛出
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        // 输入验证
        validateInputs(inputs);

        // 确保模型已初始化
        if (!modelInitialized) {
            init();
        }

        try {
            // 提取输入
            Variable encoderInput = inputs[0];
            Variable decoderInput = inputs[1];

            // 第一步：编码阶段
            Variable encoderState = encoder.layerForward(encoderInput);

            // 第二步：初始化解码器状态
            decoder.initState(encoderState.getValue());

            // 第三步：解码阶段
            Variable decoderOutput = decoder.layerForward(decoderInput, encoderState);

            return decoderOutput;

        } catch (Exception e) {
            throw new RuntimeException(String.format(
                    "EncoderDecoder '%s' 前向传播失败: %s", name, e.getMessage()), e);
        }
    }

    /**
     * 验证输入参数
     */
    private void validateInputs(Variable... inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("输入参数不能为null");
        }
        if (inputs.length < 2) {
            throw new IllegalArgumentException(String.format(
                    "EncoderDecoder需要至少2个输入参数（编码器输入、解码器输入），但实际提供了%d个",
                    inputs.length));
        }
        if (inputs[0] == null) {
            throw new IllegalArgumentException("编码器输入不能为null");
        }
        if (inputs[1] == null) {
            throw new IllegalArgumentException("解码器输入不能为null");
        }
    }

    /**
     * 获取编码器实例
     *
     * @return 编码器实例
     */
    public Encoder getEncoder() {
        return encoder;
    }

    /**
     * 获取解码器实例
     *
     * @return 解码器实例
     */
    public Decoder getDecoder() {
        return decoder;
    }

    /**
     * 检查模型是否已初始化
     *
     * @return 如果模型已初始化则返回true，否则返回false
     */
    public boolean isInitialized() {
        return modelInitialized;
    }

    /**
     * 重置模型状态
     * <p>
     * 清除所有内部状态，为处理新序列做准备。
     */
    @Override
    public void resetState() {
        super.resetState();
        encoder.resetState();
        decoder.resetState();
    }

    /**
     * 获取模型的详细信息
     *
     * @return 模型信息字符串
     */
    @Override
    public String toString() {
        return String.format(
                "EncoderDecoder(name='%s', encoder=%s, decoder=%s, initialized=%s)",
                name, encoder.getName(), decoder.getName(), modelInitialized
        );
    }
}
