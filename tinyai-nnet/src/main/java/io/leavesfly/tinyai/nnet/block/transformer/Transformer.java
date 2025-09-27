package io.leavesfly.tinyai.nnet.block.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.seq2seq.Decoder;
import io.leavesfly.tinyai.nnet.block.seq2seq.Encoder;


/**
 * Transformer模型完整实现
 * <p>
 * 这是完整的Transformer架构实现，包含编码器-解码器结构。
 * 基于论文 "Attention Is All You Need" (Vaswani et al., 2017)
 * <p>
 * 主要特性：
 * 1. 完整的编码器-解码器架构
 * 2. 多头自注意力机制
 * 3. 位置编码
 * 4. 残差连接和层归一化
 * 5. 前馈神经网络
 * <p>
 * 使用方法：
 * ```java
 * // 创建编码器和解码器
 * TransformerEncoder encoder = new TransformerEncoder("encoder", 512, 8, 6, 2048, 5000, 0.1);
 * TransformerDecoder decoder = new TransformerDecoder("decoder", 512, 8, 6, 2048, 5000, 0.1);
 * <p>
 * // 创建完整的Transformer模型
 * Transformer transformer = new Transformer("transformer", encoder, decoder);
 * <p>
 * // 前向传播
 * Variable output = transformer.layerForward(encoderInput, decoderInput);
 * ```
 * <p>
 * 参考链接：
 * - https://aistudio.baidu.com/aistudio/projectdetail/3034732?channelType=0&channel=0
 * - https://arxiv.org/abs/1706.03762
 */
public class Transformer extends Block {
    private Encoder encoder;
    private Decoder decoder;


    public Transformer(String _name, Encoder encoder, Decoder decoder) {
        super(_name, encoder.getInputShape(), decoder.getOutputShape());

        this.encoder = encoder;
        this.decoder = decoder;
        addLayer(encoder);
        addLayer(decoder);

    }

    /**
     * 便捷构造函数：创建标准的Transformer模型
     *
     * @param name         模型名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param numLayers    编码器和解码器层数
     * @param dFF          前馈网络隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate  dropout比率
     */
    public Transformer(String name, int dModel, int numHeads, int numLayers,
                       int dFF, int maxSeqLength, double dropoutRate) {
        this(name,
                new TransformerEncoder(name + "_encoder", dModel, numHeads, numLayers, dFF, maxSeqLength, dropoutRate),
                new TransformerDecoder(name + "_decoder", dModel, numHeads, numLayers, dFF, maxSeqLength, dropoutRate));
    }

    /**
     * 使用默认参数的便捷构造函数
     *
     * @param name         模型名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param numLayers    层数
     * @param maxSeqLength 最大序列长度
     */
    public Transformer(String name, int dModel, int numHeads, int numLayers, int maxSeqLength) {
        this(name, dModel, numHeads, numLayers, dModel * 4, maxSeqLength, 0.1);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("Transformer requires at least 2 inputs: encoder input and decoder input");
        }

        Variable encoderInput = inputs[0];
        Variable decoderInput = inputs[1];

        // 验证输入维度是否匹配
        NdArray encInputData = encoderInput.getValue();
        NdArray decInputData = decoderInput.getValue();

        if (encInputData.getShape().getDimension(2) != decInputData.getShape().getDimension(2)) {
            throw new IllegalArgumentException("Encoder and decoder inputs must have the same feature dimension");
        }

        // 编码器前向传播
        Variable encoderOutput = encoder.layerForward(encoderInput);

        // 初始化解码器状态
        decoder.initState(encoderOutput.getValue());

        // 解码器前向传播
        return decoder.layerForward(decoderInput, encoderOutput);
    }

    /**
     * 获取编码器
     *
     * @return 编码器实例
     */
    public Encoder getEncoder() {
        return encoder;
    }

    /**
     * 获取解码器
     *
     * @return 解码器实例
     */
    public Decoder getDecoder() {
        return decoder;
    }

    /**
     * 仅编码（不需要解码器输入）
     *
     * @param encoderInput 编码器输入
     * @return 编码器输出
     */
    public Variable encodeOnly(Variable encoderInput) {
        return encoder.layerForward(encoderInput);
    }

    /**
     * 仅解码（需要预先设置编码器输出）
     *
     * @param decoderInput  解码器输入
     * @param encoderOutput 编码器输出
     * @return 解码器输出
     */
    public Variable decodeOnly(Variable decoderInput, Variable encoderOutput) {
        decoder.initState(encoderOutput.getValue());
        return decoder.layerForward(decoderInput, encoderOutput);
    }

    /**
     * 重置模型状态
     */
    public void resetModelState() {
        super.resetState();
        if (decoder instanceof TransformerDecoder) {
            ((TransformerDecoder) decoder).resetState();
        }
    }
}
