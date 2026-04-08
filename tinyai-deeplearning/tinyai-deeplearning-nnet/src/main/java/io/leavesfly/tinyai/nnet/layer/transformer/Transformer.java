package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的完整Transformer模型
 * <p>
 * Transformer是一种基于注意力机制的序列到序列模型架构，
 * 由Vaswani等人在"Attention Is All You Need"论文中提出。
 * <p>
 * 完整的Transformer包含：
 * - 编码器（Encoder）：处理源序列
 * - 解码器（Decoder）：生成目标序列
 * <p>
 * 结构：
 * src -> Encoder -> memory
 * tgt, memory -> Decoder -> output
 * <p>
 * 参考PyTorch的nn.Transformer实现
 *
 * @author leavesfly
 * @version 2.0
 */
public class Transformer extends Module {

    /**
     * 模型维度
     */
    private final int dModel;

    /**
     * 注意力头数
     */
    private final int numHeads;

    /**
     * 编码器层数
     */
    private final int numEncoderLayers;

    /**
     * 解码器层数
     */
    private final int numDecoderLayers;

    /**
     * 前馈网络隐藏层维度
     */
    private final int dFF;

    /**
     * Dropout比率
     */
    private final float dropout;

    /**
     * 编码器
     */
    private TransformerEncoder encoder;

    /**
     * 解码器
     */
    private TransformerDecoder decoder;

    /**
     * 完整构造函数
     *
     * @param name             模型名称
     * @param dModel           模型维度
     * @param numHeads         注意力头数
     * @param numEncoderLayers 编码器层数
     * @param numDecoderLayers 解码器层数
     * @param dFF              前馈网络隐藏层维度
     * @param dropout          dropout比率
     * @param preLayerNorm     是否使用Pre-LayerNorm
     */
    public Transformer(String name, int dModel, int numHeads,
                       int numEncoderLayers, int numDecoderLayers,
                       int dFF, float dropout, boolean preLayerNorm) {
        super(name);
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numEncoderLayers = numEncoderLayers;
        this.numDecoderLayers = numDecoderLayers;
        this.dFF = dFF;
        this.dropout = dropout;

        // 创建编码器和解码器
        this.encoder = new TransformerEncoder(
                "encoder", numEncoderLayers, dModel, numHeads, dFF, dropout, preLayerNorm);
        this.decoder = new TransformerDecoder(
                "decoder", numDecoderLayers, dModel, numHeads, dFF, dropout, preLayerNorm);

        // 注册子模块
        registerModule("encoder", encoder);
        registerModule("decoder", decoder);

        init();
    }

    /**
     * 构造函数（使用默认参数）
     *
     * @param name             模型名称
     * @param dModel           模型维度
     * @param numHeads         注意力头数
     * @param numEncoderLayers 编码器层数
     * @param numDecoderLayers 解码器层数
     */
    public Transformer(String name, int dModel, int numHeads,
                       int numEncoderLayers, int numDecoderLayers) {
        this(name, dModel, numHeads, numEncoderLayers, numDecoderLayers,
                dModel * 4, 0.1f, true);
    }

    /**
     * 构造函数（对称层数）
     *
     * @param name      模型名称
     * @param dModel    模型维度
     * @param numHeads  注意力头数
     * @param numLayers 编码器和解码器层数（相同）
     */
    public Transformer(String name, int dModel, int numHeads, int numLayers) {
        this(name, dModel, numHeads, numLayers, numLayers);
    }

    /**
     * 构造函数（使用自定义编码器和解码器）
     *
     * @param name    模型名称
     * @param encoder 自定义编码器
     * @param decoder 自定义解码器
     */
    public Transformer(String name, TransformerEncoder encoder, TransformerDecoder decoder) {
        super(name);
        this.encoder = encoder;
        this.decoder = decoder;
        this.dModel = encoder.getDModel();
        this.numHeads = 0;  // 未知
        this.numEncoderLayers = encoder.getNumLayers();
        this.numDecoderLayers = decoder.getNumLayers();
        this.dFF = 0;  // 未知
        this.dropout = 0;  // 未知

        // 注册子模块
        registerModule("encoder", encoder);
        registerModule("decoder", decoder);

        init();
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    /**
     * 前向传播
     * <p>
     * 输入参数：
     * - inputs[0]: src - 源序列 (batch, src_seq_len, d_model)
     * - inputs[1]: tgt - 目标序列 (batch, tgt_seq_len, d_model)
     * - inputs[2]: src_mask - 源序列掩码（可选）
     * - inputs[3]: tgt_mask - 目标序列掩码（可选，因果掩码）
     * - inputs[4]: memory_mask - 编码器输出掩码（可选）
     * - inputs[5]: src_key_padding_mask - 源序列padding掩码（可选）
     * - inputs[6]: tgt_key_padding_mask - 目标序列padding掩码（可选）
     * - inputs[7]: memory_key_padding_mask - 编码器输出padding掩码（可选）
     *
     * @param inputs 输入变量数组
     * @return 解码器输出 (batch, tgt_seq_len, d_model)
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable src = inputs[0];
        Variable tgt = inputs[1];
        Variable srcMask = inputs.length > 2 ? inputs[2] : null;
        Variable tgtMask = inputs.length > 3 ? inputs[3] : null;
        Variable memoryMask = inputs.length > 4 ? inputs[4] : null;
        Variable srcKeyPaddingMask = inputs.length > 5 ? inputs[5] : null;
        Variable tgtKeyPaddingMask = inputs.length > 6 ? inputs[6] : null;
        Variable memoryKeyPaddingMask = inputs.length > 7 ? inputs[7] : null;

        // 编码
        Variable memory = encoder.forward(src, srcMask, srcKeyPaddingMask);

        // 解码
        Variable output = decoder.forward(tgt, memory, tgtMask, memoryMask, 
                                          tgtKeyPaddingMask, memoryKeyPaddingMask);

        return output;
    }

    /**
     * 简化的前向传播
     *
     * @param src 源序列 (batch, src_seq_len, d_model)
     * @param tgt 目标序列 (batch, tgt_seq_len, d_model)
     * @return 解码器输出
     */
    public Variable forward(Variable src, Variable tgt) {
        return forward(new Variable[]{src, tgt});
    }

    /**
     * 带掩码的前向传播
     *
     * @param src     源序列
     * @param tgt     目标序列
     * @param tgtMask 目标序列掩码（因果掩码）
     * @return 解码器输出
     */
    public Variable forward(Variable src, Variable tgt, Variable tgtMask) {
        return forward(new Variable[]{src, tgt, null, tgtMask});
    }

    /**
     * 仅编码（用于编码器-解码器分离推理）
     *
     * @param src     源序列
     * @param srcMask 源序列掩码
     * @return 编码器输出
     */
    public Variable encode(Variable src, Variable srcMask) {
        return encoder.forward(src, srcMask);
    }

    /**
     * 仅编码（无掩码）
     *
     * @param src 源序列
     * @return 编码器输出
     */
    public Variable encode(Variable src) {
        return encoder.forward(src);
    }

    /**
     * 仅解码（用于编码器-解码器分离推理）
     *
     * @param tgt     目标序列
     * @param memory  编码器输出
     * @param tgtMask 目标序列掩码
     * @return 解码器输出
     */
    public Variable decode(Variable tgt, Variable memory, Variable tgtMask) {
        return decoder.forward(tgt, memory, tgtMask);
    }

    /**
     * 仅解码（无掩码）
     *
     * @param tgt    目标序列
     * @param memory 编码器输出
     * @return 解码器输出
     */
    public Variable decode(Variable tgt, Variable memory) {
        return decoder.forward(tgt, memory);
    }

    /**
     * 生成因果掩码（用于目标序列）
     *
     * @param tgtSeqLen 目标序列长度
     * @return 因果掩码
     */
    public static Variable generateSquareSubsequentMask(int tgtSeqLen) {
        return MultiHeadAttention.generateCausalMask(tgtSeqLen);
    }

    // Getters

    public int getDModel() {
        return dModel;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getNumEncoderLayers() {
        return numEncoderLayers;
    }

    public int getNumDecoderLayers() {
        return numDecoderLayers;
    }

    public int getDFF() {
        return dFF;
    }

    public float getDropout() {
        return dropout;
    }

    public TransformerEncoder getEncoder() {
        return encoder;
    }

    public TransformerDecoder getDecoder() {
        return decoder;
    }

    @Override
    public String toString() {
        return "Transformer{" +
                "name='" + name + '\'' +
                ", dModel=" + dModel +
                ", numHeads=" + numHeads +
                ", numEncoderLayers=" + numEncoderLayers +
                ", numDecoderLayers=" + numDecoderLayers +
                ", dFF=" + dFF +
                ", dropout=" + dropout +
                '}';
    }
}

