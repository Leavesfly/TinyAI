package io.leavesfly.tinyai.nnet.v2.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.container.ModuleList;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * V2版本的TransformerEncoder（编码器堆栈）
 * <p>
 * TransformerEncoder是多个TransformerEncoderLayer的堆叠。
 * 每一层接收上一层的输出作为输入，逐层进行编码。
 * <p>
 * 结构：
 * input -> Layer_1 -> Layer_2 -> ... -> Layer_N -> [FinalLayerNorm] -> output
 * <p>
 * 特性：
 * - 支持配置层数
 * - 支持可选的最终层归一化（Pre-LN架构需要）
 * - 支持传入自定义的EncoderLayer列表
 * - 支持源序列掩码（src_mask）
 *
 * @author leavesfly
 * @version 2.0
 */
public class TransformerEncoder extends Module {

    /**
     * 编码器层列表
     */
    private final ModuleList layers;

    /**
     * 最终的层归一化（可选，用于Pre-LN架构）
     */
    private LayerNorm finalNorm;

    /**
     * 模型维度
     */
    private final int dModel;

    /**
     * 层数
     */
    private final int numLayers;

    /**
     * 构造函数（使用自定义的层列表）
     *
     * @param name      编码器名称
     * @param layers    编码器层列表
     * @param finalNorm 最终层归一化（可选）
     */
    public TransformerEncoder(String name, ModuleList layers, LayerNorm finalNorm) {
        super(name);
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.numLayers = layers.size();

        // 尝试获取dModel
        if (numLayers > 0 && layers.get(0) instanceof TransformerEncoderLayer) {
            this.dModel = ((TransformerEncoderLayer) layers.get(0)).getDModel();
        } else {
            this.dModel = 0;
        }

        // 注册子模块
        registerModule("layers", layers);
        if (finalNorm != null) {
            registerModule("final_norm", finalNorm);
        }

        init();
    }

    /**
     * 构造函数（自动创建层）
     *
     * @param name        编码器名称
     * @param numLayers   层数
     * @param dModel      模型维度
     * @param numHeads    注意力头数
     * @param dFF         前馈网络隐藏层维度
     * @param dropout     dropout比率
     * @param preLayerNorm 是否使用Pre-LayerNorm
     */
    public TransformerEncoder(String name, int numLayers, int dModel, int numHeads,
                              int dFF, float dropout, boolean preLayerNorm) {
        super(name);
        this.dModel = dModel;
        this.numLayers = numLayers;

        // 创建层列表
        this.layers = new ModuleList("layers");
        for (int i = 0; i < numLayers; i++) {
            TransformerEncoderLayer layer = new TransformerEncoderLayer(
                    "layer_" + i, dModel, numHeads, dFF, dropout, preLayerNorm);
            layers.add(layer);
        }

        // Pre-LN架构需要最终的层归一化
        if (preLayerNorm) {
            this.finalNorm = new LayerNorm("final_norm", dModel);
            registerModule("final_norm", finalNorm);
        }

        // 注册子模块
        registerModule("layers", layers);

        init();
    }

    /**
     * 构造函数（使用默认参数）
     *
     * @param name      编码器名称
     * @param numLayers 层数
     * @param dModel    模型维度
     * @param numHeads  注意力头数
     */
    public TransformerEncoder(String name, int numLayers, int dModel, int numHeads) {
        this(name, numLayers, dModel, numHeads, dModel * 4, 0.1f, true);
    }

    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }

    /**
     * 前向传播
     * <p>
     * 输入参数：
     * - inputs[0]: src - 源序列 (batch, seq_len, d_model)
     * - inputs[1]: src_mask - 源序列掩码（可选）
     * - inputs[2]: src_key_padding_mask - 源序列padding掩码（可选）
     *
     * @param inputs 输入变量数组
     * @return 编码器输出 (batch, seq_len, d_model)
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable output = inputs[0];
        Variable srcMask = inputs.length > 1 ? inputs[1] : null;
        Variable srcKeyPaddingMask = inputs.length > 2 ? inputs[2] : null;

        // 逐层处理，将掩码传递给每个 EncoderLayer
        for (int i = 0; i < numLayers; i++) {
            Module layer = layers.get(i);
            if (layer instanceof TransformerEncoderLayer encoderLayer) {
                output = encoderLayer.forward(output, srcMask, srcKeyPaddingMask);
            } else {
                output = layer.forward(output);
            }
        }

        // 应用最终的层归一化
        if (finalNorm != null) {
            output = finalNorm.forward(output);
        }

        return output;
    }

    /**
     * 简化的前向传播（仅输入，无掩码）
     *
     * @param src 源序列 (batch, seq_len, d_model)
     * @return 编码器输出
     */
    public Variable forward(Variable src) {
        return forward(new Variable[]{src});
    }

    /**
     * 带掩码的前向传播
     *
     * @param src     源序列 (batch, seq_len, d_model)
     * @param srcMask 源序列掩码
     * @return 编码器输出
     */
    public Variable forward(Variable src, Variable srcMask) {
        return forward(src, srcMask, null);
    }

    /**
     * 带掩码的前向传播
     *
     * @param src               源序列 (batch, seq_len, d_model)
     * @param srcMask           源序列掩码
     * @param srcKeyPaddingMask 源序列padding掩码
     * @return 编码器输出
     */
    public Variable forward(Variable src, Variable srcMask, Variable srcKeyPaddingMask) {
        return forward(new Variable[]{src, srcMask, srcKeyPaddingMask});
    }

    /**
     * 获取模型维度
     *
     * @return 模型维度
     */
    public int getDModel() {
        return dModel;
    }

    /**
     * 获取层数
     *
     * @return 层数
     */
    public int getNumLayers() {
        return numLayers;
    }

    /**
     * 获取层列表
     *
     * @return 层列表
     */
    public ModuleList getLayers() {
        return layers;
    }

    /**
     * 获取指定层
     *
     * @param index 层索引
     * @return 指定的编码器层
     */
    public Module getLayer(int index) {
        return layers.get(index);
    }

    @Override
    public String toString() {
        return "TransformerEncoder{" +
                "name='" + name + '\'' +
                ", numLayers=" + numLayers +
                ", dModel=" + dModel +
                ", hasFinalNorm=" + (finalNorm != null) +
                '}';
    }
}

