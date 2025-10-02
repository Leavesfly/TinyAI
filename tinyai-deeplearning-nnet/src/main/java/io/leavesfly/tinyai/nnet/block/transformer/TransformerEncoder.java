package io.leavesfly.tinyai.nnet.block.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.block.seq2seq.Encoder;
import io.leavesfly.tinyai.nnet.layer.transformer.PositionalEncoding;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerEncoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer编码器实现
 * <p>
 * 完整的Transformer编码器包含：
 * 1. 输入嵌入（由外部提供）
 * 2. 位置编码
 * 3. N层Transformer编码器层
 * 4. 可选的最终层归一化
 */
public class TransformerEncoder extends Encoder {

    private PositionalEncoding positionalEncoding;
    private List<TransformerEncoderLayer> encoderLayers;
    private int numLayers;
    private int dModel;
    private int numHeads;
    private int dFF;
    private double dropoutRate;
    private int maxSeqLength;

    /**
     * 构造Transformer编码器
     *
     * @param name         编码器名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param numLayers    编码器层数
     * @param dFF          前馈网络隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate  dropout比率
     */
    public TransformerEncoder(String name, int dModel, int numHeads, int numLayers,
                              int dFF, int maxSeqLength, double dropoutRate) {
        super(name, Shape.of(-1, maxSeqLength, dModel), Shape.of(-1, maxSeqLength, dModel));

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dFF = dFF;
        this.dropoutRate = dropoutRate;
        this.maxSeqLength = maxSeqLength;

        init();
    }

    /**
     * 使用默认参数的构造函数
     */
    public TransformerEncoder(String name, int dModel, int numHeads, int numLayers, int maxSeqLength) {
        this(name, dModel, numHeads, numLayers, dModel * 4, maxSeqLength, 0.1);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化位置编码
            positionalEncoding = new PositionalEncoding(name + "_pos_encoding", dModel, maxSeqLength, dropoutRate);
            addLayer(positionalEncoding);

            // 初始化编码器层
            encoderLayers = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                TransformerEncoderLayer layer = new TransformerEncoderLayer(
                        name + "_encoder_layer_" + i, dModel, numHeads, dFF, dropoutRate
                );
                encoderLayers.add(layer);
                addLayer(layer);
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];

        // 验证输入维度
        NdArray inputData = x.getValue();
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                    String.format("Input dimension mismatch. Expected %d, got %d",
                            dModel, inputData.getShape().getDimension(2))
            );
        }

        // 1. 添加位置编码
        Variable encoded = positionalEncoding.layerForward(x);

        // 2. 通过所有编码器层
        Variable output = encoded;
        for (TransformerEncoderLayer layer : encoderLayers) {
            output = layer.layerForward(output);
        }

        return output;
    }

    /**
     * 获取位置编码层
     */
    public PositionalEncoding getPositionalEncoding() {
        return positionalEncoding;
    }

    /**
     * 获取编码器层列表
     */
    public List<TransformerEncoderLayer> getEncoderLayers() {
        return encoderLayers;
    }

    /**
     * 获取指定索引的编码器层
     */
    public TransformerEncoderLayer getEncoderLayer(int index) {
        if (index < 0 || index >= encoderLayers.size()) {
            throw new IndexOutOfBoundsException("Encoder layer index out of bounds: " + index);
        }
        return encoderLayers.get(index);
    }

    /**
     * 获取编码器层数
     */
    public int getNumLayers() {
        return numLayers;
    }

    /**
     * 获取模型维度
     */
    public int getDModel() {
        return dModel;
    }

    /**
     * 获取注意力头数
     */
    public int getNumHeads() {
        return numHeads;
    }

    /**
     * 获取前馈网络隐藏维度
     */
    public int getDFF() {
        return dFF;
    }

    /**
     * 获取最大序列长度
     */
    public int getMaxSeqLength() {
        return maxSeqLength;
    }

    /**
     * 获取dropout比率
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
}