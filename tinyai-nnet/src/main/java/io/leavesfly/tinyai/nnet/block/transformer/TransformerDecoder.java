package io.leavesfly.tinyai.nnet.block.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.block.seq2seq.Decoder;
import io.leavesfly.tinyai.nnet.layer.transformer.PositionalEncoding;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerDecoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer解码器实现
 * <p>
 * 完整的Transformer解码器包含：
 * 1. 输入嵌入（由外部提供）
 * 2. 位置编码
 * 3. N层Transformer解码器层
 * 4. 可选的最终层归一化
 */
public class TransformerDecoder extends Decoder {

    private PositionalEncoding positionalEncoding;
    private List<TransformerDecoderLayer> decoderLayers;
    private int numLayers;
    private int dModel;
    private int numHeads;
    private int dFF;
    private double dropoutRate;
    private int maxSeqLength;
    private NdArray encoderOutput; // 存储编码器输出状态

    /**
     * 构造Transformer解码器
     *
     * @param name         解码器名称
     * @param dModel       模型维度
     * @param numHeads     注意力头数
     * @param numLayers    解码器层数
     * @param dFF          前馈网络隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate  dropout比率
     */
    public TransformerDecoder(String name, int dModel, int numHeads, int numLayers,
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
    public TransformerDecoder(String name, int dModel, int numHeads, int numLayers, int maxSeqLength) {
        this(name, dModel, numHeads, numLayers, dModel * 4, maxSeqLength, 0.1);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化位置编码
            positionalEncoding = new PositionalEncoding(name + "_pos_encoding", dModel, maxSeqLength, dropoutRate);
            addLayer(positionalEncoding);

            // 初始化解码器层
            decoderLayers = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                TransformerDecoderLayer layer = new TransformerDecoderLayer(
                        name + "_decoder_layer_" + i, dModel, numHeads, dFF, dropoutRate
                );
                decoderLayers.add(layer);
                addLayer(layer);
            }

            alreadyInit = true;
        }
    }

    @Override
    public void initState(NdArray init) {
        // 存储编码器输出作为解码器的记忆状态
        this.encoderOutput = init;
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        Variable memory = null;

        // 如果有两个输入，第二个是编码器输出
        if (inputs.length > 1) {
            memory = inputs[1];
        } else if (encoderOutput != null) {
            // 使用存储的编码器输出
            memory = new Variable(encoderOutput);
        } else {
            throw new IllegalStateException("Decoder requires encoder output as memory");
        }

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

        // 2. 通过所有解码器层
        Variable output = encoded;
        for (TransformerDecoderLayer layer : decoderLayers) {
            output = layer.layerForward(output, memory);
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
     * 获取解码器层列表
     */
    public List<TransformerDecoderLayer> getDecoderLayers() {
        return decoderLayers;
    }

    /**
     * 获取指定索引的解码器层
     */
    public TransformerDecoderLayer getDecoderLayer(int index) {
        if (index < 0 || index >= decoderLayers.size()) {
            throw new IndexOutOfBoundsException("Decoder layer index out of bounds: " + index);
        }
        return decoderLayers.get(index);
    }

    /**
     * 获取解码器层数
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

    /**
     * 获取存储的编码器输出
     */
    public NdArray getEncoderOutput() {
        return encoderOutput;
    }

    /**
     * 重置解码器状态
     */
    public void resetState() {
        this.encoderOutput = null;
    }
}