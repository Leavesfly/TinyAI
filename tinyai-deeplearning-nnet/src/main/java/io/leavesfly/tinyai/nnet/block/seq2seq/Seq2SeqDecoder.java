package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.rnn.LstmLayer;

/**
 * 基于LSTM的序列到序列解码器实现
 * <p>
 * 这个解码器采用了经典的RNN-based架构，由以下组件按顺序组成：
 * 1. Embedding层：将离散的词索引转换为稠密的词向量表示
 * 2. LSTM层：对目标序列进行建模，结合编码器的上下文信息
 * 3. Linear层：将LSTM的隐藏状态映射到目标词汇表空间
 */
public class Seq2SeqDecoder extends Decoder {

    private Embedding embedding;
    private LstmLayer lstmLayer;
    private LinearLayer linearLayer;

    // 模型参数
    private final int targetVocabSize;
    private final int embeddingDim;
    private final int hiddenSize;
    private final int outputVocabSize;
    private boolean layersInitialized = false;

    /**
     * 构造序列到序列解码器
     */
    public Seq2SeqDecoder(String _name, Shape _xInputShape, Shape _yOutputShape, int targetVocabSize, int embeddingDim, int hiddenSize, int outputVocabSize) {

        super(_name, _xInputShape, _yOutputShape);
        validateParameters(targetVocabSize, embeddingDim, hiddenSize, outputVocabSize);

        this.targetVocabSize = targetVocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenSize = hiddenSize;
        this.outputVocabSize = outputVocabSize;
    }

    @Deprecated
    public Seq2SeqDecoder(String _name, Shape _xInputShape, Shape _yOutputShape) {

        super(_name, _xInputShape, _yOutputShape);
        this.targetVocabSize = 10000;
        this.embeddingDim = 128;
        this.hiddenSize = 256;
        this.outputVocabSize = 10000;
    }

    private void validateParameters(int targetVocabSize, int embeddingDim, int hiddenSize, int outputVocabSize) {

        if (targetVocabSize <= 0 || embeddingDim <= 0 || hiddenSize <= 0 || outputVocabSize <= 0) {
            throw new IllegalArgumentException("所有参数必须大于0");
        }
    }

    @Override
    public void init() {
        if (!layersInitialized) {
            this.embedding = new Embedding(name + "_embedding", targetVocabSize, embeddingDim);
            this.lstmLayer = new LstmLayer(name + "_lstm", Shape.of(-1, embeddingDim), Shape.of(-1, hiddenSize));
            this.linearLayer = new LinearLayer(name + "_linear", hiddenSize, outputVocabSize, true);

            addLayer(embedding);
            addLayer(lstmLayer);
            addLayer(linearLayer);
            layersInitialized = true;
        }
    }

    @Override
    public void initState(NdArray encoderOutput) {
        if (encoderOutput == null) {
            throw new IllegalArgumentException("编码器输出状态不能为null");
        }
        if (!layersInitialized) init();

        this.encoderState = encoderOutput;
        this.stateInitialized = true;
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        validateForwardInputs(inputs);
        validateForwardPreconditions();

        if (!layersInitialized) init();

        Variable input = inputs[0];
        Variable y = embedding.layerForward(input);
        y = lstmLayer.layerForward(y);
        y = linearLayer.layerForward(y);
        return y;
    }

    private void validateForwardInputs(Variable... inputs) {
        if (inputs == null || inputs.length == 0 || inputs[0] == null) {
            throw new IllegalArgumentException("解码器需要有效的输入参数");
        }
    }

    // Getter方法
    public int getTargetVocabSize() {
        return targetVocabSize;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getOutputVocabSize() {
        return outputVocabSize;
    }

    public Embedding getEmbedding() {
        return embedding;
    }

    public LstmLayer getLstmLayer() {
        return lstmLayer;
    }

    public LinearLayer getLinearLayer() {
        return linearLayer;
    }

    public boolean isLayersInitialized() {
        return layersInitialized;
    }
}
