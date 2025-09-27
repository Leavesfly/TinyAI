package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 输出头实现
 * <p>
 * 将Transformer的输出转换为词汇表上的概率分布
 * 通常使用线性变换将隐藏状态映射到词汇表大小的logits
 */
public class GPT2OutputHead extends Layer {

    private Parameter outputWeight;  // 输出权重矩阵
    private boolean useBias;         // 是否使用偏置
    private Parameter outputBias;    // 输出偏置
    private int vocabSize;           // 词汇表大小
    private int dModel;              // 模型维度

    /**
     * 构造GPT-2输出头
     *
     * @param name      层名称
     * @param dModel    模型维度
     * @param vocabSize 词汇表大小
     * @param useBias   是否使用偏置
     */
    public GPT2OutputHead(String name, int dModel, int vocabSize, boolean useBias) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, vocabSize));
        this.dModel = dModel;
        this.vocabSize = vocabSize;
        this.useBias = useBias;
        init();
    }

    /**
     * 不使用偏置的构造函数
     */
    public GPT2OutputHead(String name, int dModel, int vocabSize) {
        this(name, dModel, vocabSize, false);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化输出权重矩阵 (dModel, vocabSize)
            // 使用较小的随机初始化
            outputWeight = new Parameter(NdArray.likeRandomN(Shape.of(dModel, vocabSize)).mulNum(0.02f));
            outputWeight.setName(name + "_weight");
            addParam(outputWeight.getName(), outputWeight);

            // 如果使用偏置，初始化偏置项
            if (useBias) {
                outputBias = new Parameter(NdArray.zeros(Shape.of(vocabSize)));
                outputBias.setName(name + "_bias");
                addParam(outputBias.getName(), outputBias);
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len, dModel)
        NdArray inputData = input.getValue();

        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);

        // 将输入重塑为二维矩阵进行矩阵乘法
        NdArray inputReshaped = reshapeTo2D(inputData, batchSize, seqLen);

        // 执行线性变换: input @ weight
        Variable output = matmul2D(new Variable(inputReshaped), outputWeight);

        // 添加偏置（如果有）
        if (useBias) {
            output = output.add(outputBias);
        }

        // 重塑回三维: (batch_size, seq_len, vocab_size)
        NdArray outputData = output.getValue();
        NdArray result = reshapeFrom2D(outputData, batchSize, seqLen, vocabSize);

        return new Variable(result);
    }

    /**
     * 将三维张量重塑为二维矩阵
     */
    private NdArray reshapeTo2D(NdArray input, int batchSize, int seqLen) {
        // (batch_size, seq_len, dModel) -> (batch_size * seq_len, dModel)
        return input.reshape(Shape.of(batchSize * seqLen, dModel));
    }

    /**
     * 将二维矩阵重塑回三维张量
     */
    private NdArray reshapeFrom2D(NdArray input, int batchSize, int seqLen, int outputDim) {
        // (batch_size * seq_len, outputDim) -> (batch_size, seq_len, outputDim)
        return input.reshape(Shape.of(batchSize, seqLen, outputDim));
    }

    /**
     * 执行二维矩阵乘法
     */
    private Variable matmul2D(Variable a, Variable b) {
        return a.matMul(b);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 获取输出权重参数
     */
    public Parameter getOutputWeight() {
        return outputWeight;
    }

    /**
     * 获取输出偏置参数
     */
    public Parameter getOutputBias() {
        return outputBias;
    }

    /**
     * 是否使用偏置
     */
    public boolean isUseBias() {
        return useBias;
    }

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取模型维度
     */
    public int getDModel() {
        return dModel;
    }
}