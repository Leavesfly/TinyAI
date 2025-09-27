package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 前馈神经网络层（Feed Forward Network）实现
 * <p>
 * Transformer中的前馈网络是一个两层的全连接网络，通常包含：
 * 1. 第一个线性层：将输入维度扩展到更大的隐藏维度
 * 2. ReLU激活函数
 * 3. 第二个线性层：将隐藏维度压缩回原始维度
 * <p>
 * FFN(x) = max(0, xW1 + b1)W2 + b2
 */
public class FeedForward extends Layer {

    private LinearLayer firstLinear;
    private ReLuLayer activation;
    private LinearLayer secondLinear;
    private int dModel;
    private int dFF;

    /**
     * 构造前馈神经网络层
     *
     * @param name   层名称
     * @param dModel 模型维度（输入和输出维度）
     * @param dFF    隐藏层维度（通常是dModel的4倍）
     */
    public FeedForward(String name, int dModel, int dFF) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        this.dModel = dModel;
        this.dFF = dFF;
        init();
    }

    /**
     * 使用默认隐藏维度的构造函数（4倍dModel）
     */
    public FeedForward(String name, int dModel) {
        this(name, dModel, dModel * 4);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 第一个线性层：dModel -> dFF
            firstLinear = new LinearLayer(name + "_linear1", dModel, dFF, true);

            // ReLU激活函数
            activation = new ReLuLayer(name + "_relu", Shape.of(-1, -1, dFF));

            // 第二个线性层：dFF -> dModel
            secondLinear = new LinearLayer(name + "_linear2", dFF, dModel, true);

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];

        // 获取输入形状
        NdArray inputData = input.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);

        // 将三维输入重塑为二维以进行矩阵乘法
        NdArray inputReshaped = inputData.reshape(Shape.of(batchSize * seqLen, dModel));
        Variable reshapedInput = new Variable(inputReshaped);

        // 第一个线性变换
        Variable hidden = firstLinear.layerForward(reshapedInput);

        // ReLU激活
        Variable activated = activation.layerForward(hidden);

        // 第二个线性变换
        Variable output2D = secondLinear.layerForward(activated);

        // 重塑回三维
        NdArray output3D = output2D.getValue().reshape(Shape.of(batchSize, seqLen, dModel));

        return new Variable(output3D);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 前馈网络的反向传播需要依次通过各层进行
        // 这里提供简化版本
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 获取第一个线性层
     */
    public LinearLayer getFirstLinear() {
        return firstLinear;
    }

    /**
     * 获取激活函数层
     */
    public ReLuLayer getActivation() {
        return activation;
    }

    /**
     * 获取第二个线性层
     */
    public LinearLayer getSecondLinear() {
        return secondLinear;
    }
}