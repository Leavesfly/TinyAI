package io.leavesfly.tinyai.nnet.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.rnn.SimpleRnnLayer;

/**
 * LSTM块，包含一个LSTM层和一个线性输出层
 * 注意：由于编译器问题，暂时使用SimpleRnnLayer作为占位符
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * LstmBlock是一个组合块，包含一个LSTM层和一个线性输出层，
 * 用于构建基于LSTM的序列模型。当前实现使用SimpleRnnLayer作为占位符。
 */
public class LstmBlock extends Block {
    /**
     * LSTM层，用于处理序列数据（当前为占位符）
     */
    private SimpleRnnLayer lstmLayer;  // 临时使用SimpleRnnLayer
    
    /**
     * 线性输出层，用于将LSTM的输出映射到目标维度
     */
    private LinearLayer linearLayer;

    /**
     * 构造函数，创建一个LSTM块
     * 
     * @param name 块的名称
     * @param inputSize 输入特征维度
     * @param hiddenSize 隐藏状态维度
     * @param outputSize 输出维度
     */
    public LstmBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, Shape.of(-1, inputSize), Shape.of(-1, outputSize));

        lstmLayer = new SimpleRnnLayer("lstm", Shape.of(-1, inputSize), Shape.of(-1, hiddenSize));
        addLayer(lstmLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        addLayer(linearLayer);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable state = lstmLayer.layerForward(inputs);
        return linearLayer.layerForward(state);
    }

    /**
     * 重置LSTM层的内部状态
     */
    public void resetState() {
        lstmLayer.resetState();
    }
}