package io.leavesfly.tinyai.nnet.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.rnn.GruLayer;

/**
 * GRU块，包含一个GRU层和一个线性输出层
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * GruBlock是一个组合块，包含一个GRU层和一个线性输出层，
 * 用于构建基于GRU的序列模型。
 */
public class GruBlock extends Block {
    /**
     * GRU层，用于处理序列数据
     */
    private GruLayer gruLayer;
    
    /**
     * 线性输出层，用于将GRU的输出映射到目标维度
     */
    private LinearLayer linearLayer;

    /**
     * 构造函数，创建一个GRU块
     * 
     * @param name 块的名称
     * @param inputSize 输入特征维度
     * @param hiddenSize 隐藏状态维度
     * @param outputSize 输出维度
     */
    public GruBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name, Shape.of(-1, inputSize), Shape.of(-1, outputSize));

        gruLayer = new GruLayer("gru", Shape.of(-1, inputSize), Shape.of(-1, hiddenSize));
        addLayer(gruLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        addLayer(linearLayer);
    }

    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable state = gruLayer.layerForward(inputs);
        return linearLayer.layerForward(state);
    }

    /**
     * 重置GRU层的内部状态
     */
    public void resetState() {
        gruLayer.resetState();
    }
}