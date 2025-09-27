package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.SoftMax;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Layer;


import java.util.List;

/**
 * SoftMax激活函数层
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * SoftMaxLayer实现了SoftMax激活函数。
 * SoftMax函数定义为：f(x_i) = e^(x_i) / Σ(e^(x_j))
 * 该激活函数常用于多分类问题的输出层，能够将输入转换为概率分布。
 */
public class SoftMaxLayer extends Layer {

    /**
     * 构造一个SoftMax激活函数层
     * 
     * @param _name 层名称
     */
    public SoftMaxLayer(String _name) {
        super(_name, null, null);
    }

    /**
     * 初始化方法（空实现，SoftMax层无参数需要初始化）
     */
    @Override
    public void init() {

    }

    /**
     * SoftMax激活函数的前向传播方法
     * 
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 经过SoftMax激活函数处理后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        return new SoftMax().call(inputs[0]);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return new SoftMax().forward(inputs[0]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return new SoftMax().backward(yGrad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }


}