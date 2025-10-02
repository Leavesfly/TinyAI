package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Layer;


import java.util.List;

/**
 * Tanh激活函数层
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * TanhLayer实现了双曲正切（Tanh）激活函数。
 * Tanh函数定义为：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * 该激活函数将输入映射到(-1, 1)区间，输出以0为中心。
 */
public class TanhLayer extends Layer {
    
    /**
     * 构造一个Tanh激活函数层
     * 
     * @param _name 层名称
     */
    public TanhLayer(String _name) {
        super(_name, null, null);
    }

    /**
     * 初始化方法（空实现，Tanh层无参数需要初始化）
     */
    @Override
    public void init() {

    }

    /**
     * Tanh激活函数的前向传播方法
     * 
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 经过Tanh激活函数处理后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        return new Tanh().call(inputs[0]);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return new Tanh().forward(inputs[0]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return new Tanh().backward(yGrad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}