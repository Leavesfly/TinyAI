package io.leavesfly.tinyai.nnet.layer.activate;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Sigmoid;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.List;

/**
 * Sigmoid激活函数层
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * SigmoidLayer实现了Sigmoid激活函数。
 * Sigmoid函数定义为：f(x) = 1 / (1 + e^(-x))
 * 该激活函数将输入映射到(0, 1)区间，常用于二分类问题的输出层。
 */
public class SigmoidLayer extends Layer {
    
    /**
     * 构造一个Sigmoid激活函数层
     * 
     * @param _name 层名称
     */
    public SigmoidLayer(String _name) {
        super(_name, null, null);
    }

    /**
     * 初始化方法（空实现，Sigmoid层无参数需要初始化）
     */
    @Override
    public void init() {

    }

    /**
     * Sigmoid激活函数的前向传播方法
     * 
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 经过Sigmoid激活函数处理后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        return new Sigmoid().call(inputs[0]);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}