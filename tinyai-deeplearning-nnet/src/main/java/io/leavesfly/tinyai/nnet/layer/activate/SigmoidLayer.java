package io.leavesfly.tinyai.nnet.layer.activate;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Sigmoid;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.List;
import java.util.Collections;

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
     * Sigmoid函数实例，用于复用
     */
    private Sigmoid sigmoidFunc;
    
    /**
     * 最后一次前向传播的输入，用于反向传播
     */
    private NdArray lastInput;
    
    /**
     * 构造一个Sigmoid激活函数层
     * 
     * @param _name 层名称
     */
    public SigmoidLayer(String _name) {
        super(_name, null, null);
        this.sigmoidFunc = new Sigmoid();
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
        this.lastInput = inputs[0].getValue();
        return sigmoidFunc.call(inputs[0]);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        this.lastInput = inputs[0];
        return sigmoidFunc.forward(inputs);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 手动计算Sigmoid的反向传播：梯度 = yGrad * sigmoid(x) * (1 - sigmoid(x))
        if (lastInput != null) {
            NdArray sigmoidX = lastInput.sigmoid();
            NdArray grad = yGrad.mul(sigmoidX).mul(NdArray.ones(sigmoidX.getShape()).sub(sigmoidX));
            return Collections.singletonList(grad);
        } else {
            throw new RuntimeException("SigmoidLayer backward called without forward pass");
        }
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}