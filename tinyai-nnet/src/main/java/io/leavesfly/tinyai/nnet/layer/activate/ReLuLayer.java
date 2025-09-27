package io.leavesfly.tinyai.nnet.layer.activate;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.ReLu;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.List;

/**
 * ReLU激活函数层
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * ReLuLayer实现了ReLU（Rectified Linear Unit）激活函数。
 * ReLU函数定义为：f(x) = max(0, x)
 * 该激活函数能够有效缓解梯度消失问题，并且计算简单。
 */
public class ReLuLayer extends Layer {

    /**
     * 构造一个ReLU激活函数层（指定输入形状）
     * 
     * @param _name 层名称
     * @param _xInputShape 输入形状
     */
    public ReLuLayer(String _name, Shape _xInputShape) {
        super(_name, _xInputShape, _xInputShape);
    }

    /**
     * 构造一个ReLU激活函数层（不指定输入形状）
     * 
     * @param _name 层名称
     */
    public ReLuLayer(String _name) {
        super(_name, null, null);

    }


    /**
     * 初始化方法（空实现，ReLU层无参数需要初始化）
     */
    @Override
    public void init() {
    }

    /**
     * ReLU激活函数的前向传播方法
     * 
     * @param inputs 输入变量数组，通常只包含一个输入变量
     * @return 经过ReLU激活函数处理后的输出变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        return new ReLu().call(inputs[0]);
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
        return 0;
    }


}