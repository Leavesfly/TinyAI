package io.leavesfly.tinyai.nnet;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 神经网络中需要训练的参数，对应数学中的函数就是变量
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Parameter类继承自Variable类，用于表示神经网络中需要训练的参数。
 * 在前向传播和反向传播过程中，Parameter会参与计算并更新其值。
 */
public class Parameter extends Variable {

    /**
     * 构造函数，使用指定的NdArray值创建Parameter实例
     *
     * @param value 参数的初始值
     */
    public Parameter(NdArray value) {
        super(value);
    }
}