package io.leavesfly.tinyai.nnet;


import io.leavesfly.tinyai.ndarr.Shape;

import java.util.HashMap;

/**
 * 表示神经网络中具体的层，对应数学中的一个函数
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Layer是神经网络的基本组成单元，每个Layer实现特定的数学变换功能。
 * Layer继承自LayerAble抽象类，提供了参数管理和梯度清零等基础功能。
 */
public abstract class Layer extends LayerAble {

    /**
     * 构造函数，初始化Layer的基本属性
     * 
     * @param _name Layer的名称
     * @param _inputShape 输入数据的形状
     */
    public Layer(String _name, Shape _inputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
    }

    /**
     * 构造函数，初始化Layer的基本属性（包含输出形状）
     * 
     * @param _name Layer的名称
     * @param _inputShape 输入数据的形状
     * @param _outputShape 输出数据的形状
     */
    public Layer(String _name, Shape _inputShape, Shape _outputShape) {
        name = _name;
        this.params = new HashMap<>();
        inputShape = _inputShape;
        outputShape = _outputShape;
    }

    @Override
    public void clearGrads() {
        for (Parameter parameter : params.values()) {
            parameter.clearGrad();
        }
    }

}