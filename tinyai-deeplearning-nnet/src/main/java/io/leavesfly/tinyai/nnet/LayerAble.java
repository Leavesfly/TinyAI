package io.leavesfly.tinyai.nnet;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;
import java.util.Map;

/**
 * 神经网络层的抽象基类
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * LayerAble是所有神经网络层的抽象基类，继承自Function类。
 * 定义了神经网络层的基本接口和属性，包括参数管理、前向传播、梯度清零等功能。
 */
public abstract class LayerAble extends Function {


    /**
     * 层的名称
     */
    protected String name;

    /**
     * 层的参数集合，以参数名到Parameter对象的映射形式存储
     */
    protected Map<String, Parameter> params;


    /**
     * 输入数据的形状
     */
    protected Shape inputShape;

    /**
     * 输出数据的形状
     */
    protected Shape outputShape;

    /**
     * 标记层是否已经初始化
     */
    protected boolean alreadyInit = false;


    /**
     * 初始化层的参数
     */
    abstract public void init();

    /**
     * 层的前向传播，是func的前向传播一种应用
     * inputs 不包含内部的参数部分
     * <p>
     * <p>
     * 可以基于Variable inputs上操作，生成 Variable result
     * 否则计算图断了。
     *
     * 如果重写了layerForward 就不需要实现
     * backward和forward。
     *
     * @param inputs 输入变量数组
     * @return 前向传播结果变量
     */
    public Variable layerForward(Variable... inputs) {
        return call(inputs);
    }


    /**
     * 清除所有参数的梯度
     */
    public abstract void clearGrads();


    /**
     * 获取层的名称
     *
     * @return 层的名称
     */
    public String getName() {
        return name;
    }

    /**
     * 获取层的所有参数
     *
     * @return 参数映射表
     */
    public Map<String, Parameter> getParams() {
        return params;
    }

    /**
     * 添加参数到层中
     *
     * @param paramName 参数名称
     * @param value     参数值
     */
    public void addParam(String paramName, Parameter value) {
        getParams().put(name + "." + paramName, value);
    }

    /**
     * 根据参数名称获取参数
     *
     * @param paramName 参数名称
     * @return 对应的参数对象
     */
    public Parameter getParamBy(String paramName) {
        return getParams().get(name + "." + paramName);
    }

    /**
     * 获取输入数据的形状
     *
     * @return 输入形状
     */
    public Shape getInputShape() {
        return inputShape;
    }

    /**
     * 获取输出数据的形状
     *
     * @return 输出形状
     */
    public Shape getOutputShape() {
        return outputShape;
    }


    @Override
    public int requireInputNum() {
        return -1;
    }

}