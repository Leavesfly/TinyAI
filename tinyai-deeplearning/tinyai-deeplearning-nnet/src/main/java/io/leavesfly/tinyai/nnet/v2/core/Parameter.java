package io.leavesfly.tinyai.nnet.v2.core;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * V2版本的神经网络参数类
 * <p>
 * Parameter继承自Variable，表示神经网络中需要训练的参数。
 * 与V1版本相比，增加了requiresGrad控制和数据访问方法。
 *
 * @author leavesfly
 * @version 2.0
 */
public class Parameter extends Variable {


    /**
     * 构造函数，使用指定的NdArray值创建Parameter实例
     * 默认需要计算梯度
     *
     * @param value 参数的初始值
     */
    public Parameter(NdArray value) {
        this(value, true);
    }

    /**
     * 构造函数，指定是否需要计算梯度
     *
     * @param value        参数的初始值
     * @param requiresGrad 是否需要计算梯度
     */
    public Parameter(NdArray value, boolean requiresGrad) {
        super(value);
        this.requireGrad = requiresGrad;
    }

    /**
     * 获取参数数据（NdArray形式）
     *
     * @return 参数的NdArray数据
     */
    public NdArray data() {
        return getValue();
    }

    /**
     * 设置参数数据
     *
     * @param data 新的参数数据
     */
    public void setData(NdArray data) {
        setValue(data);
    }

    /**
     * 获取参数梯度
     *
     * @return 参数的梯度，如果没有梯度返回null
     */
    public NdArray grad() {
        return getGrad();
    }

    /**
     * 判断是否需要计算梯度
     *
     * @return true表示需要计算梯度
     */
    public boolean requiresGrad() {
        return requireGrad;
    }

    /**
     * 设置是否需要计算梯度
     *
     * @param requiresGrad 是否需要计算梯度
     */
    public void setRequiresGrad(boolean requiresGrad) {
        this.requireGrad = requiresGrad;
    }

    /**
     * 清除参数梯度
     */
    @Override
    public void clearGrad() {
        super.clearGrad();
    }

    /**
     * 纯数据深拷贝，返回一个与计算图完全无关的独立Parameter副本
     *
     * 拷贝内容：value（深拷贝）、requiresGrad
     * 不拷贝：grad、creator（计算图关系）
     *
     * @return 数据独立的新Parameter实例
     */
    @Override
    public Parameter deepCopy() {
        NdArray copiedValue = getValue().copy();
        return new Parameter(copiedValue, this.requireGrad);
    }

    @Override
    public String toString() {
        return "Parameter{" +
                "data=" + (getValue() != null ? getValue().getShape() : "null") +
                ", requiresGrad=" + requireGrad +
                ", hasGrad=" + (getGrad() != null) +
                '}';
    }
}
