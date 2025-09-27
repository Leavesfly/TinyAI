package io.leavesfly.tinyai.func.base;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 加法函数
 * 
 * 实现两个变量的加法运算，支持广播操作。
 * 当两个输入变量的形状不同时，会自动进行广播以匹配形状。
 */
public class Add extends Function {

    private Shape x0Shape;
    private Shape x1Shape;

    /**
     * 前向传播计算加法
     * 
     * 执行两个NdArray的加法运算。如果两个输入的形状不同，
     * 则对第二个输入进行广播以匹配第一个输入的形状。
     * 
     * @param inputs 输入的NdArray数组，长度为2
     * @return 加法运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        x0Shape = inputs[0].getShape();
        x1Shape = inputs[1].getShape();
        
        if (!x1Shape.equals(x0Shape)) {
            return inputs[0].add(inputs[1].broadcastTo(x0Shape));
        } else {
            return inputs[0].add(inputs[1]);
        }
    }

    /**
     * 反向传播计算梯度
     * 
     * 计算加法运算的梯度。对于加法运算，梯度直接传递给两个输入变量。
     * 如果进行了广播操作，则需要对梯度进行相应的sumTo操作。
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray gx0 = yGrad;
        NdArray gx1 = x1Shape.equals(x0Shape) ? yGrad : yGrad.sumTo(x1Shape);
        return Arrays.asList(gx0, gx1);
    }

    /**
     * 获取所需输入参数个数
     * 
     * 加法运算需要两个输入参数。
     * 
     * @return 输入参数个数，固定为2
     */
    @Override
    public int requireInputNum() {
        return 2;
    }
}
