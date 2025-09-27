package io.leavesfly.tinyai.func.base;



import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 取反函数
 * 
 * 实现变量的取反运算。
 */
public class Neg extends Function {
    
    /**
     * 前向传播计算取反
     * 
     * 执行NdArray的取反运算：-inputs[0]
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return 取反运算结果的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].neg();
    }

    /**
     * 反向传播计算梯度
     * 
     * 计算取反运算的梯度。
     * 对于 z = -x，有：∂z/∂x = -1
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return Collections.singletonList(yGrad.neg());
    }

    /**
     * 获取所需输入参数个数
     * 
     * 取反运算需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
