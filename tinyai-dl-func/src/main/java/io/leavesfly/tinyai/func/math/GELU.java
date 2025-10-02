package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * GELU激活函数（Gaussian Error Linear Unit）
 * 
 * GELU激活函数，常用于深度学习中，特别是Transformer模型。
 * GELU(x) = x * Φ(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
 * 
 * @author leavesfly
 * @version 1.0
 */
public class GELU extends Function {
    
    /**
     * 前向传播计算GELU
     * 
     * 计算GELU函数值：x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
     * 
     * @param inputs 输入的NdArray数组，长度为1
     * @return GELU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        
        // GELU近似公式: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        NdArray x3 = x.mul(x).mul(x); // x^3
        NdArray inner = x.add(x3.mulNum(0.044715f)); // x + 0.044715 * x^3
        NdArray scaled = inner.mulNum((float) Math.sqrt(2.0 / Math.PI)); // √(2/π) * (...)
        NdArray tanhResult = scaled.tanh(); // tanh(...)
        NdArray onePlusTanh = tanhResult.add(NdArray.ones(tanhResult.getShape())); // 1 + tanh(...)
        NdArray halfOnePlusTanh = onePlusTanh.mulNum(0.5f); // 0.5 * (1 + tanh(...))
        
        return x.mul(halfOnePlusTanh); // x * 0.5 * (1 + tanh(...))
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于GELU函数，梯度计算使用数值近似。
     * GELU'(x) ≈ 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3))) + 
     *              x * 0.5 * sech²(√(2/π) * (x + 0.044715 * x^3)) * √(2/π) * (1 + 3 * 0.044715 * x^2)
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        
        // 简化的梯度计算：使用GELU函数的数值近似导数
        // 这里使用一个更简单的近似公式
        NdArray x2 = x.mul(x); // x^2
        NdArray x3 = x2.mul(x); // x^3
        
        // 计算tanh项
        NdArray inner = x.add(x3.mulNum(0.044715f));
        NdArray scaled = inner.mulNum((float) Math.sqrt(2.0 / Math.PI));
        NdArray tanhResult = scaled.tanh();
        
        // 计算sech²项（1 - tanh²）
        NdArray sech2 = NdArray.ones(tanhResult.getShape()).sub(tanhResult.mul(tanhResult));
        
        // 计算导数的两个主要部分
        NdArray part1 = NdArray.ones(x.getShape()).add(tanhResult).mulNum(0.5f);
        NdArray innerDerivative = NdArray.ones(x.getShape()).add(x2.mulNum(3 * 0.044715f));
        NdArray part2 = x.mul(sech2).mul(innerDerivative).mulNum((float) (0.5 * Math.sqrt(2.0 / Math.PI)));
        
        NdArray grad = part1.add(part2);
        
        return Collections.singletonList(yGrad.mul(grad));
    }

    /**
     * 获取所需输入参数个数
     * 
     * GELU函数需要一个输入参数。
     * 
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}