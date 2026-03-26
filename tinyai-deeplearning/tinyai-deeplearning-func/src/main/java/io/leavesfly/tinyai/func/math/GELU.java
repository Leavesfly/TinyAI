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
     * √(2/π) 常量，避免反复计算
     */
    private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    /**
     * 缓存前向传播的 tanh 结果和 x²，避免反向传播时重复计算
     */
    private NdArray cachedTanhResult;
    private NdArray cachedX2;
    
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
        cachedX2 = x.mul(x); // x²，缓存供 backward 使用
        NdArray x3 = cachedX2.mul(x); // x^3
        NdArray inner = x.add(x3.mulNum(0.044715f)); // x + 0.044715 * x^3
        NdArray scaled = inner.mulNum(SQRT_2_OVER_PI); // √(2/π) * (...)
        cachedTanhResult = scaled.tanh(); // tanh(...)，缓存供 backward 使用
        NdArray onePlusTanh = cachedTanhResult.add(cachedTanhResult.like(1f)); // 1 + tanh(...)
        NdArray halfOnePlusTanh = onePlusTanh.mulNum(0.5f); // 0.5 * (1 + tanh(...))
        
        return x.mul(halfOnePlusTanh); // x * 0.5 * (1 + tanh(...))
    }

    /**
     * 反向传播计算梯度
     * 
     * 对于GELU函数，梯度公式为：
     * GELU'(x) = 0.5 * (1 + tanh(s)) + x * 0.5 * sech²(s) * √(2/π) * (1 + 3 * 0.044715 * x²)
     * 其中 s = √(2/π) * (x + 0.044715 * x³)
     * 复用前向传播缓存的 tanh 和 x² 结果。
     * 
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        
        // 复用缓存的 tanh 和 x² 结果
        // 计算sech²项（1 - tanh²）
        NdArray sech2 = cachedTanhResult.like(1f).sub(cachedTanhResult.mul(cachedTanhResult));
        
        // 计算导数的两个主要部分
        NdArray part1 = cachedTanhResult.like(1f).add(cachedTanhResult).mulNum(0.5f);
        NdArray innerDerivative = cachedX2.like(1f).add(cachedX2.mulNum(3 * 0.044715f));
        NdArray part2 = x.mul(sech2).mul(innerDerivative).mulNum(0.5f * SQRT_2_OVER_PI);
        
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