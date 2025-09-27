package io.leavesfly.tinyai.func.math;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 裁剪函数
 * <p>
 * 将输入值限制在指定的最小值和最大值之间。
 */
public class Clip extends Function {
    private float min;
    private float max;

    /**
     * 构造函数
     *
     * @param min 最小值
     * @param max 最大值
     */
    public Clip(float min, float max) {
        this.min = min;
        this.max = max;
    }

    /**
     * 前向传播计算裁剪
     * <p>
     * 将输入值限制在[min, max]范围内。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 裁剪后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].clip(min, max);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于裁剪函数，梯度计算规则为：
     * - 当输入值在[min, max]范围内时，梯度为1
     * - 当输入值小于min或大于max时，梯度为0
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        // 直接创建一个新的数组来存储结果
        float[][] xMatrix = x.getMatrix();
        float[][] yGradMatrix = yGrad.getMatrix();
        float[][] result = new float[xMatrix.length][xMatrix[0].length];
        
        for (int i = 0; i < xMatrix.length; i++) {
            for (int j = 0; j < xMatrix[i].length; j++) {
                // 只有在范围内的值才保留梯度
                if (xMatrix[i][j] >= min && xMatrix[i][j] <= max) {
                    result[i][j] = yGradMatrix[i][j];
                } else {
                    result[i][j] = 0f;
                }
            }
        }
        
        return Collections.singletonList(NdArray.of(result));
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 裁剪函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
