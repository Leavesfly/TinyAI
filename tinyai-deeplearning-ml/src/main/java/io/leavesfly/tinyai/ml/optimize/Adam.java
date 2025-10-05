package io.leavesfly.tinyai.ml.optimize;

import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam优化器
 * 
 * 实现了Adam优化算法，融合了Momentum和AdaGrad的优点。
 * Adam通过计算梯度的一阶矩估计和二阶矩估计来动态调整学习率。
 * 
 * 更新公式：
 * m = β1 * m + (1 - β1) * g
 * v = β2 * v + (1 - β2) * g^2
 * θ = θ - lr * m_hat / (sqrt(v_hat) + ε)
 * 
 * @author TinyDL
 * @version 1.0
 */
public class Adam extends Optimizer {
    private float learningRate = 0.001f;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float epsilon = 1e-3f; // 使用更大的epsilon值确保数值稳定性，避免除零异常

    private Map<Integer, NdArray> ms;
    private Map<Integer, NdArray> vs;
    private int t = 0;

    /**
     * 构造函数
     * @param target 目标模型
     * @param _learningRate 学习率
     * @param _beta1 一阶矩估计衰减率
     * @param _beta2 二阶矩估计衰减率
     * @param _epsilon 防止除零的小常数
     */
    public Adam(Model target, float _learningRate, float _beta1, float _beta2, float _epsilon) {
        super(target);
        learningRate = _learningRate;
        beta1 = _beta1;
        beta2 = _beta2;
        epsilon = _epsilon;
        ms = new HashMap<>();
        vs = new HashMap<>();
    }

    /**
     * 构造函数（使用默认参数）
     * @param target 目标模型
     */
    public Adam(Model target) {
        super(target);
        ms = new HashMap<>();
        vs = new HashMap<>();
    }

    /**
     * 更新所有参数
     */
    public void update() {
        t++;
        super.update();
    }

    @Override
    public void updateOne(Parameter parameter) {
        // 检查参数的梯度是否为null，如果为null则跳过更新
        NdArray grad = parameter.getGrad();
        if (grad == null) {
            return;
        }

        int key = parameter.hashCode();
        if (!ms.containsKey(key)) {
            ms.put(key, NdArray.zeros(parameter.getValue().getShape()));
            vs.put(key, NdArray.zeros(parameter.getValue().getShape()));
        }
        NdArray m = ms.get(key);
        NdArray v = vs.get(key);

        m = m.add(grad.sub(m).mulNum(1 - beta1));
        v = v.add(grad.mul(grad).sub(v).mulNum(1 - beta2));
        ms.put(key, m);
        vs.put(key, v);

        // 为了数值稳定性，使用更大的epsilon值
        NdArray denominator = v.pow(0.5f).add(NdArray.like(v.getShape(), epsilon));
        NdArray delat = m.mulNum(lr()).div(denominator);
        parameter.setValue(parameter.getValue().sub(delat));

    }

    /**
     * 计算调整后的学习率
     * @return 调整后的学习率
     */
    private float lr() {
        float fix1 = (float) (1. - Math.pow(beta1, t));
        float fix2 = (float) (1. - Math.pow(beta2, t));
        return (float) (learningRate * Math.sqrt(fix2) / fix1);
    }
}