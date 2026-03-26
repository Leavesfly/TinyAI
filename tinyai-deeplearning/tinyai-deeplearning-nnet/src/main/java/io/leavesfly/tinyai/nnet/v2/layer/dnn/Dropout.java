package io.leavesfly.tinyai.nnet.v2.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;

import java.util.Random;

/**
 * V2版本的Dropout层
 * <p>
 * Dropout是一种正则化技术，在训练时随机将部分神经元输出置为0
 * <p>
 * 特性：
 * - 训练模式：应用dropout
 * - 推理模式：直接返回输入（不应用dropout）
 * - 使用inverted dropout：训练时缩放以保持期望值不变
 *
 * @author leavesfly
 * @version 2.0
 */
public class Dropout extends Module {

    private final float p;
    private final Random random;
    /** 缓存的 inverted dropout 缩放因子，避免每次 forward 都创建新对象 */
    private final Variable cachedScaleVar;

    /**
     * 构造函数
     *
     * @param name 层名称
     * @param p    dropout概率（0到1之间）
     */
    public Dropout(String name, float p) {
        super(name);
        if (p < 0 || p >= 1) {
            throw new IllegalArgumentException("Dropout probability must be in [0, 1), got: " + p);
        }
        this.p = p;
        this.random = new Random();

        // 预计算并缓存 scale 常量
        float scale = 1.0f / (1 - p);
        this.cachedScaleVar = new Variable(scale);
        this.cachedScaleVar.setRequireGrad(false);
    }

    /**
     * 默认构造函数（p=0.5）
     *
     * @param name 层名称
     */
    public Dropout(String name) {
        this(name, 0.5f);
    }

    /**
     * 默认构造函数
     */
    public Dropout() {
        this("dropout", 0.5f);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 推理模式：直接返回输入
        if (!_training) {
            return x;
        }

        // 训练模式：应用dropout
        if (p == 0) {
            return x;
        }

        // 生成dropout mask（使用Variable的形状属性）
        Variable mask = generateMask(x);

        // 应用mask并缩放（inverted dropout）
        Variable masked = x.mul(mask);
        return masked.mul(cachedScaleVar);
    }

    /**
     * 生成dropout mask
     * <p>
     * 改进后：直接接收Variable，利用Variable的shape属性
     *
     * @param input 输入Variable
     * @return mask Variable（0或1）
     */
    private Variable generateMask(Variable input) {
        // 使用Variable的形状属性，不需要getValue()
        int totalElements = input.getElementCount();
        float[] maskData = new float[totalElements];
        for (int i = 0; i < maskData.length; i++) {
            maskData[i] = random.nextFloat() > p ? 1.0f : 0.0f;
        }
        NdArray maskArray = NdArray.of(maskData, input.getShape());
        Variable maskVar = new Variable(maskArray);
        maskVar.setRequireGrad(false);  // mask不需要梯度
        return maskVar;
    }

    /**
     * 获取dropout概率
     *
     * @return dropout概率
     */
    public float getP() {
        return p;
    }

    @Override
    public String toString() {
        return "Dropout{name='" + name + "', p=" + p + ", training=" + _training + '}';
    }
}
