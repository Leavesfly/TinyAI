package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 层归一化（Layer Normalization）实现
 * <p>
 * 层归一化对每个样本的特征维度进行归一化，公式如下：
 * LayerNorm(x) = γ * (x - μ) / σ + β
 * <p>
 * 其中：
 * - μ 是均值
 * - σ 是标准差
 * - γ 是学习的缩放参数
 * - β 是学习的偏移参数
 * <p>
 * todo
 */
public class LayerNorm extends Layer {


    public LayerNorm(String _name, Shape _inputShape) {
        super(_name, _inputShape);
    }

    public LayerNorm(String _name, Shape _inputShape, Shape _outputShape) {
        super(_name, _inputShape, _outputShape);
    }

    /**
     * 构造层归一化层
     *
     * @param name            层名称
     * @param normalizedShape 归一化的特征维度大小
     * @param epsilon         防止除零的小常数
     */
    public LayerNorm(String name, int normalizedShape, double epsilon) {
        super(name, Shape.of(-1, -1, normalizedShape), Shape.of(-1, -1, normalizedShape));
        //todo
    }

    public LayerNorm(String name, int normalizedShape) {
        this(name, normalizedShape, 1e-6);
    }


    @Override
    public void init() {

    }

    @Override
    public Variable layerForward(Variable... inputs) {
        return null;
    }
}