package io.leavesfly.tinyai.nnet.v2.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.LazyModule;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

/**
 * V2版本的延迟初始化线性层
 * <p>
 * LazyLinear在构造时不需要指定输入维度，
 * 在首次前向传播时根据实际输入自动推断并初始化参数。
 * <p>
 * 使用示例：
 * <pre>
 * LazyLinear layer = new LazyLinear("fc", 64, true);
 * Variable output = layer.forward(input);  // 根据input.shape自动推断输入维度
 * </pre>
 *
 * @author leavesfly
 * @version 2.0
 */
public class LazyLinear extends LazyModule {

    private Parameter weight;
    private Parameter bias;
    private Integer inFeatures;  // 延迟推断，初始为null
    private final int outFeatures;
    private final boolean useBias;

    /**
     * 构造函数
     *
     * @param name        层名称
     * @param outFeatures 输出特征数
     * @param useBias     是否使用偏置
     */
    public LazyLinear(String name, int outFeatures, boolean useBias) {
        super(name);
        this.inFeatures = null;  // 延迟推断
        this.outFeatures = outFeatures;
        this.useBias = useBias;
    }

    /**
     * 构造函数（默认使用偏置）
     *
     * @param name        层名称
     * @param outFeatures 输出特征数
     */
    public LazyLinear(String name, int outFeatures) {
        this(name, outFeatures, true);
    }

    @Override
    protected void initialize(Shape... inputShapes) {
        if (inputShapes.length == 0) {
            throw new IllegalArgumentException("LazyLinear requires at least one input");
        }

        Shape inputShape = inputShapes[0];
        
        // 推断输入维度（最后一维）
        int dimNum = inputShape.getDimNum();
        if (dimNum < 2) {
            throw new IllegalArgumentException(
                    "LazyLinear requires input with at least 2 dimensions, got: " + dimNum);
        }
        this.inFeatures = inputShape.getDimension(dimNum - 1);

        // 创建参数
        NdArray weightData = NdArray.of(Shape.of(outFeatures, inFeatures));
        this.weight = registerParameter("weight", new Parameter(weightData));

        if (useBias) {
            NdArray biasData = NdArray.of(Shape.of(outFeatures));
            this.bias = registerParameter("bias", new Parameter(biasData));
        }
    }

    @Override
    public void resetParameters() {
        if (weight != null) {
            // 使用Kaiming均匀初始化
            Initializers.kaimingUniform(weight.data(), 0, "fan_in", "relu");
        }
        if (bias != null) {
            Initializers.zeros(bias.data());
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 检查并触发延迟初始化
        checkLazyInitialization(inputs);

        Variable x = inputs[0];

        // 线性变换：y = xW^T + b
        Variable y = x.matMul(weight.transpose());

        if (bias != null) {
            y = y.add(bias);
        }

        return y;
    }

    /**
     * 获取权重参数
     *
     * @return 权重参数，如果未初始化返回null
     */
    public Parameter getWeight() {
        return weight;
    }

    /**
     * 获取偏置参数
     *
     * @return 偏置参数，如果不使用偏置或未初始化返回null
     */
    public Parameter getBias() {
        return bias;
    }

    /**
     * 获取输入特征数
     *
     * @return 输入特征数，如果未初始化返回null
     */
    public Integer getInFeatures() {
        return inFeatures;
    }

    /**
     * 获取输出特征数
     *
     * @return 输出特征数
     */
    public int getOutFeatures() {
        return outFeatures;
    }

    @Override
    public String toString() {
        return "LazyLinear{" +
                "name='" + name + '\'' +
                ", inFeatures=" + (inFeatures != null ? inFeatures : "uninitialized") +
                ", outFeatures=" + outFeatures +
                ", useBias=" + useBias +
                ", initialized=" + _initialized +
                '}';
    }
}
