package io.leavesfly.tinyai.nnet.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.LazyModule;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的LazyConv2d层
 * <p>
 * 延迟初始化的二维卷积层，构造时无需指定输入通道数，
 * 首次前向传播时根据输入形状自动推断并初始化参数。
 * <p>
 * 使用示例：
 * ```java
 * // 无需指定输入通道数
 * LazyConv2d conv = new LazyConv2d("conv", outChannels=64, kernelSize=3);
 * 
 * // 首次前向传播时自动推断
 * Variable output = conv.forward(input);  // input.shape = (batch, 3, 32, 32)
 * // 自动创建weight(64, 3, 3, 3), bias(64)
 * ```
 *
 * @author leavesfly
 * @version 2.0
 */
public class LazyConv2d extends LazyModule {

    private Parameter weight;  // 卷积核权重
    private Parameter bias;    // 偏置（可选）

    private int inChannels = -1;    // 输入通道数（延迟推断）
    private final int outChannels;  // 输出通道数
    private final int kernelHeight; // 卷积核高度
    private final int kernelWidth;  // 卷积核宽度
    private final int stride;       // 步长
    private final int padding;      // 填充
    private final boolean useBias;  // 是否使用偏置

    /**
     * 构造函数（正方形卷积核）
     *
     * @param name        层名称
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     * @param stride      步长
     * @param padding     填充
     * @param useBias     是否使用偏置
     */
    public LazyConv2d(String name, int outChannels, int kernelSize,
                      int stride, int padding, boolean useBias) {
        this(name, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }

    /**
     * 构造函数（非正方形卷积核）
     *
     * @param name         层名称
     * @param outChannels  输出通道数
     * @param kernelHeight 卷积核高度
     * @param kernelWidth  卷积核宽度
     * @param stride       步长
     * @param padding      填充
     * @param useBias      是否使用偏置
     */
    public LazyConv2d(String name, int outChannels, int kernelHeight, int kernelWidth,
                      int stride, int padding, boolean useBias) {
        super(name);
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;
    }

    /**
     * 构造函数（默认参数）
     *
     * @param name        层名称
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     */
    public LazyConv2d(String name, int outChannels, int kernelSize) {
        this(name, outChannels, kernelSize, 1, 0, true);
    }

    @Override
    protected void initialize(Shape... inputShapes) {
        if (inputShapes.length == 0) {
            throw new IllegalArgumentException("LazyConv2d需要至少一个输入");
        }

        Shape inputShape = inputShapes[0];
        int[] dims = inputShape.getShapeDims();

        if (dims.length != 4) {
            throw new IllegalArgumentException(
                    String.format("Expected 4D input (batch, channels, height, width), but got %dD", dims.length));
        }

        // 从输入形状推断输入通道数
        this.inChannels = dims[1];

        // 创建权重参数
        Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
        weight = registerParameter("weight", new Parameter(NdArray.of(weightShape)));

        // 创建偏置参数
        if (useBias) {
            bias = registerParameter("bias", new Parameter(NdArray.of(Shape.of(outChannels))));
        }
    }

    @Override
    public void resetParameters() {
        if (weight == null) {
            return;  // 尚未初始化
        }

        // 使用Kaiming初始化
        Initializers.kaimingUniform(weight.data());

        if (useBias && bias != null) {
            Initializers.zeros(bias.data());
        }
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 检查并触发延迟初始化
        checkLazyInitialization(inputs);

        Variable x = inputs[0];

        // 验证输入维度
        int dim = x.ndim();
        if (dim != 4) {
            throw new IllegalArgumentException(
                    String.format("Expected 4D input (batch, channels, height, width), but got %dD", dim));
        }

        // 创建底层卷积 Function（使用 Im2Col 优化实现，自动构建计算图）
        io.leavesfly.tinyai.func.matrix.Conv2d convFunc =
            new io.leavesfly.tinyai.func.matrix.Conv2d(stride, padding);
        Variable output = convFunc.call(x, weight);

        // 添加偏置
        if (useBias) {
            output = addBias(output);
        }

        return output;
    }



    /**
     * 添加偏置
     * <p>
     * 使用Variable层级的加法操作
     */
    private Variable addBias(Variable output) {
        int outCh = output.size(1);
        Variable biasReshaped = bias.reshape(Shape.of(1, outCh, 1, 1));
        return output.add(biasReshaped);
    }

    public int getInChannels() {
        return inChannels;
    }

    public int getOutChannels() {
        return outChannels;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    @Override
    public String toString() {
        String inChannelsStr = inChannels == -1 ? "?" : String.valueOf(inChannels);
        return "LazyConv2d{" +
                "name='" + name + '\'' +
                ", inChannels=" + inChannelsStr +
                ", outChannels=" + outChannels +
                ", kernelSize=(" + kernelHeight + ", " + kernelWidth + ")" +
                ", stride=" + stride +
                ", padding=" + padding +
                ", useBias=" + useBias +
                ", initialized=" + _initialized +
                '}';
    }
}
