package io.leavesfly.tinyai.nnet.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * V2版本的MaxPool2d层
 * <p>
 * 二维最大池化层，对输入的每个窗口取最大值。
 * <p>
 * 主要用途：
 * - 降低特征图的空间维度
 * - 减少参数数量和计算量
 * - 提供平移不变性
 * - 防止过拟合
 * <p>
 * 输出尺寸计算：
 * out_height = (height + 2*padding - kernel_height) / stride + 1
 * out_width = (width + 2*padding - kernel_width) / stride + 1
 *
 * @author leavesfly
 * @version 2.0
 */
public class MaxPool2d extends Module {

    private final int kernelHeight;  // 池化窗口高度
    private final int kernelWidth;   // 池化窗口宽度
    private final int stride;        // 步长
    private final int padding;       // 填充

    /**
     * 构造函数（正方形池化窗口）
     *
     * @param name       层名称
     * @param kernelSize 池化窗口尺寸
     * @param stride     步长
     * @param padding    填充
     */
    public MaxPool2d(String name, int kernelSize, int stride, int padding) {
        this(name, kernelSize, kernelSize, stride, padding);
    }

    /**
     * 构造函数（非正方形池化窗口）
     *
     * @param name         层名称
     * @param kernelHeight 池化窗口高度
     * @param kernelWidth  池化窗口宽度
     * @param stride       步长
     * @param padding      填充
     */
    public MaxPool2d(String name, int kernelHeight, int kernelWidth, int stride, int padding) {
        super(name);
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        init();
    }

    /**
     * 构造函数（默认参数：2x2，stride=2，无填充）
     *
     * @param name 层名称
     */
    public MaxPool2d(String name) {
        this(name, 2, 2, 0);
    }

    /**
     * 构造函数（指定kernel_size，stride默认等于kernel_size）
     *
     * @param name       层名称
     * @param kernelSize 池化窗口尺寸
     */
    public MaxPool2d(String name, int kernelSize) {
        this(name, kernelSize, kernelSize, 0);
    }

    @Override
    public void resetParameters() {
        // 池化层没有可训练参数
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 验证输入维度
        int dim = x.ndim();
        if (dim != 4) {
            throw new IllegalArgumentException(
                    String.format("Expected 4D input (batch, channels, height, width), but got %dD", dim));
        }

        // 创建池化 Function 并通过 call 自动构建计算图
        io.leavesfly.tinyai.func.matrix.MaxPool2d poolFunc =
            new io.leavesfly.tinyai.func.matrix.MaxPool2d(kernelHeight, kernelWidth, stride, padding);
        return poolFunc.call(x);
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
        return "MaxPool2d{" +
                "name='" + name + '\'' +
                ", kernelSize=(" + kernelHeight + ", " + kernelWidth + ")" +
                ", stride=" + stride +
                ", padding=" + padding +
                '}';
    }
}