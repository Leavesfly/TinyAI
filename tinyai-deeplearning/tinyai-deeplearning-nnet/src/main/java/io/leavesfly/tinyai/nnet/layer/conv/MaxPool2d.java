package io.leavesfly.tinyai.nnet.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
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
        // 使用Variable的形状属性
        int dim = x.ndim();
        if (dim != 4) {
            throw new IllegalArgumentException(
                    String.format("Expected 4D input (batch, channels, height, width), but got %dD", dim));
        }

        int batchSize = x.size(0);
        int channels = x.size(1);
        int height = x.size(2);
        int width = x.size(3);

        // 计算输出尺寸
        int outputHeight = (height + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - kernelWidth) / stride + 1;

        // 执行最大池化（需要复杂的索引操作）
        NdArray inputData = x.getValue();
        NdArray output = performMaxPooling(inputData, batchSize, channels,
                                          height, width, outputHeight, outputWidth);

        // 返回新的Variable
        Variable result = new Variable(output);
        result.setRequireGrad(x.isRequireGrad());  // 继承梯度设置
        return result;
    }

    /**
     * 执行最大池化操作
     */
    private NdArray performMaxPooling(NdArray inputData, int batchSize, int channels,
                                      int height, int width, int outHeight, int outWidth) {
        Shape outputShape = Shape.of(batchSize, channels, outHeight, outWidth);
        float[] outputData = new float[batchSize * channels * outHeight * outWidth];

        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        float maxVal = Float.NEGATIVE_INFINITY;

                        // 在池化窗口内找最大值
                        for (int ph = 0; ph < kernelHeight; ph++) {
                            for (int pw = 0; pw < kernelWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    float val = inputData.get(n, c, ih, iw);
                                    maxVal = Math.max(maxVal, val);
                                }
                            }
                        }

                        int outputIndex = ((n * channels + c) * outHeight + oh) * outWidth + ow;
                        outputData[outputIndex] = maxVal == Float.NEGATIVE_INFINITY ? 0.0f : maxVal;
                    }
                }
            }
        }

        return NdArray.of(outputData, outputShape);
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
