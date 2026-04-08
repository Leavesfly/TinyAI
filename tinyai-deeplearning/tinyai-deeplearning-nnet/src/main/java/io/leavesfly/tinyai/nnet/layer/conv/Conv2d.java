package io.leavesfly.tinyai.nnet.layer.conv;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;

/**
 * V2版本的Conv2d层
 * <p>
 * 二维卷积层，用于处理图像等二维数据。
 * <p>
 * 本实现委托给底层优化的 {@link io.leavesfly.tinyai.func.matrix.Conv2d} Function，
 * 该Function使用Im2Col技术将卷积转换为高效的矩阵乘法。
 * <p>
 * 公式：
 * output = Conv2d(input, weight) + bias
 * <p>
 * 其中：
 * - input: (batch_size, in_channels, height, width)
 * - weight: (out_channels, in_channels, kernel_height, kernel_width)
 * - bias: (out_channels,)
 * - output: (batch_size, out_channels, out_height, out_width)
 * <p>
 * 输出尺寸计算：
 * out_height = (height + 2*padding - kernel_height) / stride + 1
 * out_width = (width + 2*padding - kernel_width) / stride + 1
 *
 * @author leavesfly
 * @version 2.0 (优化版 - 委托给func层)
 */
public class Conv2d extends Module {

    private Parameter weight;  // 卷积核权重
    private Parameter bias;    // 偏置（可选）

    private final int inChannels;   // 输入通道数
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
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     * @param stride      步长
     * @param padding     填充
     * @param useBias     是否使用偏置
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelSize,
                  int stride, int padding, boolean useBias) {
        this(name, inChannels, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }

    /**
     * 构造函数（非正方形卷积核）
     *
     * @param name         层名称
     * @param inChannels   输入通道数
     * @param outChannels  输出通道数
     * @param kernelHeight 卷积核高度
     * @param kernelWidth  卷积核宽度
     * @param stride       步长
     * @param padding      填充
     * @param useBias      是否使用偏置
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
                  int stride, int padding, boolean useBias) {
        super(name);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;

        initializeParameters();
        init();
    }

    /**
     * 构造函数（默认参数）
     *
     * @param name        层名称
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸
     */
    public Conv2d(String name, int inChannels, int outChannels, int kernelSize) {
        this(name, inChannels, outChannels, kernelSize, 1, 0, true);
    }

    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 权重形状: (out_channels, in_channels, kernel_height, kernel_width)
        Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
        weight = registerParameter("weight", new Parameter(NdArray.of(weightShape)));

        if (useBias) {
            // 偏置形状: (out_channels,)
            bias = registerParameter("bias", new Parameter(NdArray.of(Shape.of(outChannels))));
        }
    }

    @Override
    public void resetParameters() {
        // 使用Kaiming初始化（He初始化）
        // 卷积层适合使用ReLU激活函数
        Initializers.kaimingUniform(weight.data());

        if (useBias) {
            // 偏置初始化为0
            Initializers.zeros(bias.data());
        }
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

        int inputChannels = x.size(1);
        if (inputChannels != inChannels) {
            throw new IllegalArgumentException(
                    String.format("Expected %d input channels, but got %d", inChannels, inputChannels));
        }

        // 创建底层卷积Function（使用Im2Col优化实现）
        io.leavesfly.tinyai.func.matrix.Conv2d convFunc = 
            new io.leavesfly.tinyai.func.matrix.Conv2d(stride, padding);
        
        // 执行卷积运算（自动构建计算图）
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
     * 广播偏置到所有空间维度 [B, OC, OH, OW]
     * bias形状: [OC] -> 广播到 [B, OC, OH, OW]
     */
    private Variable addBias(Variable output) {
        // 获取输出形状
        int batchSize = output.size(0);
        int outChannels = output.size(1);
        int outHeight = output.size(2);
        int outWidth = output.size(3);
        
        // 重塑bias为 [1, OC, 1, 1] 以便广播
        Variable biasReshaped = bias.reshape(Shape.of(1, outChannels, 1, 1));
        
        // 广播并相加
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
        return "Conv2d{" +
                "name='" + name + '\'' +
                ", inChannels=" + inChannels +
                ", outChannels=" + outChannels +
                ", kernelSize=(" + kernelHeight + ", " + kernelWidth + ")" +
                ", stride=" + stride +
                ", padding=" + padding +
                ", useBias=" + useBias +
                '}';
    }
}
