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
        NdArray inputData = x.getValue();

        int[] dims = inputData.getShape().getShapeDims();
        int batchSize = dims[0];
        int inputHeight = dims[2];
        int inputWidth = dims[3];

        // 计算输出尺寸
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;

        // 执行Im2Col转换
        NdArray im2colResult = performIm2Col(inputData, batchSize, inChannels,
                                             inputHeight, inputWidth, outputHeight, outputWidth);

        // 重塑权重为二维矩阵
        NdArray weightReshaped = reshapeWeight();

        // 矩阵乘法计算卷积 - 使用Variable层级操作
        Variable im2colVar = new Variable(im2colResult);
        im2colVar.setRequireGrad(false);  // 中间结果不需要梯度
        Variable weightVar = new Variable(weightReshaped.transpose());
        weightVar.setRequireGrad(false);  // 这个临时变量不需要梯度
        Variable output = im2colVar.matMul(weightVar);

        // 添加偏置
        if (useBias) {
            output = addBias(output);
        }

        // 重塑输出为4维
        Shape outputShape = Shape.of(batchSize, outChannels, outputHeight, outputWidth);
        output = output.reshape(outputShape);

        return output;
    }

    /**
     * 执行Im2Col转换
     */
    private NdArray performIm2Col(NdArray inputData, int batchSize, int channels,
                                   int height, int width, int outHeight, int outWidth) {
        int outputRows = batchSize * outHeight * outWidth;
        int outputCols = channels * kernelHeight * kernelWidth;

        float[] outputData = new float[outputRows * outputCols];

        int outputRowIndex = 0;
        for (int n = 0; n < batchSize; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int colIndex = 0;

                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < kernelHeight; fh++) {
                            int imRow = h * stride + fh - padding;
                            for (int fw = 0; fw < kernelWidth; fw++) {
                                int imCol = w * stride + fw - padding;

                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    outputData[outputRowIndex * outputCols + colIndex] =
                                            inputData.get(n, c, imRow, imCol);
                                } else {
                                    outputData[outputRowIndex * outputCols + colIndex] = 0.0f;
                                }
                                colIndex++;
                            }
                        }
                    }
                    outputRowIndex++;
                }
            }
        }

        Shape outputShape = Shape.of(outputRows, outputCols);
        return NdArray.of(outputData, outputShape);
    }

    /**
     * 重塑权重为二维矩阵
     */
    private NdArray reshapeWeight() {
        NdArray weightData = weight.data();
        Shape newShape = Shape.of(outChannels, inChannels * kernelHeight * kernelWidth);
        return weightData.reshape(newShape);
    }

    /**
     * 添加偏置
     * <p>
     * 使用Variable层级的加法操作
     */
    private Variable addBias(Variable output) {
        // bias是Parameter，直接作为Variable参与运算
        return output.add(bias);
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
