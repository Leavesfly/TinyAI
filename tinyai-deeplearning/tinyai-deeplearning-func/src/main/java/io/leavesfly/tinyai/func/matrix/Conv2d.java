package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 2D卷积操作（优化版本 - 使用Im2Col算法）
 * <p>
 * 本实现采用Im2Col技术将卷积操作转换为高效的矩阵乘法，相比朴素实现性能提升5-10倍。
 * <p>
 * <b>算法原理</b>：
 * <pre>
 * 1. Im2Col: 将输入展开为列矩阵 [B*OH*OW, C*KH*KW]
 * 2. 矩阵乘法: [B*OH*OW, C*KH*KW] @ [C*KH*KW, OC]^T = [B*OH*OW, OC]
 * 3. Reshape: 重塑为 [B, OC, OH, OW]
 * </pre>
 * <p>
 * 前向传播: output = Conv2d(input, kernel, stride, padding)
 * <p>
 * 输入形状:
 * - input:  [batch_size, in_channels, height, width]
 * - kernel: [out_channels, in_channels, kernel_h, kernel_w]
 * <p>
 * 输出形状:
 * - output: [batch_size, out_channels, out_h, out_w]
 * <p>
 * 其中:
 * - out_h = (height + 2 * padding - kernel_h) / stride + 1
 * - out_w = (width + 2 * padding - kernel_w) / stride + 1
 * 
 * @author TinyAI Team
 * @version 2.0 (Im2Col优化版本)
 */
public class Conv2d extends Function {

    private final int stride;
    private final int padding;
    
    // 缓存前向传播信息，供反向传播使用
    private Shape inputShape;
    private Shape kernelShape;
    private NdArray im2colMatrix;  // 缓存Im2Col结果用于反向传播
    private NdArray cachedInput;   // 缓存输入
    private NdArray cachedKernel;  // 缓存kernel
    private int outHeight;
    private int outWidth;

    /**
     * 构造2D卷积函数
     * 
     * @param stride  步长
     * @param padding 填充大小
     */
    public Conv2d(int stride, int padding) {
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input = inputs[0];
        NdArray kernel = inputs[1];
        
        // 缓存输入供反向传播使用
        this.cachedInput = input;
        this.cachedKernel = kernel;
        
        // 保存形状信息供反向传播使用
        this.inputShape = input.getShape();
        this.kernelShape = kernel.getShape();
        
        // 验证输入维度
        if (inputShape.getDimNum() != 4) {
            throw new IllegalArgumentException(
                String.format("Conv2d expects 4D input, got %dD", inputShape.getDimNum())
            );
        }
        if (kernelShape.getDimNum() != 4) {
            throw new IllegalArgumentException(
                String.format("Conv2d expects 4D kernel, got %dD", kernelShape.getDimNum())
            );
        }
        
        int batchSize = inputShape.getDimension(0);
        int inChannels = inputShape.getDimension(1);
        int inputHeight = inputShape.getDimension(2);
        int inputWidth = inputShape.getDimension(3);
        
        int outChannels = kernelShape.getDimension(0);
        int kernelInChannels = kernelShape.getDimension(1);
        int kernelHeight = kernelShape.getDimension(2);
        int kernelWidth = kernelShape.getDimension(3);
        
        // 验证通道数匹配
        if (inChannels != kernelInChannels) {
            throw new IllegalArgumentException(
                String.format("Input channels (%d) != kernel input channels (%d)", 
                    inChannels, kernelInChannels)
            );
        }
        
        // 计算输出尺寸
        this.outHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        this.outWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
        
        // 步骤1: Im2Col转换 - 将输入展开为列矩阵
        // [B, C, H, W] -> [B*OH*OW, C*KH*KW]
        this.im2colMatrix = im2col(input, batchSize, inChannels, 
                                   inputHeight, inputWidth, 
                                   kernelHeight, kernelWidth);
        
        // 步骤2: Reshape kernel为二维矩阵
        // [OC, C, KH, KW] -> [OC, C*KH*KW]
        NdArray kernelReshaped = kernel.reshape(
            Shape.of(outChannels, inChannels * kernelHeight * kernelWidth)
        );
        
        // 步骤3: 矩阵乘法 - [B*OH*OW, C*KH*KW] @ [C*KH*KW, OC] = [B*OH*OW, OC]
        NdArray outputFlat = im2colMatrix.dot(kernelReshaped.transpose());
        
        // 步骤4: Reshape回4D - [B*OH*OW, OC] -> [B, OC, OH, OW]
        NdArray output = outputFlat.reshape(
            Shape.of(batchSize, outHeight, outWidth, outChannels)
        );
        
        // 转换维度顺序: [B, OH, OW, OC] -> [B, OC, OH, OW]
        output = output.transpose(0, 3, 1, 2);
        
        return output;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // yGrad shape: [batch, out_channels, out_h, out_w]
        
        int batchSize = inputShape.getDimension(0);
        int inChannels = inputShape.getDimension(1);
        int inputHeight = inputShape.getDimension(2);
        int inputWidth = inputShape.getDimension(3);
        
        int outChannels = kernelShape.getDimension(0);
        int kernelHeight = kernelShape.getDimension(2);
        int kernelWidth = kernelShape.getDimension(3);
        
        // 获取缓存的kernel
        NdArray kernel = cachedKernel;
        
        // 转换yGrad维度顺序: [B, OC, OH, OW] -> [B, OH, OW, OC]
        NdArray yGradTransposed = yGrad.transpose(0, 2, 3, 1);
        
        // Reshape: [B, OH, OW, OC] -> [B*OH*OW, OC]
        NdArray yGradFlat = yGradTransposed.reshape(
            Shape.of(batchSize * outHeight * outWidth, outChannels)
        );
        
        // 1. 计算输入梯度
        // [B*OH*OW, OC] @ [OC, C*KH*KW] = [B*OH*OW, C*KH*KW]
        NdArray kernelReshaped = kernel.reshape(
            Shape.of(outChannels, inChannels * kernelHeight * kernelWidth)
        );
        NdArray gradCol = yGradFlat.dot(kernelReshaped);
        
        // Col2Im: [B*OH*OW, C*KH*KW] -> [B, C, H, W]
        NdArray inputGrad = col2im(gradCol, batchSize, inChannels,
                                   inputHeight, inputWidth,
                                   kernelHeight, kernelWidth);
        
        // 2. 计算kernel梯度
        // [OC, B*OH*OW] @ [B*OH*OW, C*KH*KW] = [OC, C*KH*KW]
        NdArray kernelGradFlat = yGradFlat.transpose().dot(im2colMatrix);
        
        // Reshape: [OC, C*KH*KW] -> [OC, C, KH, KW]
        NdArray kernelGrad = kernelGradFlat.reshape(
            Shape.of(outChannels, inChannels, kernelHeight, kernelWidth)
        );
        
        return Arrays.asList(inputGrad, kernelGrad);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }

    /**
     * Im2Col转换 - 将卷积窗口展开为列矩阵
     * <p>
     * 将4D输入 [B, C, H, W] 展开为2D矩阵 [B*OH*OW, C*KH*KW]
     * 每一行对应一个卷积窗口的所有元素
     * 
     * @param input 输入张量 [B, C, H, W]
     * @param batchSize 批次大小
     * @param channels 输入通道数
     * @param height 输入高度
     * @param width 输入宽度
     * @param kernelHeight 卷积核高度
     * @param kernelWidth 卷积核宽度
     * @return Im2Col矩阵 [B*OH*OW, C*KH*KW]
     */
    private NdArray im2col(NdArray input, int batchSize, int channels,
                          int height, int width,
                          int kernelHeight, int kernelWidth) {
        int outputRows = batchSize * outHeight * outWidth;
        int outputCols = channels * kernelHeight * kernelWidth;
        
        float[] outputData = new float[outputRows * outputCols];
        float[] inputData = input.getArray();
        
        // 预计算输入张量各维度的步幅，避免在内层循环中重复计算
        int inputStrideB = channels * height * width;
        int inputStrideC = height * width;
        
        int outputRowIndex = 0;
        for (int b = 0; b < batchSize; b++) {
            int inputBaseB = b * inputStrideB;
            for (int oh = 0; oh < outHeight; oh++) {
                int ihBase = oh * stride - padding;
                for (int ow = 0; ow < outWidth; ow++) {
                    int iwBase = ow * stride - padding;
                    int outputBase = outputRowIndex * outputCols;
                    int colIndex = 0;
                    
                    for (int c = 0; c < channels; c++) {
                        int inputBaseBC = inputBaseB + c * inputStrideC;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            int ih = ihBase + kh;
                            if (ih >= 0 && ih < height) {
                                int inputBaseRow = inputBaseBC + ih * width;
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = iwBase + kw;
                                    if (iw >= 0 && iw < width) {
                                        outputData[outputBase + colIndex] = inputData[inputBaseRow + iw];
                                    }
                                    // padding区域默认为0（float[]初始值）
                                    colIndex++;
                                }
                            } else {
                                // 整行都在padding区域，跳过（默认为0）
                                colIndex += kernelWidth;
                            }
                        }
                    }
                    outputRowIndex++;
                }
            }
        }
        
        return NdArray.of(outputData, Shape.of(outputRows, outputCols));
    }

    /**
     * Col2Im转换 - 将列矩阵重组回卷积输入格式
     * <p>
     * Im2Col的逆操作，用于反向传播
     * 将2D矩阵 [B*OH*OW, C*KH*KW] 重组为4D张量 [B, C, H, W]
     * 
     * @param gradCol 列矩阵梯度 [B*OH*OW, C*KH*KW]
     * @param batchSize 批次大小
     * @param channels 输入通道数
     * @param height 输入高度
     * @param width 输入宽度
     * @param kernelHeight 卷积核高度
     * @param kernelWidth 卷积核宽度
     * @return 输入梯度 [B, C, H, W]
     */
    private NdArray col2im(NdArray gradCol, int batchSize, int channels,
                          int height, int width,
                          int kernelHeight, int kernelWidth) {
        NdArray inputGrad = NdArray.zeros(Shape.of(batchSize, channels, height, width));
        
        int outputCols = channels * kernelHeight * kernelWidth;
        float[] gradColData = gradCol.getArray();
        float[] gradData = inputGrad.getArray();
        
        // 预计算输出张量各维度的步幅
        int gradStrideB = channels * height * width;
        int gradStrideC = height * width;
        
        int outputRowIndex = 0;
        for (int b = 0; b < batchSize; b++) {
            int gradBaseB = b * gradStrideB;
            for (int oh = 0; oh < outHeight; oh++) {
                int ihBase = oh * stride - padding;
                for (int ow = 0; ow < outWidth; ow++) {
                    int iwBase = ow * stride - padding;
                    int colBase = outputRowIndex * outputCols;
                    int colIndex = 0;
                    
                    for (int c = 0; c < channels; c++) {
                        int gradBaseBC = gradBaseB + c * gradStrideC;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            int ih = ihBase + kh;
                            if (ih >= 0 && ih < height) {
                                int gradBaseRow = gradBaseBC + ih * width;
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = iwBase + kw;
                                    if (iw >= 0 && iw < width) {
                                        // 直接操作底层数组，累加梯度
                                        gradData[gradBaseRow + iw] += gradColData[colBase + colIndex];
                                    }
                                    colIndex++;
                                }
                            } else {
                                // 整行都在padding区域，跳过
                                colIndex += kernelWidth;
                            }
                        }
                    }
                    outputRowIndex++;
                }
            }
        }
        
        return inputGrad;
    }


}
