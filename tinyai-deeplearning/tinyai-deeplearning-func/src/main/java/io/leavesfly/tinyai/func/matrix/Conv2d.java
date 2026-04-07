package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;
/**
 * 2D卷积操作（高性能版本 - Im2Col + 按Batch矩阵乘法）
 * <p>
 * 本实现采用Im2Col技术将卷积操作转换为高效的矩阵乘法。
 * <p>
 * <b>性能优化要点</b>：
 * <ol>
 *   <li>按Batch拆分Im2Col + 矩阵乘法，消除前向/反向中的transpose操作</li>
 *   <li>padding=0时使用无边界检查的快速路径，减少分支预测开销</li>
 *   <li>直接操作底层float[]数组，避免NdArray API的额外开销</li>
 * </ol>
 * <p>
 * <b>算法流程（每个batch独立）</b>：
 * <pre>
 * 1. Im2Col:    input_b[C, H, W] → col_b[C*KH*KW, OH*OW]
 * 2. 矩阵乘法: kernel[OC, C*KH*KW] @ col_b[C*KH*KW, OH*OW] = out_b[OC, OH*OW]
 * 3. 直接写入:  out_b 写入输出的第b个batch位置，无需transpose
 * </pre>
 * <p>
 * 输入形状:
 * - input:  [batch_size, in_channels, height, width]
 * - kernel: [out_channels, in_channels, kernel_h, kernel_w]
 * <p>
 * 输出形状:
 * - output: [batch_size, out_channels, out_h, out_w]
 *
 * @author TinyAI Team
 * @version 3.0 (按Batch拆分 + 零拷贝优化)
 */
public class Conv2d extends Function {

    private final int stride;
    private final int padding;

    // 缓存前向传播信息，供反向传播使用
    private Shape inputShape;
    private Shape kernelShape;
    private NdArray[] im2colPerBatch;  // 按batch缓存Im2Col结果
    private NdArray cachedKernel;
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

        this.cachedKernel = kernel;
        this.inputShape = input.getShape();
        this.kernelShape = kernel.getShape();

        validateShapes();

        int batchSize = inputShape.getDimension(0);
        int inChannels = inputShape.getDimension(1);
        int inputHeight = inputShape.getDimension(2);
        int inputWidth = inputShape.getDimension(3);

        int outChannels = kernelShape.getDimension(0);
        int kernelHeight = kernelShape.getDimension(2);
        int kernelWidth = kernelShape.getDimension(3);

        this.outHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        this.outWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;

        int colRows = inChannels * kernelHeight * kernelWidth;
        int colCols = outHeight * outWidth;

        // kernel reshape: [OC, C*KH*KW]
        NdArray kernelReshaped = kernel.reshape(Shape.of(outChannels, colRows));

        // 输出直接按 [B, OC, OH, OW] 布局写入，无需 transpose
        int outputBatchStride = outChannels * colCols;
        float[] outputData = new float[batchSize * outputBatchStride];
        float[] inputData = input.getArray();

        this.im2colPerBatch = new NdArray[batchSize];
        int inputBatchStride = inChannels * inputHeight * inputWidth;

        for (int b = 0; b < batchSize; b++) {
            // Im2Col: 每个batch生成 [C*KH*KW, OH*OW] 的列矩阵
            NdArray colMatrix = im2colSingleBatch(inputData, b * inputBatchStride,
                    inChannels, inputHeight, inputWidth, kernelHeight, kernelWidth,
                    colRows, colCols);
            im2colPerBatch[b] = colMatrix;

            // 矩阵乘法: kernel[OC, C*KH*KW] @ col[C*KH*KW, OH*OW] = [OC, OH*OW]
            // 结果直接就是 [OC, OH*OW]，即输出的 [OC, OH, OW] 展平形式，无需 transpose
            NdArray outBatch = kernelReshaped.dot(colMatrix);

            // 直接拷贝到输出数组的对应 batch 位置
            System.arraycopy(outBatch.getArray(), 0, outputData, b * outputBatchStride, outputBatchStride);
        }

        return NdArray.of(outputData, Shape.of(batchSize, outChannels, outHeight, outWidth));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        int batchSize = inputShape.getDimension(0);
        int inChannels = inputShape.getDimension(1);
        int inputHeight = inputShape.getDimension(2);
        int inputWidth = inputShape.getDimension(3);

        int outChannels = kernelShape.getDimension(0);
        int kernelHeight = kernelShape.getDimension(2);
        int kernelWidth = kernelShape.getDimension(3);

        int colRows = inChannels * kernelHeight * kernelWidth;
        int colCols = outHeight * outWidth;

        // kernel reshape: [OC, C*KH*KW]
        NdArray kernelReshaped = cachedKernel.reshape(Shape.of(outChannels, colRows));
        // kernel 转置: [C*KH*KW, OC]
        NdArray kernelTransposed = kernelReshaped.transpose();

        float[] yGradData = yGrad.getArray();
        int gradBatchStride = outChannels * colCols;

        // 输入梯度
        float[] inputGradData = new float[batchSize * inChannels * inputHeight * inputWidth];
        int inputBatchStride = inChannels * inputHeight * inputWidth;

        // kernel梯度累加
        float[] kernelGradData = new float[outChannels * colRows];

        for (int b = 0; b < batchSize; b++) {
            // yGrad 的第b个batch: [OC, OH*OW]
            NdArray yGradBatch = NdArray.of(
                    Arrays.copyOfRange(yGradData, b * gradBatchStride, (b + 1) * gradBatchStride),
                    Shape.of(outChannels, colCols)
            );

            // 1. 输入梯度: kernel^T[C*KH*KW, OC] @ yGrad_b[OC, OH*OW] = [C*KH*KW, OH*OW]
            NdArray gradCol = kernelTransposed.dot(yGradBatch);
            col2imSingleBatch(gradCol.getArray(), inputGradData, b * inputBatchStride,
                    inChannels, inputHeight, inputWidth, kernelHeight, kernelWidth,
                    colRows, colCols);

            // 2. kernel梯度: yGrad_b[OC, OH*OW] @ col_b^T[OH*OW, C*KH*KW] = [OC, C*KH*KW]
            NdArray kernelGradBatch = yGradBatch.dot(im2colPerBatch[b].transpose());
            float[] kgBatch = kernelGradBatch.getArray();
            for (int i = 0; i < kernelGradData.length; i++) {
                kernelGradData[i] += kgBatch[i];
            }
        }

        NdArray inputGrad = NdArray.of(inputGradData,
                Shape.of(batchSize, inChannels, inputHeight, inputWidth));
        NdArray kernelGrad = NdArray.of(kernelGradData,
                Shape.of(outChannels, inChannels, kernelHeight, kernelWidth));

        return Arrays.asList(inputGrad, kernelGrad);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }

    /**
     * 验证输入和kernel的形状
     */
    private void validateShapes() {
        if (inputShape.getDimNum() != 4) {
            throw new IllegalArgumentException(
                    String.format("Conv2d expects 4D input, got %dD", inputShape.getDimNum()));
        }
        if (kernelShape.getDimNum() != 4) {
            throw new IllegalArgumentException(
                    String.format("Conv2d expects 4D kernel, got %dD", kernelShape.getDimNum()));
        }
        int inChannels = inputShape.getDimension(1);
        int kernelInChannels = kernelShape.getDimension(1);
        if (inChannels != kernelInChannels) {
            throw new IllegalArgumentException(
                    String.format("Input channels (%d) != kernel input channels (%d)",
                            inChannels, kernelInChannels));
        }
    }

    /**
     * 单Batch的Im2Col转换（列优先布局）
     * <p>
     * 将单个batch的输入 [C, H, W] 展开为列矩阵 [C*KH*KW, OH*OW]。
     * 每一列对应一个卷积窗口的所有元素，与kernel[OC, C*KH*KW]直接做矩阵乘法，
     * 结果为 [OC, OH*OW]，天然就是输出的 [OC, OH, OW] 展平形式，无需transpose。
     * <p>
     * padding=0时使用无边界检查的快速路径，减少分支预测开销。
     *
     * @param inputData     输入数组的底层float[]
     * @param inputOffset   当前batch在inputData中的起始偏移
     * @param channels      输入通道数
     * @param height        输入高度
     * @param width         输入宽度
     * @param kernelHeight  卷积核高度
     * @param kernelWidth   卷积核宽度
     * @param colRows       列矩阵行数 = C*KH*KW
     * @param colCols       列矩阵列数 = OH*OW
     * @return 列矩阵 [C*KH*KW, OH*OW]
     */
    private NdArray im2colSingleBatch(float[] inputData, int inputOffset,
                                      int channels, int height, int width,
                                      int kernelHeight, int kernelWidth,
                                      int colRows, int colCols) {
        float[] colData = new float[colRows * colCols];
        int channelStride = height * width;

        if (padding == 0) {
            // 快速路径：无padding时所有访问都在有效范围内，跳过边界检查
            for (int c = 0; c < channels; c++) {
                int inputChannelBase = inputOffset + c * channelStride;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        int rowIndex = (c * kernelHeight + kh) * kernelWidth + kw;
                        int rowBase = rowIndex * colCols;
                        int colIdx = 0;
                        for (int oh = 0; oh < outHeight; oh++) {
                            int ih = oh * stride + kh;
                            int inputRowBase = inputChannelBase + ih * width;
                            for (int ow = 0; ow < outWidth; ow++) {
                                colData[rowBase + colIdx] = inputData[inputRowBase + ow * stride + kw];
                                colIdx++;
                            }
                        }
                    }
                }
            }
        } else {
            // 通用路径：有padding时需要边界检查
            for (int c = 0; c < channels; c++) {
                int inputChannelBase = inputOffset + c * channelStride;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        int rowIndex = (c * kernelHeight + kh) * kernelWidth + kw;
                        int rowBase = rowIndex * colCols;
                        int colIdx = 0;
                        for (int oh = 0; oh < outHeight; oh++) {
                            int ih = oh * stride + kh - padding;
                            for (int ow = 0; ow < outWidth; ow++) {
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    colData[rowBase + colIdx] = inputData[inputChannelBase + ih * width + iw];
                                }
                                colIdx++;
                            }
                        }
                    }
                }
            }
        }

        return NdArray.of(colData, Shape.of(colRows, colCols));
    }

    /**
     * 单Batch的Col2Im转换（反向传播用）
     * <p>
     * 将列矩阵梯度 [C*KH*KW, OH*OW] 累加回输入梯度 [C, H, W]。
     * padding=0时使用无边界检查的快速路径。
     *
     * @param colData       列矩阵梯度数据
     * @param gradData      输入梯度数组（累加目标）
     * @param gradOffset    当前batch在gradData中的起始偏移
     * @param channels      输入通道数
     * @param height        输入高度
     * @param width         输入宽度
     * @param kernelHeight  卷积核高度
     * @param kernelWidth   卷积核宽度
     * @param colRows       列矩阵行数 = C*KH*KW
     * @param colCols       列矩阵列数 = OH*OW
     */
    private void col2imSingleBatch(float[] colData, float[] gradData, int gradOffset,
                                   int channels, int height, int width,
                                   int kernelHeight, int kernelWidth,
                                   int colRows, int colCols) {
        int channelStride = height * width;

        if (padding == 0) {
            for (int c = 0; c < channels; c++) {
                int gradChannelBase = gradOffset + c * channelStride;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        int rowIndex = (c * kernelHeight + kh) * kernelWidth + kw;
                        int rowBase = rowIndex * colCols;
                        int colIdx = 0;
                        for (int oh = 0; oh < outHeight; oh++) {
                            int ih = oh * stride + kh;
                            int gradRowBase = gradChannelBase + ih * width;
                            for (int ow = 0; ow < outWidth; ow++) {
                                gradData[gradRowBase + ow * stride + kw] += colData[rowBase + colIdx];
                                colIdx++;
                            }
                        }
                    }
                }
            }
        } else {
            for (int c = 0; c < channels; c++) {
                int gradChannelBase = gradOffset + c * channelStride;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        int rowIndex = (c * kernelHeight + kh) * kernelWidth + kw;
                        int rowBase = rowIndex * colCols;
                        int colIdx = 0;
                        for (int oh = 0; oh < outHeight; oh++) {
                            int ih = oh * stride + kh - padding;
                            for (int ow = 0; ow < outWidth; ow++) {
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    gradData[gradChannelBase + ih * width + iw] += colData[rowBase + colIdx];
                                }
                                colIdx++;
                            }
                        }
                    }
                }
            }
        }
    }

}
