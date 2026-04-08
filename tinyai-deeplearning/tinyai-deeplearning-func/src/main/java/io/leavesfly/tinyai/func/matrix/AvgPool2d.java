package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 二维平均池化操作的 Function 实现
 * <p>
 * 支持自动微分：反向传播时将梯度均匀分配到池化窗口内的所有位置。
 *
 * @author TinyAI Team
 */
public class AvgPool2d extends Function {

    private final int kernelHeight;
    private final int kernelWidth;
    private final int stride;
    private final int padding;

    // 缓存前向传播信息，供反向传播使用
    private Shape inputShape;
    private int outHeight;
    private int outWidth;

    public AvgPool2d(int kernelHeight, int kernelWidth, int stride, int padding) {
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input = inputs[0];
        this.inputShape = input.getShape();

        int[] dims = inputShape.getShapeDims();
        int batchSize = dims[0];
        int channels = dims[1];
        int height = dims[2];
        int width = dims[3];

        this.outHeight = (height + 2 * padding - kernelHeight) / stride + 1;
        this.outWidth = (width + 2 * padding - kernelWidth) / stride + 1;

        float[] outputData = new float[batchSize * channels * outHeight * outWidth];

        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        float sum = 0.0f;
                        int count = 0;

                        for (int ph = 0; ph < kernelHeight; ph++) {
                            for (int pw = 0; pw < kernelWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    sum += input.get(n, c, ih, iw);
                                    count++;
                                }
                            }
                        }

                        int outputIndex = ((n * channels + c) * outHeight + oh) * outWidth + ow;
                        outputData[outputIndex] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }

        return NdArray.of(outputData, Shape.of(batchSize, channels, outHeight, outWidth));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        int[] dims = inputShape.getShapeDims();
        int batchSize = dims[0];
        int channels = dims[1];
        int height = dims[2];
        int width = dims[3];

        float[] inputGradData = new float[batchSize * channels * height * width];

        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        int count = 0;
                        for (int ph = 0; ph < kernelHeight; ph++) {
                            for (int pw = 0; pw < kernelWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    count++;
                                }
                            }
                        }

                        float gradVal = yGrad.get(n, c, oh, ow) / count;

                        for (int ph = 0; ph < kernelHeight; ph++) {
                            for (int pw = 0; pw < kernelWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int inputIndex = ((n * channels + c) * height + ih) * width + iw;
                                    inputGradData[inputIndex] += gradVal;
                                }
                            }
                        }
                    }
                }
            }
        }

        NdArray inputGrad = NdArray.of(inputGradData, inputShape);
        return Collections.singletonList(inputGrad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}
