package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.Collections;
import java.util.List;

/**
 * 池化层
 * 支持最大池化、平均池化和自适应池化
 * 
 * 池化操作可以减少特征图的空间尺寸，降低计算复杂度并提供平移不变性
 */
public class PoolingLayer extends Layer {

    /**
     * 池化类型枚举
     */
    public enum PoolingType {
        MAX,      // 最大池化
        AVERAGE,  // 平均池化
        ADAPTIVE  // 自适应池化
    }
    
    private PoolingType poolingType;  // 池化类型
    private int poolHeight;           // 池化窗口高度
    private int poolWidth;            // 池化窗口宽度
    private int stride;               // 步长
    private int padding;              // 填充
    
    /**
     * 构造池化层
     * 
     * @param name 层名称
     * @param poolingType 池化类型
     * @param poolSize 池化窗口尺寸(正方形)
     * @param stride 步长
     * @param padding 填充
     */
    public PoolingLayer(String name, PoolingType poolingType, int poolSize, int stride, int padding) {
        this(name, poolingType, poolSize, poolSize, stride, padding);
    }
    
    /**
     * 构造池化层(非正方形窗口)
     */
    public PoolingLayer(String name, PoolingType poolingType, int poolHeight, int poolWidth, 
                        int stride, int padding) {
        super(name, null, null);  // 输入输出形状将在运行时确定
        
        this.poolingType = poolingType;
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;
        this.padding = padding;
        
        init();
    }
    
    /**
     * 简化构造函数(默认参数)
     */
    public PoolingLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 默认为2x2最大池化，步长2，无填充
        this.poolingType = PoolingType.MAX;
        this.poolHeight = 2;
        this.poolWidth = 2;
        this.stride = 2;
        this.padding = 0;
        init();
    }

    @Override
    public void init() {
        // 池化层没有可训练的参数，无需初始化
        alreadyInit = true;
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();
        
        // 检查输入形状 (batch_size, channels, height, width)
        if (inputData.getShape().getDimNum() != 4) {
            throw new RuntimeException("池化层输入必须是4维的: (batch_size, channels, height, width)");
        }
        
        NdArray output;
        switch (poolingType) {
            case MAX:
                output = performMaxPooling(inputData);
                break;
            case AVERAGE:
                output = performAveragePooling(inputData);
                break;
            case ADAPTIVE:
                output = performAdaptivePooling(inputData);
                break;
            default:
                throw new RuntimeException("不支持的池化类型: " + poolingType);
        }
        
        return new Variable(output);
    }
    
    /**
     * 执行最大池化
     */
    private NdArray performMaxPooling(NdArray inputData) {
        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);
        
        int outputHeight = (height + 2 * padding - poolHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - poolWidth) / stride + 1;
        
        Shape outputShape = Shape.of(batchSize, channels, outputHeight, outputWidth);
        float[] outputData = new float[batchSize * channels * outputHeight * outputWidth];
        
        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float maxVal = Float.NEGATIVE_INFINITY;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int inputIndex = ((n * channels + c) * height + ih) * width + iw;
                                    maxVal = Math.max(maxVal, inputData.get(n, c, ih, iw));
                                }
                            }
                        }
                        
                        int outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIndex] = maxVal == Float.NEGATIVE_INFINITY ? 0.0f : maxVal;
                    }
                }
            }
        }
        
        return NdArray.of(outputData, outputShape);
    }
    
    /**
     * 执行平均池化
     */
    private NdArray performAveragePooling(NdArray inputData) {
        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);
        
        int outputHeight = (height + 2 * padding - poolHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - poolWidth) / stride + 1;
        
        Shape outputShape = Shape.of(batchSize, channels, outputHeight, outputWidth);
        float[] outputData = new float[batchSize * channels * outputHeight * outputWidth];
        
        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0.0f;
                        int count = 0;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * stride + ph - padding;
                                int iw = ow * stride + pw - padding;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int inputIndex = ((n * channels + c) * height + ih) * width + iw;
                                    sum += inputData.get(n, c, ih, iw);
                                    count++;
                                }
                            }
                        }
                        
                        int outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIndex] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }
        
        return NdArray.of(outputData, outputShape);
    }
    
    /**
     * 执行自适应池化(简化实现)
     * 自适应池化将输入池化为固定尺寸的输出
     */
    private NdArray performAdaptivePooling(NdArray inputData) {
        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);
        
        // 自适应池化输出尺寸(默认1x1)
        int outputHeight = 1;
        int outputWidth = 1;
        
        Shape outputShape = Shape.of(batchSize, channels, outputHeight, outputWidth);
        float[] outputData = new float[batchSize * channels * outputHeight * outputWidth];
        
        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                // 计算整个特征图的平均值
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int inputIndex = ((n * channels + c) * height + h) * width + w;
                        sum += inputData.get(n, c, h, w);
                    }
                }
                
                int outputIndex = n * channels + c;
                outputData[outputIndex] = sum / (height * width);
            }
        }
        
        return NdArray.of(outputData, outputShape);
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 池化层的反向传播比较复杂，这里提供简化版本
        List<NdArray> result = new java.util.ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
}
