package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 卷积层实现类
 * <p>
 * 实现了标准的卷积操作，支持步长、填充、偏置等参数。
 * 使用Im2Col技术将卷积操作转换为矩阵乘法，提高计算效率。
 * 
 * 卷积公式：output = input * weight + bias
 * 其中 weight形状为 (out_channels, in_channels, kernel_height, kernel_width)
 */
public class ConvLayer extends Layer {
    
    private Parameter weight;        // 卷积核参数
    private Parameter bias;          // 偏置参数(可选)
    
    private int inChannels;          // 输入通道数
    private int outChannels;         // 输出通道数
    private int kernelHeight;        // 卷积核高度
    private int kernelWidth;         // 卷积核宽度
    private int stride;              // 步长
    private int padding;             // 填充
    private boolean useBias;         // 是否使用偏置
    
    /**
     * 构造卷积层
     * 
     * @param name 层名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize 卷积核尺寸(正方形)
     * @param stride 步长
     * @param padding 填充
     * @param useBias 是否使用偏置
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelSize,
                     int stride, int padding, boolean useBias) {
        this(name, inChannels, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }
    
    /**
     * 构造卷积层(非正方形卷积核)
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
                     int stride, int padding, boolean useBias) {
        super(name, null, null);  // 输入输出形状将在运行时确定
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;
        
        init();
    }

    public ConvLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 从输入形状推断参数(默认值)
        if (_inputShape != null && _inputShape.size() == 4) {
            this.inChannels = _inputShape.getDimension(1);
            this.outChannels = 32;  // 默认输出通道数
        } else {
            this.inChannels = 1;
            this.outChannels = 32;
        }
        this.kernelHeight = 3;
        this.kernelWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.useBias = true;
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重参数 (out_channels, in_channels, kernel_height, kernel_width)
            // 使用He初始化
            double fan_in = inChannels * kernelHeight * kernelWidth;
            double std = Math.sqrt(2.0 / fan_in);
            
            Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
            NdArray weightData = NdArray.likeRandomN(weightShape).mulNum(std);
            
            weight = new Parameter(weightData);
            weight.setName(name + "_weight");
            addParam("weight", weight);
            
            // 初始化偏置参数(如果使用)
            if (useBias) {
                bias = new Parameter(NdArray.zeros(Shape.of(outChannels)));
                bias.setName(name + "_bias");
                addParam("bias", bias);
            }
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();
        
        // 检查输入形状 (batch_size, channels, height, width)
        if (inputData.getShape().size() != 4) {
            throw new RuntimeException("卷积层输入必须是4维的: (batch_size, channels, height, width)");
        }
        
        int batchSize = inputData.getShape().getDimension(0);
        int inputChannels = inputData.getShape().getDimension(1);
        int inputHeight = inputData.getShape().getDimension(2);
        int inputWidth = inputData.getShape().getDimension(3);
        
        // 检查通道数匹配
        if (inputChannels != inChannels) {
            throw new RuntimeException("输入通道数不匹配: 期望" + inChannels + ", 实际" + inputChannels);
        }
        
        // 计算输出尺寸
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
        
        // 进行Im2Col转换
        NdArray im2colResult = performIm2Col(inputData, kernelHeight, kernelWidth, stride, padding);
        
        // 重塑权重为二维矩阵
        NdArray weightReshaped = reshapeWeight();
        
        // 矩阵乘法计算卷积
        Variable im2colVar = new Variable(im2colResult);
        Variable weightVar = new Variable(weightReshaped);
        Variable output = im2colVar.matMul(weightVar);
        
        // 添加偏置(如果有)
        if (useBias) {
            output = output.add(bias);
        }
        
        // 重塑输出为4维 (batch_size, output_channels, output_height, output_width)
        Shape outputShape = Shape.of(batchSize, outChannels, outputHeight, outputWidth);
        output = output.reshape(outputShape);
        
        return output;
    }
    
    /**
     * 执行Im2Col转换
     */
    private NdArray performIm2Col(NdArray inputData, int kernelH, int kernelW, int stride, int pad) {
        // 这里实现一个简化版本的Im2Col
        // 实际应用中应该使用Im2ColUtil类
        
        int batchSize = inputData.getShape().getDimension(0);
        int channels = inputData.getShape().getDimension(1);
        int height = inputData.getShape().getDimension(2);
        int width = inputData.getShape().getDimension(3);
        
        int outHeight = (height + 2 * pad - kernelH) / stride + 1;
        int outWidth = (width + 2 * pad - kernelW) / stride + 1;
        
        // 创建输出矩阵
        int outputRows = batchSize * outHeight * outWidth;
        int outputCols = channels * kernelH * kernelW;
        
        Shape outputShape = Shape.of(outputRows, outputCols);
        float[] outputData = new float[outputRows * outputCols];
        
        int outputRowIndex = 0;
        for (int n = 0; n < batchSize; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int colIndex = 0;
                    
                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < kernelH; fh++) {
                            int imRow = h * stride + fh - pad;
                            for (int fw = 0; fw < kernelW; fw++) {
                                int imCol = w * stride + fw - pad;
                                
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    // 计算输入数据中的索引
                                    int inputIndex = ((n * channels + c) * height + imRow) * width + imCol;
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
        
        return NdArray.of(outputData, outputShape);
    }
    
    /**
     * 重塑权重为二维矩阵
     */
    private NdArray reshapeWeight() {
        // 权重形状从 (out_channels, in_channels, kernel_h, kernel_w)
        // 重塑为 (in_channels * kernel_h * kernel_w, out_channels)
        NdArray weightData = weight.getValue();
        Shape newShape = Shape.of(inChannels * kernelHeight * kernelWidth, outChannels);
        return weightData.reshape(newShape).transpose();
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 卷积层的反向传播比较复杂，这里提供简化版本
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
}