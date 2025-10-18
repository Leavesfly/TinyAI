package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.VisionInput;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.cnn.ConvLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 视觉编码器
 * 基于卷积神经网络提取图像特征，并转换为Token序列
 * 架构：ResNet-style CNN + Linear投影
 * 
 * @author TinyAI
 */
public class VisionEncoder extends Block {
    
    private final int inputChannels;
    private final int hiddenDim;
    private final int patchSize;
    private final int numPatches;
    
    // 卷积特征提取层
    private ConvLayer conv1;
    private ConvLayer conv2;
    private ConvLayer conv3;
    
    // 投影层：将卷积特征投影到隐藏维度
    private LinearLayer projection;
    
    /**
     * 构造函数
     * 
     * @param inputChannels RGB图像通道数（通常为3）
     * @param hiddenDim 隐藏层维度（与Transformer对齐，如768）
     * @param imageSize 输入图像尺寸
     * @param patchSize Patch大小
     */
    public VisionEncoder(int inputChannels, int hiddenDim, int imageSize, int patchSize) {
        super("VisionEncoder", null);
        this.inputChannels = inputChannels;
        this.hiddenDim = hiddenDim;
        this.patchSize = patchSize;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        
        // 初始化卷积层
        // Conv1: 3 -> 64 channels
        this.conv1 = new ConvLayer("conv1", inputChannels, 64, 7, 2, 3, true);
        
        // Conv2: 64 -> 128 channels
        this.conv2 = new ConvLayer("conv2", 64, 128, 3, 2, 1, true);
        
        // Conv3: 128 -> 256 channels
        this.conv3 = new ConvLayer("conv3", 128, 256, 3, 2, 1, true);
        
        // 投影层：256 * (patch_size^2) -> hiddenDim
        int flattenedDim = 256;
        this.projection = new LinearLayer("projection", flattenedDim, hiddenDim, true);
    }
    
    @Override
    public void init() {
        // 初始化已在构造函数中完成
    }
    
    /**
     * 编码视觉输入
     * 
     * @param visionInput 视觉输入
     * @return 视觉Token序列，维度 [num_patches, hiddenDim]
     */
    public NdArray encode(VisionInput visionInput) {
        NdArray rgbImage = visionInput.getRgbImage();
        
        // 前向传播通过卷积层
        Variable input = new Variable(rgbImage);
        Variable conv1Out = conv1.layerForward(input);
        Variable relu1Out = conv1Out.relu();
        
        Variable conv2Out = conv2.layerForward(relu1Out);
        Variable relu2Out = conv2Out.relu();
        
        Variable conv3Out = conv3.layerForward(relu2Out);
        Variable relu3Out = conv3Out.relu();
        
        // 获取卷积输出的形状 [batch, channels, height, width]
        NdArray convFeatures = relu3Out.getValue();
        io.leavesfly.tinyai.ndarr.Shape shape = convFeatures.getShape();
        
        // 将空间维度展平为patches
        // 假设输入形状为 [height, width, channels]
        // 输出形状应为 [num_patches, channels]
        int numSpatialTokens = shape.getDimension(0) * shape.getDimension(1); // H * W
        int channels = shape.getDimension(2);
        
        // Reshape: [H, W, C] -> [H*W, C]
        NdArray flattenedFeatures = convFeatures.reshape(io.leavesfly.tinyai.ndarr.Shape.of(numSpatialTokens, channels));
        
        // 通过投影层
        Variable projInput = new Variable(flattenedFeatures);
        Variable projOutput = projection.layerForward(projInput);
        
        // 添加位置编码
        NdArray posEncoding = createPositionalEncoding(numSpatialTokens, hiddenDim);
        NdArray visualTokens = projOutput.getValue().add(posEncoding);
        
        // 保存到visionInput
        visionInput.setImageFeatures(visualTokens);
        
        return visualTokens;
    }
    
    /**
     * 创建2D正弦位置编码
     * 
     * @param numPositions 位置数量
     * @param dim 维度
     * @return 位置编码矩阵 [numPositions, dim]
     */
    private NdArray createPositionalEncoding(int numPositions, int dim) {
        float[][] encoding = new float[numPositions][dim];
        
        for (int pos = 0; pos < numPositions; pos++) {
            for (int i = 0; i < dim; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * i) / dim);
                encoding[pos][i] = (float)(i % 2 == 0 ? Math.sin(angle) : Math.cos(angle));
            }
        }
        
        return NdArray.of(encoding);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 简化的前向传播接口
        VisionInput visionInput = new VisionInput(inputs[0].getValue());
        NdArray output = encode(visionInput);
        return new Variable(output);
    }
}
