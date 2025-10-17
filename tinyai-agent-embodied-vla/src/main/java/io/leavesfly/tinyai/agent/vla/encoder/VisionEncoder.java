package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.VisionInput;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.block.Block;
import io.leavesfly.tinyai.nnet.layer.Linear;
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
    private Linear projection;
    
    /**
     * 构造函数
     * 
     * @param inputChannels RGB图像通道数（通常为3）
     * @param hiddenDim 隐藏层维度（与Transformer对齐，如768）
     * @param imageSize 输入图像尺寸
     * @param patchSize Patch大小
     */
    public VisionEncoder(int inputChannels, int hiddenDim, int imageSize, int patchSize) {
        this.inputChannels = inputChannels;
        this.hiddenDim = hiddenDim;
        this.patchSize = patchSize;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        
        // 初始化卷积层
        // Conv1: 3 -> 64 channels
        this.conv1 = new ConvLayer(inputChannels, 64, 7, 2, 3);
        
        // Conv2: 64 -> 128 channels
        this.conv2 = new ConvLayer(64, 128, 3, 2, 1);
        
        // Conv3: 128 -> 256 channels
        this.conv3 = new ConvLayer(128, 256, 3, 2, 1);
        
        // 投影层：256 * (patch_size^2) -> hiddenDim
        int flattenedDim = 256;
        this.projection = new Linear(flattenedDim, hiddenDim, true);
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
        Variable input = new Variable(rgbImage, true);
        Variable conv1Out = conv1.forward(input);
        Variable relu1Out = conv1Out.relu();
        
        Variable conv2Out = conv2.forward(relu1Out);
        Variable relu2Out = conv2Out.relu();
        
        Variable conv3Out = conv3.forward(relu2Out);
        Variable relu3Out = conv3Out.relu();
        
        // 获取卷积输出的形状 [batch, channels, height, width]
        NdArray convFeatures = relu3Out.getData();
        int[] shape = convFeatures.getShape();
        
        // 将空间维度展平为patches
        // 假设输入形状为 [height, width, channels]
        // 输出形状应为 [num_patches, channels]
        int numSpatialTokens = shape[0] * shape[1]; // H * W
        int channels = shape[2];
        
        // Reshape: [H, W, C] -> [H*W, C]
        NdArray flattenedFeatures = convFeatures.reshape(numSpatialTokens, channels);
        
        // 通过投影层
        Variable projInput = new Variable(flattenedFeatures, true);
        Variable projOutput = projection.forward(projInput);
        
        // 添加位置编码
        NdArray posEncoding = createPositionalEncoding(numSpatialTokens, hiddenDim);
        NdArray visualTokens = projOutput.getData().add(posEncoding);
        
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
        double[][] encoding = new double[numPositions][dim];
        
        for (int pos = 0; pos < numPositions; pos++) {
            for (int i = 0; i < dim; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * i) / dim);
                encoding[pos][i] = i % 2 == 0 ? Math.sin(angle) : Math.cos(angle);
            }
        }
        
        return new NdArray(encoding);
    }
    
    @Override
    public Variable forward(Variable input) {
        // 简化的前向传播接口
        VisionInput visionInput = new VisionInput(input.getData());
        NdArray output = encode(visionInput);
        return new Variable(output, input.requiresGrad());
    }
    
    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.addAll(conv1.parameters());
        params.addAll(conv2.parameters());
        params.addAll(conv3.parameters());
        params.addAll(projection.parameters());
        return params;
    }
}
