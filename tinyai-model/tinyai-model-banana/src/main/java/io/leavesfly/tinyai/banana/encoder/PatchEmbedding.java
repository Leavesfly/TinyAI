package io.leavesfly.tinyai.banana.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.conv.Conv2d;

/**
 * Patch嵌入层
 * 
 * 将图像分割成patches并嵌入到向量空间,这是Vision Transformer的核心组件。
 * 
 * 工作原理:
 * 1. 使用卷积操作将图像分割成不重叠的patches
 * 2. 每个patch通过线性投影变换到嵌入空间
 * 3. 卷积的stride等于patch_size,保证patches不重叠
 * 
 * 例如: 256x256图像, patch_size=16
 * - 分割成 16x16 = 256 个patches
 * - 每个patch大小为 16x16x3 (RGB)
 * - 投影到 hidden_size 维向量
 * 
 * 输入: 图像 [batch, channels, height, width]
 * 输出: Patch序列 [batch, num_patches, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class PatchEmbedding extends Module {
    
    private final int imageSize;     // 图像尺寸(假设正方形)
    private final int patchSize;     // Patch尺寸
    private final int imageChannels; // 图像通道数(3 for RGB)
    private final int hiddenSize;    // 嵌入维度
    private final int numPatches;    // Patch数量
    
    // 使用Conv2D实现Patch嵌入
    private final Conv2d patchConv;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param imageSize 图像尺寸(高度和宽度)
     * @param patchSize Patch尺寸
     * @param imageChannels 图像通道数
     * @param hiddenSize 嵌入维度
     */
    public PatchEmbedding(String name, int imageSize, int patchSize, 
                         int imageChannels, int hiddenSize) {
        super(name);
        
        // 验证参数
        if (imageSize % patchSize != 0) {
            throw new IllegalArgumentException(
                "imageSize必须能被patchSize整除: " + imageSize + " % " + patchSize + " != 0"
            );
        }
        
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.imageChannels = imageChannels;
        this.hiddenSize = hiddenSize;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        
        // 使用Conv2D实现Patch嵌入
        // 卷积核大小 = patch_size
        // 步长 = patch_size (确保patches不重叠)
        // 输入通道 = image_channels
        // 输出通道 = hidden_size
        this.patchConv = new Conv2d(
            name + "_patch_conv",
            imageChannels,     // 输入通道: 3 (RGB)
            hiddenSize,        // 输出通道: hidden_size
            patchSize,         // 卷积核大小: patch_size x patch_size
            patchSize,         // 步长: patch_size (不重叠)
            0,                 // 无padding
            true               // 使用bias
        );
        registerModule("patch_conv", patchConv);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // Conv2D会自动初始化参数
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为图像 [batch, channels, height, width]
     * @return Patch序列 [batch, num_patches, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("PatchEmbedding需要输入图像");
        }
        
        Variable image = inputs[0];
        validateInput(image);
        
        // 1. 使用卷积提取patches
        // 输入: [batch, channels, height, width]
        // 输出: [batch, hidden_size, num_patches_h, num_patches_w]
        Variable patchFeatures = patchConv.forward(image);
        
        // 2. 重塑为序列格式
        // [batch, hidden_size, num_patches_h, num_patches_w] 
        // -> [batch, hidden_size, num_patches]
        // -> [batch, num_patches, hidden_size]
        Variable patchSequence = reshapeToPatchSequence(patchFeatures);
        
        return patchSequence;
    }
    
    /**
     * 重塑为Patch序列
     * 
     * 将卷积输出从 [batch, hidden_size, h', w'] 
     * 重塑为 [batch, num_patches, hidden_size]
     * 
     * 使用 Permute Function 保持计算图连通，确保梯度可以回传到 patchConv。
     */
    private Variable reshapeToPatchSequence(Variable patchFeatures) {
        int batchSize = patchFeatures.size(0);
        int hiddenSize = patchFeatures.size(1);
        int numPatchesH = patchFeatures.size(2);
        int numPatchesW = patchFeatures.size(3);
        int totalPatches = numPatchesH * numPatchesW;
        
        // 验证patch数量
        if (totalPatches != numPatches) {
            throw new IllegalStateException(
                "计算的patch数量(" + totalPatches + ")与预期(" + numPatches + ")不匹配"
            );
        }
        
        // [B, H, h', w'] -> [B, H, h'*w']
        Variable reshaped = patchFeatures.reshape(
            Shape.of(batchSize, hiddenSize, totalPatches)
        );
        
        // [B, H, N] -> [B, N, H]
        // 使用 Permute Function（支持自动微分），保持梯度计算图完整
        return new Permute(0, 2, 1).call(reshaped);
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable image) {
        if (image == null) {
            throw new IllegalArgumentException("输入图像不能为null");
        }
        
        int[] shape = image.getValue().getShape().getShapeDims();
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                "图像必须是4维 [batch, channels, height, width], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
        
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        if (channels != imageChannels) {
            throw new IllegalArgumentException(
                "图像通道数不匹配: 期望" + imageChannels + ", 实际" + channels
            );
        }
        
        if (height != imageSize || width != imageSize) {
            throw new IllegalArgumentException(
                "图像尺寸不匹配: 期望" + imageSize + "x" + imageSize + 
                ", 实际" + height + "x" + width
            );
        }
    }
    
    // ==================== Getter方法 ====================
    
    public int getImageSize() {
        return imageSize;
    }
    
    public int getPatchSize() {
        return patchSize;
    }
    
    public int getImageChannels() {
        return imageChannels;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public int getNumPatches() {
        return numPatches;
    }
    
    @Override
    public String toString() {
        return String.format(
            "PatchEmbedding{imageSize=%dx%d, patchSize=%dx%d, numPatches=%d, hiddenSize=%d}",
            imageSize, imageSize, patchSize, patchSize, numPatches, hiddenSize
        );
    }
}
