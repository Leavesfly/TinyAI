package io.leavesfly.tinyai.banana.encoder;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.TransformerEncoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 图像编码器 (Vision Transformer)
 * 
 * 使用Transformer架构处理图像输入:
 * 1. Patch嵌入层 - 将图像分割成patches并嵌入
 * 2. 2D位置编码 - 添加空间位置信息
 * 3. Transformer编码器层堆叠 - 提取图像特征
 * 
 * 架构流程:
 * 图像 [batch, 3, 256, 256]
 *   ↓ PatchEmbedding
 * Patches [batch, 256, 512]
 *   ↓ + Position2D
 * Positioned Patches [batch, 256, 512]
 *   ↓ Transformer Layers
 * 图像特征 [batch, 256, 512]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ImageEncoder extends Module {
    
    private final BananaConfig config;
    
    // Patch嵌入层
    private final PatchEmbedding patchEmbedding;
    
    // 2D位置编码
    private final Position2D position2D;
    
    // Dropout层
    private final Dropout embeddingDropout;
    
    // Transformer编码器层列表
    private final List<TransformerEncoderLayer> encoderLayers;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public ImageEncoder(String name, BananaConfig config) {
        super(name);
        if (config == null) {
            throw new IllegalArgumentException("config不能为null");
        }
        this.config = config;
        
        // 1. 初始化Patch嵌入层
        this.patchEmbedding = new PatchEmbedding(
            name + "_patch_emb",
            config.getImageSize(),
            config.getPatchSize(),
            config.getImageChannels(),
            config.getHiddenSize()
        );
        registerModule("patch_emb", patchEmbedding);
        
        // 2. 初始化2D位置编码
        this.position2D = new Position2D(
            name + "_pos_2d",
            config.getNumPatches(),
            config.getHiddenSize()
        );
        registerModule("pos_2d", position2D);
        
        // 3. 初始化嵌入Dropout
        this.embeddingDropout = new Dropout(
            name + "_emb_dropout",
            config.getEmbeddingDropout()
        );
        registerModule("emb_dropout", embeddingDropout);
        
        // 4. 初始化Transformer编码器层
        this.encoderLayers = new ArrayList<>();
        for (int i = 0; i < config.getNumEncoderLayers(); i++) {
            TransformerEncoderLayer layer = new TransformerEncoderLayer(
                name + "_encoder_" + i,
                config.getHiddenSize(),
                config.getNumHeads(),
                config.getFfnHiddenSize(),
                config.getDropoutRate(),
                true  // 使用Pre-LayerNorm
            );
            encoderLayers.add(layer);
            registerModule("encoder_" + i, layer);
        }
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为图像像素 [batch, channels, height, width]
     * @return 图像特征向量 [batch, num_patches, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("ImageEncoder需要输入图像");
        }
        
        Variable imagePixels = inputs[0];
        validateInput(imagePixels);
        
        // 1. Patch嵌入: [batch, C, H, W] -> [batch, num_patches, hidden_size]
        Variable patches = patchEmbedding.forward(imagePixels);
        
        // 2. 添加2D位置编码
        Variable posEncodings = position2D.forward(patches);
        Variable x = patches.add(posEncodings);
        
        // 3. 应用嵌入Dropout
        x = embeddingDropout.forward(x);
        
        // 4. 通过Transformer编码器层
        for (TransformerEncoderLayer layer : encoderLayers) {
            x = layer.forward(x);
        }
        
        return x;
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable imagePixels) {
        if (imagePixels == null) {
            throw new IllegalArgumentException("imagePixels不能为null");
        }
        
        int[] shape = imagePixels.getValue().getShape().getShapeDims();
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                "imagePixels必须是4维 [batch, channels, height, width], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
        
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        if (channels != config.getImageChannels()) {
            throw new IllegalArgumentException(
                "图像通道数不匹配: 期望" + config.getImageChannels() + ", 实际" + channels
            );
        }
        
        if (height != config.getImageSize() || width != config.getImageSize()) {
            throw new IllegalArgumentException(
                "图像尺寸不匹配: 期望" + config.getImageSize() + "x" + config.getImageSize() + 
                ", 实际" + height + "x" + width
            );
        }
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return encoderLayers.size();
    }
    
    public PatchEmbedding getPatchEmbedding() {
        return patchEmbedding;
    }
    
    public Position2D getPosition2D() {
        return position2D;
    }
    
    public List<TransformerEncoderLayer> getEncoderLayers() {
        return encoderLayers;
    }
    
    @Override
    public String toString() {
        return String.format(
            "ImageEncoder{numLayers=%d, hiddenSize=%d, imageSize=%dx%d, patchSize=%dx%d, numPatches=%d}",
            config.getNumEncoderLayers(),
            config.getHiddenSize(),
            config.getImageSize(),
            config.getImageSize(),
            config.getPatchSize(),
            config.getPatchSize(),
            config.getNumPatches()
        );
    }
}
