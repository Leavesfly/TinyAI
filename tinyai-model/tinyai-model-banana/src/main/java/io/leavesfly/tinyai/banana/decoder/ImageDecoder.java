package io.leavesfly.tinyai.banana.decoder;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerDecoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 图像解码器 (Image Decoder)
 * 
 * 将融合后的多模态特征解码为图像像素:
 * 1. Transformer解码器层 - 提取高层特征
 * 2. 特征重塑 - 将序列特征重塑为2D特征图
 * 3. 上采样模块 - 逐步恢复图像分辨率
 * 4. 像素投影 - 输出RGB图像
 * 
 * 架构流程:
 * 融合特征 [batch, 256, 512]
 *   ↓ Transformer Decoder Layers
 * 解码特征 [batch, 256, 512]
 *   ↓ Reshape
 * 特征图 [batch, 512, 16, 16]
 *   ↓ Upsample Blocks (16→32→64→128→256)
 * 高分辨率特征 [batch, 64, 256, 256]
 *   ↓ Pixel Projection
 * 图像 [batch, 3, 256, 256]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ImageDecoder extends Module {
    
    private final BananaConfig config;
    
    // Transformer解码器层列表
    private final List<TransformerDecoderLayer> decoderLayers;
    
    // Dropout层
    private final Dropout featureDropout;
    
    // 特征投影层（降维准备上采样）
    private final Linear featureProjection;
    private final LayerNorm featureNorm;
    
    // 上采样模块列表
    private final List<UpsampleBlock> upsampleBlocks;
    
    // 像素投影层（最终输出层）
    private final PixelProjection pixelProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public ImageDecoder(String name, BananaConfig config) {
        super(name);
        this.config = config;
        
        // 1. 初始化Transformer解码器层
        this.decoderLayers = new ArrayList<>();
        int numDecoderLayers = config.getNumEncoderLayers(); // 与编码器层数相同
        for (int i = 0; i < numDecoderLayers; i++) {
            TransformerDecoderLayer layer = new TransformerDecoderLayer(
                name + "_decoder_" + i,
                config.getHiddenSize(),
                config.getNumHeads(),
                config.getFfnHiddenSize(),
                config.getDropoutRate(),
                true  // 使用Pre-LayerNorm
            );
            decoderLayers.add(layer);
            registerModule("decoder_" + i, layer);
        }
        
        // 2. 特征Dropout
        this.featureDropout = new Dropout(
            name + "_feat_dropout",
            config.getDropoutRate()
        );
        registerModule("feat_dropout", featureDropout);
        
        // 3. 特征投影层（降维以便上采样）
        // 动态计算 upsamplingBaseDim，适配不同规模（Tiny=256, Small=384, Base=512）
        int upsamplingBaseDim = Math.max(config.getHiddenSize() / 2, 64);
        this.featureProjection = new Linear(
            name + "_feat_proj",
            config.getHiddenSize(),
            upsamplingBaseDim,
            true
        );
        registerModule("feat_proj", featureProjection);
        
        this.featureNorm = new LayerNorm(
            name + "_feat_norm",
            upsamplingBaseDim,
            config.getLayerNormEpsilon()
        );
        registerModule("feat_norm", featureNorm);
        
        // 4. 初始化上采样模块（动态计算步数）
        //
        // 上采样链路：从 patchGrid 起步，每步 ×2 直到 imageSize。
        // 通道链路：baseDim → baseDim/2 → baseDim/4 → ... → finalChannels(=16)。
        //
        // 示例：
        //   Nano :  4 ×2→ 8  ×2→ 16  ×2→ 32  ×2→ 64         (imageSize=64,  4 步)
        //   Tiny : 16 ×2→ 32 ×2→ 64 ×2→ 128 ×2→ 256         (imageSize=256, 4 步)
        //   Small: 24 ×2→ 48 ×2→ 96 ×2→ 192 ×2→ 384         (imageSize=384, 4 步)  ← 注意 384 不是 2 的整数幂
        //   Base : 32 ×2→ 64 ×2→ 128 ×2→ 256 ×2→ 512        (imageSize=512, 4 步)
        //
        // 对 Small 的 384 情况：patchGrid=24，24×2^4=384，刚好整除；
        // 更一般地要求 imageSize / patchGrid 必须是 2 的幂次，否则抛出异常。
        this.upsampleBlocks = new ArrayList<>();

        int patchGridSize = config.getImageSize() / config.getPatchSize();
        int upsampleSteps = computeUpsampleSteps(patchGridSize, config.getImageSize());
        int finalChannels = 16;  // PixelProjection 的输入通道

        int curSize = patchGridSize;
        int curChannels = upsamplingBaseDim;
        for (int i = 0; i < upsampleSteps; i++) {
            int nextSize = curSize * 2;
            // 最后一步强制 finalChannels，其余步按 2 倍衰减但不低于 finalChannels
            int nextChannels = (i == upsampleSteps - 1)
                    ? finalChannels
                    : Math.max(curChannels / 2, finalChannels);

            UpsampleBlock block = new UpsampleBlock(
                    name + "_upsample_" + i,
                    curChannels,
                    nextChannels,
                    curSize,
                    nextSize
            );
            upsampleBlocks.add(block);
            registerModule("upsample_" + i, block);

            curSize = nextSize;
            curChannels = nextChannels;
        }

        // 5. 像素投影层（最终输出）
        this.pixelProjection = new PixelProjection(
            name + "_pixel_proj",
            curChannels,                 // 实际上等于 finalChannels
            config.getImageChannels()    // 3 (RGB)
        );
        registerModule("pixel_proj", pixelProjection);

        init();
    }

    /**
     * 计算从 patchGrid 到 imageSize 需要多少次 ×2 上采样步骤。
     *
     * <p>要求 imageSize / patchGrid 必须是 2 的正整数幂次，否则抛出
     * {@link IllegalArgumentException}，避免后续前向传播出现维度不匹配的隐式错误。</p>
     */
    private static int computeUpsampleSteps(int patchGrid, int imageSize) {
        if (patchGrid <= 0 || imageSize <= 0 || imageSize < patchGrid) {
            throw new IllegalArgumentException(
                    "非法的 patchGrid/imageSize: patchGrid=" + patchGrid + ", imageSize=" + imageSize
            );
        }
        int steps = 0;
        int cur = patchGrid;
        while (cur < imageSize) {
            cur *= 2;
            steps++;
        }
        if (cur != imageSize) {
            throw new IllegalArgumentException(String.format(
                    "imageSize / patchGrid 必须为 2 的整数幂次，但 imageSize=%d, patchGrid=%d, " +
                            "请调整 imageSize 或 patchSize 使得两者比值为 2 的幂。",
                    imageSize, patchGrid));
        }
        return steps;
    }
    
    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为融合特征 [batch, num_patches, hidden_size]
     *               inputs[1]为编码器输出（用于cross-attention）[batch, num_patches, hidden_size]
     * @return 生成的图像 [batch, channels, height, width]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException("ImageDecoder需要至少2个输入: fusedFeatures和encoderOutput");
        }
        
        Variable fusedFeatures = inputs[0];
        Variable encoderOutput = inputs[1];
        
        validateInput(fusedFeatures);
        
        // 1. 通过Transformer解码器层
        Variable x = fusedFeatures;
        for (TransformerDecoderLayer layer : decoderLayers) {
            x = layer.forward(x, encoderOutput);
        }
        
        // 2. 应用Dropout
        x = featureDropout.forward(x);
        
        // 3. 特征投影和归一化
        // [batch, num_patches, hidden_size] -> [batch, num_patches, upsampling_base_dim]
        x = featureProjection.forward(x);
        x = featureNorm.forward(x);
        
        // 4. 重塑为2D特征图
        // [batch, num_patches, dim] -> [batch, dim, grid_h, grid_w]
        x = reshapeTo2D(x);
        
        // 5. 通过上采样模块
        for (UpsampleBlock block : upsampleBlocks) {
            x = block.forward(x);
        }
        
        // 6. 像素投影，输出RGB图像
        Variable image = pixelProjection.forward(x);
        
        // 7. 应用Tanh激活，将像素值归一化到[-1, 1]
        image = image.tanh();
        
        return image;
    }
    
    /**
     * 将序列特征重塑为2D特征图
     * 
     * 使用 Permute Function 保持梯度计算图连通。
     *
     * @param x 输入特征 [batch, num_patches, dim]
     * @return 2D特征图 [batch, dim, grid_h, grid_w]
     */
    private Variable reshapeTo2D(Variable x) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int numPatches = shape[1];
        int dim = shape[2];
        
        // 计算网格尺寸
        int gridSize = (int) Math.sqrt(numPatches); // 16 for 256 patches
        
        if (gridSize * gridSize != numPatches) {
            throw new IllegalStateException(
                "numPatches必须是完全平方数: " + numPatches
            );
        }
        
        // [B, N, D] -> [B, D, N] -> [B, D, gridH, gridW]
        // 使用 Permute Function（支持自动微分），保持梯度计算图完整
        Variable permuted = new Permute(0, 2, 1).call(x);
        return permuted.reshape(Shape.of(batchSize, dim, gridSize, gridSize));
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable fusedFeatures) {
        if (fusedFeatures == null) {
            throw new IllegalArgumentException("fusedFeatures不能为null");
        }
        
        int[] shape = fusedFeatures.getValue().getShape().getShapeDims();
        if (shape.length != 3) {
            throw new IllegalArgumentException(
                "fusedFeatures必须是3维 [batch, num_patches, hidden_size], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
        
        int numPatches = shape[1];
        int hiddenSize = shape[2];
        
        if (numPatches != config.getNumPatches()) {
            throw new IllegalArgumentException(
                "patch数量不匹配: 期望" + config.getNumPatches() + ", 实际" + numPatches
            );
        }
        
        if (hiddenSize != config.getHiddenSize()) {
            throw new IllegalArgumentException(
                "隐藏层维度不匹配: 期望" + config.getHiddenSize() + ", 实际" + hiddenSize
            );
        }
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return decoderLayers.size();
    }
    
    public List<TransformerDecoderLayer> getDecoderLayers() {
        return decoderLayers;
    }
    
    public List<UpsampleBlock> getUpsampleBlocks() {
        return upsampleBlocks;
    }
    
    @Override
    public String toString() {
        return String.format(
            "ImageDecoder{numLayers=%d, hiddenSize=%d, outputSize=%dx%d}",
            decoderLayers.size(),
            config.getHiddenSize(),
            config.getImageSize(),
            config.getImageSize()
        );
    }
}
