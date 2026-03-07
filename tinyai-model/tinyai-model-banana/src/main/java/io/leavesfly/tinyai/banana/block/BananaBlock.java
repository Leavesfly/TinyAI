package io.leavesfly.tinyai.banana.block;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.banana.decoder.ImageDecoder;
import io.leavesfly.tinyai.banana.encoder.ImageEncoder;
import io.leavesfly.tinyai.banana.encoder.TextEncoder;
import io.leavesfly.tinyai.banana.fusion.MultiModalFusion;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * Banana模型主体块
 * 
 * 整合所有Banana组件,构建完整的多模态架构:
 * 1. 文本编码器 - 处理文本输入
 * 2. 图像编码器(ViT) - 处理图像输入
 * 3. 多模态融合层 - 跨模态注意力机制
 * 4. 输出投影层 - 生成最终输出
 * 
 * 数据流:
 * text_tokens → TextEncoder → multimodal_fusion ↘
 * image_pixels → ImageEncoder → multimodal_fusion → output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class BananaBlock extends Module {
    
    private final BananaConfig config;
    
    // 核心组件
    private TextEncoder textEncoder;        // 文本编码器
    private ImageEncoder imageEncoder;      // 图像编码器
    private MultiModalFusion fusionLayer;   // 多模态融合层
    private ImageDecoder imageDecoder;      // 图像解码器
    
    // 基础组件
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public BananaBlock(String name, BananaConfig config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件
     */
    private void initializeComponents() {
        // 初始化文本编码器
        textEncoder = new TextEncoder(name + "_text_encoder", config);
        registerModule("text_encoder", textEncoder);
        
        // 初始化图像编码器
        imageEncoder = new ImageEncoder(name + "_image_encoder", config);
        registerModule("image_encoder", imageEncoder);
        
        // 初始化多模态融合层
        if (config.isEnableCrossModalAttention()) {
            fusionLayer = new MultiModalFusion(name + "_fusion", config);
            registerModule("fusion", fusionLayer);
        }
        
        // 初始化图像解码器
        imageDecoder = new ImageDecoder(name + "_image_decoder", config);
        registerModule("image_decoder", imageDecoder);
        
        // 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln",
            config.getHiddenSize(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 初始化输出投影层
        // 根据任务类型,可能输出到不同的空间
        // 文本生成: vocab_size
        // 图像生成: image_vocab_size
        outputProjection = new Linear(
            name + "_output_proj",
            config.getHiddenSize(),
            config.getVocabSize(),  // 默认文本词汇表
            false
        );
        registerModule("output_proj", outputProjection);
    }
    
    /**
     * 前向传播（文本模式）
     * 
     * 完整流程: 文本Token -> TextEncoder -> LayerNorm -> outputProjection
     *
     * @param inputs inputs[0]为文本token IDs [batch, seq_len]
     * @return 输出Variable [batch, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        
        // 完整的文本前向传播流程
        Variable textFeatures = textEncoder.forward(input);   // [batch, seq_len, hidden_size]
        Variable normalized = finalLayerNorm.forward(textFeatures);
        return outputProjection.forward(normalized);          // [batch, seq_len, vocab_size]
    }
    
    /**
     * 文本编码前向传播
     * 
     * @param textTokenIds 文本token IDs [batch, seq_len]
     * @return 文本特征 [batch, seq_len, hidden_size]
     */
    public Variable forwardText(Variable textTokenIds) {
        return textEncoder.forward(textTokenIds);
    }
    
    /**
     * 图像编码前向传播
     * 
     * @param imagePixels 图像像素 [batch, channels, height, width]
     * @return 图像特征 [batch, num_patches, hidden_size]
     */
    public Variable forwardImage(Variable imagePixels) {
        return imageEncoder.forward(imagePixels);
    }
    
    /**
     * 多模态融合前向传播
     * 
     * @param textFeatures 文本特征 [batch, text_len, hidden_size]
     * @param imageFeatures 图像特征 [batch, num_patches, hidden_size]
     * @param taskType 任务类型
     * @return 融合后的特征 [batch, text_len, hidden_size] (不做输出投影)
     */
    public Variable forwardMultiModal(Variable textFeatures, 
                                     Variable imageFeatures, 
                                     TaskType taskType) {
        if (fusionLayer != null) {
            // 跨模态融合 - 返回融合后的文本特征
            Variable fusedTextFeatures = fusionLayer.forward(textFeatures, imageFeatures);
            
            // 最终LayerNorm (不做输出投影,保持在特征空间)
            return finalLayerNorm.forward(fusedTextFeatures);
        } else {
            // 如果未启用跨模态，直接返回归一化后的文本特征
            return finalLayerNorm.forward(textFeatures);
        }
    }
    
    /**
     * 多模态融合前向传播(带输出投影)
     * 
     * 用于生成任务,将特征投影到词汇表空间
     * 
     * @param textFeatures 文本特征 [batch, text_len, hidden_size]
     * @param imageFeatures 图像特征 [batch, num_patches, hidden_size]
     * @param taskType 任务类型
     * @return 投影后的输出 [batch, text_len, vocab_size]
     */
    public Variable forwardMultiModalWithProjection(Variable textFeatures, 
                                                    Variable imageFeatures, 
                                                    TaskType taskType) {
        // 先进行融合
        Variable fusedFeatures = forwardMultiModal(textFeatures, imageFeatures, taskType);
        
        // 再进行输出投影
        return outputProjection.forward(fusedFeatures);
    }
    
    /**
     * 带详细信息的前向传播
     * 
     * @param textTokenIds 文本token IDs
     * @param imagePixels 图像像素(可选，为null则仅文本模式)
     * @param taskType 任务类型
     * @return 详细的前向结果
     */
    public DetailedForwardResult forwardWithDetails(Variable textTokenIds,
                                                    Variable imagePixels,
                                                    TaskType taskType) {
        // 1. 编码文本
        Variable textFeatures = textEncoder.forward(textTokenIds);
        
        // 2. 编码图像（可选）
        Variable imageFeatures = null;
        if (imagePixels != null) {
            imageFeatures = imageEncoder.forward(imagePixels);
        }
        
        // 3. 跨模态融合
        Variable fusedFeatures;
        if (fusionLayer != null && imageFeatures != null) {
            fusedFeatures = fusionLayer.forward(textFeatures, imageFeatures);
        } else {
            fusedFeatures = textFeatures;
        }
        
        // 4. 归一化 + 输出投影
        Variable normalized = finalLayerNorm.forward(fusedFeatures);
        Variable output = outputProjection.forward(normalized);
        
        return new DetailedForwardResult(output, textFeatures, imageFeatures, fusedFeatures, taskType);
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable input) {
        if (input == null) {
            throw new IllegalArgumentException("输入Variable不能为null");
        }
        
        int[] shape = input.getValue().getShape().getShapeDims();
        if (shape.length < 2) {
            throw new IllegalArgumentException(
                "输入shape至少需要2维 [batch, seq_len], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
    }
    
    /**
     * 文本到图像生成
     * 
     * 设计说明:
     * 由于 text-to-image 场景中没有原始图像作为参考，
     * 我们使用文本特征进行自注意力（self-attention）操作来增强特征间的关联。
     * imageDecoder.forward(fusedFeatures, fusedFeatures) 中传入相同特征，
     * 让解码器在文本融合特征上执行自注意力，而非交叉注意力。
     * 
     * @param textTokenIds 文本描述token IDs [batch, text_len]
     * @return 生成的图像 [batch, 3, image_size, image_size]
     */
    public Variable textToImage(Variable textTokenIds) {
        // 1. 编码文本
        Variable textFeatures = forwardText(textTokenIds);
            
        // 2. 融合（使用文本特征作自注意力）
        Variable fusedFeatures;
        if (fusionLayer != null) {
            fusedFeatures = fusionLayer.forward(textFeatures, textFeatures);
        } else {
            // 未启用跨模态融合时，直接使用文本特征
            fusedFeatures = textFeatures;
        }
            
        // 3. 将融合后的文本特征扩展到patch数量
        // [batch, text_len, hidden_size] -> [batch, num_patches, hidden_size]
        fusedFeatures = expandToPatches(fusedFeatures);
            
        // 4. 解码为图像
        // 注意: 传入相同的fusedFeatures作为解码器输入和交叉注意力上下文，
        // 这样解码器将在文本融合特征上执行自注意力操作
        Variable generatedImage = imageDecoder.forward(fusedFeatures, fusedFeatures);
            
        return generatedImage;
    }
    
    /**
     * 将文本特征扩展到patch数量
     * 
     * 使用平均池化聚合序列信息，再通过 expand 复制到所有patch位置。
     * 保持梯度计算图连通。
     * 
     * @param textFeatures 文本特征 [batch, text_len, hidden_size]
     * @return 扩展后的特征 [batch, num_patches, hidden_size]
     */
    private Variable expandToPatches(Variable textFeatures) {
        int[] shape = textFeatures.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int hiddenSize = shape[2];
        int numPatches = config.getNumPatches();
        
        // 平均池化: [batch, text_len, hidden] -> [batch, 1, hidden]
        Variable meanPooled = textFeatures.mean(1, true);
        
        // 扩展到 num_patches: [batch, 1, hidden] -> [batch, num_patches, hidden]
        return meanPooled.expand(Shape.of(batchSize, numPatches, hiddenSize));
    }
    
    /**
     * 图像编辑
     * 
     * 使用原始图像特征与编辑指令融合后生成编辑后的图像。
     * 设计说明：
     * 1. 编码原始图像获得视觉特征
     * 2. 编码编辑指令获得文本特征
     * 3. 通过跨模态注意力融合文本指令和图像特征
     * 4. 解码生成编辑后的图像
     * 
     * @param imagePixels 原始图像 [batch, channels, height, width]
     * @param editInstructions 编辑指令token IDs [batch, text_len]
     * @return 编辑后的图像 [batch, 3, image_size, image_size]
     */
    public Variable imageEditing(Variable imagePixels, Variable editInstructions) {
        // 1. 编码原始图像
        Variable imageFeatures = forwardImage(imagePixels);
        
        // 2. 编码编辑指令
        Variable textFeatures = forwardText(editInstructions);
        
        // 3. 跨模态融合：文本指令注意到图像特征
        Variable fusedFeatures;
        if (fusionLayer != null) {
            fusedFeatures = fusionLayer.forward(textFeatures, imageFeatures);
        } else {
            // 未启用跨模态融合时，拼接文本和图像特征
            fusedFeatures = textFeatures;
        }
        
        // 4. 将融合特征扩展到patch数量
        fusedFeatures = expandToPatches(fusedFeatures);
        
        // 5. 使用原始图像特征作为解码器的交叉注意力上下文
        // 这样解码器可以参考原始图像生成编辑结果
        Variable editedImage = imageDecoder.forward(fusedFeatures, imageFeatures);
        
        return editedImage;
    }
    
    /**
     * 图像理解（图像到文本）
     * 
     * 将图像编码后通过输出投影生成文本描述的logits。
     * 设计说明：
     * 1. 编码图像获得视觉特征
     * 2. 通过LayerNorm归一化
     * 3. 通过输出投影层映射到词汇表空间
     * 
     * @param imagePixels 图像 [batch, channels, height, width]
     * @return 文本logits [batch, num_patches, vocab_size]
     */
    public Variable imageToText(Variable imagePixels) {
        // 1. 编码图像
        Variable imageFeatures = forwardImage(imagePixels);
        
        // 2. 如果启用跨模态注意力，使用自注意力增强特征
        if (fusionLayer != null) {
            imageFeatures = fusionLayer.forward(imageFeatures, imageFeatures);
        }
        
        // 3. LayerNorm归一化
        Variable normalized = finalLayerNorm.forward(imageFeatures);
        
        // 4. 投影到词汇表空间
        Variable textLogits = outputProjection.forward(normalized);
        
        return textLogits;
    }
    
    /**
     * 多图像组合
     * 
     * 将多张图像的特征融合后生成组合图像。
     * 设计说明：
     * 1. 分别编码每张图像
     * 2. 聚合多图像特征（平均池化）
     * 3. 使用组合指令引导融合
     * 4. 解码生成组合图像
     * 
     * @param images 多张图像堆叠 [num_images, channels, height, width]
     * @param compositionInstructions 组合指令 [batch, text_len]
     * @return 组合后的图像 [batch, 3, image_size, image_size]
     */
    public Variable composeMultipleImages(Variable images, Variable compositionInstructions) {
        int[] imageShape = images.getValue().getShape().getShapeDims();
        int numImages = imageShape[0];
        int hiddenSize = config.getHiddenSize();
        int numPatches = config.getNumPatches();
        
        // 1. 编码所有图像（作为单个batch处理）
        // images: [num_images, channels, height, width]
        Variable allImageFeatures = forwardImage(images);
        // allImageFeatures: [num_images, num_patches, hidden_size]
        
        // 2. 聚合多图像特征：在图像维度上平均池化
        // [num_images, num_patches, hidden_size] -> [1, num_patches, hidden_size]
        Variable aggregatedFeatures = allImageFeatures.mean(0, true);
        
        // 3. 编码组合指令
        Variable textFeatures = forwardText(compositionInstructions);
        
        // 4. 跨模态融合：组合指令注意到聚合的图像特征
        Variable fusedFeatures;
        if (fusionLayer != null) {
            fusedFeatures = fusionLayer.forward(textFeatures, aggregatedFeatures);
        } else {
            fusedFeatures = textFeatures;
        }
        
        // 5. 将融合特征扩展到patch数量
        fusedFeatures = expandToPatches(fusedFeatures);
        
        // 6. 解码生成组合图像，使用聚合特征作为交叉注意力上下文
        Variable composedImage = imageDecoder.forward(fusedFeatures, aggregatedFeatures);
        
        return composedImage;
    }
    
    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(80));
        System.out.println("Banana模型架构");
        System.out.println("=".repeat(80));
        System.out.println("配置: " + config);
        System.out.println("-".repeat(80));
        System.out.println("组件状态:");
        System.out.println("  文本编码器: ✓ " + textEncoder);
        System.out.println("  图像编码器: ✓ " + imageEncoder);
        System.out.println("  多模态融合: " + 
            (fusionLayer != null ? "✓ " + fusionLayer : "未启用"));
        System.out.println("  图像解码器: ✓ " + imageDecoder);
        System.out.println("  最终LayerNorm: ✓");
        System.out.println("  输出投影层: ✓");
        System.out.println("=".repeat(80));
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public TextEncoder getTextEncoder() {
        return textEncoder;
    }
    
    public ImageEncoder getImageEncoder() {
        return imageEncoder;
    }
    
    public MultiModalFusion getFusionLayer() {
        return fusionLayer;
    }
    
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    public Linear getOutputProjection() {
        return outputProjection;
    }
    
    public ImageDecoder getImageDecoder() {
        return imageDecoder;
    }
    
    // ==================== 内部结果类 ====================
    
    /**
     * 详细前向传播结果
     */
    public static class DetailedForwardResult {
        public final Variable output;
        public final Variable textFeatures;
        public final Variable imageFeatures;
        public final Variable fusedFeatures;
        public final TaskType taskType;
        
        public DetailedForwardResult(Variable output,
                                    Variable textFeatures,
                                    Variable imageFeatures,
                                    Variable fusedFeatures,
                                    TaskType taskType) {
            this.output = output;
            this.textFeatures = textFeatures;
            this.imageFeatures = imageFeatures;
            this.fusedFeatures = fusedFeatures;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format(
                "DetailedForwardResult{taskType=%s, hasTextFeatures=%b, hasImageFeatures=%b}",
                taskType, 
                textFeatures != null, 
                imageFeatures != null
            );
        }
    }
}
