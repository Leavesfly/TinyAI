package io.leavesfly.tinyai.banana.model;

import io.leavesfly.tinyai.banana.block.BananaBlock;
import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;

/**
 * Banana多模态图像生成模型
 * 
 * Banana是一个教育性的多模态模型,支持:
 * - 文本到图像生成
 * - 图像编辑
 * - 图像理解
 * - 多图像组合
 * 
 * 模型规模:
 * - Tiny: ~50M 参数 (教学用)
 * - Small: ~200M 参数 (实验用)
 * - Base: ~500M 参数 (标准)
 * 
 * 继承自TinyAI Model类,提供统一的模型接口
 * 
 * @author leavesfly
 * @version 1.0
 */
public class BananaModel extends Model {
    
    /**
     * 模型配置
     */
    private final BananaConfig config;
    
    /**
     * 模型主体
     */
    private final BananaBlock bananaBlock;
    
    /**
     * 构造Banana模型
     * 
     * @param name 模型名称
     * @param config 模型配置
     */
    public BananaModel(String name, BananaConfig config) {
        super(name, new BananaBlock(name + "_main", config));
        this.config = config;
        this.bananaBlock = (BananaBlock) getModule();
        
        // 设置模型描述
        setDescription(buildDescription());
    }
    
    /**
     * 使用预设配置创建模型
     * 
     * @param name 模型名称
     * @param preset 预设配置 ("nano", "tiny", "small", "base")
     * @return Banana模型实例
     */
    public static BananaModel create(String name, String preset) {
        BananaConfig config;
        switch (preset.toLowerCase()) {
            case "nano":
                config = BananaConfig.createNanoConfig();
                break;
            case "tiny":
                config = BananaConfig.createTinyConfig();
                break;
            case "small":
                config = BananaConfig.createSmallConfig();
                break;
            case "base":
                config = BananaConfig.createBaseConfig();
                break;
            default:
                throw new IllegalArgumentException(
                    "未知的预设配置: " + preset + ". 可选: nano, tiny, small, base"
                );
        }
        return new BananaModel(name, config);
    }
    
    /**
     * 构建模型描述信息
     */
    private String buildDescription() {
        return String.format(
            "Banana多模态图像生成模型 | 参数量: %s | 层数: %d | 维度: %d | " +
            "图像尺寸: %dx%d | Patch数量: %d",
            config.formatParameters(),
            config.getNumLayers(),
            config.getHiddenSize(),
            config.getImageSize(),
            config.getImageSize(),
            config.getNumPatches()
        );
    }
    
    // ==================== 推理方法 ====================
    
    /**
     * 标准预测方法
     * 
     * @param input 输入Variable(文本tokens或图像pixels)
     * @return 输出Variable
     */
    public Variable predict(Variable input) {
        return forward(input);
    }
    
    /**
     * 文本编码
     * 
     * @param textTokenIds 文本token IDs [batch, seq_len]
     * @return 文本特征 [batch, seq_len, hidden_size]
     */
    public Variable encodeText(Variable textTokenIds) {
        return bananaBlock.forwardText(textTokenIds);
    }
    
    /**
     * 图像编码
     * 
     * @param imagePixels 图像像素 [batch, channels, height, width]
     * @return 图像特征 [batch, num_patches, hidden_size]
     */
    public Variable encodeImage(Variable imagePixels) {
        return bananaBlock.forwardImage(imagePixels);
    }
    
    /**
     * 文本到图像生成
     * 
     * @param textTokenIds 文本描述token IDs [batch, text_len]
     * @return 生成的图像 [batch, channels, height, width]
     */
    public Variable generateImage(Variable textTokenIds) {
        return bananaBlock.textToImage(textTokenIds);
    }
    
    /**
     * 图像编辑
     * 
     * 使用编辑指令修改原始图像。
     * 设计说明：
     * 1. 编码原始图像获得视觉特征
     * 2. 编码编辑指令获得文本特征
     * 3. 通过跨模态注意力融合文本指令和图像特征
     * 4. 解码生成编辑后的图像
     * 
     * @param imagePixels 原始图像 [batch, channels, height, width]
     * @param editInstructions 编辑指令token IDs [batch, text_len]
     * @return 编辑后的图像 [batch, channels, height, width]
     */
    public Variable editImage(Variable imagePixels, Variable editInstructions) {
        return bananaBlock.imageEditing(imagePixels, editInstructions);
    }
    
    /**
     * 图像理解
     * 
     * 将图像转换为文本描述的logits。
     * 输出的logits可以通过argmax或sampling解码为文本token。
     * 
     * @param imagePixels 图像 [batch, channels, height, width]
     * @return 图像描述logits [batch, num_patches, vocab_size]
     */
    public Variable understandImage(Variable imagePixels) {
        return bananaBlock.imageToText(imagePixels);
    }
    
    /**
     * 多图像组合
     * 
     * 将多张图像按照组合指令融合为一张新图像。
     * 设计说明：
     * 1. 编码所有输入图像
     * 2. 聚合多图像特征（平均池化）
     * 3. 使用组合指令引导融合
     * 4. 解码生成组合图像
     * 
     * @param images 多张图像 [num_images, channels, height, width]
     * @param compositionInstructions 组合指令 [batch, text_len]
     * @return 组合后的图像 [batch, channels, height, width]
     */
    public Variable composeImages(Variable images, Variable compositionInstructions) {
        return bananaBlock.composeMultipleImages(images, compositionInstructions);
    }
    
    /**
     * 带详细信息的推理
     * 
     * @param textTokenIds 文本token IDs
     * @param imagePixels 图像像素(可选)
     * @param taskType 任务类型
     * @return 详细推理结果
     */
    public InferenceResult inferWithDetails(Variable textTokenIds,
                                           Variable imagePixels,
                                           TaskType taskType) {
        BananaBlock.DetailedForwardResult blockResult = 
            bananaBlock.forwardWithDetails(textTokenIds, imagePixels, taskType);
        
        return new InferenceResult(
            blockResult.output,
            blockResult.textFeatures,
            blockResult.imageFeatures,
            taskType
        );
    }
    
    // ==================== 模型信息 ====================
    
    /**
     * 打印模型详细信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(80));
        System.out.println("Banana模型详细信息");
        System.out.println("=".repeat(80));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(80));
        System.out.println(config);
        System.out.println("-".repeat(80));
        if (bananaBlock != null) {
            bananaBlock.printArchitecture();
        }
        System.out.println("=".repeat(80));
    }
    
    /**
     * 获取配置摘要
     */
    public String getConfigSummary() {
        return String.format(
            "Banana配置摘要:\n" +
            "  - 文本词汇表大小: %,d\n" +
            "  - 隐藏层维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - FFN隐藏维度: %d\n" +
            "  - 图像尺寸: %dx%d\n" +
            "  - Patch尺寸: %dx%d\n" +
            "  - Patch数量: %d\n" +
            "  - 图像编码器层数: %d\n" +
            "  - 跨模态注意力: %s\n" +
            "  - 估算参数量: %s\n",
            config.getVocabSize(),
            config.getHiddenSize(),
            config.getNumLayers(),
            config.getNumHeads(),
            config.getFfnHiddenSize(),
            config.getImageSize(),
            config.getImageSize(),
            config.getPatchSize(),
            config.getPatchSize(),
            config.getNumPatches(),
            config.getNumEncoderLayers(),
            config.isEnableCrossModalAttention() ? "启用" : "禁用",
            config.formatParameters()
        );
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public BananaBlock getBananaBlock() {
        return bananaBlock;
    }
    
    @Override
    public String toString() {
        return String.format(
            "BananaModel{name='%s', params=%s, layers=%d, hiddenSize=%d, imageSize=%dx%d}",
            getName(),
            config.formatParameters(),
            config.getNumLayers(),
            config.getHiddenSize(),
            config.getImageSize(),
            config.getImageSize()
        );
    }
    
    // ==================== 内部结果类 ====================
    
    /**
     * 推理结果类
     */
    public static class InferenceResult {
        public final Variable output;
        public final Variable textFeatures;
        public final Variable imageFeatures;
        public final TaskType taskType;
        
        public InferenceResult(Variable output,
                              Variable textFeatures,
                              Variable imageFeatures,
                              TaskType taskType) {
            this.output = output;
            this.textFeatures = textFeatures;
            this.imageFeatures = imageFeatures;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format(
                "InferenceResult{taskType=%s, hasText=%b, hasImage=%b}",
                taskType,
                textFeatures != null,
                imageFeatures != null
            );
        }
    }
}