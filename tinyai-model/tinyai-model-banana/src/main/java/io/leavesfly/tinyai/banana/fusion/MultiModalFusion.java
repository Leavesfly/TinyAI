package io.leavesfly.tinyai.banana.fusion;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;

/**
 * 多模态融合模块
 * 
 * 整合文本和图像特征,实现跨模态交互和信息融合。
 * 
 * 架构设计:
 * 1. Text → Image 跨模态注意力: 文本特征attend to图像特征
 * 2. Image → Text 跨模态注意力: 图像特征attend to文本特征
 * 3. 残差连接和LayerNorm
 * 
 * 融合流程:
 * - 文本特征通过CrossModalAttention关注图像特征
 * - 图像特征通过CrossModalAttention关注文本特征
 * - 使用Pre-LayerNorm架构保证训练稳定性
 * 
 * 输入:
 * - textFeatures: [batch, text_len, hidden_size]
 * - imageFeatures: [batch, num_patches, hidden_size]
 * 
 * 输出:
 * - fusedTextFeatures: [batch, text_len, hidden_size]
 * - fusedImageFeatures: [batch, num_patches, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiModalFusion extends Module {
    
    private final BananaConfig config;
    
    // Text → Image 跨模态注意力
    private final CrossModalAttention text2ImageAttn;
    private final LayerNorm text2ImageNorm;
    private final Dropout text2ImageDropout;
    
    // Image → Text 跨模态注意力
    private final CrossModalAttention image2TextAttn;
    private final LayerNorm image2TextNorm;
    private final Dropout image2TextDropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public MultiModalFusion(String name, BananaConfig config) {
        super(name);
        this.config = config;
        
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumHeads();
        float dropout = config.getDropoutRate();
        
        // 初始化Text → Image跨模态注意力
        this.text2ImageAttn = new CrossModalAttention(
            name + "_text2image_attn",
            hiddenSize,
            numHeads,
            dropout
        );
        registerModule("text2image_attn", text2ImageAttn);
        
        this.text2ImageNorm = new LayerNorm(
            name + "_text2image_norm",
            hiddenSize
        );
        registerModule("text2image_norm", text2ImageNorm);
        
        this.text2ImageDropout = new Dropout(
            name + "_text2image_dropout",
            dropout
        );
        registerModule("text2image_dropout", text2ImageDropout);
        
        // 初始化Image → Text跨模态注意力
        this.image2TextAttn = new CrossModalAttention(
            name + "_image2text_attn",
            hiddenSize,
            numHeads,
            dropout
        );
        registerModule("image2text_attn", image2TextAttn);
        
        this.image2TextNorm = new LayerNorm(
            name + "_image2text_norm",
            hiddenSize
        );
        registerModule("image2text_norm", image2TextNorm);
        
        this.image2TextDropout = new Dropout(
            name + "_image2text_dropout",
            dropout
        );
        registerModule("image2text_dropout", image2TextDropout);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }
    
    /**
     * 前向传播（仅返回 text 侧融合结果，单侧开销）。
     *
     * <p>由于 {@link Module#forward(Variable...)} 签名只能返回单个 {@link Variable}，
     * 此方法仅执行 <b>text → image</b> 方向的跨模态注意力并返回。</p>
     *
     * <p><b>注意</b>：此方法 <i>不会</i> 计算 image→text 分支，避免白算开销。
     * 需要同时拿到双向融合结果时，请调用 {@link #forwardBoth(Variable, Variable)}。</p>
     *
     * @param inputs {@code inputs[0]}: textFeatures {@code [batch, text_len, hidden_size]}；
     *               {@code inputs[1]}: imageFeatures {@code [batch, num_patches, hidden_size]}
     * @return fusedTextFeatures {@code [batch, text_len, hidden_size]}
     * @see #forwardBoth(Variable, Variable)
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException(
                    "MultiModalFusion需要2个输入: textFeatures和imageFeatures"
            );
        }
        Variable textFeatures = inputs[0];
        Variable imageFeatures = inputs[1];
        return fuseTextWithImage(textFeatures, imageFeatures);
    }

    /**
     * 双向融合前向传播：同时返回 text 和 image 两侧的融合结果。
     *
     * <p>两侧分支彼此独立、参数互不共享，因此返回的两个张量可以同时参与下游 loss，
     * 让两组 CrossModalAttention 的参数都获得梯度。</p>
     *
     * @param textFeatures  文本特征 {@code [batch, text_len, hidden_size]}
     * @param imageFeatures 图像特征 {@code [batch, num_patches, hidden_size]}
     * @return 长度为 2 的数组 {@code [fusedTextFeatures, fusedImageFeatures]}
     */
    public Variable[] forwardBoth(Variable textFeatures, Variable imageFeatures) {
        Variable fusedTextFeatures = fuseTextWithImage(textFeatures, imageFeatures);
        Variable fusedImageFeatures = fuseImageWithText(imageFeatures, textFeatures);
        return new Variable[]{fusedTextFeatures, fusedImageFeatures};
    }

    /**
     * 反向融合：仅执行 <b>image → text</b> 方向的跨模态注意力。
     *
     * <p>适用于图像理解类任务（如看图说话）：以图像 patch 序列作为 Query、文本作为 K/V，
     * 输出形状为 {@code [batch, num_patches, hidden_size]}。</p>
     *
     * @param textFeatures  文本特征 {@code [batch, text_len, hidden_size]}，作为 K/V
     * @param imageFeatures 图像特征 {@code [batch, num_patches, hidden_size]}，作为 Q
     * @return fusedImageFeatures {@code [batch, num_patches, hidden_size]}
     */
    public Variable forwardReverse(Variable textFeatures, Variable imageFeatures) {
        return fuseImageWithText(imageFeatures, textFeatures);
    }
    
    /**
     * 文本特征融合图像信息
     * 
     * 使用Pre-LayerNorm架构:
     * output = input + Dropout(CrossModalAttn(LayerNorm(input), context))
     * 
     * @param textFeatures 文本特征(Query)
     * @param imageFeatures 图像特征(Key/Value)
     * @return 融合后的文本特征
     */
    private Variable fuseTextWithImage(Variable textFeatures, Variable imageFeatures) {
        // Pre-LayerNorm
        Variable normalized = text2ImageNorm.forward(textFeatures);
        
        // 跨模态注意力: text attend to image
        Variable attnOutput = text2ImageAttn.forward(normalized, imageFeatures);
        
        // Dropout
        Variable dropped = text2ImageDropout.forward(attnOutput);
        
        // 残差连接
        Variable output = textFeatures.add(dropped);
        
        return output;
    }
    
    /**
     * 图像特征融合文本信息
     * 
     * 使用Pre-LayerNorm架构:
     * output = input + Dropout(CrossModalAttn(LayerNorm(input), context))
     * 
     * @param imageFeatures 图像特征(Query)
     * @param textFeatures 文本特征(Key/Value)
     * @return 融合后的图像特征
     */
    private Variable fuseImageWithText(Variable imageFeatures, Variable textFeatures) {
        // Pre-LayerNorm
        Variable normalized = image2TextNorm.forward(imageFeatures);
        
        // 跨模态注意力: image attend to text
        Variable attnOutput = image2TextAttn.forward(normalized, textFeatures);
        
        // Dropout
        Variable dropped = image2TextDropout.forward(attnOutput);
        
        // 残差连接
        Variable output = imageFeatures.add(dropped);
        
        return output;
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public CrossModalAttention getText2ImageAttn() {
        return text2ImageAttn;
    }
    
    public CrossModalAttention getImage2TextAttn() {
        return image2TextAttn;
    }
    
    @Override
    public String toString() {
        return String.format(
            "MultiModalFusion{hiddenSize=%d, numHeads=%d, dropout=%.2f}",
            config.getHiddenSize(),
            config.getNumHeads(),
            config.getDropoutRate()
        );
    }
}
