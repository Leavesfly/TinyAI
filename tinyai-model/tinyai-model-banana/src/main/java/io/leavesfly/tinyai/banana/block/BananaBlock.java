package io.leavesfly.tinyai.banana.block;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.config.TaskType;
import io.leavesfly.tinyai.banana.decoder.ImageDecoder;
import io.leavesfly.tinyai.banana.encoder.ImageEncoder;
import io.leavesfly.tinyai.banana.encoder.TextEncoder;
import io.leavesfly.tinyai.banana.fusion.CrossModalAttention;
import io.leavesfly.tinyai.banana.fusion.MultiModalFusion;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

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

    // ========= 文本 → 图像 空间扩展相关 =========
    /**
     * 可学习的 patch queries，形状 {@code [1, numPatches, hiddenSize]}。
     * <p>
     * 用于 text-to-image / image-editing / multi-image-composition 路径，通过
     * cross-attention 从文本/图像上下文中"读"出每个 patch 位置独立的特征，
     * 避免旧版 {@code mean + expand} 导致的空间同质化问题。
     * </p>
     */
    private Parameter patchQueries;

    /**
     * 专用于 patch queries 的跨模态注意力层：Q=patchQueries, K/V=上下文（文本或图像）。
     */
    private CrossModalAttention textToPatchAttn;

    /** 对 patch queries 做 LayerNorm，提升训练稳定性。 */
    private LayerNorm patchQueryNorm;
    
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
            config.getLayerNormEpsilon()
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

        // 初始化可学习的 patch queries（空间扩展用）
        // 形状 [1, numPatches, hiddenSize]，可通过 expand 广播到任意 batch_size。
        // 用小方差正态分布初始化，避免训练初期位置同质化。
        NdArray initQueries = NdArray.randn(
                Shape.of(1, config.getNumPatches(), config.getHiddenSize())
        ).mulNum(config.getInitializerRange());
        patchQueries = registerParameter("patch_queries", new Parameter(initQueries));

        // 初始化 text-to-patch 跨模态注意力
        textToPatchAttn = new CrossModalAttention(
                name + "_text2patch_attn",
                config.getHiddenSize(),
                config.getNumHeads(),
                config.getDropoutRate()
        );
        registerModule("text2patch_attn", textToPatchAttn);

        // 初始化 patch queries 的 LayerNorm
        patchQueryNorm = new LayerNorm(
                name + "_patch_query_ln",
                config.getHiddenSize(),
                config.getLayerNormEpsilon()
        );
        registerModule("patch_query_ln", patchQueryNorm);
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
        validateTextTokens(input);

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
        validateTextTokens(textTokenIds);
        return textEncoder.forward(textTokenIds);
    }

    /**
     * 图像编码前向传播
     *
     * @param imagePixels 图像像素 [batch, channels, height, width]
     * @return 图像特征 [batch, num_patches, hidden_size]
     */
    public Variable forwardImage(Variable imagePixels) {
        validateImagePixels(imagePixels);
        return imageEncoder.forward(imagePixels);
    }
    
    /**
     * 多模态融合前向传播，按 {@link TaskType} 做方向路由。
     *
     * <p>路由规则：</p>
     * <ul>
     *   <li>{@link TaskType#IMAGE_UNDERSTANDING}：使用 <b>image→text</b> 方向融合
     *       （以图像 patch 为 Query、文本为 K/V），返回形状 {@code [batch, num_patches, hidden_size]}；</li>
     *   <li>其它所有任务（{@link TaskType#TEXT_TO_IMAGE}、{@link TaskType#IMAGE_EDITING}、
     *       {@link TaskType#MULTI_IMAGE_COMPOSITION}、{@link TaskType#GENERAL_MULTIMODAL}、{@code null}）：
     *       使用 <b>text→image</b> 方向融合，返回形状 {@code [batch, text_len, hidden_size]}。</li>
     * </ul>
     *
     * <p>当 {@code fusionLayer == null}（即 config 关闭了 cross-modal）时退化为直接 LayerNorm
     * 文本特征，与历史行为保持一致。</p>
     *
     * @param textFeatures  文本特征 {@code [batch, text_len, hidden_size]}
     * @param imageFeatures 图像特征 {@code [batch, num_patches, hidden_size]}
     * @param taskType      任务类型；{@code null} 视为 {@link TaskType#GENERAL_MULTIMODAL}
     * @return 融合后的特征（形状见路由规则），<b>不做输出投影</b>
     */
    public Variable forwardMultiModal(Variable textFeatures,
                                     Variable imageFeatures,
                                     TaskType taskType) {
        if (fusionLayer == null) {
            // 未启用跨模态，退回文本路径
            return finalLayerNorm.forward(textFeatures);
        }

        Variable fused;
        if (taskType == TaskType.IMAGE_UNDERSTANDING) {
            // 图像理解：image→text 方向，以图像 patch 序列为主
            fused = fusionLayer.forwardReverse(textFeatures, imageFeatures);
        } else {
            // 默认及图像生成类任务：text→image 方向，以文本序列为主
            fused = fusionLayer.forward(textFeatures, imageFeatures);
        }
        return finalLayerNorm.forward(fused);
    }

    /**
     * 多模态融合前向传播（带输出投影）。
     *
     * <p>用于生成类任务，将融合特征投影到词汇表空间。输出形状由 {@code taskType} 决定：</p>
     * <ul>
     *   <li>{@link TaskType#IMAGE_UNDERSTANDING}：{@code [batch, num_patches, vocab_size]}；</li>
     *   <li>其它任务：{@code [batch, text_len, vocab_size]}。</li>
     * </ul>
     *
     * @param textFeatures  文本特征 {@code [batch, text_len, hidden_size]}
     * @param imageFeatures 图像特征 {@code [batch, num_patches, hidden_size]}
     * @param taskType      任务类型
     * @return 投影后的输出（形状见上）
     */
    public Variable forwardMultiModalWithProjection(Variable textFeatures,
                                                    Variable imageFeatures,
                                                    TaskType taskType) {
        // 先进行融合（已按 taskType 路由）
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
        validateTextTokens(textTokenIds);
        if (imagePixels != null) {
            validateImagePixels(imagePixels);
        }

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
     * 校验文本 token ids 输入：必须非 null、形状为 {@code [batch, seq_len]}，且 {@code seq_len <= maxTextLength}。
     *
     * <p>Token id 的取值范围会在 {@link TextEncoder} 内部做细粒度校验，这里只做结构校验以避免重复扫描。</p>
     */
    private void validateTextTokens(Variable tokenIds) {
        if (tokenIds == null) {
            throw new IllegalArgumentException("文本 tokenIds 不能为 null");
        }
        int[] shape = tokenIds.getValue().getShape().getShapeDims();
        if (shape.length != 2) {
            throw new IllegalArgumentException(
                    "文本 tokenIds 必须是 2 维 [batch, seq_len], 当前 shape: " +
                            java.util.Arrays.toString(shape));
        }
        int seqLen = shape[1];
        if (seqLen > config.getMaxTextLength()) {
            throw new IllegalArgumentException(String.format(
                    "文本序列长度 %d 超过配置上限 maxTextLength=%d", seqLen, config.getMaxTextLength()));
        }
    }

    /**
     * 校验图像像素输入：必须非 null、形状为 {@code [batch, C, H, W]}，且 {@code C/H/W} 与 config 对齐。
     */
    private void validateImagePixels(Variable imagePixels) {
        if (imagePixels == null) {
            throw new IllegalArgumentException("imagePixels 不能为 null");
        }
        int[] shape = imagePixels.getValue().getShape().getShapeDims();
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                    "imagePixels 必须是 4 维 [batch, channels, height, width], 当前 shape: " +
                            java.util.Arrays.toString(shape));
        }
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int expectedC = config.getImageChannels();
        int expectedHW = config.getImageSize();
        if (channels != expectedC || height != expectedHW || width != expectedHW) {
            throw new IllegalArgumentException(String.format(
                    "imagePixels 形状与 config 不匹配：期望 [B, %d, %d, %d], 实际 [B, %d, %d, %d]",
                    expectedC, expectedHW, expectedHW, channels, height, width));
        }
    }

    /**
     * @deprecated 请改用更明确的 {@link #validateTextTokens(Variable)} 或 {@link #validateImagePixels(Variable)}。
     *             保留仅为向后兼容，语义上等价于文本 token 校验。
     */
    @Deprecated
    private void validateInput(Variable input) {
        validateTextTokens(input);
    }
    
    /**
     * 文本到图像生成。
     *
     * <p>关键设计：</p>
     * <ul>
     *   <li>不再对文本特征做 <code>mean + expand</code> 造成空间同质化；</li>
     *   <li>使用 <b>learnable patch queries</b>（形状 {@code [1, numPatches, hidden]}）
     *       通过 cross-attention 从文本上下文里"读"出每个 patch 位置独立的特征；</li>
     *   <li>decoder 的 cross-attn context 使用 <b>文本特征</b>（而非和 query 相同的张量），
     *       让 decoder 的 cross-attn 真正发挥作用。</li>
     * </ul>
     *
     * @param textTokenIds 文本描述 token IDs，形状 {@code [batch, text_len]}
     * @return 生成的图像，形状 {@code [batch, 3, image_size, image_size]}
     */
    public Variable textToImage(Variable textTokenIds) {
        validateTextTokens(textTokenIds);

        // 1. 编码文本：[batch, text_len, hidden]
        Variable textFeatures = forwardText(textTokenIds);

        // 2. 可选：对文本特征做一次自注意力增强（MultiModalFusion self-self）
        //    这里保留自增强语义，但不会被当作"空间"特征使用
        Variable textContext = (fusionLayer != null)
                ? fusionLayer.forward(textFeatures, textFeatures)
                : textFeatures;

        // 3. 用 learnable patch queries + cross-attention 构造空间差异化的 patch 特征
        //    queries: [batch, numPatches, hidden], context: textContext
        Variable patchFeatures = buildPatchFeaturesFromContext(textTokenIds, textContext);

        // 4. 解码为图像；cross-attn 的上下文使用文本特征，避免冗余退化
        return imageDecoder.forward(patchFeatures, textContext);
    }

    /**
     * 基于可学习 patch queries + cross-attention 从给定上下文中构造 patch 级别特征。
     *
     * <p>流程：</p>
     * <ol>
     *   <li>将 {@link #patchQueries}（{@code [1, numPatches, hidden]}）广播到 batch 维；</li>
     *   <li>先做 LayerNorm 稳定训练；</li>
     *   <li>以 queries 作为 Q、以 context 作为 K/V 做 cross-attention；</li>
     *   <li>返回 {@code [batch, numPatches, hidden]}，空间位置之间是差异化的。</li>
     * </ol>
     *
     * @param batchRef 仅用于读取 batch_size 的参考张量（通常是 textTokenIds 或 imageFeatures）
     * @param context  cross-attention 的 K/V 上下文，形状 {@code [batch, ctx_len, hidden]}
     * @return patch 级别特征 {@code [batch, numPatches, hidden]}
     */
    private Variable buildPatchFeaturesFromContext(Variable batchRef, Variable context) {
        int[] refShape = batchRef.getValue().getShape().getShapeDims();
        int batchSize = refShape[0];
        int numPatches = config.getNumPatches();
        int hiddenSize = config.getHiddenSize();

        // 广播 patch queries 到 batch 维：[1, N, H] -> [B, N, H]
        // 注意：必须直接在 Parameter 上调用 expand（Parameter extends Variable），
        // 而不能用 new Variable(patchQueries.getValue()).expand(...)——后者会切断
        // patchQueries 与计算图的连接，导致反向传播时梯度无法回传到这个可学习参数。
        Variable queries = patchQueries.expand(Shape.of(batchSize, numPatches, hiddenSize));
        queries.setName("patch_queries_expanded");

        // LayerNorm 稳定
        Variable normalizedQueries = patchQueryNorm.forward(queries);

        // cross-attention: Q=patchQueries, K/V=context
        return textToPatchAttn.forward(normalizedQueries, context);
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
        validateImagePixels(imagePixels);
        validateTextTokens(editInstructions);

        // 1. 编码原始图像：[batch, numPatches, hidden]
        Variable imageFeatures = forwardImage(imagePixels);

        // 2. 编码编辑指令：[batch, text_len, hidden]
        Variable textFeatures = forwardText(editInstructions);

        // 3. 跨模态融合：让文本指令注意到图像特征，得到"编辑意图"融合特征
        //    结果形状 [batch, text_len, hidden]
        Variable editIntent = (fusionLayer != null)
                ? fusionLayer.forward(textFeatures, imageFeatures)
                : textFeatures;

        // 4. 用 learnable patch queries + cross-attention 构造空间差异化的 patch 特征
        //    这里 context 选择原始图像特征（已是 patch 粒度），让 queries 从图像空间读信息
        Variable patchFeatures = buildPatchFeaturesFromContext(imagePixels, imageFeatures);

        // 5. decoder 的 cross-attn 上下文使用"编辑意图" editIntent，让解码过程受文本引导
        return imageDecoder.forward(patchFeatures, editIntent);
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
        validateImagePixels(imagePixels);
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
        validateImagePixels(images);
        validateTextTokens(compositionInstructions);

        // 1. 编码所有图像（作为单个 batch 处理）
        // images: [num_images, channels, height, width]
        // allImageFeatures: [num_images, num_patches, hidden_size]
        Variable allImageFeatures = forwardImage(images);

        // 2. 聚合多图像特征：在图像维度上平均池化
        // [num_images, num_patches, hidden] -> [1, num_patches, hidden]
        Variable aggregatedFeatures = allImageFeatures.mean(0, true);

        // 3. 编码组合指令：[batch, text_len, hidden]
        Variable textFeatures = forwardText(compositionInstructions);

        // 4. 跨模态融合：组合指令注意到聚合的图像特征，得到"组合意图"
        Variable compositionIntent = (fusionLayer != null)
                ? fusionLayer.forward(textFeatures, aggregatedFeatures)
                : textFeatures;

        // 5. 用 learnable patch queries + cross-attention 构造空间差异化的 patch 特征
        //    这里 context 选择聚合图像特征（已是 patch 粒度），保留空间结构
        Variable patchFeatures = buildPatchFeaturesFromContext(compositionInstructions, aggregatedFeatures);

        // 6. decoder 的 cross-attn 上下文使用组合意图 compositionIntent
        return imageDecoder.forward(patchFeatures, compositionIntent);
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
