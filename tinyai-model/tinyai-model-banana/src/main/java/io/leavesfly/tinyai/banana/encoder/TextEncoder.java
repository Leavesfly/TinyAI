package io.leavesfly.tinyai.banana.encoder;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.embedding.Embedding;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.transformer.PositionalEncoding;
import io.leavesfly.tinyai.nnet.layer.transformer.TransformerEncoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 文本编码器
 * 
 * 使用Transformer Encoder架构处理文本输入:
 * 1. Token嵌入层 - 将token IDs转换为向量
 * 2. 位置编码 - 添加位置信息
 * 3. Transformer编码器层堆叠 - 提取文本特征
 * 
 * 输入: 文本token IDs [batch, seq_len]
 * 输出: 文本特征向量 [batch, seq_len, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TextEncoder extends Module {
    
    private final BananaConfig config;
    
    // Token嵌入层
    private final Embedding tokenEmbedding;
    
    // 位置编码
    private final PositionalEncoding positionalEncoding;
    
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
    public TextEncoder(String name, BananaConfig config) {
        super(name);
        if (config == null) {
            throw new IllegalArgumentException("config不能为null");
        }
        this.config = config;
        
        // 1. 初始化Token嵌入层
        this.tokenEmbedding = new Embedding(
            name + "_token_emb",
            config.getVocabSize(),
            config.getHiddenSize()
        );
        registerModule("token_emb", tokenEmbedding);
        
        // 2. 初始化位置编码
        this.positionalEncoding = new PositionalEncoding(
            name + "_pos_enc",
            config.getHiddenSize(),
            config.getMaxTextLength()
        );
        registerModule("pos_enc", positionalEncoding);
        
        // 3. 初始化嵌入Dropout
        this.embeddingDropout = new Dropout(
            name + "_emb_dropout",
            config.getEmbeddingDropout()
        );
        registerModule("emb_dropout", embeddingDropout);
        
        // 4. 初始化Transformer编码器层
        this.encoderLayers = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
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
     * @param inputs inputs[0]为文本token IDs [batch, seq_len]
     * @return 文本特征向量 [batch, seq_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("TextEncoder需要输入token IDs");
        }
        
        Variable tokenIds = inputs[0];
        validateInput(tokenIds);
        
        // 1. Token嵌入: [batch, seq_len] -> [batch, seq_len, hidden_size]
        Variable embeddings = tokenEmbedding.forward(tokenIds);
        
        // 2. 添加位置编码
        Variable posEncodings = positionalEncoding.forward(embeddings);
        Variable x = embeddings.add(posEncodings);
        
        // 3. 应用嵌入Dropout
        x = embeddingDropout.forward(x);
        
        // 4. 通过Transformer编码器层
        for (TransformerEncoderLayer layer : encoderLayers) {
            x = layer.forward(x);
        }
        
        return x;
    }
    
    /**
     * 验证输入有效性：形状、序列长度以及 token id 取值范围。
     *
     * <p>Token id 越界时，{@link Embedding} 的查表会静默读到错误数据或直接抛 {@link ArrayIndexOutOfBoundsException}，
     * 出错信息对用户非常不友好。此处做一次 O(batch*seq) 的扫描，在训练端给出明确报错。</p>
     */
    private void validateInput(Variable tokenIds) {
        if (tokenIds == null) {
            throw new IllegalArgumentException("tokenIds不能为null");
        }

        int[] shape = tokenIds.getValue().getShape().getShapeDims();
        if (shape.length != 2) {
            throw new IllegalArgumentException(
                "tokenIds必须是2维 [batch, seq_len], 当前shape: " +
                java.util.Arrays.toString(shape)
            );
        }

        int seqLen = shape[1];
        if (seqLen > config.getMaxTextLength()) {
            throw new IllegalArgumentException(
                "序列长度 " + seqLen + " 超过最大长度 " + config.getMaxTextLength()
            );
        }

        // Token id 范围校验：要求 id ∈ [0, vocabSize)
        int vocabSize = config.getVocabSize();
        float[] flat = tokenIds.getValue().getArray();
        for (int i = 0; i < flat.length; i++) {
            float f = flat[i];
            int id = (int) f;
            if (f != id || id < 0 || id >= vocabSize) {
                int row = i / seqLen;
                int col = i % seqLen;
                throw new IllegalArgumentException(String.format(
                    "tokenIds[%d, %d]=%s 越界或非整数；要求 id ∈ [0, vocabSize=%d)。" +
                        "请检查 tokenizer 是否与 BananaConfig.vocabSize 对齐。",
                    row, col, f, vocabSize));
            }
        }
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return encoderLayers.size();
    }
    
    public List<TransformerEncoderLayer> getEncoderLayers() {
        return encoderLayers;
    }
    
    @Override
    public String toString() {
        return String.format(
            "TextEncoder{numLayers=%d, hiddenSize=%d, numHeads=%d, vocabSize=%d}",
            config.getNumLayers(),
            config.getHiddenSize(),
            config.getNumHeads(),
            config.getVocabSize()
        );
    }
}
