package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.embedding.TokenEmbedding;
import io.leavesfly.tinyai.minimind.model.transformer.MiniMindTransformerLayer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * MiniMind 模型主体结构
 * <p>
 * 架构:
 * 1. Token Embedding 层 - 将 token IDs 转换为向量
 * 2. N 个 Transformer 层 - 堆叠的自注意力层和前馈网络
 * 3. Final LayerNorm - 最终归一化层
 * 4. Language Model Head - 输出层,映射到词汇表
 * <p>
 * 继承自 V2 Module,可作为 Model 的核心组件
 *
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindBlock extends Module {

    /**
     * 模型配置
     */
    private final MiniMindConfig config;

    /**
     * Token 嵌入层
     */
    private final TokenEmbedding tokenEmbedding;

    /**
     * Transformer 层列表
     */
    private final List<MiniMindTransformerLayer> layers;

    /**
     * 最终归一化层
     */
    private final LayerNorm finalNorm;

    /**
     * 语言模型头 (LM Head) - 将隐藏状态映射回词汇表
     */
    private final Linear lmHead;

    /**
     * 是否处于训练模式
     */
    private boolean training = true;

    /**
     * 构造 MiniMindBlock
     *
     * @param config 模型配置
     */
    public MiniMindBlock(MiniMindConfig config) {
        super("MiniMindBlock");
        this.config = config;

        // 1. 创建 Token Embedding 层
        this.tokenEmbedding = new TokenEmbedding(config.getVocabSize(), config.getHiddenSize());
        registerModule("token_embedding", tokenEmbedding);

        // 2. 创建 Transformer 层列表
        this.layers = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
            MiniMindTransformerLayer layer = new MiniMindTransformerLayer(
                "layer_" + i,
                config.getHiddenSize(),
                config.getNumHeads(),
                config.getFfnHiddenSize(),
                config.getMaxSeqLen(),
                config.getDropout(),
                config.getEpsilon(),
                config.getActivationFunction(),
                config.isPreLayerNorm()
            );
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }

        // 3. 创建最终归一化层
        this.finalNorm = new LayerNorm("final_norm", config.getHiddenSize(), config.getEpsilon());
        registerModule("final_norm", finalNorm);

        // 4. 创建 LM Head (Language Model Head)
        // 将隐藏状态 [batch, seq_len, hidden_size] 映射到 [batch, seq_len, vocab_size]
        this.lmHead = new Linear("lm_head", config.getHiddenSize(), config.getVocabSize(), false);
        registerModule("lm_head", lmHead);

        // 初始化参数
        init();
    }

    /**
     * 前向传播（不使用 KV-Cache）
     *
     * @param inputs 输入 Variable 数组,inputs[0] 为 token IDs
     * @return 输出 Variable,形状 [batch_size, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable tokenIds = inputs[0];
        return forwardWithCache(tokenIds, null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播
     *
     * @param tokenIds  Token IDs,形状 [batch_size, seq_len]
     * @param kvCaches  KV-Cache 列表（每层一个）,可为 null
     * @param startPos  起始位置（用于 RoPE 和因果掩码）
     * @return 输出 logits,形状 [batch_size, seq_len, vocab_size]
     */
    public Variable forwardWithCache(Variable tokenIds, List<KVCache> kvCaches, int startPos) {
        // 1. Token Embedding: [batch, seq_len] -> [batch, seq_len, hidden_size]
        Variable x = tokenEmbedding.forward(tokenIds);

        // 2. 通过所有 Transformer 层
        for (int i = 0; i < layers.size(); i++) {
            MiniMindTransformerLayer layer = layers.get(i);
            KVCache kvCache = (kvCaches != null && i < kvCaches.size()) ? kvCaches.get(i) : null;
            x = layer.forwardWithCache(x, kvCache, startPos);
        }

        // 3. 最终归一化
        x = finalNorm.forward(x);

        // 4. LM Head: [batch, seq_len, hidden_size] -> [batch, seq_len, vocab_size]
        Variable logits = lmHead.forward(x);

        return logits;
    }

    /**
     * 生成时的前向传播（使用 KV-Cache 优化）
     * <p>
     * 用于自回归文本生成,每次只处理一个新 token
     *
     * @param tokenId  当前 token ID,形状 [batch_size, 1]
     * @param kvCaches KV-Cache 列表（每层一个）
     * @param position 当前位置
     * @return 输出 logits,形状 [batch_size, 1, vocab_size]
     */
    public Variable forwardGeneration(Variable tokenId, List<KVCache> kvCaches, int position) {
        return forwardWithCache(tokenId, kvCaches, position);
    }

    /**
     * 创建 KV-Cache 列表
     *
     * @param batchSize 批次大小
     * @return KV-Cache 列表
     */
    public List<KVCache> createKVCaches(int batchSize) {
        List<KVCache> kvCaches = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
            KVCache cache = new KVCache(
                batchSize,
                config.getNumHeads(),
                config.getHiddenSize() / config.getNumHeads(),
                config.getMaxSeqLen()
            );
            kvCaches.add(cache);
        }
        return kvCaches;
    }

    /**
     * 清空所有 KV-Cache
     *
     * @param kvCaches KV-Cache 列表
     */
    public void clearKVCaches(List<KVCache> kvCaches) {
        if (kvCaches != null) {
            for (KVCache cache : kvCaches) {
                cache.clear();
            }
        }
    }

    /**
     * 设置训练模式
     *
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
        for (MiniMindTransformerLayer layer : layers) {
            layer.setTraining(training);
        }
    }

    /**
     * 获取模型配置
     */
    public MiniMindConfig getConfig() {
        return config;
    }

    /**
     * 获取 Transformer 层列表
     */
    public List<MiniMindTransformerLayer> getLayers() {
        return layers;
    }

    /**
     * 获取参数数量估算
     *
     * @return 参数数量
     */
    public long estimateParameters() {
        return config.estimateParameters();
    }

    /**
     * 打印模型结构信息
     */
    public void printModelInfo() {
        System.out.println("=== MiniMind Model Structure ===");
        System.out.println("Vocabulary Size: " + config.getVocabSize());
        System.out.println("Max Sequence Length: " + config.getMaxSeqLen());
        System.out.println("Hidden Size: " + config.getHiddenSize());
        System.out.println("Number of Layers: " + config.getNumLayers());
        System.out.println("Number of Heads: " + config.getNumHeads());
        System.out.println("Head Dimension: " + (config.getHiddenSize() / config.getNumHeads()));
        System.out.println("FFN Hidden Size: " + config.getFfnHiddenSize());
        System.out.println("Dropout: " + config.getDropout());
        System.out.println("Activation: " + config.getActivationFunction());
        System.out.println("Estimated Parameters: " + estimateParameters());
        System.out.println("================================");
    }
}
