package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.moe.MoETransformerBlock;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.embedding.TokenEmbedding;
import io.leavesfly.tinyai.minimind.model.transformer.TransformerBlock;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * MiniMind 模型主体结构（统一标准模式和 MoE 模式）
 * <p>
 * 架构:
 * 1. Token Embedding 层 - 将 token IDs 转换为向量
 * 2. N 个 Transformer 层 - 标准 FFN 或 MoE FFN（由配置决定）
 * 3. Final LayerNorm - 最终归一化层
 * 4. Language Model Head - 输出层,映射到词汇表
 * <p>
 * 当 config.isUseMoE() 为 true 时,使用 MoE Transformer 层;
 * 否则使用标准 Transformer 层。
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
     * 标准 Transformer 层列表（非 MoE 模式使用）
     */
    private final List<TransformerBlock> layers;

    /**
     * MoE Transformer 层列表（MoE 模式使用）
     */
    private final List<MoETransformerBlock> moeLayers;

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
     * 累积的负载均衡损失（MoE 模式使用）
     */
    private float totalBalanceLoss = 0.0f;

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

        // 2. 根据配置创建标准或 MoE Transformer 层
        if (config.isUseMoE()) {
            this.layers = null;
            this.moeLayers = new ArrayList<>();
            for (int i = 0; i < config.getNumLayers(); i++) {
                MoETransformerBlock layer = new MoETransformerBlock(
                    "moe_layer_" + i, config);
                moeLayers.add(layer);
                registerModule("moe_layer_" + i, layer);
            }
        } else {
            this.moeLayers = null;
            this.layers = new ArrayList<>();
            for (int i = 0; i < config.getNumLayers(); i++) {
                TransformerBlock layer = new TransformerBlock(
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
        }

        // 3. 创建最终归一化层
        this.finalNorm = new LayerNorm("final_norm", config.getHiddenSize(), config.getEpsilon());
        registerModule("final_norm", finalNorm);

        // 4. 创建 LM Head
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
        return forwardWithMoEOutput(tokenIds, null, 0).getOutput();
    }

    /**
     * 带 KV-Cache 的前向传播（返回 MoEOutput，包含负载均衡损失）
     *
     * @param tokenIds  Token IDs,形状 [batch_size, seq_len]
     * @param kvCaches  KV-Cache 列表（每层一个）,可为 null
     * @param startPos  起始位置（用于 RoPE 和因果掩码）
     * @return MoE 输出结果（标准模式下 balanceLoss 为 0）
     */
    public MoEOutput forwardWithMoEOutput(Variable tokenIds, List<KVCache> kvCaches, int startPos) {
        totalBalanceLoss = 0.0f;

        // 1. Token Embedding
        Variable x = tokenEmbedding.forward(tokenIds);

        // 2. 通过所有 Transformer 层（添加防御性空值检查）
        if (config.isUseMoE()) {
            if (moeLayers == null) {
                throw new IllegalStateException("MoE模式配置为true，但moeLayers为null");
            }
            for (int i = 0; i < moeLayers.size(); i++) {
                MoETransformerBlock layer = moeLayers.get(i);
                KVCache kvCache = (kvCaches != null && i < kvCaches.size()) ? kvCaches.get(i) : null;
                MoETransformerBlock.LayerOutput layerOutput =
                    layer.forwardWithCache(x, kvCache, startPos);
                x = layerOutput.getOutput();
                totalBalanceLoss += layerOutput.getBalanceLoss();
            }
        } else {
            if (layers == null) {
                throw new IllegalStateException("标准模式配置为false，但layers为null");
            }
            for (int i = 0; i < layers.size(); i++) {
                TransformerBlock layer = layers.get(i);
                KVCache kvCache = (kvCaches != null && i < kvCaches.size()) ? kvCaches.get(i) : null;
                x = layer.forwardWithCache(x, kvCache, startPos);
            }
        }

        // 3. 最终归一化
        x = finalNorm.forward(x);

        // 4. LM Head
        Variable logits = lmHead.forward(x);

        return new MoEOutput(logits, totalBalanceLoss);
    }

    /**
     * 带 KV-Cache 的前向传播（仅返回 logits，兼容旧接口）
     */
    public Variable forwardWithCache(Variable tokenIds, List<KVCache> kvCaches, int startPos) {
        return forwardWithMoEOutput(tokenIds, kvCaches, startPos).getOutput();
    }

    /**
     * 生成时的前向传播（使用 KV-Cache 优化）
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
     */
    public void setTraining(boolean training) {
        this.training = training;
        if (config.isUseMoE()) {
            for (MoETransformerBlock layer : moeLayers) {
                layer.setTraining(training);
            }
        } else {
            for (TransformerBlock layer : layers) {
                layer.setTraining(training);
            }
        }
    }

    /**
     * 获取模型配置
     */
    public MiniMindConfig getConfig() {
        return config;
    }

    /**
     * 获取标准 Transformer 层列表（非 MoE 模式）
     */
    public List<TransformerBlock> getLayers() {
        return layers;
    }

    /**
     * 获取 MoE Transformer 层列表（MoE 模式）
     */
    public List<MoETransformerBlock> getMoeLayers() {
        return moeLayers;
    }

    /**
     * 获取总的负载均衡损失（MoE 模式）
     */
    public float getTotalBalanceLoss() {
        return totalBalanceLoss;
    }

    /**
     * 获取专家使用统计（MoE 模式）
     */
    public String getExpertUsageStats() {
        if (!config.isUseMoE()) {
            return "Not a MoE model";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Expert Usage Statistics:\n");
        for (int i = 0; i < moeLayers.size(); i++) {
            sb.append(String.format("Layer %d: %s\n", i, moeLayers.get(i).getUsageStats()));
        }
        return sb.toString();
    }

    /**
     * 重置所有层的统计信息（MoE 模式）
     */
    public void resetStats() {
        if (config.isUseMoE()) {
            for (MoETransformerBlock layer : moeLayers) {
                layer.resetStats();
            }
        }
        totalBalanceLoss = 0.0f;
    }

    /**
     * 获取参数数量估算
     */
    public long estimateParameters() {
        return config.estimateParameters();
    }

    /**
     * 打印模型结构信息
     */
    public void printModelInfo() {
        System.out.println("=== MiniMind Model Structure ===");
        System.out.println("Mode: " + (config.isUseMoE() ? "MoE" : "Standard"));
        System.out.println("Vocabulary Size: " + config.getVocabSize());
        System.out.println("Max Sequence Length: " + config.getMaxSeqLen());
        System.out.println("Hidden Size: " + config.getHiddenSize());
        System.out.println("Number of Layers: " + config.getNumLayers());
        System.out.println("Number of Heads: " + config.getNumHeads());
        System.out.println("Head Dimension: " + (config.getHiddenSize() / config.getNumHeads()));
        System.out.println("FFN Hidden Size: " + config.getFfnHiddenSize());
        if (config.isUseMoE()) {
            System.out.println("Num Experts: " + config.getNumExperts());
            System.out.println("Experts Per Token: " + config.getNumExpertsPerToken());
        }
        System.out.println("Dropout: " + config.getDropout());
        System.out.println("Activation: " + config.getActivationFunction());
        System.out.println("Estimated Parameters: " + estimateParameters());
        System.out.println("================================");
    }

    /**
     * 打印完整的网络层级架构
     * <p>
     * 以树状结构展示每一层的名称、类型和关键维度参数
     */
    public void printArchitecture() {
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumHeads();
        int headDim = hiddenSize / numHeads;
        int ffnHiddenSize = config.getFfnHiddenSize();
        int vocabSize = config.getVocabSize();
        int numLayers = config.getNumLayers();
        boolean useMoE = config.isUseMoE();

        String separator = "=".repeat(60);
        System.out.println(separator);
        System.out.println("  MiniMind Network Architecture");
        System.out.println("  Mode: " + (useMoE ? "MoE (Mixture of Experts)" : "Standard"));
        System.out.println(separator);

        // Token Embedding
        System.out.println("MiniMindBlock");
        System.out.printf("  ├── TokenEmbedding          [vocab=%d, hidden=%d]%n",
            vocabSize, hiddenSize);

        // Transformer 层（后面还有 final_norm 和 lm_head，所以所有层都用 ├──）
        String layerPrefix = useMoE ? "MoETransformerLayer" : "TransformerLayer";
        for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
            String layerConnector = "  ├──";
            String childConnector = "  │    ";

            System.out.printf("%s %s[%d]%n", layerConnector, layerPrefix, layerIndex);
            System.out.printf("%s  ├── LayerNorm(norm1)       [hidden=%d]%n",
                childConnector, hiddenSize);
            System.out.printf("%s  ├── MultiHeadAttention     [hidden=%d, heads=%d, head_dim=%d]%n",
                childConnector, hiddenSize, numHeads, headDim);
            System.out.printf("%s  │     ├── Q_proj           [%d → %d]%n",
                childConnector, hiddenSize, hiddenSize);
            System.out.printf("%s  │     ├── K_proj           [%d → %d]%n",
                childConnector, hiddenSize, hiddenSize);
            System.out.printf("%s  │     ├── V_proj           [%d → %d]%n",
                childConnector, hiddenSize, hiddenSize);
            System.out.printf("%s  │     └── O_proj           [%d → %d]%n",
                childConnector, hiddenSize, hiddenSize);
            System.out.printf("%s  ├── LayerNorm(norm2)       [hidden=%d]%n",
                childConnector, hiddenSize);

            if (useMoE) {
                int numExperts = config.getNumExperts();
                int numExpertsPerToken = config.getNumExpertsPerToken();
                System.out.printf("%s  └── MoEBlock              [experts=%d, top_k=%d]%n",
                    childConnector, numExperts, numExpertsPerToken);
                System.out.printf("%s        ├── Router           [hidden=%d → %d experts]%n",
                    childConnector, hiddenSize, numExperts);
                for (int expertIndex = 0; expertIndex < numExperts; expertIndex++) {
                    boolean isLastExpert = (expertIndex == numExperts - 1);
                    String expertConnector = isLastExpert ? "└──" : "├──";
                    System.out.printf("%s        %s Expert[%d]       [%d → %d → %d]%n",
                        childConnector, expertConnector, expertIndex,
                        hiddenSize, ffnHiddenSize, hiddenSize);
                }
            } else {
                System.out.printf("%s  └── FeedForward           [%d → %d → %d]%n",
                    childConnector, hiddenSize, ffnHiddenSize, hiddenSize);
            }
        }

        // Final LayerNorm
        System.out.printf("  ├── LayerNorm(final_norm)  [hidden=%d]%n", hiddenSize);

        // LM Head
        System.out.printf("  └── Linear(lm_head)        [%d → %d]%n", hiddenSize, vocabSize);

        System.out.println(separator);
        System.out.printf("  Total Estimated Parameters: %,d%n", estimateParameters());
        System.out.println(separator);
    }

    /**
     * MoE 输出结果（标准模式下 balanceLoss 为 0）
     */
    public static class MoEOutput {
        private final Variable output;
        private final float balanceLoss;

        public MoEOutput(Variable output, float balanceLoss) {
            this.output = output;
            this.balanceLoss = balanceLoss;
        }

        public Variable getOutput() {
            return output;
        }

        public float getBalanceLoss() {
            return balanceLoss;
        }

        @Override
        public String toString() {
            return String.format("MoEOutput(shape=%s, balance_loss=%.6f)",
                output.getShape(), balanceLoss);
        }
    }
}