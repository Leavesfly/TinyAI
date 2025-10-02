package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;

/**
 * Qwen3核心块
 * 
 * 实现了完整的Qwen3 Transformer模型的核心部分，包含：
 * 1. 词嵌入层（Token Embedding）
 * 2. 多层Transformer解码器
 * 3. 最终层归一化
 * 
 * 根据TinyAI架构规范，该类继承自Block类，
 * 提供了模型的主要前向传播逻辑。
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Block extends Block {
    
    /**
     * 配置信息
     */
    private Qwen3Config config;
    
    /**
     * 词汇表大小
     */
    private int vocabSize;
    
    /**
     * 隐藏层维度
     */
    private int hiddenSize;
    
    /**
     * Transformer层数
     */
    private int numHiddenLayers;
    
    /**
     * 词嵌入层
     */
    private Embedding embedTokens;
    
    /**
     * Transformer解码器层数组
     */
    private Qwen3DecoderLayer[] decoderLayers;
    
    /**
     * 最终层归一化
     */
    private RMSNorm norm;

    /**
     * 构造Qwen3核心块
     * 
     * @param name 块名称
     * @param config 配置信息
     */
    public Qwen3Block(String name, Qwen3Config config) {
        super(name, Shape.of(-1, -1), Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.vocabSize = config.getVocabSize();
        this.hiddenSize = config.getHiddenSize();
        this.numHiddenLayers = config.getNumHiddenLayers();
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化词嵌入层
            embedTokens = new Embedding(name + "_embed_tokens", vocabSize, hiddenSize);
            addLayer(embedTokens);
            
            // 初始化Transformer解码器层
            decoderLayers = new Qwen3DecoderLayer[numHiddenLayers];
            for (int i = 0; i < numHiddenLayers; i++) {
                decoderLayers[i] = new Qwen3DecoderLayer(name + "_layer_" + i, config, i);
                addLayer(decoderLayers[i]);
            }
            
            // 初始化最终层归一化
            norm = new RMSNorm(name + "_norm", hiddenSize, config.getRmsNormEps());
            addLayer(norm);
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable inputIds = inputs[0];
        
        // 验证输入
        NdArray inputData = inputIds.getValue();
        validateInput(inputData);
        
        // 1. 词嵌入
        Variable hiddenStates = embedTokens.layerForward(inputIds);
        
        // 2. 通过所有Transformer解码器层
        for (int i = 0; i < numHiddenLayers; i++) {
            hiddenStates = decoderLayers[i].layerForward(hiddenStates);
        }
        
        // 3. 最终层归一化
        hiddenStates = norm.layerForward(hiddenStates);
        
        return hiddenStates;
    }
    
    /**
     * 验证输入格式
     * 
     * @param inputData 输入数据
     */
    private void validateInput(NdArray inputData) {
        Shape inputShape = inputData.getShape();
        
        // 输入应该是2D: (batch_size, seq_len) 或 1D: (seq_len)
        if (inputShape.getDimNum() != 1 && inputShape.getDimNum() != 2) {
            throw new IllegalArgumentException(
                "输入维度错误: 期望1D或2D，实际" + inputShape.getDimNum() + "D"
            );
        }
        
        // 检查token ID范围
        float[] flatData = inputData.flatten();
        for (float value : flatData) {
            int tokenId = (int) value;
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException(
                    "Token ID超出范围: " + tokenId + "，期望范围[0, " + (vocabSize - 1) + "]"
                );
            }
        }
    }
    
    /**
     * 获取模型统计信息
     * 
     * @return 模型统计信息字符串
     */
    public String getModelStats() {
        StringBuilder stats = new StringBuilder();
        stats.append("=== Qwen3Block 模型统计 ===\n");
        stats.append("配置信息:\n");
        stats.append("  - 词汇表大小: ").append(vocabSize).append("\n");
        stats.append("  - 隐藏层维度: ").append(hiddenSize).append("\n");
        stats.append("  - Transformer层数: ").append(numHiddenLayers).append("\n");
        stats.append("  - 注意力头数: ").append(config.getNumAttentionHeads()).append("\n");
        stats.append("  - KV头数: ").append(config.getNumKeyValueHeads()).append("\n");
        stats.append("  - 中间层维度: ").append(config.getIntermediateSize()).append("\n");
        stats.append("  - 最大位置编码: ").append(config.getMaxPositionEmbeddings()).append("\n");
        
        // 计算参数量
        long totalParams = getTotalParameters();
        stats.append("参数统计:\n");
        stats.append("  - 总参数量: ").append(formatNumber(totalParams)).append("\n");
        stats.append("  - 嵌入参数: ").append(formatNumber((long) vocabSize * hiddenSize)).append("\n");
        
        return stats.toString();
    }
    
    /**
     * 计算总参数量
     * 
     * @return 总参数量
     */
    private long getTotalParameters() {
        long total = 0;
        
        // 词嵌入参数
        total += (long) vocabSize * hiddenSize;
        
        // 每层的参数量
        long layerParams = calculateLayerParameters();
        total += layerParams * numHiddenLayers;
        
        // 最终层归一化参数
        total += hiddenSize;
        
        return total;
    }
    
    /**
     * 计算单层参数量
     * 
     * @return 单层参数量
     */
    private long calculateLayerParameters() {
        long layerParams = 0;
        
        // 注意力层参数
        layerParams += (long) hiddenSize * hiddenSize; // Q投影
        layerParams += (long) hiddenSize * config.getNumKeyValueHeads() * config.getKvHeadDim(); // K投影  
        layerParams += (long) hiddenSize * config.getNumKeyValueHeads() * config.getKvHeadDim(); // V投影
        layerParams += (long) hiddenSize * hiddenSize; // O投影
        
        // MLP层参数
        layerParams += (long) hiddenSize * config.getIntermediateSize(); // gate_proj
        layerParams += (long) hiddenSize * config.getIntermediateSize(); // up_proj
        layerParams += (long) config.getIntermediateSize() * hiddenSize; // down_proj
        
        // 层归一化参数 (2个RMSNorm)
        layerParams += hiddenSize * 2;
        
        return layerParams;
    }
    
    /**
     * 格式化数字显示
     * 
     * @param number 数字
     * @return 格式化后的字符串
     */
    private String formatNumber(long number) {
        if (number >= 1_000_000_000) {
            return String.format("%.2fB", number / 1_000_000_000.0);
        } else if (number >= 1_000_000) {
            return String.format("%.2fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.2fK", number / 1_000.0);
        } else {
            return String.valueOf(number);
        }
    }
    
    /**
     * 打印模型架构信息
     */
    public void printArchitecture() {
        System.out.println(getModelStats());
        
        System.out.println("架构结构:");
        System.out.println("  Input (token_ids)");
        System.out.println("  └─ EmbedTokens: " + vocabSize + " -> " + hiddenSize);
        
        for (int i = 0; i < numHiddenLayers; i++) {
            System.out.println("  └─ Layer" + i + ":");
            System.out.println("     ├─ RMSNorm");
            System.out.println("     ├─ SelfAttention (" + config.getNumAttentionHeads() + " heads)");
            System.out.println("     ├─ RMSNorm"); 
            System.out.println("     └─ MLP (" + hiddenSize + " -> " + config.getIntermediateSize() + " -> " + hiddenSize + ")");
        }
        
        System.out.println("  └─ FinalNorm");
        System.out.println("  Output: (" + hiddenSize + ",)");
    }
    
    // Getters
    
    /**
     * 获取配置信息
     * 
     * @return 配置信息
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取词嵌入层
     * 
     * @return 词嵌入层
     */
    public Embedding getEmbedTokens() {
        return embedTokens;
    }
    
    /**
     * 获取解码器层数组
     * 
     * @return 解码器层数组
     */
    public Qwen3DecoderLayer[] getDecoderLayers() {
        return decoderLayers;
    }
    
    /**
     * 获取最终层归一化
     * 
     * @return 最终层归一化
     */
    public RMSNorm getNorm() {
        return norm;
    }
    
    /**
     * 获取指定索引的解码器层
     * 
     * @param index 层索引
     * @return 解码器层
     */
    public Qwen3DecoderLayer getLayer(int index) {
        if (index < 0 || index >= numHiddenLayers) {
            throw new IndexOutOfBoundsException("层索引超出范围: " + index);
        }
        return decoderLayers[index];
    }
}