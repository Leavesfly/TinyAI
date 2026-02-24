package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.deepseek.DeepSeekTokenEmbeddingBase;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * DeepSeek-R1 Token嵌入层
 * 
 * 负责将输入的token ID序列转换为稠密的向量表示，包括：
 * 1. Token嵌入 - 词汇表中每个token的向量表示
 * 2. 位置嵌入 - 序列中每个位置的向量表示
 * 3. Dropout - 嵌入层的正则化
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1TokenEmbedding extends DeepSeekTokenEmbeddingBase {
    
    private final DeepSeekR1Config config;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象
     */
    public DeepSeekR1TokenEmbedding(String name, DeepSeekR1Config config) {
        super(name,
                config.getVocabSize(),
                config.getNEmbd(),
                config.getNPositions(),
                (float) config.getEmbdPdrop(),
                config.getInitializerRange());
        this.config = config;
    }
    
    /**
     * 获取R1配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
}
