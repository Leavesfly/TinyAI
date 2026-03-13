package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTokenEmbeddingBase;

/**
 * DeepSeek-R1 Token嵌入层
 *
 * R1 与 V3 使用完全相同的嵌入架构（对标论文 arXiv:2501.12948）：
 * 1. Token Embedding - 将 token ID 映射到嵌入向量
 * 2. Dropout - 嵌入层正则化
 *
 * 位置信息由注意力层中的 RoPE 提供，嵌入层不包含学习式位置嵌入。
 *
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1TokenEmbedding extends DeepSeekTokenEmbeddingBase {

    private final DeepSeekR1Config config;

    /**
     * 构造函数
     *
     * @param name   模块名称
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
