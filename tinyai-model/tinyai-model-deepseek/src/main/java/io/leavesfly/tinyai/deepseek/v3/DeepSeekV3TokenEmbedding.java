package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.DeepSeekTokenEmbeddingBase;

/**
 * DeepSeek-V3 Token嵌入层
 *
 * 继承统一的 DeepSeekTokenEmbeddingBase 基类，与 R1 共享相同的嵌入架构：
 * 1. Token Embedding - 将 token ID 映射到嵌入向量
 * 2. Dropout - 嵌入层正则化
 *
 * 位置信息由注意力层中的 RoPE 提供，嵌入层不包含学习式位置嵌入。
 *
 * @author leavesfly
 * @version 3.0
 */
public class DeepSeekV3TokenEmbedding extends DeepSeekTokenEmbeddingBase {

    private final DeepSeekV3Config config;

    /**
     * 构造函数
     *
     * @param name   模块名称
     * @param config V3 配置对象
     */
    public DeepSeekV3TokenEmbedding(String name, DeepSeekV3Config config) {
        super(name,
                config.getVocabSize(),
                config.getNEmbd(),
                config.getNPositions(),
                (float) config.getEmbdPdrop(),
                config.getInitializerRange());
        this.config = config;
    }

    /**
     * 获取配置对象
     */
    public DeepSeekV3Config getConfig() {
        return config;
    }


}
