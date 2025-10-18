package io.leavesfly.tinyai.agent.vla.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 语言输入数据模型
 * 封装VLA智能体的语言模态输入
 *
 * @author TinyAI
 */
public class LanguageInput {

    /**
     * 自然语言指令
     */
    private String instruction;

    /**
     * Token ID序列，维度 [seq_len]
     */
    private NdArray tokenIds;

    /**
     * 注意力掩码，维度 [seq_len]
     */
    private NdArray attentionMask;

    /**
     * 文本嵌入向量，维度 [seq_len, 768]
     */
    private NdArray embeddings;

    /**
     * 构造函数 - 仅指令文本
     */
    public LanguageInput(String instruction) {
        this.instruction = instruction;
    }

    /**
     * 构造函数 - 指令 + Token IDs
     */
    public LanguageInput(String instruction, NdArray tokenIds) {
        this.instruction = instruction;
        this.tokenIds = tokenIds;
    }

    /**
     * 完整构造函数
     */
    public LanguageInput(String instruction, NdArray tokenIds, NdArray attentionMask) {
        this.instruction = instruction;
        this.tokenIds = tokenIds;
        this.attentionMask = attentionMask;
    }

    // Getters and Setters
    public String getInstruction() {
        return instruction;
    }

    public void setInstruction(String instruction) {
        this.instruction = instruction;
    }

    public NdArray getTokenIds() {
        return tokenIds;
    }

    public void setTokenIds(NdArray tokenIds) {
        this.tokenIds = tokenIds;
    }

    public NdArray getAttentionMask() {
        return attentionMask;
    }

    public void setAttentionMask(NdArray attentionMask) {
        this.attentionMask = attentionMask;
    }

    public NdArray getEmbeddings() {
        return embeddings;
    }

    public void setEmbeddings(NdArray embeddings) {
        this.embeddings = embeddings;
    }

    @Override
    public String toString() {
        return "LanguageInput{" +
                "instruction='" + instruction + '\'' +
                ", tokenIdsShape=" + (tokenIds != null ? tokenIds.getShape() : "null") +
                ", hasAttentionMask=" + (attentionMask != null) +
                ", hasEmbeddings=" + (embeddings != null) +
                '}';
    }
}
