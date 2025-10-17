package io.leavesfly.tinyai.agent.vla.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import java.util.HashMap;
import java.util.Map;

/**
 * VLA统一状态表示
 * 封装视觉、语言、本体感知三种模态的完整状态
 * 
 * @author TinyAI
 */
public class VLAState {
    
    /** 视觉模态输入 */
    private VisionInput visionInput;
    
    /** 语言模态输入 */
    private LanguageInput languageInput;
    
    /** 本体感知输入 */
    private ProprioceptionInput proprioceptionInput;
    
    /** 融合后的多模态特征，维度 [total_seq_len, hidden_dim] */
    private NdArray fusedFeatures;
    
    /** 各模态注意力权重 */
    private Map<String, NdArray> attentionWeights;
    
    /** 状态时间戳 */
    private long timestamp;
    
    /**
     * 构造函数 - 完整三模态输入
     */
    public VLAState(VisionInput visionInput, LanguageInput languageInput, 
                    ProprioceptionInput proprioceptionInput) {
        this.visionInput = visionInput;
        this.languageInput = languageInput;
        this.proprioceptionInput = proprioceptionInput;
        this.attentionWeights = new HashMap<>();
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 构造函数 - 视觉 + 语言
     */
    public VLAState(VisionInput visionInput, LanguageInput languageInput) {
        this.visionInput = visionInput;
        this.languageInput = languageInput;
        this.attentionWeights = new HashMap<>();
        this.timestamp = System.currentTimeMillis();
    }
    
    // Getters and Setters
    public VisionInput getVisionInput() {
        return visionInput;
    }
    
    public void setVisionInput(VisionInput visionInput) {
        this.visionInput = visionInput;
    }
    
    public LanguageInput getLanguageInput() {
        return languageInput;
    }
    
    public void setLanguageInput(LanguageInput languageInput) {
        this.languageInput = languageInput;
    }
    
    public ProprioceptionInput getProprioceptionInput() {
        return proprioceptionInput;
    }
    
    public void setProprioceptionInput(ProprioceptionInput proprioceptionInput) {
        this.proprioceptionInput = proprioceptionInput;
    }
    
    public NdArray getFusedFeatures() {
        return fusedFeatures;
    }
    
    public void setFusedFeatures(NdArray fusedFeatures) {
        this.fusedFeatures = fusedFeatures;
    }
    
    public Map<String, NdArray> getAttentionWeights() {
        return attentionWeights;
    }
    
    public void setAttentionWeights(Map<String, NdArray> attentionWeights) {
        this.attentionWeights = attentionWeights;
    }
    
    public void addAttentionWeight(String modalityName, NdArray weight) {
        this.attentionWeights.put(modalityName, weight);
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    @Override
    public String toString() {
        return "VLAState{" +
                "hasVision=" + (visionInput != null) +
                ", hasLanguage=" + (languageInput != null) +
                ", hasProprioception=" + (proprioceptionInput != null) +
                ", hasFusedFeatures=" + (fusedFeatures != null) +
                ", attentionWeightsCount=" + attentionWeights.size() +
                ", timestamp=" + timestamp +
                '}';
    }
}
