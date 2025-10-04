package io.leavesfly.tinyai.agent.research;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 研究洞察
 * 记录研究过程中产生的重要洞察和发现
 * 
 * @author 山泽
 */
public class ResearchInsight {
    /** 洞察内容 */
    private String content;
    
    /** 洞察类型 */
    private String insightType;
    
    /** 置信度 (0.0-1.0) */
    private double confidence;
    
    /** 支持证据 */
    private List<String> supportingEvidence;
    
    /** 含义和影响 */
    private List<String> implications;
    
    /** 创建时间戳 */
    private LocalDateTime timestamp;
    
    /**
     * 完整构造函数
     */
    public ResearchInsight(String content, String insightType, double confidence) {
        this.content = content;
        this.insightType = insightType;
        this.confidence = validateConfidence(confidence);
        this.supportingEvidence = new ArrayList<>();
        this.implications = new ArrayList<>();
        this.timestamp = LocalDateTime.now();
    }
    
    /**
     * 简化构造函数
     */
    public ResearchInsight(String content, String insightType) {
        this(content, insightType, 0.7);
    }
    
    /**
     * 验证置信度范围
     */
    private double validateConfidence(double confidence) {
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("置信度必须在0.0-1.0范围内，当前值: " + confidence);
        }
        return confidence;
    }
    
    // Getter和Setter方法
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public String getInsightType() {
        return insightType;
    }
    
    public void setInsightType(String insightType) {
        this.insightType = insightType;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = validateConfidence(confidence);
    }
    
    public List<String> getSupportingEvidence() {
        return supportingEvidence;
    }
    
    public void setSupportingEvidence(List<String> supportingEvidence) {
        this.supportingEvidence = supportingEvidence != null ? supportingEvidence : new ArrayList<>();
    }
    
    public List<String> getImplications() {
        return implications;
    }
    
    public void setImplications(List<String> implications) {
        this.implications = implications != null ? implications : new ArrayList<>();
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * 添加支持证据
     */
    public void addSupportingEvidence(String evidence) {
        this.supportingEvidence.add(evidence);
    }
    
    /**
     * 添加含义
     */
    public void addImplication(String implication) {
        this.implications.add(implication);
    }
    
    /**
     * 检查是否有足够的证据支持
     */
    public boolean hasStrongEvidence() {
        return supportingEvidence.size() >= 2 && confidence >= 0.7;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ResearchInsight that = (ResearchInsight) o;
        return Double.compare(that.confidence, confidence) == 0 &&
               Objects.equals(content, that.content) &&
               Objects.equals(insightType, that.insightType);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(content, insightType, confidence);
    }
    
    @Override
    public String toString() {
        return String.format("ResearchInsight{type='%s', confidence=%.2f, content='%s'}", 
            insightType, confidence, content);
    }
}