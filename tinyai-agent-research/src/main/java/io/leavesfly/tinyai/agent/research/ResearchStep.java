package io.leavesfly.tinyai.agent.research;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 研究步骤记录
 * 记录研究过程中的每个步骤和相关信息
 * 
 * @author 山泽
 */
public class ResearchStep {
    /** 研究阶段 */
    private ResearchPhase phase;
    
    /** 步骤类型 */
    private String stepType;
    
    /** 步骤内容 */
    private String content;
    
    /** 置信度 (0.0-1.0) */
    private double confidence;
    
    /** 信息来源 */
    private List<String> sources;
    
    /** 时间戳 */
    private LocalDateTime timestamp;
    
    /** 元数据 */
    private Map<String, Object> metadata;
    
    /**
     * 构造函数
     */
    public ResearchStep(ResearchPhase phase, String stepType, String content) {
        this(phase, stepType, content, 0.0);
    }
    
    /**
     * 完整构造函数
     */
    public ResearchStep(ResearchPhase phase, String stepType, String content, double confidence) {
        this.phase = phase;
        this.stepType = stepType;
        this.content = content;
        this.confidence = validateConfidence(confidence);
        this.sources = new ArrayList<>();
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>();
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
    public ResearchPhase getPhase() {
        return phase;
    }
    
    public void setPhase(ResearchPhase phase) {
        this.phase = phase;
    }
    
    public String getStepType() {
        return stepType;
    }
    
    public void setStepType(String stepType) {
        this.stepType = stepType;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = validateConfidence(confidence);
    }
    
    public List<String> getSources() {
        return sources;
    }
    
    public void setSources(List<String> sources) {
        this.sources = sources != null ? sources : new ArrayList<>();
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata != null ? metadata : new HashMap<>();
    }
    
    /**
     * 添加信息来源
     */
    public void addSource(String source) {
        this.sources.add(source);
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        this.metadata.put(key, value);
    }
    
    /**
     * 获取元数据
     */
    public Object getMetadata(String key) {
        return this.metadata.get(key);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ResearchStep that = (ResearchStep) o;
        return Double.compare(that.confidence, confidence) == 0 &&
               phase == that.phase &&
               Objects.equals(stepType, that.stepType) &&
               Objects.equals(content, that.content);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(phase, stepType, content, confidence);
    }
    
    @Override
    public String toString() {
        return String.format("ResearchStep{phase=%s, stepType='%s', content='%s', confidence=%.2f}", 
            phase, stepType, content, confidence);
    }
}