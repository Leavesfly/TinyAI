package io.leavesfly.tinyai.agent.research;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 研究查询结构
 * 封装研究请求的所有参数和元数据
 * 
 * @author 山泽
 */
public class ResearchQuery {
    /** 查询内容 */
    private String query;
    
    /** 领域分类 */
    private String domain;
    
    /** 复杂度等级 (1-5) */
    private int complexity;
    
    /** 紧急程度 (1-5) */
    private int urgency;
    
    /** 深度要求 (1-5) */
    private int depthRequired;
    
    /** 查询时间戳 */
    private LocalDateTime timestamp;
    
    /** 元数据 */
    private Map<String, Object> metadata;
    
    /**
     * 默认构造函数
     */
    public ResearchQuery() {
        this("", "general", 1, 1, 3);
    }
    
    /**
     * 基础构造函数
     */
    public ResearchQuery(String query) {
        this(query, "general", 3, 2, 3);
    }
    
    /**
     * 完整构造函数
     */
    public ResearchQuery(String query, String domain, int complexity, int urgency, int depthRequired) {
        this.query = query;
        this.domain = domain;
        this.complexity = validateRange(complexity, 1, 5, "复杂度");
        this.urgency = validateRange(urgency, 1, 5, "紧急程度");
        this.depthRequired = validateRange(depthRequired, 1, 5, "深度要求");
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>();
    }
    
    /**
     * 验证参数范围
     */
    private int validateRange(int value, int min, int max, String paramName) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(String.format("%s必须在%d-%d范围内，当前值: %d", 
                paramName, min, max, value));
        }
        return value;
    }
    
    // Getter和Setter方法
    public String getQuery() {
        return query;
    }
    
    public void setQuery(String query) {
        this.query = query;
    }
    
    public String getDomain() {
        return domain;
    }
    
    public void setDomain(String domain) {
        this.domain = domain;
    }
    
    public int getComplexity() {
        return complexity;
    }
    
    public void setComplexity(int complexity) {
        this.complexity = validateRange(complexity, 1, 5, "复杂度");
    }
    
    public int getUrgency() {
        return urgency;
    }
    
    public void setUrgency(int urgency) {
        this.urgency = validateRange(urgency, 1, 5, "紧急程度");
    }
    
    public int getDepthRequired() {
        return depthRequired;
    }
    
    public void setDepthRequired(int depthRequired) {
        this.depthRequired = validateRange(depthRequired, 1, 5, "深度要求");
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
        ResearchQuery that = (ResearchQuery) o;
        return complexity == that.complexity &&
               urgency == that.urgency &&
               depthRequired == that.depthRequired &&
               Objects.equals(query, that.query) &&
               Objects.equals(domain, that.domain);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(query, domain, complexity, urgency, depthRequired);
    }
    
    @Override
    public String toString() {
        return String.format("ResearchQuery{query='%s', domain='%s', complexity=%d, urgency=%d, depthRequired=%d}", 
            query, domain, complexity, urgency, depthRequired);
    }
}