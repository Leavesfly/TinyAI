package io.leavesfly.tinyai.agent.research.v2.model;

import java.time.LocalDateTime;
import java.util.*;

/**
 * 研究报告
 * 表示最终生成的研究成果
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchReport {
    
    /**
     * 报告ID
     */
    private String reportId;
    
    /**
     * 报告标题
     */
    private String title;
    
    /**
     * 执行摘要
     */
    private String summary;
    
    /**
     * 完整报告内容（Markdown格式）
     */
    private String fullContent;
    
    /**
     * 研究洞察列表
     */
    private List<Insight> insights;
    
    /**
     * 引用来源列表
     */
    private List<Reference> references;
    
    /**
     * 质量评分（0-100）
     */
    private double qualityScore;
    
    /**
     * 生成时间
     */
    private LocalDateTime generatedAt;
    
    /**
     * 元数据
     */
    private Map<String, Object> metadata;
    
    public ResearchReport() {
        this.reportId = UUID.randomUUID().toString();
        this.insights = new ArrayList<>();
        this.references = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.generatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加洞察
     */
    public void addInsight(Insight insight) {
        this.insights.add(insight);
    }
    
    /**
     * 添加引用来源
     */
    public void addReference(Reference reference) {
        this.references.add(reference);
    }
    
    /**
     * 洞察类
     */
    public static class Insight {
        private String title;
        private String content;
        private double confidence;
        private List<String> supportingEvidence;
        
        public Insight() {
            this.supportingEvidence = new ArrayList<>();
        }
        
        public Insight(String title, String content) {
            this();
            this.title = title;
            this.content = content;
        }
        
        // Getters and Setters
        public String getTitle() {
            return title;
        }
        
        public void setTitle(String title) {
            this.title = title;
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
            this.confidence = confidence;
        }
        
        public List<String> getSupportingEvidence() {
            return supportingEvidence;
        }
        
        public void setSupportingEvidence(List<String> supportingEvidence) {
            this.supportingEvidence = supportingEvidence;
        }
    }
    
    /**
     * 引用来源类
     */
    public static class Reference {
        private String source;
        private String url;
        private String title;
        private LocalDateTime accessedAt;
        
        public Reference() {
        }
        
        public Reference(String source, String url) {
            this.source = source;
            this.url = url;
            this.accessedAt = LocalDateTime.now();
        }
        
        // Getters and Setters
        public String getSource() {
            return source;
        }
        
        public void setSource(String source) {
            this.source = source;
        }
        
        public String getUrl() {
            return url;
        }
        
        public void setUrl(String url) {
            this.url = url;
        }
        
        public String getTitle() {
            return title;
        }
        
        public void setTitle(String title) {
            this.title = title;
        }
        
        public LocalDateTime getAccessedAt() {
            return accessedAt;
        }
        
        public void setAccessedAt(LocalDateTime accessedAt) {
            this.accessedAt = accessedAt;
        }
    }
    
    // Getters and Setters
    
    public String getReportId() {
        return reportId;
    }
    
    public void setReportId(String reportId) {
        this.reportId = reportId;
    }
    
    public String getTitle() {
        return title;
    }
    
    public void setTitle(String title) {
        this.title = title;
    }
    
    public String getSummary() {
        return summary;
    }
    
    public void setSummary(String summary) {
        this.summary = summary;
    }
    
    public String getFullContent() {
        return fullContent;
    }
    
    public void setFullContent(String fullContent) {
        this.fullContent = fullContent;
    }
    
    public List<Insight> getInsights() {
        return insights;
    }
    
    public void setInsights(List<Insight> insights) {
        this.insights = insights;
    }
    
    public List<Reference> getReferences() {
        return references;
    }
    
    public void setReferences(List<Reference> references) {
        this.references = references;
    }
    
    public double getQualityScore() {
        return qualityScore;
    }
    
    public void setQualityScore(double qualityScore) {
        this.qualityScore = qualityScore;
    }
    
    public LocalDateTime getGeneratedAt() {
        return generatedAt;
    }
    
    public void setGeneratedAt(LocalDateTime generatedAt) {
        this.generatedAt = generatedAt;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
}
