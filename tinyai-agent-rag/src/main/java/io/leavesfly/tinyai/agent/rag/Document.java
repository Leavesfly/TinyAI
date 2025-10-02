package io.leavesfly.tinyai.agent.rag;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 文档结构类
 * 表示RAG系统中的一个文档，包含内容、元数据、向量嵌入等信息
 */
public class Document {
    private String id;                          // 文档唯一标识
    private String content;                     // 文档内容
    private Map<String, Object> metadata;      // 文档元数据
    private List<Double> embedding;             // 文档向量嵌入
    private LocalDateTime createdAt;            // 创建时间

    /**
     * 构造函数
     */
    public Document(String id, String content, Map<String, Object> metadata) {
        this.id = id;
        this.content = content;
        this.metadata = metadata;
        this.createdAt = LocalDateTime.now();
    }

    /**
     * 完整构造函数
     */
    public Document(String id, String content, Map<String, Object> metadata, 
                   List<Double> embedding, LocalDateTime createdAt) {
        this.id = id;
        this.content = content;
        this.metadata = metadata;
        this.embedding = embedding;
        this.createdAt = createdAt != null ? createdAt : LocalDateTime.now();
    }

    // Getter和Setter方法
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public List<Double> getEmbedding() {
        return embedding;
    }

    public void setEmbedding(List<Double> embedding) {
        this.embedding = embedding;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public String toString() {
        return "Document{" +
                "id='" + id + '\'' +
                ", content='" + (content.length() > 50 ? content.substring(0, 50) + "..." : content) + '\'' +
                ", metadata=" + metadata +
                ", embeddingSize=" + (embedding != null ? embedding.size() : 0) +
                ", createdAt=" + createdAt +
                '}';
    }
}