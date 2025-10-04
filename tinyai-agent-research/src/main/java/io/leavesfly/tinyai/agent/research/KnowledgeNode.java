package io.leavesfly.tinyai.agent.research;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 知识图谱节点
 * 表示知识图谱中的一个知识点
 * 
 * @author 山泽
 */
public class KnowledgeNode {
    /** 节点唯一标识 */
    private String id;
    
    /** 节点内容 */
    private String content;
    
    /** 节点类型 */
    private String nodeType;
    
    /** 所属领域 */
    private String domain;
    
    /** 置信度 (0.0-1.0) */
    private double confidence;
    
    /** 连接的其他节点ID */
    private Set<String> connections;
    
    /** 支持证据 */
    private List<String> evidence;
    
    /** 创建时间戳 */
    private LocalDateTime timestamp;
    
    /**
     * 完整构造函数
     */
    public KnowledgeNode(String id, String content, String nodeType, String domain, double confidence) {
        this.id = id;
        this.content = content;
        this.nodeType = nodeType;
        this.domain = domain;
        this.confidence = validateConfidence(confidence);
        this.connections = new HashSet<>();
        this.evidence = new ArrayList<>();
        this.timestamp = LocalDateTime.now();
    }
    
    /**
     * 简化构造函数
     */
    public KnowledgeNode(String id, String content, String nodeType, String domain) {
        this(id, content, nodeType, domain, 0.5);
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
    
    public String getNodeType() {
        return nodeType;
    }
    
    public void setNodeType(String nodeType) {
        this.nodeType = nodeType;
    }
    
    public String getDomain() {
        return domain;
    }
    
    public void setDomain(String domain) {
        this.domain = domain;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = validateConfidence(confidence);
    }
    
    public Set<String> getConnections() {
        return connections;
    }
    
    public void setConnections(Set<String> connections) {
        this.connections = connections != null ? connections : new HashSet<>();
    }
    
    public List<String> getEvidence() {
        return evidence;
    }
    
    public void setEvidence(List<String> evidence) {
        this.evidence = evidence != null ? evidence : new ArrayList<>();
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * 添加连接
     */
    public void addConnection(String nodeId) {
        this.connections.add(nodeId);
    }
    
    /**
     * 移除连接
     */
    public void removeConnection(String nodeId) {
        this.connections.remove(nodeId);
    }
    
    /**
     * 添加证据
     */
    public void addEvidence(String evidence) {
        this.evidence.add(evidence);
    }
    
    /**
     * 检查是否连接到指定节点
     */
    public boolean isConnectedTo(String nodeId) {
        return this.connections.contains(nodeId);
    }
    
    /**
     * 获取连接数量
     */
    public int getConnectionCount() {
        return this.connections.size();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KnowledgeNode that = (KnowledgeNode) o;
        return Objects.equals(id, that.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
    
    @Override
    public String toString() {
        return String.format("KnowledgeNode{id='%s', nodeType='%s', domain='%s', connections=%d}", 
            id, nodeType, domain, connections.size());
    }
}