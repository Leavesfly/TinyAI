package io.leavesfly.tinyai.agent.context;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 记忆单元
 * 
 * @author 山泽
 */
public class Memory {
    
    private String id;                      // 记忆ID
    private String content;                 // 记忆内容
    private String memoryType;             // 记忆类型：'working', 'episodic', 'semantic'
    private LocalDateTime timestamp;        // 创建时间戳
    private double importance;              // 重要性分数
    private int accessCount;                // 访问次数
    private LocalDateTime lastAccessed;     // 最后访问时间
    private List<Double> embedding;         // 嵌入向量
    private Map<String, Object> metadata;   // 元数据
    
    // 构造函数
    public Memory() {
        this.timestamp = LocalDateTime.now();
        this.lastAccessed = LocalDateTime.now();
        this.importance = 0.0;
        this.accessCount = 0;
        this.metadata = new HashMap<>();
    }
    
    public Memory(String id, String content, String memoryType) {
        this();
        this.id = id;
        this.content = content;
        this.memoryType = memoryType;
    }
    
    public Memory(String id, String content, String memoryType, double importance) {
        this(id, content, memoryType);
        this.importance = importance;
    }
    
    // Getter 和 Setter 方法
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
    
    public String getMemoryType() {
        return memoryType;
    }
    
    public void setMemoryType(String memoryType) {
        this.memoryType = memoryType;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public double getImportance() {
        return importance;
    }
    
    public void setImportance(double importance) {
        this.importance = importance;
    }
    
    public int getAccessCount() {
        return accessCount;
    }
    
    public void setAccessCount(int accessCount) {
        this.accessCount = accessCount;
    }
    
    public LocalDateTime getLastAccessed() {
        return lastAccessed;
    }
    
    public void setLastAccessed(LocalDateTime lastAccessed) {
        this.lastAccessed = lastAccessed;
    }
    
    public List<Double> getEmbedding() {
        return embedding;
    }
    
    public void setEmbedding(List<Double> embedding) {
        this.embedding = embedding;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    /**
     * 增加访问次数并更新最后访问时间
     */
    public void incrementAccess() {
        this.accessCount++;
        this.lastAccessed = LocalDateTime.now();
    }
    
    @Override
    public String toString() {
        return String.format("Memory{id='%s', type='%s', content='%s', importance=%.2f, accessCount=%d}", 
                           id, memoryType, content.length() > 50 ? content.substring(0, 50) + "..." : content, 
                           importance, accessCount);
    }
}