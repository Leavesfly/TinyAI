package io.leavesfly.tinyai.agent.context;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 文档类
 * RAG系统中的文档实体
 * 
 * @author 山泽
 */
public class Document {
    
    private String id;                      // 文档ID
    private String content;                 // 文档内容
    private Map<String, Object> metadata;   // 元数据
    private LocalDateTime timestamp;        // 时间戳
    
    // 构造函数
    public Document(String id, String content, Map<String, Object> metadata) {
        this.id = id;
        this.content = content;
        this.metadata = metadata != null ? metadata : new HashMap<>();
        this.timestamp = LocalDateTime.now();
    }
    
    public Document(String id, String content) {
        this(id, content, null);
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
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * 获取指定键的元数据
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    @Override
    public String toString() {
        return String.format("Document{id='%s', contentLength=%d, timestamp=%s}",
                           id, content.length(), timestamp);
    }
}