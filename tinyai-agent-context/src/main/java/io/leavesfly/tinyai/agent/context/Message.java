package io.leavesfly.tinyai.agent.context;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 消息结构
 * 
 * @author 山泽
 */
public class Message {
    
    private String role;                    // 角色：'user', 'assistant', 'system', 'tool'
    private String content;                 // 消息内容
    private LocalDateTime timestamp;        // 时间戳
    private Map<String, Object> metadata;   // 元数据
    
    // 构造函数
    public Message() {
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>();
    }
    
    public Message(String role, String content) {
        this();
        this.role = role;
        this.content = content;
    }
    
    public Message(String role, String content, Map<String, Object> metadata) {
        this(role, content);
        this.metadata = metadata != null ? metadata : new HashMap<>();
    }
    
    // Getter 和 Setter 方法
    public String getRole() {
        return role;
    }
    
    public void setRole(String role) {
        this.role = role;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
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
        this.metadata = metadata;
    }
    
    /**
     * 添加元数据
     */
    public void addMetadata(String key, Object value) {
        this.metadata.put(key, value);
    }
    
    /**
     * 获取指定键的元数据
     */
    public Object getMetadata(String key) {
        return this.metadata.get(key);
    }
    
    @Override
    public String toString() {
        return String.format("Message{role='%s', content='%s', timestamp=%s}", 
                           role, 
                           content != null && content.length() > 100 ? content.substring(0, 100) + "..." : content, 
                           timestamp);
    }
}