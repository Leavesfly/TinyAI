package io.leavesfly.tinyai.agent.multi;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Agent间通信的消息类
 * 扩展了原有的Message类以支持多Agent系统
 * 
 * @author 山泽
 */
public class AgentMessage {
    
    private String id;                           // 消息唯一标识
    private String senderId;                     // 发送者Agent ID
    private String receiverId;                   // 接收者Agent ID
    private MessageType messageType;             // 消息类型
    private Object content;                      // 消息内容
    private Map<String, Object> metadata;        // 元数据
    private LocalDateTime timestamp;             // 时间戳
    private int priority;                        // 优先级 (1-10, 10最高)
    
    // 构造函数
    public AgentMessage() {
        this.id = UUID.randomUUID().toString();
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>();
        this.priority = 1;
    }
    
    public AgentMessage(String senderId, String receiverId, MessageType messageType, Object content) {
        this();
        this.senderId = senderId;
        this.receiverId = receiverId;
        this.messageType = messageType;
        this.content = content;
    }
    
    public AgentMessage(String senderId, String receiverId, MessageType messageType, 
                       Object content, int priority) {
        this(senderId, receiverId, messageType, content);
        this.priority = priority;
    }
    
    // Getter 和 Setter 方法
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getSenderId() {
        return senderId;
    }
    
    public void setSenderId(String senderId) {
        this.senderId = senderId;
    }
    
    public String getReceiverId() {
        return receiverId;
    }
    
    public void setReceiverId(String receiverId) {
        this.receiverId = receiverId;
    }
    
    public MessageType getMessageType() {
        return messageType;
    }
    
    public void setMessageType(MessageType messageType) {
        this.messageType = messageType;
    }
    
    public Object getContent() {
        return content;
    }
    
    public void setContent(Object content) {
        this.content = content;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata != null ? metadata : new HashMap<>();
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = Math.max(1, Math.min(10, priority)); // 限制在1-10范围内
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
    
    /**
     * 转换为Map格式
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("id", id);
        map.put("senderId", senderId);
        map.put("receiverId", receiverId);
        map.put("messageType", messageType.getValue());
        map.put("content", content);
        map.put("metadata", metadata);
        map.put("timestamp", timestamp.toString());
        map.put("priority", priority);
        return map;
    }
    
    @Override
    public String toString() {
        return String.format("AgentMessage{id='%s', sender='%s', receiver='%s', type=%s, priority=%d, timestamp=%s}",
                id, senderId, receiverId, messageType, priority, timestamp);
    }
}