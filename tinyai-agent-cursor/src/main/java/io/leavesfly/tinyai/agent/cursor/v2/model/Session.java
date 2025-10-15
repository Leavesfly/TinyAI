package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 会话数据结构
 * 管理用户与AI编程助手的会话状态
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class Session {
    
    /**
     * 会话唯一标识
     */
    private String sessionId;
    
    /**
     * 项目ID
     */
    private String projectId;
    
    /**
     * 用户ID
     */
    private String userId;
    
    /**
     * 对话历史
     */
    private List<Message> messages;
    
    /**
     * 工作记忆（当前会话的临时数据）
     */
    private Map<String, Object> workingMemory;
    
    /**
     * 会话状态
     */
    private SessionState state;
    
    /**
     * 创建时间戳
     */
    private long createdAt;
    
    /**
     * 最后活跃时间
     */
    private long lastActiveAt;
    
    /**
     * 生存时间（秒，默认30分钟）
     */
    private int ttl = 1800;
    
    /**
     * 消息计数
     */
    private int messageCount;
    
    /**
     * 会话元数据
     */
    private Map<String, Object> metadata;
    
    public Session() {
        this.sessionId = UUID.randomUUID().toString();
        this.messages = new ArrayList<>();
        this.workingMemory = new HashMap<>();
        this.metadata = new HashMap<>();
        this.state = SessionState.ACTIVE;
        this.createdAt = System.currentTimeMillis();
        this.lastActiveAt = this.createdAt;
        this.messageCount = 0;
    }
    
    public Session(String projectId, String userId) {
        this();
        this.projectId = projectId;
        this.userId = userId;
    }
    
    /**
     * 添加消息
     */
    public void addMessage(Message message) {
        this.messages.add(message);
        this.messageCount++;
        this.lastActiveAt = System.currentTimeMillis();
    }
    
    /**
     * 获取最近N条消息
     */
    public List<Message> getRecentMessages(int count) {
        int size = messages.size();
        int fromIndex = Math.max(0, size - count);
        return new ArrayList<>(messages.subList(fromIndex, size));
    }
    
    /**
     * 设置工作记忆
     */
    public void putWorkingMemory(String key, Object value) {
        this.workingMemory.put(key, value);
        this.lastActiveAt = System.currentTimeMillis();
    }
    
    /**
     * 获取工作记忆
     */
    public Object getWorkingMemory(String key) {
        return this.workingMemory.get(key);
    }
    
    /**
     * 清除工作记忆
     */
    public void clearWorkingMemory() {
        this.workingMemory.clear();
    }
    
    /**
     * 设置元数据
     */
    public void putMetadata(String key, Object value) {
        this.metadata.put(key, value);
    }
    
    /**
     * 获取元数据
     */
    public Object getMetadata(String key) {
        return this.metadata.get(key);
    }
    
    /**
     * 更新活跃时间
     */
    public void updateActiveTime() {
        this.lastActiveAt = System.currentTimeMillis();
    }
    
    /**
     * 是否过期
     */
    public boolean isExpired() {
        long currentTime = System.currentTimeMillis();
        return (currentTime - lastActiveAt) > (ttl * 1000L);
    }
    
    /**
     * 是否活跃
     */
    public boolean isActive() {
        return state == SessionState.ACTIVE && !isExpired();
    }
    
    /**
     * 关闭会话
     */
    public void close() {
        this.state = SessionState.CLOSED;
    }
    
    /**
     * 标记为过期
     */
    public void expire() {
        this.state = SessionState.EXPIRED;
    }
    
    // Getters and Setters
    
    public String getSessionId() {
        return sessionId;
    }
    
    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }
    
    public String getProjectId() {
        return projectId;
    }
    
    public void setProjectId(String projectId) {
        this.projectId = projectId;
    }
    
    public String getUserId() {
        return userId;
    }
    
    public void setUserId(String userId) {
        this.userId = userId;
    }
    
    public List<Message> getMessages() {
        return messages;
    }
    
    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }
    
    public Map<String, Object> getWorkingMemory() {
        return workingMemory;
    }
    
    public void setWorkingMemory(Map<String, Object> workingMemory) {
        this.workingMemory = workingMemory;
    }
    
    public SessionState getState() {
        return state;
    }
    
    public void setState(SessionState state) {
        this.state = state;
    }
    
    public long getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }
    
    public long getLastActiveAt() {
        return lastActiveAt;
    }
    
    public void setLastActiveAt(long lastActiveAt) {
        this.lastActiveAt = lastActiveAt;
    }
    
    public int getTtl() {
        return ttl;
    }
    
    public void setTtl(int ttl) {
        this.ttl = ttl;
    }
    
    public int getMessageCount() {
        return messageCount;
    }
    
    public void setMessageCount(int messageCount) {
        this.messageCount = messageCount;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    /**
     * 会话状态枚举
     */
    public enum SessionState {
        /** 活跃状态 */
        ACTIVE,
        /** 已关闭 */
        CLOSED,
        /** 已过期 */
        EXPIRED
    }
    
    @Override
    public String toString() {
        return "Session{" +
                "sessionId='" + sessionId + '\'' +
                ", projectId='" + projectId + '\'' +
                ", userId='" + userId + '\'' +
                ", messageCount=" + messageCount +
                ", state=" + state +
                ", createdAt=" + createdAt +
                ", lastActiveAt=" + lastActiveAt +
                '}';
    }
}
