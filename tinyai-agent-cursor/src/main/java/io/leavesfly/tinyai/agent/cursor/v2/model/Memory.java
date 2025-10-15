package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 记忆数据结构
 * 支持工作记忆、短期记忆、长期记忆和语义记忆
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class Memory {
    
    /**
     * 记忆唯一标识
     */
    private String id;
    
    /**
     * 会话ID（可选，仅工作记忆和短期记忆使用）
     */
    private String sessionId;
    
    /**
     * 项目ID（可选，长期记忆和语义记忆使用）
     */
    private String projectId;
    
    /**
     * 记忆类型
     */
    private MemoryType type;
    
    /**
     * 记忆内容
     */
    private String content;
    
    /**
     * 创建时间戳
     */
    private long createdAt;
    
    /**
     * 更新时间戳
     */
    private long updatedAt;
    
    /**
     * 重要性评分（0-1之间）
     */
    private double importance;
    
    /**
     * 访问次数
     */
    private int accessCount;
    
    /**
     * 最后访问时间
     */
    private long lastAccessAt;
    
    /**
     * 向量表示（仅语义记忆使用）
     */
    private double[] embedding;
    
    /**
     * 元数据
     */
    private Map<String, Object> metadata;
    
    public Memory() {
        this.createdAt = System.currentTimeMillis();
        this.updatedAt = this.createdAt;
        this.lastAccessAt = this.createdAt;
        this.accessCount = 0;
        this.importance = 0.5;
        this.metadata = new HashMap<>();
    }
    
    public Memory(String id, MemoryType type, String content) {
        this();
        this.id = id;
        this.type = type;
        this.content = content;
    }
    
    /**
     * 创建工作记忆
     */
    public static Memory working(String id, String sessionId, String content) {
        Memory memory = new Memory(id, MemoryType.WORKING, content);
        memory.setSessionId(sessionId);
        return memory;
    }
    
    /**
     * 创建短期记忆
     */
    public static Memory shortTerm(String id, String sessionId, String content) {
        Memory memory = new Memory(id, MemoryType.SHORT_TERM, content);
        memory.setSessionId(sessionId);
        return memory;
    }
    
    /**
     * 创建长期记忆
     */
    public static Memory longTerm(String id, String projectId, String content) {
        Memory memory = new Memory(id, MemoryType.LONG_TERM, content);
        memory.setProjectId(projectId);
        return memory;
    }
    
    /**
     * 创建语义记忆
     */
    public static Memory semantic(String id, String projectId, String content, double[] embedding) {
        Memory memory = new Memory(id, MemoryType.SEMANTIC, content);
        memory.setProjectId(projectId);
        memory.setEmbedding(embedding);
        return memory;
    }
    
    /**
     * 记录访问
     */
    public void recordAccess() {
        this.accessCount++;
        this.lastAccessAt = System.currentTimeMillis();
    }
    
    /**
     * 更新内容
     */
    public void updateContent(String newContent) {
        this.content = newContent;
        this.updatedAt = System.currentTimeMillis();
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
     * 是否过期（根据类型判断）
     */
    public boolean isExpired(long currentTime, long ttl) {
        return (currentTime - createdAt) > ttl;
    }
    
    // Getters and Setters
    
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
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
    
    public MemoryType getType() {
        return type;
    }
    
    public void setType(MemoryType type) {
        this.type = type;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public long getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }
    
    public long getUpdatedAt() {
        return updatedAt;
    }
    
    public void setUpdatedAt(long updatedAt) {
        this.updatedAt = updatedAt;
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
    
    public long getLastAccessAt() {
        return lastAccessAt;
    }
    
    public void setLastAccessAt(long lastAccessAt) {
        this.lastAccessAt = lastAccessAt;
    }
    
    public double[] getEmbedding() {
        return embedding;
    }
    
    public void setEmbedding(double[] embedding) {
        this.embedding = embedding;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    /**
     * 记忆类型枚举
     */
    public enum MemoryType {
        /**
         * 工作记忆：当前对话轮次的中间状态
         * 生命周期：当前会话
         * 存储：内存
         */
        WORKING,
        
        /**
         * 短期记忆：本次会话的代码片段、分析结果
         * 生命周期：单次会话
         * 存储：内存
         */
        SHORT_TERM,
        
        /**
         * 长期记忆：项目特定规则、用户偏好设置
         * 生命周期：跨会话持久化
         * 存储：文件/数据库
         */
        LONG_TERM,
        
        /**
         * 语义记忆：代码库向量索引、API文档
         * 生命周期：项目生命周期
         * 存储：向量数据库
         */
        SEMANTIC
    }
}
