package io.leavesfly.tinyai.agent.mcp;

/**
 * 资源类型枚举
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public enum ResourceType {
    /**
     * 文件资源
     */
    FILE("file"),
    
    /**
     * 数据库资源
     */
    DATABASE("database"),
    
    /**
     * API资源
     */
    API("api"),
    
    /**
     * 内存资源
     */
    MEMORY("memory"),
    
    /**
     * 文档资源
     */
    DOCUMENT("document");
    
    private final String value;
    
    ResourceType(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return value;
    }
    
    @Override
    public String toString() {
        return value;
    }
}
