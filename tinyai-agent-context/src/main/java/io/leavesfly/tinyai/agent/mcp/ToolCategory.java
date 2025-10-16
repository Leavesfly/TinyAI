package io.leavesfly.tinyai.agent.mcp;

/**
 * 工具类别枚举
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public enum ToolCategory {
    /**
     * 计算类工具
     */
    COMPUTATION("computation"),
    
    /**
     * 搜索类工具
     */
    SEARCH("search"),
    
    /**
     * 数据访问工具
     */
    DATA_ACCESS("data_access"),
    
    /**
     * 系统工具
     */
    SYSTEM("system"),
    
    /**
     * 自定义工具
     */
    CUSTOM("custom");
    
    private final String value;
    
    ToolCategory(String value) {
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
