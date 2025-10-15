package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具执行结果数据结构
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolResult {
    
    /**
     * 工具名称
     */
    private String toolName;
    
    /**
     * 执行是否成功
     */
    private boolean success;
    
    /**
     * 结果内容
     */
    private String result;
    
    /**
     * 错误信息（如果失败）
     */
    private String error;
    
    /**
     * 执行时间（毫秒）
     */
    private long executionTime;
    
    /**
     * 额外信息
     */
    private Map<String, Object> metadata;
    
    public ToolResult() {
        this.metadata = new HashMap<>();
    }
    
    public ToolResult(String toolName, boolean success, String result) {
        this();
        this.toolName = toolName;
        this.success = success;
        this.result = result;
    }
    
    /**
     * 创建成功结果
     */
    public static ToolResult success(String toolName, String result) {
        return new ToolResult(toolName, true, result);
    }
    
    /**
     * 创建失败结果
     */
    public static ToolResult failure(String toolName, String error) {
        ToolResult toolResult = new ToolResult(toolName, false, null);
        toolResult.setError(error);
        return toolResult;
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
    
    // Getters and Setters
    
    public String getToolName() {
        return toolName;
    }
    
    public void setToolName(String toolName) {
        this.toolName = toolName;
    }
    
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public String getResult() {
        return result;
    }
    
    public void setResult(String result) {
        this.result = result;
    }
    
    public String getError() {
        return error;
    }
    
    public void setError(String error) {
        this.error = error;
    }
    
    public long getExecutionTime() {
        return executionTime;
    }
    
    public void setExecutionTime(long executionTime) {
        this.executionTime = executionTime;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    @Override
    public String toString() {
        return "ToolResult{" +
                "toolName='" + toolName + '\'' +
                ", success=" + success +
                ", result='" + result + '\'' +
                ", error='" + error + '\'' +
                ", executionTime=" + executionTime +
                '}';
    }
}
