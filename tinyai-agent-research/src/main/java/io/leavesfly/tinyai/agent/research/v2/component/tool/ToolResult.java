package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具执行结果
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolResult {
    
    private boolean success;
    private String message;
    private Map<String, Object> data;
    private long executionTimeMs;
    
    public ToolResult() {
        this.data = new HashMap<>();
    }
    
    public static ToolResult success(String message) {
        ToolResult result = new ToolResult();
        result.success = true;
        result.message = message;
        return result;
    }
    
    public static ToolResult success(Map<String, Object> data) {
        ToolResult result = new ToolResult();
        result.success = true;
        result.data = data;
        return result;
    }
    
    public static ToolResult error(String message) {
        ToolResult result = new ToolResult();
        result.success = false;
        result.message = message;
        return result;
    }
    
    public void putData(String key, Object value) {
        this.data.put(key, value);
    }
    
    @SuppressWarnings("unchecked")
    public <T> T getData(String key) {
        return (T) this.data.get(key);
    }
    
    // Getters and Setters
    
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public String getMessage() {
        return message;
    }
    
    public void setMessage(String message) {
        this.message = message;
    }
    
    public Map<String, Object> getData() {
        return data;
    }
    
    public void setData(Map<String, Object> data) {
        this.data = data;
    }
    
    public long getExecutionTimeMs() {
        return executionTimeMs;
    }
    
    public void setExecutionTimeMs(long executionTimeMs) {
        this.executionTimeMs = executionTimeMs;
    }
}
