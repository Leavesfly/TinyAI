package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具调用结果
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class ToolResult {
    /**
     * 调用ID
     */
    private String callId;
    
    /**
     * 结果内容
     */
    private Object content;
    
    /**
     * 是否错误
     */
    private boolean isError;
    
    /**
     * 错误消息
     */
    private String errorMessage;
    
    /**
     * 执行时间（秒）
     */
    private double executionTime;
    
    public ToolResult(String callId, Object content) {
        this.callId = callId;
        this.content = content;
        this.isError = false;
        this.executionTime = 0.0;
    }
    
    public ToolResult(String callId, Object content, boolean isError, String errorMessage, double executionTime) {
        this.callId = callId;
        this.content = content;
        this.isError = isError;
        this.errorMessage = errorMessage;
        this.executionTime = executionTime;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("callId", callId);
        map.put("content", content);
        map.put("isError", isError);
        map.put("errorMessage", errorMessage);
        map.put("executionTime", executionTime);
        return map;
    }
    
    // Getters and Setters
    public String getCallId() {
        return callId;
    }
    
    public void setCallId(String callId) {
        this.callId = callId;
    }
    
    public Object getContent() {
        return content;
    }
    
    public void setContent(Object content) {
        this.content = content;
    }
    
    public boolean isError() {
        return isError;
    }
    
    public void setError(boolean error) {
        isError = error;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    public double getExecutionTime() {
        return executionTime;
    }
    
    public void setExecutionTime(double executionTime) {
        this.executionTime = executionTime;
    }
}
