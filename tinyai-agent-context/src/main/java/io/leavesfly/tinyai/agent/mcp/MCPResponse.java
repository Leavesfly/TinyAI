package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;

/**
 * MCP 响应（基于 JSON-RPC 2.0）
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPResponse {
    /**
     * JSON-RPC版本
     */
    private String jsonrpc;
    
    /**
     * 请求ID
     */
    private String id;
    
    /**
     * 结果
     */
    private Object result;
    
    /**
     * 错误信息
     */
    private Map<String, Object> error;
    
    public MCPResponse() {
        this.jsonrpc = "2.0";
        this.id = "";
    }
    
    public MCPResponse(String id, Object result) {
        this.jsonrpc = "2.0";
        this.id = id;
        this.result = result;
    }
    
    public MCPResponse(String id, Map<String, Object> error) {
        this.jsonrpc = "2.0";
        this.id = id;
        this.error = error;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("jsonrpc", jsonrpc);
        map.put("id", id);
        
        if (error != null) {
            map.put("error", error);
        } else {
            map.put("result", result);
        }
        
        return map;
    }
    
    /**
     * 创建错误响应
     */
    public static MCPResponse createErrorResponse(String id, int code, String message) {
        Map<String, Object> error = new HashMap<>();
        error.put("code", code);
        error.put("message", message);
        return new MCPResponse(id, error);
    }
    
    // Getters and Setters
    public String getJsonrpc() {
        return jsonrpc;
    }
    
    public void setJsonrpc(String jsonrpc) {
        this.jsonrpc = jsonrpc;
    }
    
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public Object getResult() {
        return result;
    }
    
    public void setResult(Object result) {
        this.result = result;
    }
    
    public Map<String, Object> getError() {
        return error;
    }
    
    public void setError(Map<String, Object> error) {
        this.error = error;
    }
}
