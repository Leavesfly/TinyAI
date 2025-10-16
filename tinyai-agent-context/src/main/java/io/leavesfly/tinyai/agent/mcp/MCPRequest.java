package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * MCP 请求（基于 JSON-RPC 2.0）
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPRequest {
    /**
     * JSON-RPC版本
     */
    private String jsonrpc;
    
    /**
     * 请求ID
     */
    private String id;
    
    /**
     * 方法名
     */
    private String method;
    
    /**
     * 参数
     */
    private Map<String, Object> params;
    
    public MCPRequest() {
        this.jsonrpc = "2.0";
        this.id = UUID.randomUUID().toString();
        this.method = "";
        this.params = new HashMap<>();
    }
    
    public MCPRequest(String method, Map<String, Object> params) {
        this.jsonrpc = "2.0";
        this.id = UUID.randomUUID().toString();
        this.method = method;
        this.params = params;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("jsonrpc", jsonrpc);
        map.put("id", id);
        map.put("method", method);
        map.put("params", params);
        return map;
    }
    
    public static MCPRequest fromMap(Map<String, Object> data) {
        MCPRequest request = new MCPRequest();
        request.setJsonrpc((String) data.getOrDefault("jsonrpc", "2.0"));
        request.setId((String) data.getOrDefault("id", UUID.randomUUID().toString()));
        request.setMethod((String) data.getOrDefault("method", ""));
        @SuppressWarnings("unchecked")
        Map<String, Object> params = (Map<String, Object>) data.getOrDefault("params", new HashMap<>());
        request.setParams(params);
        return request;
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
    
    public String getMethod() {
        return method;
    }
    
    public void setMethod(String method) {
        this.method = method;
    }
    
    public Map<String, Object> getParams() {
        return params;
    }
    
    public void setParams(Map<String, Object> params) {
        this.params = params;
    }
}
