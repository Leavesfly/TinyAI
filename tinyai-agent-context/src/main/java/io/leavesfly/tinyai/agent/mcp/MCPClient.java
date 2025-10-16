package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

/**
 * MCP 客户端 - 连接并使用 MCP Server
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPClient {
    /**
     * 客户端ID
     */
    private String clientId;
    
    /**
     * 已连接的服务器
     */
    private Map<String, MCPServer> connectedServers;
    
    public MCPClient() {
        this(UUID.randomUUID().toString().substring(0, 8));
    }
    
    public MCPClient(String clientId) {
        this.clientId = clientId;
        this.connectedServers = new HashMap<>();
        System.out.println("🔌 MCP Client " + clientId + " 已创建");
    }
    
    /**
     * 连接到 MCP Server
     */
    public void connect(String serverName, MCPServer server) {
        connectedServers.put(serverName, server);
        System.out.println("✅ 已连接到服务器: " + serverName);
    }
    
    /**
     * 断开连接
     */
    public void disconnect(String serverName) {
        if (connectedServers.containsKey(serverName)) {
            connectedServers.remove(serverName);
            System.out.println("❌ 已断开连接: " + serverName);
        }
    }
    
    /**
     * 列出已连接的服务器
     */
    public List<String> listServers() {
        return new ArrayList<>(connectedServers.keySet());
    }
    
    /**
     * 发送请求到服务器
     */
    private MCPResponse sendRequest(String serverName, MCPRequest request) {
        if (!connectedServers.containsKey(serverName)) {
            return MCPResponse.createErrorResponse(
                request.getId(),
                -32000,
                "未连接到服务器: " + serverName
            );
        }
        
        MCPServer server = connectedServers.get(serverName);
        return server.handleRequest(request);
    }
    
    // ========== Resource 操作 ==========
    
    /**
     * 列出服务器的资源
     */
    public List<Map<String, Object>> listResources(String serverName) {
        MCPRequest request = new MCPRequest("resources/list", new HashMap<>());
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> result = (List<Map<String, Object>>) response.getResult();
            return result;
        }
        return new ArrayList<>();
    }
    
    /**
     * 读取资源内容
     */
    public Map<String, Object> readResource(String serverName, String uri) {
        Map<String, Object> params = new HashMap<>();
        params.put("uri", uri);
        
        MCPRequest request = new MCPRequest("resources/read", params);
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            Map<String, Object> result = (Map<String, Object>) response.getResult();
            return result;
        }
        return null;
    }
    
    // ========== Tool 操作 ==========
    
    /**
     * 列出服务器的工具
     */
    public List<Map<String, Object>> listTools(String serverName) {
        MCPRequest request = new MCPRequest("tools/list", new HashMap<>());
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> result = (List<Map<String, Object>>) response.getResult();
            return result;
        }
        return new ArrayList<>();
    }
    
    /**
     * 调用工具
     */
    public Map<String, Object> callTool(String serverName, String toolName, Map<String, Object> arguments) {
        Map<String, Object> params = new HashMap<>();
        params.put("id", UUID.randomUUID().toString());
        params.put("name", toolName);
        params.put("arguments", arguments != null ? arguments : new HashMap<>());
        
        MCPRequest request = new MCPRequest("tools/call", params);
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            Map<String, Object> result = (Map<String, Object>) response.getResult();
            return result;
        }
        return new HashMap<>();
    }
    
    // ========== Prompt 操作 ==========
    
    /**
     * 列出服务器的提示词模板
     */
    public List<Map<String, Object>> listPrompts(String serverName) {
        MCPRequest request = new MCPRequest("prompts/list", new HashMap<>());
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> result = (List<Map<String, Object>>) response.getResult();
            return result;
        }
        return new ArrayList<>();
    }
    
    /**
     * 获取提示词
     */
    public String getPrompt(String serverName, String promptName, Map<String, Object> arguments) {
        Map<String, Object> params = new HashMap<>();
        params.put("name", promptName);
        params.put("arguments", arguments != null ? arguments : new HashMap<>());
        
        MCPRequest request = new MCPRequest("prompts/get", params);
        MCPResponse response = sendRequest(serverName, request);
        
        if (response.getResult() != null) {
            @SuppressWarnings("unchecked")
            Map<String, Object> result = (Map<String, Object>) response.getResult();
            return (String) result.get("prompt");
        }
        return null;
    }
    
    // Getters
    public String getClientId() {
        return clientId;
    }
    
    public Map<String, MCPServer> getConnectedServers() {
        return connectedServers;
    }
}
