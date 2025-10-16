package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

/**
 * 支持 MCP 的 AI Agent
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPEnabledAgent {
    /**
     * Agent名称
     */
    private String name;
    
    /**
     * MCP客户端
     */
    private MCPClient client;
    
    /**
     * 对话历史
     */
    private List<Map<String, String>> conversationHistory;
    
    public MCPEnabledAgent(String name) {
        this.name = name;
        this.client = new MCPClient(name.toLowerCase().replace(" ", "_"));
        this.conversationHistory = new ArrayList<>();
        
        System.out.println("🤖 Agent '" + name + "' 已创建，支持 MCP 协议");
    }
    
    /**
     * 连接到 MCP Server
     */
    public void connectToServer(String serverName, MCPServer server) {
        client.connect(serverName, server);
        System.out.println("✅ Agent 已连接到 '" + serverName + "' 服务器");
    }
    
    /**
     * 发现所有连接服务器的能力
     */
    public Map<String, Map<String, Object>> discoverCapabilities() {
        Map<String, Map<String, Object>> capabilities = new HashMap<>();
        
        for (String serverName : client.listServers()) {
            Map<String, Object> serverCap = new HashMap<>();
            serverCap.put("resources", client.listResources(serverName));
            serverCap.put("tools", client.listTools(serverName));
            serverCap.put("prompts", client.listPrompts(serverName));
            capabilities.put(serverName, serverCap);
        }
        
        return capabilities;
    }
    
    /**
     * 处理用户查询
     */
    public String processQuery(String query) {
        // 记录用户输入
        Map<String, String> userMsg = new HashMap<>();
        userMsg.put("role", "user");
        userMsg.put("content", query);
        conversationHistory.add(userMsg);
        
        // 简单的意图识别
        String response;
        if (query.contains("搜索") || query.contains("查找")) {
            response = handleSearchQuery(query);
        } else if (query.contains("统计") || query.contains("分析") || query.contains("计算")) {
            response = handleAnalysisQuery(query);
        } else if (query.contains("读取") || query.contains("查看") || query.contains("显示")) {
            response = handleReadQuery(query);
        } else {
            response = handleGeneralQuery(query);
        }
        
        // 记录助手回复
        Map<String, String> assistantMsg = new HashMap<>();
        assistantMsg.put("role", "assistant");
        assistantMsg.put("content", response);
        conversationHistory.add(assistantMsg);
        
        return response;
    }
    
    /**
     * 处理搜索查询
     */
    @SuppressWarnings("unchecked")
    private String handleSearchQuery(String query) {
        // 提取关键词（简单实现）
        String keywords = query.replace("搜索", "").replace("查找", "").trim();
        
        try {
            Map<String, Object> args = new HashMap<>();
            args.put("keyword", keywords);
            
            Map<String, Object> result = client.callTool("filesystem", "search_files", args);
            
            if (result.containsKey("isError") && (Boolean) result.get("isError")) {
                return "搜索失败: " + result.get("errorMessage");
            }
            
            Map<String, Object> content = (Map<String, Object>) result.get("content");
            int found = (Integer) content.get("found");
            
            if (found == 0) {
                return "未找到包含 '" + keywords + "' 的文件";
            }
            
            List<Map<String, Object>> results = (List<Map<String, Object>>) content.get("results");
            StringBuilder response = new StringBuilder("找到 " + found + " 个匹配的文件：\n\n");
            
            int count = Math.min(results.size(), 3);  // 只显示前3个
            for (int i = 0; i < count; i++) {
                Map<String, Object> r = results.get(i);
                response.append("📄 ").append(r.get("uri")).append("\n");
                response.append("   ").append(r.get("preview")).append("\n\n");
            }
            
            return response.toString();
        } catch (Exception e) {
            return "搜索出错: " + e.getMessage();
        }
    }
    
    /**
     * 处理分析查询
     */
    @SuppressWarnings("unchecked")
    private String handleAnalysisQuery(String query) {
        try {
            Map<String, Object> args = new HashMap<>();
            args.put("data_uri", "db://sales");
            args.put("field", "amount");
            
            Map<String, Object> result = client.callTool("dataanalysis", "calculate_statistics", args);
            
            if (result.containsKey("isError") && (Boolean) result.get("isError")) {
                return "分析失败: " + result.get("errorMessage");
            }
            
            Map<String, Object> stats = (Map<String, Object>) result.get("content");
            
            StringBuilder response = new StringBuilder("📊 销售数据统计分析：\n\n");
            response.append("- 记录数量：").append(stats.get("count")).append("\n");
            response.append("- 总销售额：¥").append(stats.get("sum")).append("\n");
            
            double avg = (Double) stats.get("average");
            response.append(String.format("- 平均销售额：¥%.2f\n", avg));
            response.append("- 最高销售额：¥").append(stats.get("max")).append("\n");
            response.append("- 最低销售额：¥").append(stats.get("min")).append("\n");
            
            return response.toString();
        } catch (Exception e) {
            return "分析出错: " + e.getMessage();
        }
    }
    
    /**
     * 处理读取查询
     */
    @SuppressWarnings("unchecked")
    private String handleReadQuery(String query) {
        try {
            List<Map<String, Object>> resources = client.listResources("filesystem");
            
            if (resources.isEmpty()) {
                return "没有可用的资源";
            }
            
            StringBuilder response = new StringBuilder("📚 可用资源：\n\n");
            
            int count = Math.min(resources.size(), 5);
            for (int i = 0; i < count; i++) {
                Map<String, Object> resource = resources.get(i);
                response.append("- ").append(resource.get("name")).append(": ");
                response.append(resource.get("description")).append("\n");
            }
            
            return response.toString();
        } catch (Exception e) {
            return "读取出错: " + e.getMessage();
        }
    }
    
    /**
     * 处理通用查询
     */
    @SuppressWarnings("unchecked")
    private String handleGeneralQuery(String query) {
        Map<String, Map<String, Object>> capabilities = discoverCapabilities();
        
        int totalResources = 0;
        int totalTools = 0;
        
        for (Map<String, Object> cap : capabilities.values()) {
            totalResources += ((List<?>) cap.get("resources")).size();
            totalTools += ((List<?>) cap.get("tools")).size();
        }
        
        StringBuilder response = new StringBuilder("我是 " + name + "，通过 MCP 协议连接到了多个服务器。\n\n");
        response.append("当前能力：\n");
        response.append("- 📦 可访问 ").append(totalResources).append(" 个资源\n");
        response.append("- 🔧 可使用 ").append(totalTools).append(" 个工具\n");
        response.append("- 🌐 连接到 ").append(capabilities.size()).append(" 个服务器\n\n");
        response.append("你可以让我：\n");
        response.append("- 搜索文件内容\n");
        response.append("- 分析数据统计\n");
        response.append("- 读取资源信息\n\n");
        response.append("请告诉我你需要什么帮助！");
        
        return response.toString();
    }
    
    // Getters
    public String getName() {
        return name;
    }
    
    public MCPClient getClient() {
        return client;
    }
    
    public List<Map<String, String>> getConversationHistory() {
        return conversationHistory;
    }
}
