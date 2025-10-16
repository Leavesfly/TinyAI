package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

/**
 * MCP (Model Context Protocol) 完整演示
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPDemo {
    
    /**
     * 演示 1: MCP 基础功能
     */
    public static void demoBasicMCP() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("📋 演示 1: MCP 基础功能");
        System.out.println("=".repeat(70));
        
        // 创建服务器
        FileSystemMCPServer server = new FileSystemMCPServer();
        
        // 创建客户端
        MCPClient client = new MCPClient();
        client.connect("filesystem", server);
        
        // 列出资源
        System.out.println("\n📦 可用资源：");
        List<Map<String, Object>> resources = client.listResources("filesystem");
        for (Map<String, Object> res : resources) {
            System.out.println("  - " + res.get("name") + ": " + res.get("description"));
        }
        
        // 读取资源
        System.out.println("\n📖 读取资源内容：");
        Map<String, Object> content = client.readResource("filesystem", "file:///docs/readme.md");
        if (content != null) {
            System.out.println("  URI: " + content.get("uri"));
            String contentStr = content.get("content").toString();
            String preview = contentStr.length() > 100 ? contentStr.substring(0, 100) + "..." : contentStr;
            System.out.println("  内容: " + preview);
        }
        
        // 列出工具
        System.out.println("\n🔧 可用工具：");
        List<Map<String, Object>> tools = client.listTools("filesystem");
        for (Map<String, Object> tool : tools) {
            System.out.println("  - " + tool.get("name") + ": " + tool.get("description"));
        }
        
        // 调用工具
        System.out.println("\n🔍 调用搜索工具：");
        Map<String, Object> args = new HashMap<>();
        args.put("keyword", "API");
        Map<String, Object> result = client.callTool("filesystem", "search_files", args);
        System.out.println("  结果: " + formatJson(result));
        
        // 获取提示词
        System.out.println("\n📝 获取提示词模板：");
        Map<String, Object> promptArgs = new HashMap<>();
        promptArgs.put("filename", "readme.md");
        promptArgs.put("content", "示例文档内容");
        String prompt = client.getPrompt("filesystem", "analyze_file", promptArgs);
        if (prompt != null) {
            String promptPreview = prompt.length() > 200 ? prompt.substring(0, 200) + "..." : prompt;
            System.out.println("  " + promptPreview);
        }
    }
    
    /**
     * 演示 2: 数据分析 MCP Server
     */
    @SuppressWarnings("unchecked")
    public static void demoDataAnalysis() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("📊 演示 2: 数据分析 MCP Server");
        System.out.println("=".repeat(70));
        
        // 创建数据分析服务器
        DataAnalysisMCPServer server = new DataAnalysisMCPServer();
        
        // 创建客户端
        MCPClient client = new MCPClient();
        client.connect("dataanalysis", server);
        
        // 查询数据
        System.out.println("\n🔍 查询用户数据：");
        Map<String, Object> queryArgs = new HashMap<>();
        queryArgs.put("data_uri", "db://users");
        queryArgs.put("filter_field", "city");
        queryArgs.put("filter_value", "北京");
        
        Map<String, Object> result = client.callTool("dataanalysis", "query_data", queryArgs);
        System.out.println("  " + formatJson(result));
        
        // 统计分析
        System.out.println("\n📈 销售额统计分析：");
        Map<String, Object> statsArgs = new HashMap<>();
        statsArgs.put("data_uri", "db://sales");
        statsArgs.put("field", "amount");
        
        result = client.callTool("dataanalysis", "calculate_statistics", statsArgs);
        System.out.println("  " + formatJson(result));
    }
    
    /**
     * 演示 3: AI Agent 使用 MCP
     */
    public static void demoAgentWithMCP() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("🤖 演示 3: AI Agent 使用 MCP");
        System.out.println("=".repeat(70));
        
        // 创建服务器
        FileSystemMCPServer fsServer = new FileSystemMCPServer();
        DataAnalysisMCPServer daServer = new DataAnalysisMCPServer();
        
        // 创建 Agent
        MCPEnabledAgent agent = new MCPEnabledAgent("智能助手");
        agent.connectToServer("filesystem", fsServer);
        agent.connectToServer("dataanalysis", daServer);
        
        // 发现能力
        System.out.println("\n🔍 发现 Agent 能力...");
        Map<String, Map<String, Object>> capabilities = agent.discoverCapabilities();
        System.out.println("  连接的服务器: " + capabilities.keySet());
        
        // 测试查询
        String[] testQueries = {
            "你好，介绍一下自己",
            "搜索 API 相关的文档",
            "统计销售数据",
            "查看所有可用资源"
        };
        
        for (String query : testQueries) {
            System.out.println("\n👤 用户: " + query);
            String response = agent.processQuery(query);
            System.out.println("🤖 助手: " + response);
            System.out.println("-".repeat(70));
        }
    }
    
    /**
     * 交互式演示
     */
    public static void interactiveDemo() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("💬 演示 4: 交互式 MCP Agent");
        System.out.println("=".repeat(70));
        
        // 创建完整的 MCP 生态系统
        FileSystemMCPServer fsServer = new FileSystemMCPServer();
        DataAnalysisMCPServer daServer = new DataAnalysisMCPServer();
        
        MCPEnabledAgent agent = new MCPEnabledAgent("MCP 智能助手");
        agent.connectToServer("filesystem", fsServer);
        agent.connectToServer("dataanalysis", daServer);
        
        System.out.println("\n✅ MCP Agent 已准备就绪！");
        System.out.println("\n可用命令：");
        System.out.println("  - 搜索 <关键词>");
        System.out.println("  - 统计分析");
        System.out.println("  - 查看资源");
        System.out.println("  - 能力展示");
        System.out.println("  - quit 退出");
        System.out.println("\n" + "=".repeat(70));
        
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            try {
                System.out.print("\n👤 你: ");
                String userInput = scanner.nextLine().trim();
                
                if (userInput.isEmpty()) {
                    continue;
                }
                
                if (userInput.equalsIgnoreCase("quit") || 
                    userInput.equalsIgnoreCase("exit") || 
                    userInput.equals("退出")) {
                    System.out.println("\n👋 再见！");
                    break;
                }
                
                if (userInput.equals("能力展示")) {
                    Map<String, Map<String, Object>> caps = agent.discoverCapabilities();
                    System.out.println("\n📊 当前连接的 MCP 服务器：");
                    for (Map.Entry<String, Map<String, Object>> entry : caps.entrySet()) {
                        String serverName = entry.getKey();
                        Map<String, Object> cap = entry.getValue();
                        System.out.println("\n  🌐 " + serverName + ":");
                        System.out.println("    - 资源: " + ((List<?>) cap.get("resources")).size() + " 个");
                        System.out.println("    - 工具: " + ((List<?>) cap.get("tools")).size() + " 个");
                        System.out.println("    - 提示词: " + ((List<?>) cap.get("prompts")).size() + " 个");
                    }
                    continue;
                }
                
                String response = agent.processQuery(userInput);
                System.out.println("\n🤖 助手: " + response);
                
            } catch (Exception e) {
                System.out.println("\n❌ 错误: " + e.getMessage());
            }
        }
        
        scanner.close();
    }
    
    /**
     * 格式化 JSON 输出
     */
    private static String formatJson(Object obj) {
        if (obj instanceof Map) {
            StringBuilder sb = new StringBuilder("{\n");
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) obj;
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                sb.append("    \"").append(entry.getKey()).append("\": ");
                if (entry.getValue() instanceof String) {
                    sb.append("\"").append(entry.getValue()).append("\"");
                } else {
                    sb.append(entry.getValue());
                }
                sb.append(",\n");
            }
            if (sb.length() > 2) {
                sb.setLength(sb.length() - 2); // 移除最后的逗号
                sb.append("\n");
            }
            sb.append("  }");
            return sb.toString();
        }
        return obj.toString();
    }
    
    /**
     * 主函数
     */
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("🚀 MCP (Model Context Protocol) 完整演示");
        System.out.println("=".repeat(70));
        System.out.println("""
                
                MCP 是一个标准化协议，用于连接 AI 应用与外部资源、工具。
                
                本演示包含：
                1. 基础 MCP 功能（资源、工具、提示词）
                2. 数据分析 MCP Server
                3. AI Agent 使用 MCP
                4. 交互式演示
                
                选择演示模式：
                1 - 基础功能演示
                2 - 数据分析演示
                3 - Agent 使用演示
                4 - 交互式演示
                5 - 全部演示
                0 - 退出
                """);
        
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            try {
                System.out.print("\n请选择 (0-5): ");
                String choice = scanner.nextLine().trim();
                
                switch (choice) {
                    case "0":
                        System.out.println("\n👋 感谢使用 MCP 演示系统！");
                        scanner.close();
                        return;
                    case "1":
                        demoBasicMCP();
                        break;
                    case "2":
                        demoDataAnalysis();
                        break;
                    case "3":
                        demoAgentWithMCP();
                        break;
                    case "4":
                        interactiveDemo();
                        break;
                    case "5":
                        demoBasicMCP();
                        demoDataAnalysis();
                        demoAgentWithMCP();
                        break;
                    default:
                        System.out.println("❌ 无效选择，请输入 0-5");
                }
            } catch (Exception e) {
                System.out.println("\n❌ 发生错误: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
}
