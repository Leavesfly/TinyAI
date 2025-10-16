package io.leavesfly.tinyai.agent.demo;

import java.util.*;

import io.leavesfly.tinyai.agent.context.SimplifiedAdvancedAgent;

/**
 * 简化版AdvancedAgent演示
 * 展示LLMSimulator集成后的对话效果
 * 
 * @author 山泽
 */
public class SimplifiedAgentDemo {
    
    // 工具方法
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    public static void main(String[] args) {
        System.out.println(repeat("=", 60));
        System.out.println("🤖 简化版AdvancedAgent + LLMSimulator演示");
        System.out.println(repeat("=", 60));
        
        // 创建不同类型的简化Agent
        SimplifiedAdvancedAgent generalAgent = new SimplifiedAdvancedAgent("通用助手");
        SimplifiedAdvancedAgent analystAgent = new SimplifiedAdvancedAgent("数据分析师", 
            "你是一位专业的数据分析师，擅长数据分析、趋势预测和数据可视化。请用专业的方式回答问题。");
        SimplifiedAdvancedAgent researcherAgent = new SimplifiedAdvancedAgent("研究员",
            "你是一位严谨的研究员，擅长文献调研、实验设计和理论分析。请提供基于证据的回答。");
        
        // 演示基本对话功能
        demonstrateBasicConversation(generalAgent);
        
        // 演示专业Agent对话
        demonstrateSpecializedAgents(analystAgent, researcherAgent);
        
        // 演示对话历史管理
        demonstrateConversationHistory(generalAgent);
        
        // 演示统计信息
        demonstrateStats(generalAgent, analystAgent, researcherAgent);
        
        System.out.println("\n👋 演示结束！");
    }
    
    /**
     * 演示基本对话功能
     */
    private static void demonstrateBasicConversation(SimplifiedAdvancedAgent agent) {
        System.out.println("\n💬 基本对话功能演示:");
        System.out.println(repeat("-", 40));
        
        String[] queries = {
            "你好，请介绍一下你自己",
            "你能帮我做什么？",
            "谢谢你的帮助"
        };
        
        for (String query : queries) {
            System.out.println("\n👤 用户: " + query);
            String response = agent.processMessage(query);
            System.out.println("🤖 " + agent.getName() + ": " + response);
        }
    }
    
    /**
     * 演示专业Agent对话
     */
    private static void demonstrateSpecializedAgents(SimplifiedAdvancedAgent analystAgent, 
                                                   SimplifiedAdvancedAgent researcherAgent) {
        System.out.println("\n🎯 专业Agent对话演示:");
        System.out.println(repeat("-", 40));
        
        // 数据分析师对话
        System.out.println("\n📊 数据分析师对话:");
        String[] analystQueries = {
            "请分析一下电商行业的发展趋势",
            "如何提高数据可视化的效果？"
        };
        
        for (String query : analystQueries) {
            System.out.println("\n👤 用户: " + query);
            String response = analystAgent.processMessage(query);
            System.out.println("🤖 " + analystAgent.getName() + ": " + response);
        }
        
        // 研究员对话
        System.out.println("\n🔬 研究员对话:");
        String[] researchQueries = {
            "人工智能在教育领域有哪些应用？",
            "如何设计一个有效的用户体验研究？"
        };
        
        for (String query : researchQueries) {
            System.out.println("\n👤 用户: " + query);
            String response = researcherAgent.processMessage(query);
            System.out.println("🤖 " + researcherAgent.getName() + ": " + response);
        }
    }
    
    /**
     * 演示对话历史管理
     */
    private static void demonstrateConversationHistory(SimplifiedAdvancedAgent agent) {
        System.out.println("\n📚 对话历史管理演示:");
        System.out.println(repeat("-", 40));
        
        // 进行一些对话
        System.out.println("\n进行连续对话...");
        String[] conversation = {
            "我想学习编程",
            "应该从哪种语言开始？",
            "Java难学吗？"
        };
        
        for (String query : conversation) {
            System.out.println("👤 用户: " + query);
            String response = agent.processMessage(query);
            System.out.println("🤖 " + agent.getName() + ": " + response);
        }
        
        // 显示对话历史
        System.out.println("\n📋 对话历史:");
        List<Map<String, Object>> history = agent.exportConversation();
        for (int i = 0; i < history.size(); i++) {
            Map<String, Object> msg = history.get(i);
            System.out.println(String.format("%d. [%s] %s", 
                i + 1, msg.get("role"), msg.get("content")));
        }
        
        // 清空对话历史
        System.out.println("\n🗑️ 清空对话历史...");
        agent.clearConversation();
        System.out.println("✅ 对话历史已清空，当前对话数量: " + agent.getConversationHistory().size());
    }
    
    /**
     * 演示统计信息
     */
    private static void demonstrateStats(SimplifiedAdvancedAgent... agents) {
        System.out.println("\n📊 Agent统计信息:");
        System.out.println(repeat("-", 40));
        
        for (SimplifiedAdvancedAgent agent : agents) {
            System.out.println("\n🤖 " + agent.getName() + ":");
            Map<String, Object> stats = agent.getStats();
            stats.forEach((key, value) -> 
                System.out.println("  " + key + ": " + value));
        }
    }
    
    /**
     * 交互式演示
     */
    public static void interactiveDemo() {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("🚀 启动交互式简化Agent演示...");
        System.out.println("💡 输入 'help' 查看帮助，输入 'quit' 退出");
        System.out.println("🎭 输入 'switch' 切换Agent类型");
        System.out.println(repeat("-", 50));
        
        // 创建不同类型的Agent
        Map<String, SimplifiedAdvancedAgent> agents = new HashMap<>();
        agents.put("general", new SimplifiedAdvancedAgent("通用助手"));
        agents.put("analyst", new SimplifiedAdvancedAgent("数据分析师", 
            "你是一位专业的数据分析师，擅长数据分析、趋势预测和数据可视化。"));
        agents.put("researcher", new SimplifiedAdvancedAgent("研究员",
            "你是一位严谨的研究员，擅长文献调研、实验设计和理论分析。"));
        
        String currentAgentType = "general";
        SimplifiedAdvancedAgent currentAgent = agents.get(currentAgentType);
        
        while (true) {
            System.out.print(String.format("\n[%s] 👤 你: ", currentAgentType));
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                System.out.println("\n👋 再见！感谢使用简化Agent系统！");
                break;
            }
            
            if (input.equalsIgnoreCase("help")) {
                printInteractiveHelp();
                continue;
            }
            
            if (input.equalsIgnoreCase("switch")) {
                System.out.println("可用的Agent类型:");
                agents.keySet().forEach(type -> System.out.println("  - " + type));
                System.out.print("请选择Agent类型: ");
                String newType = scanner.nextLine().trim();
                
                if (agents.containsKey(newType)) {
                    currentAgentType = newType;
                    currentAgent = agents.get(currentAgentType);
                    System.out.println("✅ 已切换到 " + currentAgentType + " (" + currentAgent.getName() + ")");
                } else {
                    System.out.println("❌ 无效的Agent类型");
                }
                continue;
            }
            
            if (input.equalsIgnoreCase("stats")) {
                System.out.println("\n📊 当前Agent统计信息:");
                Map<String, Object> stats = currentAgent.getStats();
                stats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
                continue;
            }
            
            if (input.equalsIgnoreCase("clear")) {
                currentAgent.clearConversation();
                System.out.println("✅ 对话历史已清空");
                continue;
            }
            
            if (input.isEmpty()) {
                continue;
            }
            
            try {
                String response = currentAgent.processMessage(input);
                System.out.println("🤖 " + currentAgent.getName() + ": " + response);
            } catch (Exception e) {
                System.out.println("❌ 处理消息时出错: " + e.getMessage());
            }
        }
        
        scanner.close();
    }
    
    /**
     * 打印交互式帮助
     */
    private static void printInteractiveHelp() {
        System.out.println("\n💡 交互式简化Agent帮助:");
        System.out.println("- 直接输入消息与当前Agent对话");
        System.out.println("- 'switch' - 切换Agent类型");
        System.out.println("- 'stats' - 查看当前Agent统计信息");
        System.out.println("- 'clear' - 清空当前对话历史");
        System.out.println("- 'help' - 显示此帮助信息");
        System.out.println("- 'quit' 或 'exit' - 退出程序");
        System.out.println("");
        System.out.println("🎭 可用Agent类型:");
        System.out.println("- general: 通用助手，提供综合性帮助");
        System.out.println("- analyst: 数据分析师，专业的数据分析和趋势预测");
        System.out.println("- researcher: 研究员，严谨的文献调研和理论分析");
    }
}