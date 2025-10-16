package io.leavesfly.tinyai.agent.demo;

import io.leavesfly.tinyai.agent.context.AdvancedAgent;
import io.leavesfly.tinyai.agent.context.LLMSimulator;
import io.leavesfly.tinyai.agent.context.Memory;
import io.leavesfly.tinyai.agent.context.MemoryManager;
import io.leavesfly.tinyai.agent.context.RAGSystem;
import io.leavesfly.tinyai.agent.context.RetrievalResult;
import io.leavesfly.tinyai.agent.context.ToolCall;

import java.util.*;

/**
 * AdvancedAgent 演示示例
 * 展示如何使用高级LLM Agent系统
 * 
 * @author 山泽
 */
public class AgentDemo {
    
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
        System.out.println("🤖 高级LLM Agent系统演示");
        System.out.println(repeat("=", 60));
        
        // 创建高级Agent
        AdvancedAgent agent = new AdvancedAgent(
            "高级助手",
            "你是一个智能助手，拥有记忆、知识库和工具使用能力。你可以帮助用户解决各种问题。"
        );
        
        // 添加一些知识到RAG系统
        addKnowledgeBase(agent);
        
        // 显示Agent统计信息
        System.out.println("\n📊 Agent统计信息:");
        Map<String, Object> stats = agent.getStats();
        stats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
        
        // 演示对话交互
        demonstrateConversations(agent);
        
        // 演示工具使用
        demonstrateTools(agent);
        
        // 演示记忆检索
        demonstrateMemoryRetrieval(agent);
        
        // 演示RAG功能
        demonstrateRAG(agent);
        
        // 演示LLM模拟器功能
        demonstrateLLMSimulator(agent);
        
        // 最终统计
        System.out.println("\n📊 最终统计信息:");
        Map<String, Object> finalStats = agent.getStats();
        finalStats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
        
        System.out.println("\n👋 演示结束！");
    }
    
    /**
     * 添加知识库
     */
    private static void addKnowledgeBase(AdvancedAgent agent) {
        System.out.println("\n📚 添加知识库...");
        
        String[] knowledgeBase = {
            "人工智能（AI）是由人类开发的智能系统，能够执行通常需要人类智能的任务。",
            "机器学习是人工智能的一个分支，使用统计技术使计算机能够在没有明确编程的情况下学习。",
            "深度学习是机器学习的一个子集，它模仿人脑的神经网络结构。",
            "Java是一种高级编程语言，广泛用于企业级应用开发。",
            "大语言模型（LLM）是一种基于深度学习的人工智能模型，能够理解和生成人类语言。"
        };
        
        for (int i = 0; i < knowledgeBase.length; i++) {
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("topic", "AI知识");
            metadata.put("source", "知识库");
            agent.addKnowledge(knowledgeBase[i], "kb_" + (i + 1), metadata);
            System.out.println("  已添加: kb_" + (i + 1));
        }
    }
    
    /**
     * 演示对话交互
     */
    private static void demonstrateConversations(AdvancedAgent agent) {
        System.out.println("\n💬 演示对话交互:");
        System.out.println(repeat("-", 40));
        
        String[] testQueries = {
            "你好，我想了解人工智能",
            "什么是机器学习？",
            "Java语言有什么特点？",
            "请帮我记住今天要开会"
        };
        
        for (String query : testQueries) {
            System.out.println("\n👤 用户: " + query);
            String response = agent.processMessage(query);
            System.out.println("🤖 助手: " + response);
        }
    }
    
    /**
     * 演示工具使用
     */
    private static void demonstrateTools(AdvancedAgent agent) {
        System.out.println("\n🔧 演示工具使用:");
        System.out.println(repeat("-", 40));
        
        // 测试计算器工具
        System.out.println("\n测试计算器工具:");
        Map<String, Object> calcArgs = new HashMap<>();
        calcArgs.put("operation", "add");
        calcArgs.put("a", 10);
        calcArgs.put("b", 5);
        
        ToolCall calcResult = agent.getToolRegistry().callTool("calculator", calcArgs);
        System.out.println("计算结果: " + calcResult);
        System.out.println("执行结果: " + calcResult.getResult());
        
        // 测试时间工具
        System.out.println("\n测试时间工具:");
        ToolCall timeResult = agent.getToolRegistry().callTool("time", new HashMap<>());
        System.out.println("时间结果: " + timeResult);
        System.out.println("执行结果: " + timeResult.getResult());
        
        // 测试笔记工具
        System.out.println("\n测试笔记工具:");
        Map<String, Object> noteArgs = new HashMap<>();
        noteArgs.put("action", "create");
        noteArgs.put("content", "学习Java编程");
        
        ToolCall noteResult = agent.getToolRegistry().callTool("note", noteArgs);
        System.out.println("笔记结果: " + noteResult);
        System.out.println("执行结果: " + noteResult.getResult());
        
        // 列出笔记
        System.out.println("\n列出所有笔记:");
        Map<String, Object> listArgs = new HashMap<>();
        listArgs.put("action", "list");
        
        ToolCall listResult = agent.getToolRegistry().callTool("note", listArgs);
        System.out.println("笔记列表: " + listResult.getResult());
    }
    
    /**
     * 演示记忆检索
     */
    private static void demonstrateMemoryRetrieval(AdvancedAgent agent) {
        System.out.println("\n🧠 演示记忆检索:");
        System.out.println(repeat("-", 40));
        
        // 添加一些记忆
        MemoryManager memoryManager = agent.getMemoryManager();
        memoryManager.addMemory("用户喜欢学习编程", "episodic", 0.8);
        memoryManager.addMemory("今天讨论了AI话题", "episodic", 0.7);
        memoryManager.addMemory("用户询问了Java语言", "working", 0.6);
        
        // 检索相关记忆
        String[] queries = {"编程", "人工智能", "学习"};
        
        for (String query : queries) {
            System.out.println("\n查询: '" + query + "'");
            List<Memory> memories = memoryManager.retrieveMemories(query, 2);
            for (Memory memory : memories) {
                System.out.println("  - [" + memory.getMemoryType() + "] " + memory.getContent() + 
                                 " (重要性: " + memory.getImportance() + ")");
            }
        }
        
        // 显示记忆统计
        System.out.println("\n记忆统计:");
        Map<String, Object> memStats = memoryManager.getMemoryStats();
        memStats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
    }
    
    /**
     * 演示RAG功能
     */
    private static void demonstrateRAG(AdvancedAgent agent) {
        System.out.println("\n📚 演示RAG检索功能:");
        System.out.println(repeat("-", 40));
        
        RAGSystem ragSystem = agent.getRagSystem();
        
        // 测试检索
        String[] testQueries = {
            "人工智能",
            "机器学习算法",
            "Java编程语言",
            "深度学习网络"
        };
        
        for (String query : testQueries) {
            System.out.println("\n查询: '" + query + "'");
            List<RetrievalResult> results = ragSystem.retrieve(query, 2);
            
            for (int i = 0; i < results.size(); i++) {
                RetrievalResult result = results.get(i);
                System.out.println(String.format("  %d. 文档: %s (相似度: %.3f)", 
                                                i + 1, 
                                                result.getDocument().getId(), 
                                                result.getSimilarity()));
                System.out.println("     内容: " + result.getDocument().getContent());
            }
        }
        
        // 显示RAG统计
        System.out.println("\nRAG统计:");
        Map<String, Object> ragStats = ragSystem.getStats();
        ragStats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
    }
    
    /**
     * 交互式演示（控制台输入）
     */
    public static void interactiveDemo() {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("🚀 启动交互式Agent演示...");
        AdvancedAgent agent = new AdvancedAgent(
            "交互助手",
            "你是一个智能助手，能够帮助用户解决各种问题。"
        );
        
        // 添加知识库
        addKnowledgeBase(agent);
        
        System.out.println("\n💡 输入 'help' 查看帮助，输入 'quit' 退出");
        System.out.println(repeat("-", 50));
        
        while (true) {
            System.out.print("\n👤 你: ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                System.out.println("\n👋 再见！感谢使用Agent系统！");
                break;
            }
            
            if (input.equalsIgnoreCase("help")) {
                printHelp();
                continue;
            }
            
            if (input.equalsIgnoreCase("stats")) {
                System.out.println("\n📊 Agent统计信息:");
                Map<String, Object> stats = agent.getStats();
                stats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
                continue;
            }
            
            if (input.isEmpty()) {
                continue;
            }
            
            try {
                String response = agent.processMessage(input);
                System.out.println("🤖 " + agent.getName() + ": " + response);
            } catch (Exception e) {
                System.out.println("❌ 处理消息时出错: " + e.getMessage());
            }
        }
        
        scanner.close();
    }
    
    /**
     * 演示LLM模拟器功能
     */
    private static void demonstrateLLMSimulator(AdvancedAgent agent) {
        System.out.println("\n🤖 演示LLM模拟器功能:");
        System.out.println(repeat("-", 40));
        
        LLMSimulator llmSimulator = agent.getLLMSimulator();
        
        // 测试不同类型Agent的回复
        String[] agentTypes = {"analyst", "researcher", "coordinator", "executor", "critic"};
        String[] testQueries = {
            "请分析一下市场数据趋势",
            "请研究一下人工智能技术",
            "请协调一下团队任务分配",
            "请执行数据处理任务",
            "请评估产品质量"
        };
        
        for (int i = 0; i < agentTypes.length; i++) {
            String agentType = agentTypes[i];
            String query = testQueries[i];
            
            System.out.println(String.format("\n📝 测试%s类型Agent:", agentType));
            System.out.println("👤 用户: " + query);
            
            // 构建消息列表
            List<Map<String, String>> messages = new ArrayList<>();
            Map<String, String> systemMsg = new HashMap<>();
            systemMsg.put("role", "system");
            systemMsg.put("content", llmSimulator.generateSystemPrompt(agentType, 
                agentType + "助手", agentType + "专家"));
            messages.add(systemMsg);
            
            Map<String, String> userMsg = new HashMap<>();
            userMsg.put("role", "user");
            userMsg.put("content", query);
            messages.add(userMsg);
            
            // 获取响应
            String response = llmSimulator.chatCompletion(messages, agentType);
            System.out.println("🤖 " + agentType + "助手: " + response);
        }
        
        // 演示异步调用
        System.out.println("\n🚀 演示异步LLM调用:");
        
        List<Map<String, String>> asyncMessages = new ArrayList<>();
        Map<String, String> asyncUserMsg = new HashMap<>();
        asyncUserMsg.put("role", "user");
        asyncUserMsg.put("content", "请同时分析多个数据源");
        asyncMessages.add(asyncUserMsg);
        
        try {
            // 异步调用
            llmSimulator.chatCompletionAsync(asyncMessages, "analyst")
                .thenAccept(response -> {
                    System.out.println("📊 异步回复: " + response);
                })
                .get(); // 等待完成以便显示
        } catch (Exception e) {
            System.out.println("❌ 异步调用失败: " + e.getMessage());
        }
        
        // 显示LLM模拟器信息
        System.out.println("\n📊 LLM模拟器信息:");
        System.out.println("  模型名称: " + llmSimulator.getModelName());
        System.out.println("  温度参数: " + llmSimulator.getTemperature());
        System.out.println("  最大token数: " + llmSimulator.getMaxTokens());
    }
    
    /**
     * 打印帮助信息
     */
    private static void printHelp() {
        System.out.println("\n💡 帮助信息:");
        System.out.println("- 与 Agent 进行自然对话");
        System.out.println("- 'stats' - 查看 Agent 统计信息");
        System.out.println("- 'quit' 或 'exit' - 退出程序");
        System.out.println("");
        System.out.println("🔧 可用功能:");
        System.out.println("- 知识问答: '什么是人工智能?'");
        System.out.println("- 记忆对话: Agent会记住对话内容");
        System.out.println("- 上下文理解: 基于历史对话提供回答");
    }
}