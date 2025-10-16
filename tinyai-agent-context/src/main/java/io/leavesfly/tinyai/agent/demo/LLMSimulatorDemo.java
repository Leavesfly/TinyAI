package io.leavesfly.tinyai.agent.demo;

import java.util.*;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * LLMSimulator演示示例
 * 展示如何使用LLM模拟器进行对话
 *
 * @author 山泽
 */
public class LLMSimulatorDemo {

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
        System.out.println("🤖 LLM模拟器演示");
        System.out.println(repeat("=", 60));

        // 创建LLM模拟器
        LLMSimulator llmSimulator = new LLMSimulator();

        // 显示基本信息
        System.out.println("\n📊 LLM模拟器信息:");
        System.out.println("  模型名称: " + llmSimulator.getModelName());
        System.out.println("  温度参数: " + llmSimulator.getTemperature());
        System.out.println("  最大token数: " + llmSimulator.getMaxTokens());

        // 演示不同类型Agent的对话
        demonstrateAgentTypes(llmSimulator);

        // 演示异步调用
        demonstrateAsyncCall(llmSimulator);

        // 演示系统提示生成
        demonstrateSystemPrompts(llmSimulator);

        System.out.println("\n👋 演示结束！");
    }

    /**
     * 演示不同类型Agent的对话
     */
    private static void demonstrateAgentTypes(LLMSimulator llmSimulator) {
        System.out.println("\n💬 演示不同类型Agent的对话:");
        System.out.println(repeat("-", 50));

        // 定义不同类型的Agent和查询
        Map<String, String> agentQueries = new HashMap<>();
        agentQueries.put("analyst", "请分析一下最新的市场数据趋势");
        agentQueries.put("researcher", "请研究一下人工智能在医疗领域的应用");
        agentQueries.put("coordinator", "请协调团队完成项目里程碑");
        agentQueries.put("executor", "请执行数据备份和清理任务");
        agentQueries.put("critic", "请评估这个产品的用户界面设计");

        for (Map.Entry<String, String> entry : agentQueries.entrySet()) {
            String agentType = entry.getKey();
            String query = entry.getValue();

            System.out.println(String.format("\n🎭 %s类型Agent:", agentType));
            System.out.println("👤 用户: " + query);

            // 构建消息列表
            List<Map<String, String>> messages = new ArrayList<>();

            // 添加系统消息
            Map<String, String> systemMessage = new HashMap<>();
            systemMessage.put("role", "system");
            systemMessage.put("content", llmSimulator.generateSystemPrompt(agentType,
                    agentType + "助手", agentType + "专家"));
            messages.add(systemMessage);

            // 添加用户消息
            Map<String, String> userMessage = new HashMap<>();
            userMessage.put("role", "user");
            userMessage.put("content", query);
            messages.add(userMessage);

            // 获取响应
            String response = llmSimulator.chatCompletion(messages, agentType);
            System.out.println("🤖 " + agentType + "助手: " + response);
        }
    }

    /**
     * 演示异步调用
     */
    private static void demonstrateAsyncCall(LLMSimulator llmSimulator) {
        System.out.println("\n🚀 演示异步LLM调用:");
        System.out.println(repeat("-", 40));

        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", "请同时处理多个复杂的分析任务");
        messages.add(userMessage);

        System.out.println("👤 用户: " + userMessage.get("content"));
        System.out.println("⏳ 正在异步处理...");

        try {
            // 异步调用
            llmSimulator.chatCompletionAsync(messages, "analyst")
                    .thenAccept(response -> {
                        System.out.println("✅ 异步完成!");
                        System.out.println("🤖 分析师: " + response);
                    })
                    .get(); // 等待完成以便显示结果

        } catch (Exception e) {
            System.out.println("❌ 异步调用失败: " + e.getMessage());
        }
    }

    /**
     * 演示系统提示生成
     */
    private static void demonstrateSystemPrompts(LLMSimulator llmSimulator) {
        System.out.println("\n📝 演示系统提示生成:");
        System.out.println(repeat("-", 40));

        String[] agentTypes = {"analyst", "researcher", "coordinator", "executor", "critic"};
        String[] agentNames = {"数据分析师", "研究员", "项目协调员", "执行专家", "质量评审员"};
        String[] roles = {"数据分析专家", "科研专家", "项目管理专家", "任务执行专家", "质量控制专家"};

        for (int i = 0; i < agentTypes.length; i++) {
            String agentType = agentTypes[i];
            String agentName = agentNames[i];
            String role = roles[i];

            String systemPrompt = llmSimulator.generateSystemPrompt(agentType, agentName, role);

            System.out.println(String.format("\n🎯 %s类型Agent:", agentType));
            System.out.println("系统提示: " + systemPrompt);
        }
    }

    /**
     * 交互式LLM演示
     */
    public static void interactiveDemo() {
        Scanner scanner = new Scanner(System.in);
        LLMSimulator llmSimulator = new LLMSimulator();

        System.out.println("🚀 启动交互式LLM模拟器演示...");
        System.out.println("💡 输入 'help' 查看帮助，输入 'quit' 退出");
        System.out.println("🎭 可选Agent类型: analyst, researcher, coordinator, executor, critic");
        System.out.println(repeat("-", 60));

        String currentAgentType = "general";

        while (true) {
            System.out.print(String.format("\n[%s] 👤 你: ", currentAgentType));
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                System.out.println("\n👋 再见！感谢使用LLM模拟器！");
                break;
            }

            if (input.equalsIgnoreCase("help")) {
                printInteractiveHelp();
                continue;
            }

            // 切换Agent类型
            if (input.startsWith("switch ")) {
                String newType = input.substring(7).trim();
                if (Arrays.asList("analyst", "researcher", "coordinator", "executor", "critic", "general")
                        .contains(newType)) {
                    currentAgentType = newType;
                    System.out.println("✅ 已切换到 " + currentAgentType + " 类型Agent");
                } else {
                    System.out.println("❌ 无效的Agent类型。可用类型: analyst, researcher, coordinator, executor, critic, general");
                }
                continue;
            }

            if (input.isEmpty()) {
                continue;
            }

            try {
                // 构建消息
                List<Map<String, String>> messages = new ArrayList<>();

                // 添加系统消息（如果不是general类型）
                if (!"general".equals(currentAgentType)) {
                    Map<String, String> systemMessage = new HashMap<>();
                    systemMessage.put("role", "system");
                    systemMessage.put("content", llmSimulator.generateSystemPrompt(
                            currentAgentType, currentAgentType + "助手", currentAgentType + "专家"));
                    messages.add(systemMessage);
                }

                // 添加用户消息
                Map<String, String> userMessage = new HashMap<>();
                userMessage.put("role", "user");
                userMessage.put("content", input);
                messages.add(userMessage);

                // 获取响应
                String response = llmSimulator.chatCompletion(messages, currentAgentType);
                System.out.println("🤖 " + currentAgentType + "助手: " + response);

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
        System.out.println("\n💡 交互式LLM模拟器帮助:");
        System.out.println("- 直接输入消息与当前Agent类型对话");
        System.out.println("- 'switch <type>' - 切换Agent类型 (analyst/researcher/coordinator/executor/critic/general)");
        System.out.println("- 'help' - 显示此帮助信息");
        System.out.println("- 'quit' 或 'exit' - 退出程序");
        System.out.println("");
        System.out.println("🎭 Agent类型说明:");
        System.out.println("- analyst: 数据分析专家，擅长数据分析和趋势预测");
        System.out.println("- researcher: 研究专家，擅长文献调研和理论分析");
        System.out.println("- coordinator: 协调专家，擅长任务分配和团队管理");
        System.out.println("- executor: 执行专家，擅长具体任务的执行和实施");
        System.out.println("- critic: 评审专家，擅长质量评估和改进建议");
        System.out.println("- general: 通用助手，提供综合性帮助");
    }
}