package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;

import java.util.*;
import java.util.function.Function;

/**
 * OpenManus Agent系统演示程序
 * 完整演示OpenManus Agent系统的四大核心特征：
 * 1. 双执行机制演示
 * 2. 分层架构展示
 * 3. 计划驱动任务分解演示
 * 4. 动态工具调用演示
 * 
 * @author 山泽
 */
public class ManusDemo {
    
    private static final Scanner scanner = new Scanner(System.in);
    
    public static void main(String[] args) {
        System.out.println("🌟".repeat(30));
        System.out.println("OpenManus Agent系统Java版完整演示");
        System.out.println("🌟".repeat(30));
        
        System.out.println("""
                🎯 演示内容：
                1. 直接Agent模式 - 基础的推理与行动
                2. 计划驱动模式 - 复杂任务的分解与执行  
                3. Flow编排模式 - 工作流程的灵活编排
                4. 分层架构展示 - 四层架构的协同工作
                5. 工具管理系统 - 动态工具注册与调用
                6. 系统监控 - 实时状态监控与统计
                7. 交互式体验 - 与系统的实时互动
                
                OpenManus的四大核心特征：
                ✅ 双执行机制（直接Agent模式 & Flow编排模式）
                ✅ 分层架构（BaseAgent → ReActAgent → ToolCallAgent → Manus）
                ✅ 计划驱动任务分解
                ✅ 动态工具调用
                """);
        
        try {
            comprehensiveDemo();
        } catch (Exception e) {
            System.err.println("❌ 演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 综合演示
     */
    private static void comprehensiveDemo() {
        Map<String, Runnable> demoOptions = new LinkedHashMap<>();
        demoOptions.put("1", () -> demoDirectAgentMode());
        demoOptions.put("2", () -> demoPlanningDrivenMode());
        demoOptions.put("3", () -> demoFlowOrchestration());
        demoOptions.put("4", () -> demoLayeredArchitecture());
        demoOptions.put("5", () -> demoToolManagement());
        demoOptions.put("6", () -> demoSystemMonitoring());
        demoOptions.put("7", () -> interactiveDemo());
        demoOptions.put("8", () -> runAllDemos());
        
        while (true) {
            System.out.println("\n" + "=".repeat(50));
            System.out.println("选择演示内容:");
            System.out.println("1. 直接Agent模式演示");
            System.out.println("2. 计划驱动模式演示");
            System.out.println("3. Flow编排模式演示");
            System.out.println("4. 分层架构演示");
            System.out.println("5. 工具管理演示");
            System.out.println("6. 系统监控演示");
            System.out.println("7. 交互式演示");
            System.out.println("8. 全部演示");
            System.out.println("0. 退出");
            
            System.out.print("\n请选择 (0-8): ");
            String choice = scanner.nextLine().trim();
            
            if ("0".equals(choice)) {
                System.out.println("\n👋 感谢使用OpenManus Agent系统演示！");
                break;
            } else if (demoOptions.containsKey(choice)) {
                demoOptions.get(choice).run();
            } else {
                System.out.println("❌ 无效选择，请输入 0-8");
            }
        }
    }
    
    /**
     * 演示直接Agent模式
     */
    private static void demoDirectAgentMode() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🤖 OpenManus - 直接Agent模式演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Direct");
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(false); // 关闭计划模式，使用基础ReAct
        
        String[] testQueries = {
            "计算 25 * 8 + 15",
            "现在几点了？",
            "分析这个文本: 'OpenManus是一个强大的Agent系统'"
        };
        
        for (int i = 0; i < testQueries.length; i++) {
            String query = testQueries[i];
            System.out.printf("\n📝 测试 %d: %s\n", i + 1, query);
            System.out.println("-".repeat(40));
            
            Message message = new Message("user", query);
            Message response = manus.processMessage(message);
            
            System.out.printf("🎯 回答: %s\n", response.getContent());
            System.out.printf("📊 状态: %s\n", manus.getStatus().get("state"));
            
            try {
                Thread.sleep(1000); // 演示间隔
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    /**
     * 演示计划驱动模式
     */
    private static void demoPlanningDrivenMode() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("📋 OpenManus - 计划驱动模式演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Planning");
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(true); // 启用计划模式
        
        String[] complexQueries = {
            "详细分析计算 100 * 25 的结果",
            "研究当前时间并进行深入分析",
            "制定一个完整的学习计划"
        };
        
        for (int i = 0; i < complexQueries.length; i++) {
            String query = complexQueries[i];
            System.out.printf("\n📝 复杂查询 %d: %s\n", i + 1, query);
            System.out.println("-".repeat(50));
            
            Message message = new Message("user", query);
            Message response = manus.processMessage(message);
            
            System.out.printf("🎯 计划执行结果:\n%s\n", response.getContent());
            
            Map<String, Object> metadata = response.getMetadata();
            if (metadata != null && !metadata.isEmpty()) {
                System.out.printf("📊 执行信息: %s\n", metadata);
            }
            
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    /**
     * 演示Flow编排模式
     */
    private static void demoFlowOrchestration() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🔄 OpenManus - Flow编排模式演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Flow");
        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
        
        String[] flowQueries = {
            "计算 15 + 25 * 3",
            "查询当前时间",
            "分析文本内容",
            "这是一个通用查询"  // 测试回退机制
        };
        
        for (int i = 0; i < flowQueries.length; i++) {
            String query = flowQueries[i];
            System.out.printf("\n📝 Flow查询 %d: %s\n", i + 1, query);
            System.out.println("-".repeat(40));
            
            Message message = new Message("user", query);
            Message response = manus.processMessage(message);
            
            System.out.printf("🎯 Flow执行结果: %s\n", response.getContent());
            
            Map<String, Object> metadata = response.getMetadata();
            if (metadata != null && !metadata.isEmpty()) {
                System.out.printf("📊 Flow信息: %s\n", metadata);
            }
            
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    /**
     * 演示分层架构
     */
    private static void demoLayeredArchitecture() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🏗️ OpenManus - 分层架构演示");
        System.out.println("=".repeat(60));
        
        System.out.println("创建各层Agent实例...");
        
        // 创建各层实例
        ReActAgent reactAgent = new ReActAgent("ReAct层Agent");
        ToolCallAgent toolcallAgent = new ToolCallAgent("工具调用层Agent");
        Manus manusAgent = new Manus("Manus核心层Agent");
        
        List<BaseAgent> agents = Arrays.asList(reactAgent, toolcallAgent, manusAgent);
        
        // 展示各层状态
        System.out.println("\n各层Agent状态信息:");
        for (BaseAgent agent : agents) {
            Map<String, Object> status = agent.getStatus();
            System.out.printf("📊 %s: %s\n", agent.getName(), status);
        }
        
        // 测试消息处理
        Message testMessage = new Message("user", "计算 10 + 20");
        
        System.out.printf("\n测试消息: %s\n", testMessage.getContent());
        System.out.println("-".repeat(40));
        
        for (BaseAgent agent : agents) {
            System.out.printf("\n%s 处理结果:\n", agent.getName());
            try {
                Message response = agent.processMessage(testMessage);
                String content = response.getContent();
                String preview = content.length() > 100 ? content.substring(0, 100) + "..." : content;
                System.out.printf("回答: %s\n", preview);
            } catch (Exception e) {
                System.out.printf("处理失败: %s\n", e.getMessage());
            }
        }
    }
    
    /**
     * 演示工具管理系统
     */
    private static void demoToolManagement() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🔧 OpenManus - 工具管理系统演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Tools");
        
        // 展示内置工具
        System.out.println("内置工具列表:");
        var tools = manus.getToolRegistry().listTools();
        for (var tool : tools) {
            System.out.printf("  🛠️ %s: %s\n", tool.getName(), tool.getDescription());
        }
        
        // 注册自定义工具
        System.out.println("\n注册自定义工具...");
        
        // 天气查询工具
        manus.registerCustomTool("weather", args -> {
            String city = (String) args.get("city");
            Map<String, String> weatherData = Map.of(
                "北京", "晴天 25°C",
                "上海", "多云 22°C", 
                "广州", "雨天 28°C"
            );
            Map<String, Object> result = new HashMap<>();
            result.put("city", city);
            result.put("weather", weatherData.getOrDefault(city, "暂无数据"));
            return result;
        }, "天气查询工具");
        
        // 翻译工具
        manus.registerCustomTool("translator", args -> {
            String text = (String) args.get("text");
            String targetLang = (String) args.getOrDefault("target_lang", "en");
            Map<String, String> translations = Map.of(
                "你好", "Hello",
                "谢谢", "Thank you",
                "再见", "Goodbye"
            );
            Map<String, Object> result = new HashMap<>();
            result.put("original", text);
            result.put("translated", translations.getOrDefault(text, String.format("[%s] %s", targetLang, text)));
            return result;
        }, "文本翻译工具");
        
        // 展示更新后的工具列表
        System.out.println("\n更新后的工具列表:");
        tools = manus.getToolRegistry().listTools();
        for (var tool : tools) {
            System.out.printf("  🛠️ %s: %s\n", tool.getName(), tool.getDescription());
        }
        
        // 展示工具统计
        Map<String, Integer> toolStats = manus.getToolStats();
        System.out.printf("\n📊 工具使用统计: %s\n", toolStats);
    }
    
    /**
     * 演示系统监控
     */
    private static void demoSystemMonitoring() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("📊 OpenManus - 系统监控演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Monitor");
        
        // 配置系统
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(true);
        
        // 处理一些消息来生成数据
        String[] testMessages = {
            "计算 100 + 200",
            "查询时间",
            "详细分析系统状态"
        };
        
        System.out.println("处理测试消息...");
        for (String msgContent : testMessages) {
            Message message = new Message("user", msgContent);
            manus.processMessage(message);
            
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        // 展示系统状态
        System.out.println("\n系统状态监控:");
        Map<String, Object> status = manus.getSystemStatus();
        
        for (Map.Entry<String, Object> entry : status.entrySet()) {
            System.out.printf("📈 %s: %s\n", entry.getKey(), entry.getValue());
        }
        
        // 展示消息历史
        List<Message> messages = manus.getMessages();
        System.out.printf("\n消息历史 (共 %d 条):\n", messages.size());
        int showCount = Math.min(6, messages.size());
        for (int i = messages.size() - showCount; i < messages.size(); i++) {
            Message msg = messages.get(i);
            String content = msg.getContent();
            String preview = content.length() > 50 ? content.substring(0, 50) + "..." : content;
            System.out.printf("  %d. [%s] %s\n", i + 1, msg.getRole(), preview);
        }
    }
    
    /**
     * 交互式演示
     */
    private static void interactiveDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("💬 OpenManus - 交互式演示");
        System.out.println("=".repeat(60));
        
        Manus manus = new Manus("OpenManus-Interactive");
        
        // 配置系统
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(true);
        
        System.out.println("🎮 OpenManus Agent系统已启动！");
        System.out.println("\n可用功能:");
        System.out.println("- 数学计算: '计算 10 + 20'");
        System.out.println("- 时间查询: '现在几点?'");
        System.out.println("- 文本分析: '分析这段文本'");
        System.out.println("- 复杂任务: '详细研究某个主题'");
        System.out.println("- 系统控制: 'mode:flow' (切换到Flow模式), 'status' (查看状态)");
        System.out.println("\n输入 'quit' 退出演示");
        System.out.println("-".repeat(60));
        
        while (true) {
            try {
                System.out.print("\n👤 你: ");
                String userInput = scanner.nextLine().trim();
                
                if (userInput.toLowerCase().matches("quit|exit|退出")) {
                    System.out.println("\n👋 感谢使用OpenManus Agent系统演示！");
                    break;
                }
                
                if (userInput.isEmpty()) {
                    continue;
                }
                
                // 处理系统命令
                if (userInput.startsWith("mode:")) {
                    String modeName = userInput.split(":", 2)[1].trim();
                    if ("flow".equals(modeName)) {
                        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
                        System.out.println("🔄 已切换到Flow编排模式");
                    } else if ("direct".equals(modeName)) {
                        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
                        System.out.println("🤖 已切换到直接Agent模式");
                    } else {
                        System.out.println("❌ 不支持的模式");
                    }
                    continue;
                }
                
                if ("status".equals(userInput)) {
                    Map<String, Object> status = manus.getSystemStatus();
                    System.out.println("\n📊 系统状态:");
                    for (Map.Entry<String, Object> entry : status.entrySet()) {
                        System.out.printf("  %s: %s\n", entry.getKey(), entry.getValue());
                    }
                    continue;
                }
                
                // 处理用户消息
                Message message = new Message("user", userInput);
                Message response = manus.processMessage(message);
                
                System.out.printf("\n🤖 Manus: %s\n", response.getContent());
                
                Map<String, Object> metadata = response.getMetadata();
                if (metadata != null && !metadata.isEmpty()) {
                    System.out.printf("💡 执行信息: %s\n", metadata);
                }
                
            } catch (Exception e) {
                System.out.printf("\n❌ 发生错误: %s\n", e.getMessage());
            }
        }
    }
    
    /**
     * 运行所有演示
     */
    private static void runAllDemos() {
        System.out.println("\n🎬 开始全部演示...");
        
        Runnable[] demos = {
            ManusDemo::demoDirectAgentMode,
            ManusDemo::demoPlanningDrivenMode, 
            ManusDemo::demoFlowOrchestration,
            ManusDemo::demoLayeredArchitecture,
            ManusDemo::demoToolManagement,
            ManusDemo::demoSystemMonitoring
        };
        
        for (int i = 0; i < demos.length; i++) {
            System.out.printf("\n🎬 开始第%d个演示...\n", i + 1);
            demos[i].run();
            
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        System.out.println("\n🎉 全部演示完成！");
    }
}