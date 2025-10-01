package io.leavesfly.tinyai.agent.pattern;

import java.util.Scanner;

/**
 * Agent模式完全指南主程序
 * 提供交互式演示各种Agent模式
 * @author 山泽
 */
public class AgentPatternMain {
    
    private static final Scanner scanner = new Scanner(System.in);
    
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🤖 Agent模式完全指南");
        System.out.println("=".repeat(80));
        System.out.println(
                "这个演示包含了常用的Agent模式实现：\n" +
                "\n" +
                "1. ReAct Agent - 推理与行动结合\n" +
                "2. Reflect Agent - 自我反思改进\n" +
                "3. Planning Agent - 计划导向执行\n" +
                "4. Collaborative Agent - 多Agent协作\n" +
                "\n" +
                "选择要演示的模式：\n" +
                "1 - ReAct Agent 演示\n" +
                "2 - Reflect Agent 演示  \n" +
                "3 - Planning Agent 演示\n" +
                "4 - Collaborative Agent 演示\n" +
                "5 - Agent模式对比\n" +
                "6 - 高级Agent模式介绍\n" +
                "7 - 全部演示\n" +
                "8 - 自定义测试\n" +
                "0 - 退出\n");
        
        while (true) {
            try {
                System.out.print("\n请选择 (0-8): ");
                String choice = scanner.nextLine().trim();
                
                switch (choice) {
                    case "0":
                        System.out.println("\n👋 感谢使用Agent模式演示系统！");
                        return;
                    case "1":
                        AgentPatternDemo.demoReActAgent();
                        break;
                    case "2":
                        AgentPatternDemo.demoReflectAgent();
                        break;
                    case "3":
                        AgentPatternDemo.demoPlanningAgent();
                        break;
                    case "4":
                        AgentPatternDemo.demoCollaborativeAgent();
                        break;
                    case "5":
                        AgentPatternDemo.compareAgentPatterns();
                        break;
                    case "6":
                        AgentPatternDemo.advancedAgentPatterns();
                        break;
                    case "7":
                        runAllDemos();
                        break;
                    case "8":
                        runCustomTest();
                        break;
                    default:
                        System.out.println("❌ 无效选择，请输入 0-8");
                }
                
            } catch (Exception e) {
                System.out.println("❌ 发生错误: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    /**
     * 运行所有演示
     */
    private static void runAllDemos() {
        System.out.println("\n🚀 开始全部演示...");
        AgentPatternDemo.demoReActAgent();
        AgentPatternDemo.demoReflectAgent();
        AgentPatternDemo.demoPlanningAgent();
        AgentPatternDemo.demoCollaborativeAgent();
        AgentPatternDemo.compareAgentPatterns();
        AgentPatternDemo.advancedAgentPatterns();
        System.out.println("\n✅ 全部演示完成！");
    }
    
    /**
     * 运行自定义测试
     */
    private static void runCustomTest() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🔧 自定义测试模式");
        System.out.println("=".repeat(60));
        
        // 选择Agent类型
        System.out.println("选择Agent类型:");
        System.out.println("1 - ReAct Agent");
        System.out.println("2 - Reflect Agent");
        System.out.println("3 - Planning Agent");
        System.out.println("4 - Collaborative Agent");
        
        System.out.print("请选择Agent类型 (1-4): ");
        String agentChoice = scanner.nextLine().trim();
        
        BaseAgent agent = null;
        switch (agentChoice) {
            case "1":
                agent = new ReActAgent("自定义ReAct Agent");
                addCustomTools(agent);
                break;
            case "2":
                agent = new ReflectAgent("自定义Reflect Agent");
                break;
            case "3":
                agent = new PlanningAgent("自定义Planning Agent");
                break;
            case "4":
                CollaborativeAgent collaborativeAgent = new CollaborativeAgent("自定义协作Agent");
                collaborativeAgent.addSpecialist("react_expert", new ReActAgent("ReAct专家"));
                collaborativeAgent.addSpecialist("reflect_expert", new ReflectAgent("反思专家"));
                collaborativeAgent.addSpecialist("planning_expert", new PlanningAgent("规划专家"));
                agent = collaborativeAgent;
                break;
            default:
                System.out.println("❌ 无效选择，使用默认ReAct Agent");
                agent = new ReActAgent("默认Agent");
                addCustomTools(agent);
        }
        
        // 交互式测试
        System.out.println("\n✅ Agent创建完成: " + agent.getName());
        System.out.println("💡 提示: 输入 'quit' 退出自定义测试");
        
        while (true) {
            System.out.print("\n🤖 请输入您的问题: ");
            String query = scanner.nextLine().trim();
            
            if ("quit".equalsIgnoreCase(query)) {
                System.out.println("退出自定义测试");
                break;
            }
            
            if (query.isEmpty()) {
                System.out.println("❌ 问题不能为空");
                continue;
            }
            
            System.out.println("\n" + "-".repeat(40));
            System.out.println("🔍 处理中...");
            
            try {
                long startTime = System.currentTimeMillis();
                String result = agent.process(query);
                long endTime = System.currentTimeMillis();
                
                System.out.println("\n🎯 结果: " + result);
                System.out.println("\n⏱️ 处理时间: " + (endTime - startTime) + "ms");
                
                System.out.println("\n📋 执行步骤:");
                System.out.println(agent.getStepsSummary());
                
                // 显示额外信息
                if (agent instanceof ReflectAgent) {
                    ReflectAgent reflectAgent = (ReflectAgent) agent;
                    if (!reflectAgent.getReflections().isEmpty()) {
                        System.out.println("\n🪞 反思记录:");
                        for (int i = 0; i < reflectAgent.getReflections().size(); i++) {
                            System.out.println("  " + (i + 1) + ". " + reflectAgent.getReflections().get(i));
                        }
                    }
                }
                
                if (agent instanceof PlanningAgent) {
                    PlanningAgent planningAgent = (PlanningAgent) agent;
                    if (!planningAgent.getPlan().isEmpty()) {
                        System.out.println("\n📋 执行计划:");
                        for (int i = 0; i < planningAgent.getPlan().size(); i++) {
                            System.out.println("  " + (i + 1) + ". " + planningAgent.getPlan().get(i));
                        }
                    }
                }
                
                if (agent instanceof CollaborativeAgent) {
                    CollaborativeAgent collaborativeAgent = (CollaborativeAgent) agent;
                    System.out.println("\n🤝 专家状态:");
                    System.out.println(collaborativeAgent.getSpecialistSummary());
                }
                
            } catch (Exception e) {
                System.out.println("❌ 处理失败: " + e.getMessage());
            }
            
            System.out.println("\n" + "-".repeat(40));
        }
    }
    
    /**
     * 为Agent添加自定义工具
     */
    private static void addCustomTools(BaseAgent agent) {
        // 添加一些实用工具
        agent.addTool("weather", SampleTools.createWeatherTool(), "天气查询工具");
        agent.addTool("news", SampleTools.createNewsTool(), "新闻查询工具");
        agent.addTool("translate", SampleTools.createTranslateTool(), "翻译工具");
        agent.addTool("time", SampleTools.createTimeTool(), "时间工具");
        agent.addTool("convert", SampleTools.createUnitConverterTool(), "单位转换工具");
    }
    
    /**
     * 显示帮助信息
     */
    public static void showHelp() {
        System.out.println(
                "\n🆘 使用帮助：\n" +
                "\n" +
                "各Agent模式特点：\n" +
                "\n" +
                "1. ReAct Agent (推理-行动模式)\n" +
                "   - 交替进行思考和行动\n" +
                "   - 适合需要工具调用的任务\n" +
                "   - 示例查询: \"计算 10 + 20\", \"搜索Java相关信息\"\n" +
                "\n" +
                "2. Reflect Agent (反思模式)\n" +
                "   - 具有自我反思和改进能力\n" +
                "   - 适合需要质量控制的任务\n" +
                "   - 示例查询: \"分析这段代码的优缺点\", \"如何提高编程技能\"\n" +
                "\n" +
                "3. Planning Agent (规划模式)\n" +
                "   - 先制定计划再执行\n" +
                "   - 适合复杂任务分解\n" +
                "   - 示例查询: \"制定学习计划\", \"研究深度学习\"\n" +
                "\n" +
                "4. Collaborative Agent (协作模式)\n" +
                "   - 多个专家Agent协同工作\n" +
                "   - 适合需要多方面专业知识的任务\n" +
                "   - 自动根据问题类型选择合适的专家\n" +
                "\n" +
                "自定义测试中可用的工具：\n" +
                "- weather: 查询天气 (参数: city)\n" +
                "- news: 查询新闻 (参数: category)\n" +
                "- translate: 翻译文本 (参数: text, target_lang)\n" +
                "- time: 获取时间 (参数: format)\n" +
                "- convert: 单位转换 (参数: value, from_unit, to_unit)\n");
    }
}