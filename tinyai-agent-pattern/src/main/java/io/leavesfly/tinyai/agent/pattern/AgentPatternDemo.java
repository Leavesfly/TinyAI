package io.leavesfly.tinyai.agent.pattern;

/**
 * Agent模式演示类
 * 展示各种Agent模式的使用方法
 * @author 山泽
 */
public class AgentPatternDemo {
    
    /**
     * 演示ReAct Agent
     */
    public static void demoReActAgent() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🤖 ReAct Agent 演示");
        System.out.println("=".repeat(60));
        
        ReActAgent agent = new ReActAgent();
        
        String[] testQueries = {
            "计算 25 * 4 + 10",
            "搜索 Python 编程",
            "查找我的记忆中关于学习的内容"
        };
        
        for (String query : testQueries) {
            System.out.println("\n📝 查询: " + query);
            System.out.println("-".repeat(40));
            
            String result = agent.process(query);
            System.out.println("\n🎯 结果: " + result);
            
            System.out.println("\n📋 执行步骤:");
            System.out.println(agent.getStepsSummary());
            System.out.println("\n" + "=".repeat(40));
        }
    }
    
    /**
     * 演示Reflect Agent
     */
    public static void demoReflectAgent() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🪞 Reflect Agent 演示");
        System.out.println("=".repeat(60));
        
        ReflectAgent agent = new ReflectAgent();
        
        String[] testQueries = {
            "分析这段文本的特点",
            "如何提高工作效率？"
        };
        
        for (String query : testQueries) {
            System.out.println("\n📝 查询: " + query);
            System.out.println("-".repeat(40));
            
            String result = agent.process(query);
            System.out.println("\n🎯 结果: " + result);
            
            System.out.println("\n🪞 反思记录:");
            for (int i = 0; i < agent.getReflections().size(); i++) {
                System.out.println("  " + (i + 1) + ". " + agent.getReflections().get(i));
            }
            
            System.out.println("\n📋 执行步骤:");
            System.out.println(agent.getStepsSummary());
            System.out.println("\n" + "=".repeat(40));
        }
    }
    
    /**
     * 演示Planning Agent
     */
    public static void demoPlanningAgent() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("📋 Planning Agent 演示");
        System.out.println("=".repeat(60));
        
        PlanningAgent agent = new PlanningAgent();
        
        String[] testQueries = {
            "研究机器学习的应用领域",
            "学习Python编程的完整计划"
        };
        
        for (String query : testQueries) {
            System.out.println("\n📝 查询: " + query);
            System.out.println("-".repeat(40));
            
            String result = agent.process(query);
            System.out.println("\n🎯 结果: " + result);
            
            System.out.println("\n📋 执行计划:");
            for (int i = 0; i < agent.getPlan().size(); i++) {
                System.out.println("  " + (i + 1) + ". " + agent.getPlan().get(i).getDescription());
            }
            
            System.out.println("\n📋 详细步骤:");
            System.out.println(agent.getStepsSummary());
            System.out.println("\n" + "=".repeat(40));
        }
    }
    
    /**
     * 演示协作Agent
     */
    public static void demoCollaborativeAgent() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🤝 Collaborative Agent 演示");
        System.out.println("=".repeat(60));
        
        // 创建协作Agent和专家
        CollaborativeAgent coordinator = new CollaborativeAgent("协调者");
        
        // 添加专家
        coordinator.addSpecialist("calculator_expert", new ReActAgent("计算专家"));
        coordinator.addSpecialist("analysis_expert", new ReflectAgent("分析专家"));
        coordinator.addSpecialist("planning_expert", new PlanningAgent("规划专家"));
        
        System.out.println("\n" + coordinator.getSpecialistSummary());
        
        String[] testQueries = {
            "计算 15 * 8 + 25",
            "分析当前AI发展趋势",
            "制定学习深度学习的计划"
        };
        
        for (String query : testQueries) {
            System.out.println("\n📝 查询: " + query);
            System.out.println("-".repeat(40));
            
            String result = coordinator.process(query);
            System.out.println("\n🎯 结果: " + result);
            
            System.out.println("\n📋 协调步骤:");
            System.out.println(coordinator.getStepsSummary());
            System.out.println("\n" + "=".repeat(40));
        }
    }
    
    /**
     * 比较不同Agent模式
     */
    public static void compareAgentPatterns() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🔍 Agent模式对比分析");
        System.out.println("=".repeat(80));
        
        String[][] patterns = {
            {
                "ReAct Agent",
                "推理与行动交替进行，通过观察结果指导下一步",
                "逻辑清晰、可解释性强、适合需要工具调用的任务",
                "可能陷入局部循环、对复杂任务分解能力有限",
                "数学计算、信息查询、简单推理任务"
            },
            {
                "Reflect Agent", 
                "具有自我反思能力，能够评估和改进自己的回答",
                "自我改进、质量控制、持续学习",
                "计算开销较大、可能过度反思",
                "内容生成、质量要求高的任务、创意写作"
            },
            {
                "Planning Agent",
                "先制定详细计划再执行，适合复杂任务分解",
                "任务分解能力强、执行有条理、适合复杂项目",
                "规划开销大、不够灵活、可能过度规划",
                "项目管理、研究任务、学习规划"
            },
            {
                "Collaborative Agent",
                "多个专家Agent协同工作，发挥各自优势",
                "专业化分工、质量验证、互补优势",
                "协调复杂、资源消耗大、通信开销",
                "复杂问题解决、多领域任务、高质量要求"
            }
        };
        
        for (String[] pattern : patterns) {
            System.out.println("\n📊 " + pattern[0]);
            System.out.println("-".repeat(50));
            System.out.println("📝 描述: " + pattern[1]);
            System.out.println("✅ 优势: " + pattern[2]);
            System.out.println("❌ 劣势: " + pattern[3]);
            System.out.println("🎯 适用场景: " + pattern[4]);
        }
    }
    
    /**
     * 介绍高级Agent模式
     */
    public static void advancedAgentPatterns() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🚀 高级Agent模式介绍");
        System.out.println("=".repeat(80));
        
        String[][] advancedPatterns = {
            {
                "Tree of Thoughts (ToT)",
                "以树状结构探索多个思考路径，选择最优解",
                "多路径探索、回溯机制、最优解选择",
                "状态表示、搜索策略、评估函数"
            },
            {
                "Chain of Thought (CoT)",
                "逐步推理，通过中间步骤得出最终答案",
                "步骤推理、逻辑链条、透明过程",
                "提示工程、步骤分解、逻辑验证"
            },
            {
                "Multi-Agent Debate",
                "多个Agent辩论讨论，通过不同观点得出更好的结论",
                "观点对抗、论据交换、共识达成",
                "角色设定、辩论规则、结论总结"
            },
            {
                "Self-Consistency",
                "生成多个推理路径，选择最一致的答案",
                "多次采样、一致性检查、投票机制",
                "多样性生成、一致性度量、结果聚合"
            },
            {
                "AutoGPT Pattern",
                "自主设定目标、制定计划、执行任务的循环模式",
                "目标导向、自主规划、持续执行",
                "目标分解、进度跟踪、自主调整"
            }
        };
        
        for (String[] pattern : advancedPatterns) {
            System.out.println("\n🎯 " + pattern[0]);
            System.out.println("-".repeat(50));
            System.out.println("📝 描述: " + pattern[1]);
            System.out.println("⭐ 特点: " + pattern[2]);
            System.out.println("🔧 实现要点: " + pattern[3]);
        }
    }
}