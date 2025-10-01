package io.leavesfly.tinyai.agent.pattern;

/**
 * 快速演示类
 * 展示各种Agent模式的基本功能
 * @author 山泽
 */
public class QuickDemo {
    
    public static void main(String[] args) {
        System.out.println("🤖 TinyAI Agent Pattern 快速演示");
        System.out.println("=".repeat(50));
        
        // 演示 ReAct Agent
        demoReActAgent();
        
        // 演示 Reflect Agent
        demoReflectAgent();
        
        // 演示 Planning Agent
        demoPlanningAgent();
        
        // 演示 Collaborative Agent
        demoCollaborativeAgent();
        
        System.out.println("\n✅ 演示完成！");
    }
    
    private static void demoReActAgent() {
        System.out.println("\n🤖 ReAct Agent 演示");
        System.out.println("-".repeat(30));
        
        ReActAgent agent = new ReActAgent("演示ReAct Agent");
        
        // 测试数学计算
        String result = agent.process("计算 15 * 8");
        System.out.println("查询: 计算 15 * 8");
        System.out.println("结果: " + result);
        System.out.println("执行步骤数: " + agent.getSteps().size());
    }
    
    private static void demoReflectAgent() {
        System.out.println("\n🪞 Reflect Agent 演示");
        System.out.println("-".repeat(30));
        
        ReflectAgent agent = new ReflectAgent("演示Reflect Agent");
        
        String result = agent.process("分析编程学习的重要性");
        System.out.println("查询: 分析编程学习的重要性");
        System.out.println("结果: " + result.substring(0, Math.min(100, result.length())) + "...");
        System.out.println("反思记录数: " + agent.getReflections().size());
    }
    
    private static void demoPlanningAgent() {
        System.out.println("\n📋 Planning Agent 演示");
        System.out.println("-".repeat(30));
        
        PlanningAgent agent = new PlanningAgent("演示Planning Agent");
        
        String result = agent.process("研究Java编程语言");
        System.out.println("查询: 研究Java编程语言");
        System.out.println("结果: " + result.substring(0, Math.min(100, result.length())) + "...");
        System.out.println("执行计划步骤数: " + agent.getPlan().size());
    }
    
    private static void demoCollaborativeAgent() {
        System.out.println("\n🤝 Collaborative Agent 演示");
        System.out.println("-".repeat(30));
        
        CollaborativeAgent coordinator = new CollaborativeAgent("演示协作Agent");
        
        // 添加专家
        coordinator.addSpecialist("react_expert", new ReActAgent("ReAct专家"));
        coordinator.addSpecialist("reflect_expert", new ReflectAgent("反思专家"));
        
        String result = coordinator.process("计算 12 + 28");
        System.out.println("查询: 计算 12 + 28");
        System.out.println("结果: " + result.substring(0, Math.min(100, result.length())) + "...");
        System.out.println("注册专家数: " + coordinator.getSpecialistNames().size());
    }
}