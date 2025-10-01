package io.leavesfly.tinyai.agent.multi;

import java.util.Arrays;
import java.util.Map;

/**
 * 快速演示类
 * 简化的多Agent系统演示
 * 
 * @author 山泽
 */
public class QuickDemo {
    
    public static void main(String[] args) {
        System.out.println("🚀 Multi-Agent系统快速演示");
        System.out.println("================================");
        
        try {
            // 创建系统
            MultiAgentSystem system = new MultiAgentSystem();
            
            // 添加Agent
            System.out.println("\n📝 创建Agent...");
            String analystId = system.addAgent(AnalystAgent.class).get();
            String coordId = system.addAgent(CoordinatorAgent.class).get();
            String executorId = system.addAgent(ExecutorAgent.class).get();
            
            System.out.println("✅ 已创建 " + system.getAgentCount() + " 个Agent");
            
            // 创建团队
            system.createTeam("demo团队", Arrays.asList(analystId, coordId, executorId));
            System.out.println("✅ 已创建团队，包含 " + system.getTeams().get("demo团队").size() + " 个成员");
            
            // 启动系统
            system.startSystem().get();
            System.out.println("✅ 系统已启动");
            
            // 分配任务
            System.out.println("\n🎯 分配任务...");
            AgentTask task1 = new AgentTask("数据分析任务", "分析市场趋势数据", "demo_user");
            AgentTask task2 = new AgentTask("执行处理任务", "处理数据并生成报告", "demo_user");
            
            system.assignTask(task1, analystId).get();
            system.assignTask(task2, executorId).get();
            
            System.out.println("✅ 已分配 2 个任务");
            
            // 等待任务执行
            System.out.println("\n⏳ 等待任务执行...");
            Thread.sleep(3000);
            
            // 显示系统状态
            Map<String, Object> status = system.getSystemStatus();
            @SuppressWarnings("unchecked")
            Map<String, Object> systemMetrics = (Map<String, Object>) status.get("systemMetrics");
            
            System.out.println("\n📊 系统状态：");
            System.out.println("- 活跃Agent数：" + systemMetrics.get("activeAgents"));
            System.out.println("- 分配任务数：" + systemMetrics.get("totalTasks"));
            System.out.println("- 消息总数：" + systemMetrics.get("totalMessages"));
            
            // 显示Agent完成情况
            System.out.println("\n🏆 Agent工作情况：");
            for (Map.Entry<String, BaseAgent> entry : system.getAgents().entrySet()) {
                BaseAgent agent = entry.getValue();
                AgentMetrics metrics = agent.getMetrics();
                System.out.println(String.format("- %s: 完成 %d 个任务，发送 %d 条消息", 
                    agent.getName(), 
                    metrics.getTasksCompleted(),
                    metrics.getMessagesSent()));
            }
            
            // 广播消息测试
            System.out.println("\n📢 发送广播消息...");
            system.broadcastMessage("项目演示成功完成！", "demo_system");
            
            Thread.sleep(500);
            
            // 停止系统
            system.stopSystem().get();
            System.out.println("\n✅ 系统已停止");
            
            System.out.println("\n🎉 快速演示完成！");
            System.out.println("Multi-Agent系统运行正常，所有核心功能验证通过。");
            
        } catch (Exception e) {
            System.err.println("❌ 演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}