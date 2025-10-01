package io.leavesfly.tinyai.agent.multi;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * 多Agent系统演示类
 * 展示多Agent系统的各种功能和协作能力
 * 
 * @author 山泽
 */
public class MultiAgentDemo {
    
    /**
     * 创建重复字符串的工具方法（兼容Java 8）
     */
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    public static void main(String[] args) {
        System.out.println("🌟 从零构建的基于LLM的Multi-Agent系统 (Java版)");
        System.out.println(repeat("=", 60));
        System.out.println("\n这个系统展示了完整的多智能体架构：");
        System.out.println("• 🧠 智能Agent：分析师、研究员、协调员、执行员、评审员");
        System.out.println("• 📡 通信系统：消息总线、点对点通信、广播机制");
        System.out.println("• 🏗️  架构设计：模块化、可扩展、异步处理");
        System.out.println("• 🤝 协作机制：任务分配、团队协调、状态同步");
        System.out.println("• 📊 监控系统：性能指标、状态跟踪、历史记录");
        
        try {
            // 运行所有演示
            demoBasicAgentCommunication();
            
            Thread.sleep(2000);
            demoTaskAssignmentAndExecution();
            
            Thread.sleep(2000);
            demoTeamCollaboration();
            
            System.out.println("\n" + repeat("=", 60));
            System.out.println("🎉 所有演示完成！");
            System.out.println("\n这个Multi-Agent系统具备以下特点：");
            System.out.println("✨ 完全基于Java构建，最小化外部依赖");
            System.out.println("✨ 基于LLM的智能对话能力");
            System.out.println("✨ 灵活的消息通信机制");
            System.out.println("✨ 支持复杂的团队协作");
            System.out.println("✨ 实时状态监控和指标统计");
            System.out.println("✨ 异步执行，高性能处理");
            System.out.println("\n💡 可以基于这个框架继续扩展：");
            System.out.println("• 添加更多专业的Agent类型");
            System.out.println("• 集成真实的LLM API");
            System.out.println("• 增加工具调用能力");
            System.out.println("• 添加持久化存储");
            System.out.println("• 构建Web界面进行可视化管理");
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示基本的Agent通信
     */
    private static void demoBasicAgentCommunication() throws Exception {
        System.out.println("\n" + repeat("=", 50));
        System.out.println("🤖 基本Agent通信演示");
        System.out.println(repeat("=", 50));
        
        MultiAgentSystem system = new MultiAgentSystem();
        
        // 添加不同类型的Agent
        CompletableFuture<String> analystFuture = system.addAgent(AnalystAgent.class);
        CompletableFuture<String> researcherFuture = system.addAgent(ResearcherAgent.class);
        
        String analystId = analystFuture.get();
        String researcherId = researcherFuture.get();
        
        // 启动系统
        system.startSystem().get();
        
        System.out.println(String.format("\n创建了两个Agent："));
        System.out.println(String.format("- 分析师 (ID: %s)", analystId));
        System.out.println(String.format("- 研究员 (ID: %s)", researcherId));
        
        // 模拟对话
        System.out.println("\n🔄 开始Agent间对话...");
        List<AgentMessage> conversation = system.simulateConversation(
                analystId, researcherId,
                "你好，我想了解一下你的研究领域",
                3
        ).get();
        
        System.out.println("\n💬 对话记录：");
        for (int i = 0; i < conversation.size(); i++) {
            AgentMessage msg = conversation.get(i);
            String senderName = "未知";
            String receiverName = "未知";
            
            BaseAgent sender = system.getAgents().get(msg.getSenderId());
            BaseAgent receiver = system.getAgents().get(msg.getReceiverId());
            
            if (sender != null) senderName = sender.getName();
            if (receiver != null) receiverName = receiver.getName();
            
            System.out.println(String.format("%d. %s -> %s: %s", 
                    i + 1, senderName, receiverName, msg.getContent()));
        }
        
        // 停止系统
        system.stopSystem().get();
        System.out.println("\n✅ 基本通信演示完成");
    }
    
    /**
     * 演示任务分配和执行
     */
    private static void demoTaskAssignmentAndExecution() throws Exception {
        System.out.println("\n" + repeat("=", 50));
        System.out.println("📋 任务分配和执行演示");
        System.out.println(repeat("=", 50));
        
        MultiAgentSystem system = new MultiAgentSystem();
        
        // 添加各种类型的Agent
        String coordId = system.addAgent(CoordinatorAgent.class).get();
        String analystId = system.addAgent(AnalystAgent.class).get();
        String executorId = system.addAgent(ExecutorAgent.class).get();
        String criticId = system.addAgent(CriticAgent.class).get();
        
        // 创建团队
        system.createTeam("项目团队", Arrays.asList(coordId, analystId, executorId, criticId));
        
        // 启动系统
        system.startSystem().get();
        
        System.out.println(String.format("\n创建了项目团队："));
        Map<String, BaseAgent> agents = system.getAgents();
        for (String agentId : Arrays.asList(coordId, analystId, executorId, criticId)) {
            BaseAgent agent = agents.get(agentId);
            System.out.println(String.format("- %s (角色: %s)", agent.getName(), agent.getRole()));
        }
        
        // 创建和分配任务
        AgentTask[] tasks = {
            new AgentTask("市场数据分析", "分析最新的市场趋势数据", "user"),
            new AgentTask("执行数据处理任务", "处理收集到的数据并生成报告", "user"),
            new AgentTask("质量评审任务", "评审生成的分析报告质量", "user")
        };
        
        System.out.println("\n📝 分配任务：");
        system.assignTask(tasks[0], analystId).get();
        System.out.println(String.format("- 任务1分配给：%s", agents.get(analystId).getName()));
        
        system.assignTask(tasks[1], executorId).get();
        System.out.println(String.format("- 任务2分配给：%s", agents.get(executorId).getName()));
        
        system.assignTask(tasks[2], criticId).get();
        System.out.println(String.format("- 任务3分配给：%s", agents.get(criticId).getName()));
        
        // 等待任务执行
        System.out.println("\n⏳ 等待任务执行...");
        Thread.sleep(8000);
        
        // 显示系统状态
        Map<String, Object> status = system.getSystemStatus();
        @SuppressWarnings("unchecked")
        Map<String, Object> systemMetrics = (Map<String, Object>) status.get("systemMetrics");
        
        System.out.println("\n📊 系统状态：");
        System.out.println(String.format("- 总任务数：%s", systemMetrics.get("totalTasks")));
        System.out.println(String.format("- 活跃Agent数：%s", systemMetrics.get("activeAgents")));
        
        System.out.println("\n🎯 Agent任务完成情况：");
        @SuppressWarnings("unchecked")
        Map<String, Object> agentStatuses = (Map<String, Object>) status.get("agents");
        for (Map.Entry<String, BaseAgent> entry : agents.entrySet()) {
            BaseAgent agent = entry.getValue();
            int completed = agent.getMetrics().getTasksCompleted();
            System.out.println(String.format("- %s: 已完成 %d 个任务", agent.getName(), completed));
        }
        
        // 停止系统
        system.stopSystem().get();
        System.out.println("\n✅ 任务执行演示完成");
    }
    
    /**
     * 演示团队协作
     */
    private static void demoTeamCollaboration() throws Exception {
        System.out.println("\n" + repeat("=", 50));
        System.out.println("👥 团队协作演示");
        System.out.println(repeat("=", 50));
        
        MultiAgentSystem system = new MultiAgentSystem();
        
        // 创建完整的团队
        String coordId = system.addAgent(CoordinatorAgent.class).get();
        String analystId = system.addAgent(AnalystAgent.class).get();
        String researcherId = system.addAgent(ResearcherAgent.class).get();
        String executorId = system.addAgent(ExecutorAgent.class).get();
        String criticId = system.addAgent(CriticAgent.class).get();
        
        // 创建团队
        List<String> teamMembers = Arrays.asList(coordId, analystId, researcherId, executorId, criticId);
        system.createTeam("AI研发团队", teamMembers);
        
        // 启动系统
        system.startSystem().get();
        
        System.out.println("\n🏢 AI研发团队成员：");
        Map<String, BaseAgent> agents = system.getAgents();
        for (String agentId : teamMembers) {
            BaseAgent agent = agents.get(agentId);
            System.out.println(String.format("- %s: %s", agent.getName(), agent.getRole()));
            System.out.println(String.format("  能力: %s", String.join(", ", agent.getCapabilities())));
        }
        
        // 创建复杂项目任务
        AgentTask projectTask = new AgentTask(
                "AI产品开发项目",
                "协调开发一个新的AI产品，包括需求分析、技术研究、实施和质量评估",
                "product_manager"
        );
        projectTask.setPriority(5);
        
        System.out.println(String.format("\n🚀 启动项目：%s", projectTask.getTitle()));
        
        // 分配给团队（通过协调员）
        system.assignTask(projectTask, "AI研发团队", true).get();
        
        // 广播项目启动消息
        system.broadcastMessage("🎉 新项目正式启动！请各位团队成员积极配合，确保项目成功！", coordId);
        System.out.println("\n📢 已发送项目启动广播");
        
        // 模拟项目执行过程
        System.out.println("\n⚙️ 模拟项目执行过程...");
        Thread.sleep(6000);
        
        // 显示团队协作情况
        System.out.println("\n📈 团队协作情况：");
        for (String agentId : teamMembers) {
            BaseAgent agent = agents.get(agentId);
            System.out.println(String.format("\n%s (%s):", agent.getName(), agent.getRole()));
            System.out.println(String.format("  当前状态: %s", agent.getState().getValue()));
            System.out.println(String.format("  任务队列: %d 个待处理任务", agent.getTaskQueueSize()));
            System.out.println(String.format("  已完成: %d 个任务", agent.getMetrics().getTasksCompleted()));
            System.out.println(String.format("  消息统计: 发送 %d, 接收 %d", 
                    agent.getMetrics().getMessagesSent(), agent.getMetrics().getMessagesReceived()));
        }
        
        // 显示消息交互历史
        System.out.println("\n💬 最近的消息交互：");
        List<AgentMessage> recentMessages = system.getMessageBus().getRecentMessages(5);
        for (AgentMessage msg : recentMessages) {
            String senderName = msg.getSenderId();
            BaseAgent sender = agents.get(msg.getSenderId());
            if (sender != null) {
                senderName = sender.getName();
            }
            
            String content = msg.getContent().toString();
            if (content.length() > 100) {
                content = content.substring(0, 100) + "...";
            }
            
            System.out.println(String.format("  %s: %s", senderName, content));
        }
        
        // 停止系统
        system.stopSystem().get();
        System.out.println("\n✅ 团队协作演示完成");
    }
}