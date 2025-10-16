package io.leavesfly.tinyai.agent.multi;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * 多Agent系统管理器
 * 统一管理和协调多个Agent的工作
 * 
 * @author 山泽
 */
public class MultiAgentSystem {
    
    private final MessageBus messageBus;
    private final LLMSimulator llm;
    private final Map<String, BaseAgent> agents;
    private final Map<String, List<String>> teams;
    private final AtomicBoolean running;
    
    // 系统指标
    private volatile long totalTasks = 0;
    private volatile long completedTasks = 0;
    private volatile long totalMessages = 0;
    
    public MultiAgentSystem() {
        this.messageBus = new MessageBus();
        this.llm = new LLMSimulator();
        this.agents = new ConcurrentHashMap<>();
        this.teams = new ConcurrentHashMap<>();
        this.running = new AtomicBoolean(false);
    }
    
    public MultiAgentSystem(int maxHistorySize) {
        this.messageBus = new MessageBus(maxHistorySize);
        this.llm = new LLMSimulator();
        this.agents = new ConcurrentHashMap<>();
        this.teams = new ConcurrentHashMap<>();
        this.running = new AtomicBoolean(false);
    }
    
    /**
     * 添加Agent
     */
    public CompletableFuture<String> addAgent(Class<? extends BaseAgent> agentClass) {
        return addAgent(agentClass, null);
    }
    
    public CompletableFuture<String> addAgent(Class<? extends BaseAgent> agentClass, String agentId) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                String finalAgentId = agentId;
                if (finalAgentId == null) {
                    finalAgentId = generateAgentId(agentClass);
                }
                
                // 使用反射创建Agent实例
                BaseAgent agent = agentClass.getDeclaredConstructor(String.class, MessageBus.class, LLMSimulator.class)
                        .newInstance(finalAgentId, messageBus, llm);
                
                agents.put(finalAgentId, agent);
                
                // 如果系统正在运行，启动Agent
                if (running.get()) {
                    agent.start();
                }
                
                System.out.println(String.format("添加Agent: %s (ID: %s)", agent.getName(), finalAgentId));
                return finalAgentId;
                
            } catch (Exception e) {
                System.err.println("添加Agent失败: " + e.getMessage());
                throw new RuntimeException("Failed to add agent", e);
            }
        });
    }
    
    /**
     * 移除Agent
     */
    public CompletableFuture<Boolean> removeAgent(String agentId) {
        return CompletableFuture.supplyAsync(() -> {
            BaseAgent agent = agents.get(agentId);
            if (agent != null) {
                // 停止Agent
                agent.stop();
                
                // 从系统中移除
                agents.remove(agentId);
                
                // 从团队中移除
                for (List<String> teamMembers : teams.values()) {
                    teamMembers.remove(agentId);
                }
                
                System.out.println(String.format("移除Agent: %s", agent.getName()));
                return true;
            }
            return false;
        });
    }
    
    /**
     * 创建团队
     */
    public boolean createTeam(String teamName, List<String> agentIds) {
        // 验证所有Agent都存在
        for (String agentId : agentIds) {
            if (!agents.containsKey(agentId)) {
                System.err.println(String.format("Agent %s 不存在，无法创建团队", agentId));
                return false;
            }
        }
        
        teams.put(teamName, new ArrayList<>(agentIds));
        
        // 如果有协调员，通知团队成员
        List<String> coordinators = new ArrayList<>();
        for (String agentId : agentIds) {
            BaseAgent agent = agents.get(agentId);
            if (agent instanceof CoordinatorAgent) {
                coordinators.add(agentId);
            }
        }
        
        // 为协调员添加团队成员
        for (String coordId : coordinators) {
            CoordinatorAgent coordinator = (CoordinatorAgent) agents.get(coordId);
            for (String agentId : agentIds) {
                if (!agentId.equals(coordId)) {
                    coordinator.addTeamMember(agentId);
                }
            }
        }
        
        System.out.println(String.format("创建团队 '%s'，包含 %d 个成员", teamName, agentIds.size()));
        return true;
    }
    
    /**
     * 分配任务给特定Agent
     */
    public CompletableFuture<Boolean> assignTask(AgentTask task, String agentId) {
        return CompletableFuture.supplyAsync(() -> {
            if (agents.containsKey(agentId)) {
                totalTasks++;
                
                Map<String, Object> taskData = task.toMap();
                AgentMessage message = new AgentMessage("system", agentId, MessageType.TASK, taskData);
                messageBus.publishAsync(message);
                
                System.out.println(String.format("分配任务 '%s' 给Agent: %s", task.getTitle(), agentId));
                return true;
            }
            return false;
        });
    }
    
    /**
     * 分配任务给团队
     */
    public CompletableFuture<Boolean> assignTask(AgentTask task, String teamName, boolean useTeam) {
        return CompletableFuture.supplyAsync(() -> {
            if (!useTeam) {
                return false;
            }
            
            List<String> teamMembers = teams.get(teamName);
            if (teamMembers == null || teamMembers.isEmpty()) {
                System.err.println(String.format("团队 '%s' 不存在或为空", teamName));
                return false;
            }
            
            totalTasks++;
            
            // 查找协调员
            String coordinatorId = null;
            for (String memberId : teamMembers) {
                BaseAgent agent = agents.get(memberId);
                if (agent instanceof CoordinatorAgent) {
                    coordinatorId = memberId;
                    break;
                }
            }
            
            // 分配给协调员或第一个成员
            String targetAgentId = coordinatorId != null ? coordinatorId : teamMembers.get(0);
            
            Map<String, Object> taskData = task.toMap();
            AgentMessage message = new AgentMessage("system", targetAgentId, MessageType.TASK, taskData);
            messageBus.publishAsync(message);
            
            System.out.println(String.format("分配任务 '%s' 给团队 '%s' (通过Agent: %s)", 
                    task.getTitle(), teamName, targetAgentId));
            return true;
        });
    }
    
    /**
     * 启动系统
     */
    public CompletableFuture<Void> startSystem() {
        if (running.compareAndSet(false, true)) {
            List<CompletableFuture<Void>> startTasks = new ArrayList<>();
            
            // 启动所有Agent
            for (BaseAgent agent : agents.values()) {
                startTasks.add(agent.start());
            }
            
            System.out.println(String.format("多Agent系统已启动，共 %d 个Agent", agents.size()));
            
            return CompletableFuture.allOf(startTasks.toArray(new CompletableFuture[0]));
        }
        return CompletableFuture.completedFuture(null);
    }
    
    /**
     * 停止系统
     */
    public CompletableFuture<Void> stopSystem() {
        if (running.compareAndSet(true, false)) {
            List<CompletableFuture<Void>> stopTasks = new ArrayList<>();
            
            // 停止所有Agent
            for (BaseAgent agent : agents.values()) {
                stopTasks.add(agent.stop());
            }
            
            // 关闭消息总线
            messageBus.shutdown();
            
            System.out.println("多Agent系统已停止");
            
            return CompletableFuture.allOf(stopTasks.toArray(new CompletableFuture[0]));
        }
        return CompletableFuture.completedFuture(null);
    }
    
    /**
     * 广播消息
     */
    public void broadcastMessage(String content, String senderId) {
        if (senderId == null) {
            senderId = "system";
        }
        
        AgentMessage message = new AgentMessage(senderId, "broadcast", MessageType.BROADCAST, content);
        messageBus.publishAsync(message);
        totalMessages++;
        
        System.out.println(String.format("广播消息: %s", content));
    }
    
    /**
     * 模拟两个Agent之间的对话
     */
    public CompletableFuture<List<AgentMessage>> simulateConversation(String agent1Id, String agent2Id, 
                                                                     String initialMessage, int rounds) {
        return CompletableFuture.supplyAsync(() -> {
            if (!agents.containsKey(agent1Id) || !agents.containsKey(agent2Id)) {
                throw new IllegalArgumentException("指定的Agent不存在");
            }
            
            // 发送初始消息
            AgentMessage message = new AgentMessage(agent1Id, agent2Id, MessageType.TEXT, initialMessage);
            messageBus.publishAsync(message);
            
            List<AgentMessage> conversation = new ArrayList<>();
            conversation.add(message);
            
            // 等待对话轮次
            for (int i = 0; i < rounds - 1; i++) {
                try {
                    Thread.sleep(2000); // 等待回复
                    
                    // 获取最新的对话历史
                    List<AgentMessage> recentMessages = messageBus.getConversationHistory(
                            agent1Id, agent2Id, 2);
                    
                    // 添加新消息到对话中
                    for (AgentMessage msg : recentMessages) {
                        if (!conversation.contains(msg)) {
                            conversation.add(msg);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            
            return conversation;
        });
    }
    
    /**
     * 获取系统状态
     */
    public Map<String, Object> getSystemStatus() {
        Map<String, Object> status = new HashMap<>();
        
        // 系统级别统计
        Map<String, Object> systemMetrics = new HashMap<>();
        systemMetrics.put("totalTasks", totalTasks);
        systemMetrics.put("completedTasks", getCompletedTasksCount());
        systemMetrics.put("activeAgents", agents.size());
        systemMetrics.put("totalMessages", messageBus.getStatistics().get("totalMessages"));
        systemMetrics.put("teams", teams.size());
        systemMetrics.put("running", running.get());
        
        status.put("systemMetrics", systemMetrics);
        
        // Agent状态
        Map<String, Object> agentStatuses = new HashMap<>();
        for (Map.Entry<String, BaseAgent> entry : agents.entrySet()) {
            agentStatuses.put(entry.getKey(), entry.getValue().getStatus());
        }
        status.put("agents", agentStatuses);
        
        // 团队信息
        status.put("teams", new HashMap<>(teams));
        
        // 消息总线状态
        status.put("messageBus", messageBus.getStatistics());
        
        return status;
    }
    
    /**
     * 获取已完成任务数
     */
    private long getCompletedTasksCount() {
        return agents.values().stream()
                .mapToLong(agent -> agent.getMetrics().getTasksCompleted())
                .sum();
    }
    
    /**
     * 生成Agent ID
     */
    private String generateAgentId(Class<? extends BaseAgent> agentClass) {
        String className = agentClass.getSimpleName().toLowerCase();
        String prefix = className.replace("agent", "");
        return prefix + "_" + UUID.randomUUID().toString().substring(0, 8);
    }
    
    // Getter 方法
    public Map<String, BaseAgent> getAgents() {
        return new HashMap<>(agents);
    }
    
    public Map<String, List<String>> getTeams() {
        return new HashMap<>(teams);
    }
    
    public MessageBus getMessageBus() {
        return messageBus;
    }
    
    public LLMSimulator getLlm() {
        return llm;
    }
    
    public boolean isRunning() {
        return running.get();
    }
    
    public int getAgentCount() {
        return agents.size();
    }
    
    public int getTeamCount() {
        return teams.size();
    }
}