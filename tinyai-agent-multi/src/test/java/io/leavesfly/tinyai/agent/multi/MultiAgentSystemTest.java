package io.leavesfly.tinyai.agent.multi;

import org.junit.Before;
import org.junit.Test;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

import org.junit.After;
import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * 多Agent系统单元测试
 * 
 * @author 山泽
 */
public class MultiAgentSystemTest {
    
    private MultiAgentSystem system;
    private MessageBus messageBus;
    private LLMSimulator llm;
    
    @Before
    public void setUp() {
        system = new MultiAgentSystem();
        messageBus = new MessageBus();
        llm = new LLMSimulator();
    }
    
    @After
    public void tearDown() throws Exception {
        if (system != null && system.isRunning()) {
            system.stopSystem().get();
        }
        if (messageBus != null) {
            messageBus.shutdown();
        }
    }
    
    @Test
    public void testMessageBusBasicFunctionality() {
        // 测试消息总线基本功能
        AgentMessage message = new AgentMessage("sender1", "receiver1", MessageType.TEXT, "测试消息");
        
        // 验证消息属性
        assertNotNull(message.getId());
        assertEquals("sender1", message.getSenderId());
        assertEquals("receiver1", message.getReceiverId());
        assertEquals(MessageType.TEXT, message.getMessageType());
        assertEquals("测试消息", message.getContent());
        assertEquals(1, message.getPriority()); // 默认优先级
        
        // 测试消息发布
        messageBus.publish(message);
        
        // 验证消息历史
        List<AgentMessage> history = messageBus.getAllMessages();
        assertEquals(1, history.size());
        assertEquals(message.getId(), history.get(0).getId());
        
        // 验证统计信息
        Map<String, Object> stats = messageBus.getStatistics();
        assertEquals(1L, stats.get("totalMessages"));
    }
    
    @Test
    public void testAgentTaskCreationAndProperties() {
        // 测试任务创建和属性
        AgentTask task = new AgentTask("测试任务", "这是一个测试任务", "creator1");
        
        assertNotNull(task.getId());
        assertEquals("测试任务", task.getTitle());
        assertEquals("这是一个测试任务", task.getDescription());
        assertEquals("creator1", task.getCreatedBy());
        assertEquals(TaskStatus.PENDING, task.getStatus());
        assertEquals(1, task.getPriority()); // 默认优先级
        assertNotNull(task.getCreatedAt());
        assertNotNull(task.getUpdatedAt());
        
        // 测试状态变更
        task.setStatus(TaskStatus.IN_PROGRESS);
        assertEquals(TaskStatus.IN_PROGRESS, task.getStatus());
        
        // 测试子任务添加
        AgentTask subtask = new AgentTask("子任务", "子任务描述", "creator1");
        task.addSubtask(subtask);
        assertEquals(1, task.getSubtasks().size());
        
        // 测试依赖添加
        task.addDependency("dependency1");
        assertEquals(1, task.getDependencies().size());
    }
    
    @Test
    public void testAgentMetrics() {
        // 测试Agent性能指标
        AgentMetrics metrics = new AgentMetrics();
        
        // 初始状态验证
        assertEquals(0, metrics.getTasksCompleted());
        assertEquals(0, metrics.getMessagesSent());
        assertEquals(0, metrics.getErrorCount());
        assertEquals(0.0, metrics.getSuccessRate(), 0.01);
        
        // 记录任务完成
        metrics.recordTaskCompleted(1000);
        assertEquals(1, metrics.getTasksCompleted());
        assertEquals(1.0, metrics.getAverageResponseTime(), 0.01);
        
        // 记录消息发送
        metrics.recordMessageSent();
        assertEquals(1, metrics.getMessagesSent());
        
        // 记录错误
        metrics.recordError();
        assertEquals(1, metrics.getErrorCount());
        
        // 测试成功率计算
        metrics.recordTaskAssigned();
        metrics.recordTaskFailed();
        // 1 completed, 1 failed = 50% success rate
        assertEquals(0.5, metrics.getSuccessRate(), 0.01);
    }
    
    @Test
    public void testLLMSimulator() throws Exception {
        // 测试LLM模拟器
        LLMSimulator simulator = new LLMSimulator();
        
        List<Map<String, String>> messages = Arrays.asList(
            Map_of("role", "user", "content", "你好，请分析一下数据")
        );
        
        // 测试异步聊天完成
        CompletableFuture<String> future = simulator.chatCompletionAsync(messages, "analyst");
        String response = future.get();
        
        assertNotNull(response);
        assertFalse(response.isEmpty());
        assertTrue(response.length() > 10); // 确保有合理的回复长度
        
        // 测试同步聊天完成
        String syncResponse = simulator.chatCompletion(messages, "researcher");
        assertNotNull(syncResponse);
        assertFalse(syncResponse.isEmpty());
        
        // 测试系统提示生成
        String systemPrompt = simulator.generateSystemPrompt("analyst", "测试分析师", "数据分析师");
        assertNotNull(systemPrompt);
        assertTrue(systemPrompt.contains("测试分析师"));
        assertTrue(systemPrompt.contains("数据分析师"));
    }
    
    @Test
    public void testMultiAgentSystemBasics() throws Exception {
        // 测试多Agent系统基本功能
        
        // 添加Agent
        String analystId = system.addAgent(AnalystAgent.class).get();
        assertNotNull(analystId);
        
        String researcherId = system.addAgent(ResearcherAgent.class).get();
        assertNotNull(researcherId);
        
        // 验证Agent数量
        assertEquals(2, system.getAgentCount());
        
        // 测试团队创建
        boolean teamCreated = system.createTeam("测试团队", Arrays.asList(analystId, researcherId));
        assertTrue(teamCreated);
        assertEquals(1, system.getTeamCount());
        
        // 启动系统
        system.startSystem().get();
        assertTrue(system.isRunning());
        
        // 验证Agent状态
        Map<String, Object> status = system.getSystemStatus();
        assertNotNull(status);
        
        @SuppressWarnings("unchecked")
        Map<String, Object> systemMetrics = (Map<String, Object>) status.get("systemMetrics");
        assertEquals(2, systemMetrics.get("activeAgents"));
        assertTrue((Boolean) systemMetrics.get("running"));
        
        // 测试任务分配
        AgentTask task = new AgentTask("测试分析任务", "分析测试数据", "test_user");
        boolean taskAssigned = system.assignTask(task, analystId).get();
        assertTrue(taskAssigned);
        
        // 等待一段时间让任务处理
        Thread.sleep(1000);
        
        // 停止系统
        system.stopSystem().get();
        assertFalse(system.isRunning());
    }
    
    @Test
    public void testAnalystAgent() throws Exception {
        // 测试分析师Agent
        AnalystAgent analyst = new AnalystAgent("test_analyst", messageBus, llm);
        
        assertEquals("test_analyst", analyst.getAgentId());
        assertTrue(analyst.getName().contains("分析师"));
        assertEquals("数据分析师", analyst.getRole());
        assertTrue(analyst.getCapabilities().contains("数据分析"));
        assertTrue(analyst.getCapabilities().contains("趋势预测"));
        
        // 测试Agent启动和停止
        analyst.start().get();
        assertTrue(analyst.isRunning());
        assertEquals(AgentState.IDLE, analyst.getState());
        
        analyst.stop().get();
        assertFalse(analyst.isRunning());
    }
    
    @Test
    public void testCoordinatorAgent() throws Exception {
        // 测试协调员Agent
        CoordinatorAgent coordinator = new CoordinatorAgent("test_coordinator", messageBus, llm);
        
        assertEquals("test_coordinator", coordinator.getAgentId());
        assertTrue(coordinator.getName().contains("协调员"));
        assertEquals("项目协调员", coordinator.getRole());
        assertTrue(coordinator.getCapabilities().contains("任务分配"));
        
        // 测试团队成员管理
        coordinator.addTeamMember("member1");
        coordinator.addTeamMember("member2");
        assertEquals(2, coordinator.getTeamSize());
        
        List<String> members = coordinator.getTeamMembers();
        assertTrue(members.contains("member1"));
        assertTrue(members.contains("member2"));
        
        coordinator.removeTeamMember("member1");
        assertEquals(1, coordinator.getTeamSize());
    }
    
    @Test
    public void testMessageBusSubscriptionAndPublishing() throws InterruptedException {
        // 测试消息总线订阅和发布
        final boolean[] messageReceived = {false};
        final AgentMessage[] receivedMessage = {null};
        
        // 订阅消息
        messageBus.subscribe("test_agent", (message) -> {
            messageReceived[0] = true;
            receivedMessage[0] = message;
        });
        
        assertTrue(messageBus.isSubscribed("test_agent"));
        
        // 发布消息
        AgentMessage message = new AgentMessage("sender", "test_agent", MessageType.TEXT, "测试消息");
        messageBus.publish(message);
        
        // 等待异步处理
        Thread.sleep(100);
        
        // 验证消息接收
        assertTrue(messageReceived[0]);
        assertNotNull(receivedMessage[0]);
        assertEquals(message.getId(), receivedMessage[0].getId());
        
        // 测试广播
        final int[] broadcastCount = {0};
        messageBus.subscribe("agent1", (msg) -> broadcastCount[0]++);
        messageBus.subscribe("agent2", (msg) -> broadcastCount[0]++);
        
        AgentMessage broadcast = new AgentMessage("system", "broadcast", MessageType.BROADCAST, "广播消息");
        messageBus.publish(broadcast);
        
        Thread.sleep(100);
        
        // 验证广播（应该发送给2个订阅者，但不发送给发送者）
        assertEquals(2, broadcastCount[0]);
    }
    
    @Test
    public void testConversationHistory() {
        // 测试对话历史功能
        AgentMessage msg1 = new AgentMessage("agent1", "agent2", MessageType.TEXT, "消息1");
        AgentMessage msg2 = new AgentMessage("agent2", "agent1", MessageType.TEXT, "消息2");
        AgentMessage msg3 = new AgentMessage("agent1", "agent2", MessageType.TEXT, "消息3");
        
        messageBus.publish(msg1);
        messageBus.publish(msg2);
        messageBus.publish(msg3);
        
        // 获取对话历史
        List<AgentMessage> conversation = messageBus.getConversationHistory("agent1", "agent2", 10);
        assertEquals(3, conversation.size());
        
        // 验证消息顺序（应该按时间排序）
        assertEquals(msg1.getId(), conversation.get(0).getId());
        assertEquals(msg2.getId(), conversation.get(1).getId());
        assertEquals(msg3.getId(), conversation.get(2).getId());
        
        // 测试限制数量
        List<AgentMessage> limitedConversation = messageBus.getConversationHistory("agent1", "agent2", 2);
        assertEquals(2, limitedConversation.size());
    }
    
    @Test
    public void testEnumValues() {
        // 测试枚举值
        
        // MessageType枚举测试
        assertEquals("text", MessageType.TEXT.getValue());
        assertEquals("task", MessageType.TASK.getValue());
        assertEquals("broadcast", MessageType.BROADCAST.getValue());
        assertEquals(MessageType.TEXT, MessageType.fromValue("text"));
        
        // AgentState枚举测试
        assertEquals("idle", AgentState.IDLE.getValue());
        assertEquals("busy", AgentState.BUSY.getValue());
        assertEquals("offline", AgentState.OFFLINE.getValue());
        assertEquals(AgentState.IDLE, AgentState.fromValue("idle"));
        
        // TaskStatus枚举测试
        assertEquals("pending", TaskStatus.PENDING.getValue());
        assertEquals("completed", TaskStatus.COMPLETED.getValue());
        assertEquals("failed", TaskStatus.FAILED.getValue());
        assertEquals(TaskStatus.PENDING, TaskStatus.fromValue("pending"));
    }
    
    /**
     * 创建HashMap的工具方法（兼容Java 8）
     */
    private static Map<String, String> Map_of(String k1, String v1, String k2, String v2) {
        Map<String, String> map = new java.util.HashMap<>();
        map.put(k1, v1);
        map.put(k2, v2);
        return map;
    }
    
    private static Map<String, String> Map_of(String k1, String v1) {
        Map<String, String> map = new java.util.HashMap<>();
        map.put(k1, v1);
        return map;
    }
}