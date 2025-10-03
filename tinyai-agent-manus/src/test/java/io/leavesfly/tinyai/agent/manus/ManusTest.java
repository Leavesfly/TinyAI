package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.ToolCall;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Manus系统单元测试
 * 
 * @author 山泽
 */
public class ManusTest {
    
    private Manus manus;
    
    @Before
    public void setUp() {
        manus = new Manus("TestManus");
    }
    
    @Test
    public void testBasicInitialization() {
        // 测试基本初始化
        assertEquals("TestManus", manus.getName());
        assertEquals(AgentState.IDLE, manus.getState());
        assertEquals(ExecutionMode.DIRECT_AGENT, manus.getExecutionMode());
        assertFalse(manus.isPlanningEnabled());
        assertTrue(manus.getMessages().isEmpty());
        assertTrue(manus.getToolCallHistory().isEmpty());
    }
    
    @Test
    public void testExecutionModeSwitch() {
        // 测试执行模式切换
        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
        assertEquals(ExecutionMode.FLOW_ORCHESTRATION, manus.getExecutionMode());
        
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        assertEquals(ExecutionMode.DIRECT_AGENT, manus.getExecutionMode());
    }
    
    @Test
    public void testPlanningModeToggle() {
        // 测试计划模式开关
        assertFalse(manus.isPlanningEnabled());
        
        manus.setPlanningEnabled(true);
        assertTrue(manus.isPlanningEnabled());
        
        manus.setPlanningEnabled(false);
        assertFalse(manus.isPlanningEnabled());
    }
    
    @Test
    public void testDirectAgentMode() {
        // 测试直接Agent模式
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(false);
        
        Message userMessage = new Message("user", "计算 10 + 20");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertEquals("assistant", response.getRole());
        assertNotNull(response.getContent());
        assertFalse(response.getContent().isEmpty());
        
        // 验证消息被记录（ReAct模式可能产生多条消息）
        assertTrue(manus.getMessages().size() >= 2); // 至少有 user + assistant
    }
    
    @Test
    public void testCalculatorTool() {
        // 测试计算器工具
        Message userMessage = new Message("user", "计算 5 * 6");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertTrue(response.getContent().contains("30") || response.getContent().contains("结果"));
    }
    
    @Test
    public void testTimeTool() {
        // 测试时间工具
        Message userMessage = new Message("user", "现在几点了？");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertTrue(response.getContent().contains("时间") || response.getContent().contains("T"));
    }
    
    @Test
    public void testTextAnalyzerTool() {
        // 测试文本分析工具
        Message userMessage = new Message("user", "分析这个文本: '测试文本'");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertTrue(response.getContent().contains("分析") || response.getContent().contains("结果"));
    }
    
    @Test
    public void testPlanningMode() {
        // 测试计划模式
        manus.setExecutionMode(ExecutionMode.DIRECT_AGENT);
        manus.setPlanningEnabled(true);
        
        Message userMessage = new Message("user", "详细分析计算 15 + 25");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertTrue(response.getContent().contains("计划") || response.getContent().contains("任务"));
        assertNotNull(manus.getCurrentPlan());
    }
    
    @Test
    public void testFlowOrchestrationMode() {
        // 测试Flow编排模式
        manus.setExecutionMode(ExecutionMode.FLOW_ORCHESTRATION);
        
        Message userMessage = new Message("user", "计算 8 + 12");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertTrue(response.getContent().contains("Flow") || response.getContent().contains("结果"));
    }
    
    @Test
    public void testFlowRegistration() {
        // 测试Flow注册
        FlowDefinition customFlow = new FlowDefinition("自定义流程", "测试流程");
        Map<String, Object> nodes = new HashMap<>();
        nodes.put("type", "tool");
        nodes.put("name", "calculator");
        customFlow.setNodes(nodes);
        
        manus.registerFlow("test_flow", customFlow);
        
        Map<String, FlowDefinition> flows = manus.getRegisteredFlows();
        assertTrue(flows.containsKey("test_flow"));
        assertEquals("自定义流程", flows.get("test_flow").getName());
    }
    
    @Test
    public void testCustomToolRegistration() {
        // 测试自定义工具注册
        manus.registerCustomTool("test_tool", args -> {
            String input = (String) args.get("input");
            return "processed: " + input;
        }, "测试工具");
        
        assertTrue(manus.getToolRegistry().hasTool("test_tool"));
        
        Map<String, Object> args = new HashMap<>();
        args.put("input", "test");
        ToolCall result = manus.getToolRegistry().callTool("test_tool", args);
        
        assertTrue(result.isSuccess());
        assertEquals("processed: test", result.getResult());
    }
    
    @Test
    public void testSystemStatus() {
        // 测试系统状态
        Map<String, Object> status = manus.getSystemStatus();
        
        assertNotNull(status);
        assertTrue(status.containsKey("name"));
        assertTrue(status.containsKey("execution_mode"));
        assertTrue(status.containsKey("planning_enabled"));
        assertTrue(status.containsKey("current_state"));
        assertTrue(status.containsKey("total_messages"));
        
        assertEquals("TestManus", status.get("name"));
        assertEquals("直接Agent模式", status.get("execution_mode"));
        assertEquals(false, status.get("planning_enabled"));
    }
    
    @Test
    public void testErrorHandling() {
        // 测试错误处理
        Message userMessage = new Message("user", "执行一个无效的操作");
        Message response = manus.processMessage(userMessage);
        
        assertNotNull(response);
        assertNotNull(response.getContent());
        // 错误情况下仍应该返回有意义的响应
        assertFalse(response.getContent().trim().isEmpty());
    }
    
    @Test
    public void testMessageHistory() {
        // 测试消息历史
        assertTrue(manus.getMessages().isEmpty());
        
        Message userMessage1 = new Message("user", "第一条消息");
        manus.processMessage(userMessage1);
        
        assertTrue(manus.getMessages().size() >= 2); // 至少2条
        
        Message userMessage2 = new Message("user", "第二条消息");
        manus.processMessage(userMessage2);
        
        assertTrue(manus.getMessages().size() >= 4); // 至少4条（每次处理至少产生2条）
    }
    
    @Test
    public void testToolUsageStatistics() {
        // 测试工具使用统计
        Map<String, Integer> initialStats = manus.getToolStats();
        assertTrue(initialStats.isEmpty());
        
        // 触发工具调用
        Message userMessage = new Message("user", "计算 1 + 1");
        manus.processMessage(userMessage);
        
        Map<String, Integer> finalStats = manus.getToolStats();
        // 应该有工具使用记录（可能是calculator）
        assertFalse(finalStats.isEmpty());
    }
}