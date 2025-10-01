package io.leavesfly.tinyai.agent.pattern;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Agent模式测试类
 * @author 山泽
 */
public class AgentPatternTest {
    
    @Test
    public void testReActAgent() {
        ReActAgent agent = new ReActAgent("测试ReAct Agent");
        
        // 测试数学计算
        String result = agent.process("计算 10 + 20");
        assertNotNull("结果不应为空", result);
        assertTrue("结果应包含计算相关内容", result.contains("计算") || result.contains("30"));
        
        // 检查是否有执行步骤
        assertFalse("应该有执行步骤", agent.getSteps().isEmpty());
        assertTrue("步骤数应大于0", agent.getSteps().size() > 0);
        
        System.out.println("ReAct Agent测试 - 查询: 计算 10 + 20");
        System.out.println("结果: " + result);
        System.out.println("步骤数: " + agent.getSteps().size());
    }
    
    @Test
    public void testReflectAgent() {
        ReflectAgent agent = new ReflectAgent("测试Reflect Agent");
        
        String result = agent.process("分析这个问题的复杂性");
        assertNotNull("结果不应为空", result);
        
        // 检查是否有反思记录
        assertFalse("应该有反思记录", agent.getReflections().isEmpty());
        
        System.out.println("Reflect Agent测试 - 查询: 分析这个问题的复杂性");
        System.out.println("结果: " + result);
        System.out.println("反思数: " + agent.getReflections().size());
    }
    
    @Test
    public void testPlanningAgent() {
        PlanningAgent agent = new PlanningAgent("测试Planning Agent");
        
        String result = agent.process("研究人工智能发展历史");
        assertNotNull("结果不应为空", result);
        
        // 检查是否有执行计划
        assertFalse("应该有执行计划", agent.getPlan().isEmpty());
        
        System.out.println("Planning Agent测试 - 查询: 研究人工智能发展历史");
        System.out.println("结果: " + result);
        System.out.println("计划步骤数: " + agent.getPlan().size());
    }
    
    @Test
    public void testCollaborativeAgent() {
        CollaborativeAgent coordinator = new CollaborativeAgent("测试协作Agent");
        
        // 添加专家
        coordinator.addSpecialist("react_expert", new ReActAgent("ReAct专家"));
        coordinator.addSpecialist("reflect_expert", new ReflectAgent("反思专家"));
        
        String result = coordinator.process("计算 5 * 6");
        assertNotNull("结果不应为空", result);
        
        // 检查专家数量
        assertEquals("应该有2个专家", 2, coordinator.getSpecialistNames().size());
        
        System.out.println("Collaborative Agent测试 - 查询: 计算 5 * 6");
        System.out.println("结果: " + result);
        System.out.println("专家数: " + coordinator.getSpecialistNames().size());
    }
    
    @Test
    public void testAgentState() {
        ReActAgent agent = new ReActAgent("状态测试Agent");
        
        // 初始状态应该是THINKING
        assertEquals("初始状态应该是THINKING", AgentState.THINKING, agent.getState());
        
        // 处理后状态应该是DONE
        agent.process("简单测试");
        assertEquals("处理后状态应该是DONE", AgentState.DONE, agent.getState());
        
        System.out.println("Agent状态测试通过");
    }
    
    @Test
    public void testStepRecording() {
        ReActAgent agent = new ReActAgent("步骤记录测试Agent");
        
        agent.process("测试步骤记录");
        
        // 检查步骤记录
        assertFalse("应该有步骤记录", agent.getSteps().isEmpty());
        
        String summary = agent.getStepsSummary();
        assertNotNull("步骤摘要不应为空", summary);
        assertFalse("步骤摘要应有内容", summary.trim().isEmpty());
        
        System.out.println("步骤记录测试 - 步骤摘要:");
        System.out.println(summary);
    }
    
    @Test
    public void testSampleTools() {
        // 测试天气工具
        java.util.function.Function<java.util.Map<String, Object>, Object> weatherTool = SampleTools.createWeatherTool();
        java.util.Map<String, Object> weatherArgs = new java.util.HashMap<>();
        weatherArgs.put("city", "北京");
        Object weatherResult = weatherTool.apply(weatherArgs);
        assertNotNull("天气工具结果不应为空", weatherResult);
        assertTrue("天气结果应包含温度信息", weatherResult.toString().contains("°C"));
        
        // 测试时间工具
        java.util.function.Function<java.util.Map<String, Object>, Object> timeTool = SampleTools.createTimeTool();
        java.util.Map<String, Object> timeArgs = new java.util.HashMap<>();
        timeArgs.put("format", "date");
        Object timeResult = timeTool.apply(timeArgs);
        assertNotNull("时间工具结果不应为空", timeResult);
        
        System.out.println("示例工具测试:");
        System.out.println("天气: " + weatherResult);
        System.out.println("时间: " + timeResult);
    }
}