package io.leavesfly.tinyai.agent.evol;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;
import java.util.*;

/**
 * 自进化Agent测试类
 * 
 * @author 山泽
 */
public class SelfEvolvingAgentTest {
    
    private SelfEvolvingAgent agent;
    
    @Before
    public void setUp() {
        agent = new SelfEvolvingAgent("测试Agent");
    }
    
    @Test
    public void testAgentInitialization() {
        assertNotNull("Agent应该被正确初始化", agent);
        assertEquals("Agent名称应该正确", "测试Agent", agent.getName());
        assertEquals("初始任务数应该为0", 0, agent.getTotalTasks());
        assertEquals("初始成功任务数应该为0", 0, agent.getSuccessfulTasks());
        assertTrue("探索率应该在合理范围内", agent.getExplorationRate() > 0 && agent.getExplorationRate() < 1);
    }
    
    @Test
    public void testBasicTaskProcessing() {
        String task = "测试任务";
        Map<String, Object> context = new HashMap<>();
        context.put("test", "value");
        
        SelfEvolvingAgent.TaskResult result = agent.processTask(task, context);
        
        assertNotNull("任务结果不应该为null", result);
        assertEquals("任务名称应该匹配", task, result.getTask());
        assertNotNull("应该选择了某个动作", result.getAction());
        assertNotNull("应该有执行结果", result.getResult());
        
        assertEquals("任务计数应该增加", 1, agent.getTotalTasks());
    }
    
    @Test
    public void testExperienceClass() {
        Map<String, Object> context = new HashMap<>();
        context.put("key", "value");
        
        Experience exp = new Experience("测试任务", context, "search", "结果", true, 0.8);
        
        assertEquals("任务名称应该正确", "测试任务", exp.getTask());
        assertEquals("动作应该正确", "search", exp.getAction());
        assertTrue("应该标记为成功", exp.isSuccess());
        assertEquals("奖励值应该正确", 0.8, exp.getReward(), 0.001);
    }
    
    @Test
    public void testStrategyClass() {
        Map<String, Object> conditions = new HashMap<>();
        conditions.put("type", "test");
        
        List<String> actions = Arrays.asList("action1", "action2");
        Strategy strategy = new Strategy("测试策略", "描述", conditions, actions, 0.8, 5);
        
        assertEquals("策略名称应该正确", "测试策略", strategy.getName());
        assertEquals("成功率应该正确", 0.8, strategy.getSuccessRate(), 0.001);
        assertEquals("使用次数应该正确", 5, strategy.getUsageCount());
    }
    
    @Test
    public void testKnowledgeGraph() {
        KnowledgeGraph kg = new KnowledgeGraph();
        
        Map<String, Object> properties = new HashMap<>();
        properties.put("type", "task");
        kg.addConcept("概念1", properties);
        kg.addConcept("概念2", properties);
        kg.addRelation("概念1", "概念2", "related_to", 0.8);
        
        List<String> related = kg.findRelatedConcepts("概念1", 2);
        assertTrue("应该找到相关概念", related.contains("概念2"));
    }
    
    @Test
    public void testReflectionModule() {
        ReflectionModule reflection = new ReflectionModule();
        Experience exp = new Experience("测试任务", new HashMap<>(), "search", "结果", true, 0.8);
        
        String reflectionResult = reflection.reflectOnExperience(exp);
        assertNotNull("反思结果不应该为null", reflectionResult);
        assertFalse("反思结果不应该为空", reflectionResult.isEmpty());
    }
    
    @Test
    public void testPerformanceSummary() {
        for (int i = 0; i < 3; i++) {
            agent.processTask("任务" + i, new HashMap<>());
        }
        
        Map<String, Object> summary = agent.getPerformanceSummary();
        assertNotNull("性能摘要不应该为null", summary);
        assertEquals("总任务数应该正确", 3, summary.get("total_tasks"));
    }
    
    @Test
    public void testSelfEvolution() {
        for (int i = 0; i < 10; i++) {
            agent.processTask("进化测试任务" + i, new HashMap<>());
        }
        
        agent.selfEvolve();
        assertTrue("自进化后Agent应该仍然正常工作", agent.getTotalTasks() > 0);
    }
}