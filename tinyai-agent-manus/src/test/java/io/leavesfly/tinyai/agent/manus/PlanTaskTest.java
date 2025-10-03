package io.leavesfly.tinyai.agent.manus;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Plan和Task系统单元测试
 * 
 * @author 山泽
 */
public class PlanTaskTest {
    
    private Plan plan;
    private Task task;
    
    @Before
    public void setUp() {
        plan = new Plan("测试计划", "测试计划目标");
        task = new Task("测试任务", "action");
    }
    
    @Test
    public void testTaskCreation() {
        // 测试任务创建
        assertNotNull(task.getId());
        assertEquals("测试任务", task.getDescription());
        assertEquals("action", task.getType());
        assertEquals("pending", task.getStatus());
        assertNotNull(task.getCreatedAt());
        assertFalse(task.isCompleted());
        assertFalse(task.isFailed());
        assertFalse(task.isRunning());
    }
    
    @Test
    public void testTaskExecution() {
        // 测试任务执行流程
        task.start();
        assertEquals("running", task.getStatus());
        assertTrue(task.isRunning());
        assertNotNull(task.getStartedAt());
        
        task.complete("任务完成结果");
        assertEquals("completed", task.getStatus());
        assertTrue(task.isCompleted());
        assertEquals("任务完成结果", task.getResult());
        assertNotNull(task.getCompletedAt());
    }
    
    @Test
    public void testTaskFailure() {
        // 测试任务失败
        task.start();
        task.fail("测试错误");
        
        assertEquals("failed", task.getStatus());
        assertTrue(task.isFailed());
        assertEquals("测试错误", task.getErrorMessage());
        assertNotNull(task.getCompletedAt());
    }
    
    @Test
    public void testTaskParameters() {
        // 测试任务参数
        Map<String, Object> params = new HashMap<>();
        params.put("param1", "value1");
        params.put("param2", 42);
        
        task.setParameters(params);
        assertEquals(params, task.getParameters());
        
        task.addParameter("param3", true);
        assertEquals(true, task.getParameter("param3"));
    }
    
    @Test
    public void testTaskExecutionTime() {
        // 测试任务执行时间
        long startTime = System.currentTimeMillis();
        task.start();
        
        try {
            Thread.sleep(10); // 短暂等待
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        task.complete("完成");
        long executionTime = task.getExecutionTime();
        
        assertTrue(executionTime >= 0);
        assertTrue(executionTime < 1000); // 应该在1秒内完成
    }
    
    @Test
    public void testPlanCreation() {
        // 测试计划创建
        assertNotNull(plan.getId());
        assertEquals("测试计划", plan.getTitle());
        assertEquals("测试计划目标", plan.getGoal());
        assertEquals("planning", plan.getStatus());
        assertNotNull(plan.getCreatedAt());
        assertTrue(plan.getTasks().isEmpty());
        assertEquals(0.0, plan.getProgress(), 0.001);
    }
    
    @Test
    public void testPlanTaskManagement() {
        // 测试计划任务管理
        assertTrue(plan.getTasks().isEmpty());
        
        plan.addTask("任务1", "thinking");
        plan.addTask("任务2", "action");
        plan.addTask(new Task("任务3", "observation"));
        
        assertEquals(3, plan.getTasks().size());
        assertEquals("任务1", plan.getTasks().get(0).getDescription());
        assertEquals("thinking", plan.getTasks().get(0).getType());
    }
    
    @Test
    public void testPlanExecution() {
        // 测试计划执行
        plan.addTask("任务1", "thinking");
        plan.addTask("任务2", "action");
        
        assertFalse(plan.isExecuting());
        plan.start();
        assertTrue(plan.isExecuting());
        assertEquals("executing", plan.getStatus());
        assertNotNull(plan.getStartedAt());
    }
    
    @Test
    public void testPlanTaskFlow() {
        // 测试计划任务流程
        plan.addTask("任务1", "thinking");
        plan.addTask("任务2", "action");
        plan.start();
        
        // 获取第一个任务
        Task nextTask = plan.getNextTask();
        assertNotNull(nextTask);
        assertEquals("任务1", nextTask.getDescription());
        
        // 完成第一个任务
        plan.completeCurrentTask("任务1完成");
        assertEquals(0.5, plan.getProgress(), 0.001);
        
        // 获取第二个任务
        nextTask = plan.getNextTask();
        assertNotNull(nextTask);
        assertEquals("任务2", nextTask.getDescription());
        
        // 完成第二个任务
        plan.completeCurrentTask("任务2完成");
        assertEquals(1.0, plan.getProgress(), 0.001);
        assertTrue(plan.isCompleted());
    }
    
    @Test
    public void testPlanFailure() {
        // 测试计划失败
        plan.addTask("任务1", "thinking");
        plan.start();
        
        plan.failCurrentTask("任务失败");
        assertTrue(plan.isFailed());
        assertEquals("failed", plan.getStatus());
        assertNotNull(plan.getCompletedAt());
    }
    
    @Test
    public void testPlanStatistics() {
        // 测试计划统计
        plan.addTask("任务1", "thinking");
        plan.addTask("任务2", "action");
        plan.addTask("任务3", "observation");
        plan.start();
        
        Map<String, Object> stats = plan.getStatistics();
        assertNotNull(stats);
        assertEquals(3, (int) stats.get("total_tasks"));
        assertEquals(0, (int) stats.get("completed_tasks"));
        assertEquals(0, (int) stats.get("failed_tasks"));
        assertEquals(0.0, stats.get("progress"));
        assertEquals("executing", stats.get("status"));
        
        // 完成一个任务
        plan.completeCurrentTask("完成");
        stats = plan.getStatistics();
        assertEquals(1, (int) stats.get("completed_tasks"));
        
        // 失败一个任务
        plan.failCurrentTask("失败");
        stats = plan.getStatistics();
        assertEquals(1, (int) stats.get("failed_tasks"));
    }
    
    @Test
    public void testPlanMetadata() {
        // 测试计划元数据
        assertTrue(plan.getMetadata().isEmpty());
        
        plan.addMetadata("priority", "high");
        plan.addMetadata("category", "test");
        
        Map<String, Object> metadata = plan.getMetadata();
        assertEquals("high", metadata.get("priority"));
        assertEquals("test", metadata.get("category"));
    }
    
    @Test
    public void testPlanParallelMode() {
        // 测试计划并行模式
        assertFalse(plan.isAllowParallel());
        
        plan.setAllowParallel(true);
        assertTrue(plan.isAllowParallel());
        
        plan.addTask("并行任务1", "action");
        plan.addTask("并行任务2", "action");
        plan.start();
        
        // 在并行模式下应该能获取到待执行的任务
        Task task1 = plan.getNextTask();
        assertNotNull(task1);
        
        // 标记第一个任务为运行中
        task1.start();
        
        // 应该还能获取到第二个任务
        Task task2 = plan.getNextTask();
        assertNotNull(task2);
        assertNotEquals(task1.getId(), task2.getId());
    }
}