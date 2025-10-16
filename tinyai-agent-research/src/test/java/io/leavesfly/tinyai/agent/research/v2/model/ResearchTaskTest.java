package io.leavesfly.tinyai.agent.research.v2.model;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * ResearchTask 单元测试
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ResearchTaskTest {
    
    @Test
    public void testTaskCreation() {
        ResearchTask task = new ResearchTask("测试研究主题");
        
        assertNotNull(task.getTaskId());
        assertEquals("测试研究主题", task.getTopic());
        assertEquals(TaskStatus.SUBMITTED, task.getStatus());
        assertNotNull(task.getCreatedAt());
    }
    
    @Test
    public void testStatusTransition() {
        ResearchTask task = new ResearchTask("测试");
        
        // 正常状态转换
        task.updateStatus(TaskStatus.PLANNING);
        assertEquals(TaskStatus.PLANNING, task.getStatus());
        
        task.updateStatus(TaskStatus.EXECUTING);
        assertEquals(TaskStatus.EXECUTING, task.getStatus());
    }
    
    @Test(expected = IllegalStateException.class)
    public void testInvalidStatusTransition() {
        ResearchTask task = new ResearchTask("测试");
        
        // 尝试非法状态转换
        task.updateStatus(TaskStatus.COMPLETED);
    }
    
    @Test
    public void testContextOperations() {
        ResearchTask task = new ResearchTask("测试");
        
        task.putContext("key1", "value1");
        task.putContext("key2", 123);
        
        assertEquals("value1", task.getContext("key1"));
        assertEquals(Integer.valueOf(123), task.getContext("key2"));
    }
    
    @Test
    public void testDurationCalculation() {
        ResearchTask task = new ResearchTask("测试");
        
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            // ignore
        }
        
        long duration = task.getDurationSeconds();
        assertTrue(duration >= 0);
    }
}
