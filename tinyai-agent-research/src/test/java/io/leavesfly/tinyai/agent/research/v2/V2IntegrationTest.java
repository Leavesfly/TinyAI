package io.leavesfly.tinyai.agent.research.v2;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.*;
import io.leavesfly.tinyai.agent.research.v2.service.MasterAgent;
import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;
import io.leavesfly.tinyai.agent.cursor.v2.service.MockLLMGateway;

import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 端到端集成测试
 * 测试完整的研究流程
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class V2IntegrationTest {
    
    private MasterAgent masterAgent;
    
    @Before
    public void setUp() {
        // 初始化组件
        LLMGateway llmGateway = new MockLLMGateway();
        ResearchLLMAdapter llmAdapter = new ResearchLLMAdapter(llmGateway);
        
        MemoryManager memoryManager = new MemoryManager();
        RAGEngine ragEngine = new RAGEngine();
        ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);
        
        masterAgent = new MasterAgent(llmAdapter, contextEngine, memoryManager);
    }
    
    @After
    public void tearDown() {
        if (masterAgent != null) {
            masterAgent.shutdown();
        }
    }
    
    @Test(timeout = 60000) // 60秒超时
    public void testCompleteResearchWorkflow() throws InterruptedException {
        // 1. 提交研究任务
        String topic = "Java并发编程最佳实践";
        ResearchTask task = masterAgent.submitResearch(topic);
        
        assertNotNull(task);
        assertNotNull(task.getTaskId());
        assertEquals(topic, task.getTopic());
        assertEquals(TaskStatus.SUBMITTED, task.getStatus());
        
        // 2. 等待任务完成
        int maxWaitSeconds = 30;
        int waited = 0;
        while (waited < maxWaitSeconds) {
            Thread.sleep(1000);
            waited++;
            
            ResearchTask updatedTask = masterAgent.queryTask(task.getTaskId());
            assertNotNull(updatedTask);
            
            if (updatedTask.getStatus().isTerminal()) {
                break;
            }
        }
        
        // 3. 验证最终状态
        ResearchTask finalTask = masterAgent.queryTask(task.getTaskId());
        assertNotNull(finalTask);
        
        // 应该完成或失败
        assertTrue(finalTask.getStatus().isTerminal());
        
        if (finalTask.getStatus() == TaskStatus.COMPLETED) {
            // 验证研究计划
            assertNotNull(finalTask.getPlan());
            assertTrue(finalTask.getPlan().getQuestions().size() > 0);
            
            // 验证研究报告
            assertNotNull(finalTask.getReport());
            assertNotNull(finalTask.getReport().getTitle());
            assertNotNull(finalTask.getReport().getFullContent());
            
            System.out.println("研究完成！");
            System.out.println("报告标题: " + finalTask.getReport().getTitle());
            System.out.println("问题数量: " + finalTask.getPlan().getQuestions().size());
        } else {
            System.err.println("任务失败: " + finalTask.getErrorMessage());
        }
    }
    
    @Test
    public void testTaskProgressQuery() throws InterruptedException {
        // 提交任务
        ResearchTask task = masterAgent.submitResearch("测试主题");
        
        // 等待一小段时间
        Thread.sleep(500);
        
        // 查询进度
        MasterAgent.TaskProgress progress = masterAgent.getTaskProgress(task.getTaskId());
        
        assertNotNull(progress);
        assertEquals(task.getTaskId(), progress.getTaskId());
        assertNotNull(progress.getStatus());
        assertTrue(progress.getProgress() >= 0);
        assertTrue(progress.getProgress() <= 100);
    }
    
    @Test
    public void testMultipleTasksSequentially() throws InterruptedException {
        // 顺序提交多个任务
        ResearchTask task1 = masterAgent.submitResearch("主题1");
        Thread.sleep(100);
        ResearchTask task2 = masterAgent.submitResearch("主题2");
        
        assertNotNull(task1);
        assertNotNull(task2);
        assertNotEquals(task1.getTaskId(), task2.getTaskId());
        
        // 两个任务都应该可以查询到
        assertNotNull(masterAgent.queryTask(task1.getTaskId()));
        assertNotNull(masterAgent.queryTask(task2.getTaskId()));
    }
}
