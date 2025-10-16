package io.leavesfly.tinyai.agent.research.v2;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.*;
import io.leavesfly.tinyai.agent.research.v2.service.MasterAgent;
import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;
import io.leavesfly.tinyai.agent.cursor.v2.component.rag.RAGEngine;
import io.leavesfly.tinyai.agent.cursor.v2.service.LLMGateway;

/**
 * V2版本演示程序
 * 展示深度研究智能体V2的核心功能
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class V2Demo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("TinyAI 深度研究智能体 V2 演示");
        System.out.println("=".repeat(60));
        System.out.println();
        
        try {
            // 1. 初始化组件
            System.out.println("【1】初始化系统组件...");
            
            // 创建模拟的LLM网关
            LLMGateway llmGateway = createMockLLMGateway();
            ResearchLLMAdapter llmAdapter = new ResearchLLMAdapter(llmGateway);
            
            // 创建上下文引擎
            MemoryManager memoryManager = new MemoryManager();
            RAGEngine ragEngine = new RAGEngine();
            ContextEngine contextEngine = new ContextEngine(memoryManager, ragEngine);
            
            // 创建主控智能体
            MasterAgent masterAgent = new MasterAgent(llmAdapter, contextEngine, memoryManager);
            System.out.println("✓ 系统组件初始化完成\\n");
            
            // 2. 提交研究任务
            System.out.println("【2】提交研究任务...");
            String topic = "Java中的并发编程最佳实践";
            ResearchTask task = masterAgent.submitResearch(topic);
            System.out.println("✓ 任务已提交，任务ID: " + task.getTaskId());
            System.out.println("✓ 研究主题: " + topic);
            System.out.println();
            
            // 3. 等待任务完成
            System.out.println("【3】等待研究完成...");
            waitForCompletion(masterAgent, task.getTaskId(), 30);
            
            // 4. 获取研究结果
            System.out.println("\\n【4】研究结果:");
            ResearchTask completedTask = masterAgent.queryTask(task.getTaskId());
            
            if (completedTask.getStatus() == TaskStatus.COMPLETED) {
                displayResults(completedTask);
            } else {
                System.out.println("✗ 任务状态: " + completedTask.getStatus());
                if (completedTask.getErrorMessage() != null) {
                    System.out.println("✗ 错误信息: " + completedTask.getErrorMessage());
                }
            }
            
            // 5. 关闭资源
            masterAgent.shutdown();
            System.out.println("\\n✓ 系统已关闭");
            
        } catch (Exception e) {
            System.err.println("✗ 演示过程中发生错误:");
            e.printStackTrace();
        }
        
        System.out.println("\\n" + "=".repeat(60));
    }
    
    /**
     * 等待任务完成
     */
    private static void waitForCompletion(MasterAgent masterAgent, String taskId, int maxSeconds) {
        int elapsed = 0;
        while (elapsed < maxSeconds) {
            try {
                Thread.sleep(1000);
                elapsed++;
                
                MasterAgent.TaskProgress progress = masterAgent.getTaskProgress(taskId);
                if (progress != null) {
                    System.out.print("\\r进度: " + String.format("%.1f%%", progress.getProgress()) + 
                                   " | 状态: " + progress.getStatus().getName() +
                                   " | 耗时: " + elapsed + "s");
                    
                    if (progress.getStatus().isTerminal()) {
                        System.out.println();
                        break;
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
    
    /**
     * 显示研究结果
     */
    private static void displayResults(ResearchTask task) {
        System.out.println("-".repeat(60));
        System.out.println("任务ID: " + task.getTaskId());
        System.out.println("主题: " + task.getTopic());
        System.out.println("状态: " + task.getStatus().getName());
        System.out.println("耗时: " + task.getDurationSeconds() + "秒");
        
        // 显示研究计划
        if (task.getPlan() != null) {
            ResearchPlan plan = task.getPlan();
            System.out.println("\\n研究计划:");
            System.out.println("  - 策略: " + plan.getStrategy().getName());
            System.out.println("  - 问题数: " + plan.getQuestions().size());
            System.out.println("  - 研究深度: " + plan.getEstimatedDepth());
        }
        
        // 显示研究报告
        if (task.getReport() != null) {
            ResearchReport report = task.getReport();
            System.out.println("\\n研究报告:");
            System.out.println("  标题: " + report.getTitle());
            System.out.println("  摘要: " + report.getSummary());
            System.out.println("  质量评分: " + report.getQualityScore());
            System.out.println("\\n完整内容:");
            System.out.println(report.getFullContent());
        }
        
        System.out.println("-".repeat(60));
    }
    
    /**
     * 创建模拟的LLM网关(用于演示)
     */
    private static LLMGateway createMockLLMGateway() {
        return new io.leavesfly.tinyai.agent.cursor.v2.service.MockLLMGateway();
    }
}
