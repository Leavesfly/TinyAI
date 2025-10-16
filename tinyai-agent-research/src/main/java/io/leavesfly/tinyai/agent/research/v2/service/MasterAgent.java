package io.leavesfly.tinyai.agent.research.v2.service;

import io.leavesfly.tinyai.agent.research.v2.adapter.ResearchLLMAdapter;
import io.leavesfly.tinyai.agent.research.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.component.ContextEngine;
import io.leavesfly.tinyai.agent.cursor.v2.component.memory.MemoryManager;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * 主控智能体
 * 负责研究任务的整体协调和流程控制
 * 
 * 核心职责:
 * 1. 接收研究任务并创建任务实例
 * 2. 调用PlannerAgent制定研究计划
 * 3. 协调ExecutorAgent并行执行任务
 * 4. 收集和整合各阶段结果
 * 5. 生成最终研究报告
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class MasterAgent {
    
    /**
     * 任务存储
     */
    private final Map<String, ResearchTask> tasks;
    
    /**
     * LLM适配器
     */
    private final ResearchLLMAdapter llmAdapter;
    
    /**
     * 上下文引擎
     */
    private final ContextEngine contextEngine;
    
    /**
     * 记忆管理器
     */
    private final MemoryManager memoryManager;
    
    /**
     * 规划智能体
     */
    private final PlannerAgent plannerAgent;
    
    /**
     * 执行器智能体
     */
    private final ExecutorAgent executorAgent;
    
    /**
     * 写作智能体
     */
    private final WriterAgent writerAgent;
    
    /**
     * 线程池（用于并行执行）
     */
    private final ExecutorService executorService;
    
    public MasterAgent(ResearchLLMAdapter llmAdapter, 
                      ContextEngine contextEngine,
                      MemoryManager memoryManager) {
        this.llmAdapter = llmAdapter;
        this.contextEngine = contextEngine;
        this.memoryManager = memoryManager;
        this.tasks = new ConcurrentHashMap<>();
        
        // 创建子智能体
        this.plannerAgent = new PlannerAgent(llmAdapter);
        this.executorAgent = new ExecutorAgent(llmAdapter, contextEngine, memoryManager);
        this.writerAgent = new WriterAgent(llmAdapter);
        
        // 创建线程池
        int processors = Runtime.getRuntime().availableProcessors();
        this.executorService = Executors.newFixedThreadPool(Math.max(4, processors));
    }
    
    /**
     * 提交研究任务
     */
    public ResearchTask submitResearch(String topic) {
        return submitResearch(topic, null, null);
    }
    
    /**
     * 提交研究任务（带配置）
     */
    public ResearchTask submitResearch(String topic, String userId, Map<String, Object> config) {
        System.out.println("[MasterAgent] 提交研究任务: " + topic);
        
        // 创建任务
        ResearchTask task = new ResearchTask(topic);
        task.setUserId(userId);
        if (config != null) {
            task.setConfig(config);
        }
        
        // 存储任务
        tasks.put(task.getTaskId(), task);
        
        // 异步执行研究
        executorService.submit(() -> executeResearch(task));
        
        return task;
    }
    
    /**
     * 执行研究流程
     */
    private void executeResearch(ResearchTask task) {
        try {
            // 阶段1: 规划
            System.out.println("[MasterAgent] 开始规划阶段");
            task.updateStatus(TaskStatus.PLANNING);
            ResearchPlan plan = plannerAgent.createPlan(task.getTopic(), task.getContext());
            task.setPlan(plan);
            
            // 阶段2: 执行
            System.out.println("[MasterAgent] 开始执行阶段，共 " + plan.getQuestions().size() + " 个问题");
            task.updateStatus(TaskStatus.EXECUTING);
            Map<String, Object> executionResults = executeResearchPlan(task, plan);
            task.putContext("executionResults", executionResults);
            
            // 阶段3: 分析
            System.out.println("[MasterAgent] 开始分析阶段");
            task.updateStatus(TaskStatus.ANALYZING);
            Map<String, Object> analysisResults = analyzeResults(task, executionResults);
            task.putContext("analysisResults", analysisResults);
            
            // 阶段4: 报告生成
            System.out.println("[MasterAgent] 开始报告生成阶段");
            task.updateStatus(TaskStatus.REPORTING);
            ResearchReport report = generateReport(task);
            task.setReport(report);
            
            // 完成
            task.updateStatus(TaskStatus.COMPLETED);
            System.out.println("[MasterAgent] 研究任务完成: " + task.getTaskId());
            
        } catch (Exception e) {
            System.err.println("[MasterAgent] 研究任务失败: " + e.getMessage());
            e.printStackTrace();
            task.updateStatus(TaskStatus.FAILED);
            task.setErrorMessage(e.getMessage());
        }
    }
    
    /**
     * 执行研究计划
     */
    private Map<String, Object> executeResearchPlan(ResearchTask task, ResearchPlan plan) {
        Map<String, Object> allResults = new ConcurrentHashMap<>();
        
        // 获取执行层级（支持并行）
        List<List<ResearchQuestion>> levels = plan.getExecutionLevels();
        
        for (int i = 0; i < levels.size(); i++) {
            List<ResearchQuestion> currentLevel = levels.get(i);
            System.out.println(String.format("[MasterAgent] 执行第 %d 层，共 %d 个问题", 
                i + 1, currentLevel.size()));
            
            // 并行执行同层级的问题
            List<Future<Map.Entry<String, Object>>> futures = new ArrayList<>();
            
            for (ResearchQuestion question : currentLevel) {
                Future<Map.Entry<String, Object>> future = executorService.submit(() -> {
                    Object result = executorAgent.executeQuestion(question, allResults);
                    return new AbstractMap.SimpleEntry<>(question.getQuestionId(), result);
                });
                futures.add(future);
            }
            
            // 等待当前层级完成
            for (Future<Map.Entry<String, Object>> future : futures) {
                try {
                    Map.Entry<String, Object> entry = future.get(5, TimeUnit.MINUTES);
                    allResults.put(entry.getKey(), entry.getValue());
                } catch (Exception e) {
                    System.err.println("[MasterAgent] 问题执行失败: " + e.getMessage());
                }
            }
        }
        
        return allResults;
    }
    
    /**
     * 分析执行结果
     */
    private Map<String, Object> analyzeResults(ResearchTask task, Map<String, Object> executionResults) {
        Map<String, Object> analysisResults = new HashMap<>();
        
        // 简单汇总（后续可增强）
        analysisResults.put("questionCount", executionResults.size());
        analysisResults.put("completionRate", calculateCompletionRate(executionResults));
        
        // 提取关键发现
        List<String> keyFindings = extractKeyFindings(executionResults);
        analysisResults.put("keyFindings", keyFindings);
        
        return analysisResults;
    }
    
    /**
     * 生成研究报告
     */
    private ResearchReport generateReport(ResearchTask task) {
        // 收集所有信息
        String topic = task.getTopic();
        Map<String, Object> executionResults = task.getContext("executionResults");
        Map<String, Object> analysisResults = task.getContext("analysisResults");
        
        // 调用WriterAgent生成报告
        return writerAgent.generateReport(topic, executionResults, analysisResults);
    }
    
    /**
     * 查询任务状态
     */
    public ResearchTask queryTask(String taskId) {
        return tasks.get(taskId);
    }
    
    /**
     * 获取任务进度
     */
    public TaskProgress getTaskProgress(String taskId) {
        ResearchTask task = tasks.get(taskId);
        if (task == null) {
            return null;
        }
        
        TaskProgress progress = new TaskProgress();
        progress.setTaskId(taskId);
        progress.setStatus(task.getStatus());
        progress.setProgress(calculateProgress(task));
        progress.setElapsedSeconds(task.getDurationSeconds());
        
        return progress;
    }
    
    /**
     * 计算任务进度
     */
    private double calculateProgress(ResearchTask task) {
        switch (task.getStatus()) {
            case SUBMITTED: return 0.0;
            case PLANNING: return 10.0;
            case EXECUTING: return 40.0;
            case ANALYZING: return 70.0;
            case REPORTING: return 90.0;
            case COMPLETED: return 100.0;
            case FAILED: return 0.0;
            default: return 0.0;
        }
    }
    
    /**
     * 计算完成率
     */
    private double calculateCompletionRate(Map<String, Object> results) {
        if (results == null || results.isEmpty()) {
            return 0.0;
        }
        
        long successCount = results.values().stream()
            .filter(Objects::nonNull)
            .count();
        
        return (double) successCount / results.size();
    }
    
    /**
     * 提取关键发现
     */
    private List<String> extractKeyFindings(Map<String, Object> results) {
        List<String> findings = new ArrayList<>();
        
        // 简单提取（后续可使用LLM增强）
        for (Map.Entry<String, Object> entry : results.entrySet()) {
            if (entry.getValue() != null) {
                findings.add(entry.getValue().toString());
            }
        }
        
        return findings;
    }
    
    /**
     * 关闭资源
     */
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * 任务进度类
     */
    public static class TaskProgress {
        private String taskId;
        private TaskStatus status;
        private double progress;
        private long elapsedSeconds;
        
        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        
        public TaskStatus getStatus() { return status; }
        public void setStatus(TaskStatus status) { this.status = status; }
        
        public double getProgress() { return progress; }
        public void setProgress(double progress) { this.progress = progress; }
        
        public long getElapsedSeconds() { return elapsedSeconds; }
        public void setElapsedSeconds(long elapsedSeconds) { this.elapsedSeconds = elapsedSeconds; }
    }
}
