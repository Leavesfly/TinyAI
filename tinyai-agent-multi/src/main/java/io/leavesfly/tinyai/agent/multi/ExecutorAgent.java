package io.leavesfly.tinyai.agent.multi;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * 执行员Agent
 * 专门负责任务执行、工具使用和结果报告
 * 
 * @author 山泽
 */
public class ExecutorAgent extends BaseAgent {
    
    public ExecutorAgent(String agentId, MessageBus messageBus, LLMSimulator llm) {
        super(agentId, "执行员-" + agentId.substring(Math.max(0, agentId.length() - 4)), 
              "任务执行员", messageBus, llm);
    }
    
    @Override
    protected void initializeCapabilities() {
        capabilities.addAll(Arrays.asList("任务执行", "工具使用", "结果报告", "操作自动化"));
    }
    
    @Override
    protected String getAgentType() {
        return "executor";
    }
    
    @Override
    protected Object performTask(AgentTask task) throws Exception {
        String description = task.getDescription().toLowerCase();
        
        if (description.contains("数据")) {
            return executeDataTask(task);
        } else if (description.contains("文档")) {
            return executeDocumentTask(task);
        } else if (description.contains("处理")) {
            return executeProcessingTask(task);
        } else {
            return executeGeneralTask(task);
        }
    }
    
    /**
     * 执行数据相关任务
     */
    private Map<String, Object> executeDataTask(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始执行数据任务: %s", name, task.getTitle()));
        
        List<String> executionSteps = Arrays.asList(
            "连接数据源",
            "验证数据完整性",
            "清洗和预处理",
            "执行数据变换",
            "生成处理报告"
        );
        
        return executeStepsWithProgress(executionSteps, task);
    }
    
    /**
     * 执行文档相关任务
     */
    private Map<String, Object> executeDocumentTask(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始执行文档任务: %s", name, task.getTitle()));
        
        List<String> executionSteps = Arrays.asList(
            "收集相关信息",
            "整理文档结构",
            "编写内容草稿",
            "格式化和美化",
            "审核和最终确认"
        );
        
        return executeStepsWithProgress(executionSteps, task);
    }
    
    /**
     * 执行处理任务
     */
    private Map<String, Object> executeProcessingTask(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始执行处理任务: %s", name, task.getTitle()));
        
        List<String> executionSteps = Arrays.asList(
            "初始化处理环境",
            "加载处理模块",
            "执行核心处理逻辑",
            "验证处理结果",
            "输出处理报告"
        );
        
        return executeStepsWithProgress(executionSteps, task);
    }
    
    /**
     * 执行通用任务
     */
    private Map<String, Object> executeGeneralTask(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始执行通用任务: %s", name, task.getTitle()));
        
        List<String> executionSteps = Arrays.asList(
            "分析任务需求",
            "制定执行计划",
            "准备执行环境",
            "逐步执行操作",
            "完成质量验收"
        );
        
        return executeStepsWithProgress(executionSteps, task);
    }
    
    /**
     * 按步骤执行并显示进度
     */
    private Map<String, Object> executeStepsWithProgress(List<String> steps, AgentTask task) throws InterruptedException {
        for (int i = 0; i < steps.size(); i++) {
            String step = steps.get(i);
            
            System.out.println(String.format("Agent %s 执行步骤 %d/%d: %s", 
                    name, i + 1, steps.size(), step));
            
            // 模拟执行时间（每个步骤500-800ms）
            Thread.sleep(500 + (int)(Math.random() * 300));
            
            // 记录进度到任务元数据
            task.addMetadata("step_" + (i + 1), step);
            task.addMetadata("progress", String.format("%.1f%%", (i + 1) * 100.0 / steps.size()));
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("executionType", "步骤化执行");
        result.put("stepsCompleted", steps);
        result.put("totalSteps", steps.size());
        result.put("executionTime", steps.size() * 650); // 平均执行时间
        result.put("status", "成功完成");
        result.put("quality", "良好");
        result.put("outputFormat", "标准化报告");
        
        return result;
    }
    
    /**
     * 模拟工具使用
     */
    private void useTool(String toolName, String operation) throws InterruptedException {
        System.out.println(String.format("Agent %s 使用工具: %s 执行 %s", name, toolName, operation));
        Thread.sleep(200); // 模拟工具使用时间
    }
    
    @Override
    protected void performPeriodicWork() throws InterruptedException {
        // 执行员的周期性工作：检查工具状态、清理临时文件等
        if (metrics.getTasksCompleted() % 10 == 0 && metrics.getTasksCompleted() > 0) {
            System.out.println(String.format("Agent %s 执行周期性系统维护", name));
            
            // 模拟系统维护操作
            useTool("系统清理工具", "清理临时文件");
            useTool("性能监控工具", "检查系统性能");
            
            Thread.sleep(300);
        }
        
        super.performPeriodicWork();
    }
}