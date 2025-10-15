package io.leavesfly.tinyai.agent.cursor.v2.tool;

import io.leavesfly.tinyai.agent.cursor.v2.model.Message;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolCall;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;

import java.util.*;
import java.util.concurrent.*;

/**
 * 工具编排器
 * 负责工具的调用、结果收集和错误处理
 * 
 * 核心功能：
 * 1. 解析LLM的工具调用请求
 * 2. 验证和执行工具
 * 3. 收集和格式化工具执行结果
 * 4. 支持并行执行多个工具
 * 5. 提供沙箱隔离能力
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolOrchestrator {
    
    /**
     * 工具注册表
     */
    private final ToolRegistry registry;
    
    /**
     * 线程池（用于并行执行工具）
     */
    private final ExecutorService executor;
    
    /**
     * 工具执行超时时间（秒）
     */
    private int toolExecutionTimeout = 30;
    
    /**
     * 是否启用并行执行
     */
    private boolean enableParallelExecution = true;
    
    /**
     * 执行历史（用于调试和审计）
     */
    private final List<ToolExecutionRecord> executionHistory;
    
    /**
     * 最大历史记录数
     */
    private int maxHistorySize = 100;
    
    public ToolOrchestrator(ToolRegistry registry) {
        this.registry = registry;
        this.executor = Executors.newFixedThreadPool(5);
        this.executionHistory = new ArrayList<>();
    }
    
    /**
     * 执行工具调用列表
     * 
     * @param toolCalls 工具调用列表
     * @return 工具执行结果列表
     */
    public List<ToolResult> executeTools(List<ToolCall> toolCalls) {
        if (toolCalls == null || toolCalls.isEmpty()) {
            return new ArrayList<>();
        }
        
        if (enableParallelExecution && toolCalls.size() > 1) {
            return executeToolsParallel(toolCalls);
        } else {
            return executeToolsSequential(toolCalls);
        }
    }
    
    /**
     * 执行单个工具调用
     * 
     * @param toolCall 工具调用
     * @return 工具执行结果
     */
    public ToolResult executeTool(ToolCall toolCall) {
        if (toolCall == null || toolCall.getFunction() == null) {
            return ToolResult.failure("unknown", "Invalid tool call");
        }
        
        String toolName = toolCall.getFunction().getName();
        Map<String, Object> arguments = toolCall.getFunction().getArguments();
        
        return executeTool(toolName, arguments);
    }
    
    /**
     * 执行工具（按名称）
     * 
     * @param toolName 工具名称
     * @param parameters 工具参数
     * @return 工具执行结果
     */
    public ToolResult executeTool(String toolName, Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 查找工具
            Tool tool = registry.getTool(toolName);
            if (tool == null) {
                return ToolResult.failure(toolName, "Tool not found: " + toolName);
            }
            
            // 验证参数
            if (!tool.validateParameters(parameters)) {
                return ToolResult.failure(toolName, "Invalid parameters");
            }
            
            // 执行工具
            ToolResult result;
            if (tool.requiresSandbox()) {
                result = executeInSandbox(tool, parameters);
            } else {
                result = tool.execute(parameters);
            }
            
            // 记录执行历史
            recordExecution(toolName, parameters, result, System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (Exception e) {
            ToolResult result = ToolResult.failure(toolName, "Execution failed: " + e.getMessage());
            result.setExecutionTime(System.currentTimeMillis() - startTime);
            return result;
        }
    }
    
    /**
     * 顺序执行工具列表
     */
    private List<ToolResult> executeToolsSequential(List<ToolCall> toolCalls) {
        List<ToolResult> results = new ArrayList<>();
        
        for (ToolCall toolCall : toolCalls) {
            ToolResult result = executeTool(toolCall);
            results.add(result);
        }
        
        return results;
    }
    
    /**
     * 并行执行工具列表
     */
    private List<ToolResult> executeToolsParallel(List<ToolCall> toolCalls) {
        List<Future<ToolResult>> futures = new ArrayList<>();
        
        // 提交所有任务
        for (ToolCall toolCall : toolCalls) {
            Future<ToolResult> future = executor.submit(() -> executeTool(toolCall));
            futures.add(future);
        }
        
        // 收集结果
        List<ToolResult> results = new ArrayList<>();
        for (Future<ToolResult> future : futures) {
            try {
                ToolResult result = future.get(toolExecutionTimeout, TimeUnit.SECONDS);
                results.add(result);
            } catch (TimeoutException e) {
                ToolResult timeoutResult = ToolResult.failure("unknown", "Execution timeout");
                results.add(timeoutResult);
            } catch (Exception e) {
                ToolResult errorResult = ToolResult.failure("unknown", "Execution error: " + e.getMessage());
                results.add(errorResult);
            }
        }
        
        return results;
    }
    
    /**
     * 在沙箱中执行工具
     * TODO: 实现真正的沙箱隔离（如使用Docker、JVM SecurityManager等）
     */
    private ToolResult executeInSandbox(Tool tool, Map<String, Object> parameters) {
        // 当前仅做简单的超时控制
        Future<ToolResult> future = executor.submit(() -> tool.execute(parameters));
        
        try {
            return future.get(toolExecutionTimeout, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            return ToolResult.failure(tool.getName(), "Execution timeout in sandbox");
        } catch (Exception e) {
            return ToolResult.failure(tool.getName(), "Sandbox execution failed: " + e.getMessage());
        }
    }
    
    /**
     * 将工具执行结果转换为消息
     * 
     * @param results 工具执行结果列表
     * @return 工具消息列表
     */
    public List<Message> resultsToMessages(List<ToolResult> results) {
        List<Message> messages = new ArrayList<>();
        
        for (ToolResult result : results) {
            String content = result.isSuccess() ? 
                           result.getResult() : 
                           "Error: " + result.getError();
            
            // 创建工具消息（需要与对应的toolCallId关联）
            Message message = Message.tool(content, result.getToolName());
            messages.add(message);
        }
        
        return messages;
    }
    
    /**
     * 获取可用工具定义（用于传递给LLM）
     * 
     * @return 工具定义列表
     */
    public List<ToolDefinition> getAvailableTools() {
        return registry.getAllToolDefinitions();
    }
    
    /**
     * 获取指定类别的工具定义
     * 
     * @param category 工具类别
     * @return 工具定义列表
     */
    public List<ToolDefinition> getAvailableTools(Tool.ToolCategory category) {
        return registry.getToolDefinitionsByCategory(category);
    }
    
    /**
     * 记录执行历史
     */
    private void recordExecution(String toolName, Map<String, Object> parameters, 
                                 ToolResult result, long executionTime) {
        ToolExecutionRecord record = new ToolExecutionRecord(
            toolName, parameters, result, executionTime, System.currentTimeMillis()
        );
        
        synchronized (executionHistory) {
            executionHistory.add(record);
            
            // 限制历史记录大小
            if (executionHistory.size() > maxHistorySize) {
                executionHistory.remove(0);
            }
        }
    }
    
    /**
     * 获取执行历史
     * 
     * @param limit 返回的记录数量
     * @return 执行历史记录
     */
    public List<ToolExecutionRecord> getExecutionHistory(int limit) {
        synchronized (executionHistory) {
            int size = executionHistory.size();
            int fromIndex = Math.max(0, size - limit);
            return new ArrayList<>(executionHistory.subList(fromIndex, size));
        }
    }
    
    /**
     * 清除执行历史
     */
    public void clearHistory() {
        synchronized (executionHistory) {
            executionHistory.clear();
        }
    }
    
    /**
     * 获取统计信息
     */
    public OrchestratorStats getStats() {
        OrchestratorStats stats = new OrchestratorStats();
        
        synchronized (executionHistory) {
            stats.totalExecutions = executionHistory.size();
            
            for (ToolExecutionRecord record : executionHistory) {
                if (record.result.isSuccess()) {
                    stats.successfulExecutions++;
                } else {
                    stats.failedExecutions++;
                }
                stats.totalExecutionTime += record.executionTime;
            }
            
            if (stats.totalExecutions > 0) {
                stats.averageExecutionTime = stats.totalExecutionTime / stats.totalExecutions;
            }
        }
        
        return stats;
    }
    
    /**
     * 关闭编排器
     */
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
    
    /**
     * 配置项
     */
    public void setToolExecutionTimeout(int seconds) {
        this.toolExecutionTimeout = seconds;
    }
    
    public void setEnableParallelExecution(boolean enable) {
        this.enableParallelExecution = enable;
    }
    
    public void setMaxHistorySize(int size) {
        this.maxHistorySize = size;
    }
    
    /**
     * 工具执行记录
     */
    public static class ToolExecutionRecord {
        public final String toolName;
        public final Map<String, Object> parameters;
        public final ToolResult result;
        public final long executionTime;
        public final long timestamp;
        
        public ToolExecutionRecord(String toolName, Map<String, Object> parameters,
                                  ToolResult result, long executionTime, long timestamp) {
            this.toolName = toolName;
            this.parameters = parameters;
            this.result = result;
            this.executionTime = executionTime;
            this.timestamp = timestamp;
        }
        
        @Override
        public String toString() {
            return "ToolExecutionRecord{" +
                    "tool='" + toolName + '\'' +
                    ", success=" + result.isSuccess() +
                    ", time=" + executionTime + "ms" +
                    '}';
        }
    }
    
    /**
     * 编排器统计信息
     */
    public static class OrchestratorStats {
        public int totalExecutions;
        public int successfulExecutions;
        public int failedExecutions;
        public long totalExecutionTime;
        public long averageExecutionTime;
        
        @Override
        public String toString() {
            return "OrchestratorStats{" +
                    "total=" + totalExecutions +
                    ", success=" + successfulExecutions +
                    ", failed=" + failedExecutions +
                    ", avgTime=" + averageExecutionTime + "ms" +
                    '}';
        }
    }
}
