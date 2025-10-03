package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.ToolCall;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Manus核心系统
 * OpenManus分层架构的第四层（核心层）
 * 实现双执行机制、计划驱动和Flow编排
 * 
 * @author 山泽
 */
public class Manus extends ToolCallAgent {
    
    // 执行模式和配置
    private ExecutionMode executionMode;            // 当前执行模式
    private boolean planningEnabled;                // 是否启用计划模式
    private Plan currentPlan;                       // 当前执行的计划
    
    // Flow管理
    private Map<String, FlowDefinition> registeredFlows;  // 注册的Flow
    private Map<String, String> flowMappings;             // Flow映射
    
    // 统计和监控
    private Map<String, Object> systemMetrics;     // 系统指标
    private LocalDateTime systemStartTime;         // 系统启动时间
    private int totalProcessedMessages;            // 处理的消息总数
    private int totalExecutedPlans;                // 执行的计划总数
    private int totalFlowExecutions;               // Flow执行总数
    
    /**
     * 构造函数
     */
    public Manus(String name) {
        super(name);
        this.executionMode = ExecutionMode.DIRECT_AGENT;
        this.planningEnabled = false;
        this.registeredFlows = new ConcurrentHashMap<>();
        this.flowMappings = new HashMap<>();
        this.systemMetrics = new ConcurrentHashMap<>();
        this.systemStartTime = LocalDateTime.now();
        this.totalProcessedMessages = 0;
        this.totalExecutedPlans = 0;
        this.totalFlowExecutions = 0;
        
        // 初始化Flow映射
        initializeFlowMappings();
        
        // 注册示例Flow
        registerDefaultFlows();
    }
    
    /**
     * 初始化Flow映射
     */
    private void initializeFlowMappings() {
        flowMappings.put("计算", "calculation_flow");
        flowMappings.put("数学", "calculation_flow");
        flowMappings.put("运算", "calculation_flow");
        flowMappings.put("时间", "time_flow");
        flowMappings.put("几点", "time_flow");
        flowMappings.put("现在", "time_flow");
        flowMappings.put("分析", "analysis_flow");
        flowMappings.put("解析", "analysis_flow");
        flowMappings.put("检查", "analysis_flow");
    }
    
    /**
     * 注册默认Flow
     */
    private void registerDefaultFlows() {
        // 计算Flow
        FlowDefinition calculationFlow = new FlowDefinition("计算流程", "专门处理数学计算的流程");
        Map<String, Object> calcNodes = new HashMap<>();
        calcNodes.put("type", "tool");
        calcNodes.put("name", "calculator");
        calcNodes.put("description", "执行数学计算");
        calculationFlow.setNodes(calcNodes);
        registerFlow("calculation_flow", calculationFlow);
        
        // 时间查询Flow
        FlowDefinition timeFlow = new FlowDefinition("时间查询流程", "处理时间相关查询的流程");
        Map<String, Object> timeNodes = new HashMap<>();
        timeNodes.put("type", "tool");
        timeNodes.put("name", "get_time");
        timeNodes.put("description", "获取当前时间");
        timeFlow.setNodes(timeNodes);
        registerFlow("time_flow", timeFlow);
        
        // 分析Flow
        FlowDefinition analysisFlow = new FlowDefinition("分析流程", "处理分析任务的流程");
        Map<String, Object> analysisNodes = new HashMap<>();
        analysisNodes.put("type", "tool");
        analysisNodes.put("name", "text_analyzer");
        analysisNodes.put("description", "执行文本分析");
        analysisFlow.setNodes(analysisNodes);
        registerFlow("analysis_flow", analysisFlow);
    }
    
    /**
     * 核心消息处理方法 - 根据执行模式分发
     */
    @Override
    public Message processMessage(Message message) {
        addMessage(message);
        totalProcessedMessages++;
        updateState(AgentState.THINKING);
        
        try {
            Message response;
            
            // 根据执行模式选择处理方式
            switch (executionMode) {
                case DIRECT_AGENT:
                    response = processDirectAgentMode(message);
                    break;
                    
                case FLOW_ORCHESTRATION:
                    response = processFlowOrchestrationMode(message);
                    break;
                    
                default:
                    response = processDirectAgentMode(message);
                    break;
            }
            
            addMessage(response);
            updateState(AgentState.DONE);
            return response;
            
        } catch (Exception e) {
            updateState(AgentState.ERROR);
            String errorMessage = "Manus处理过程中发生错误：" + e.getMessage();
            Message errorResponse = new Message("assistant", errorMessage);
            addMessage(errorResponse);
            return errorResponse;
        }
    }
    
    /**
     * 直接Agent模式处理
     */
    private Message processDirectAgentMode(Message message) {
        if (planningEnabled) {
            return processWithPlanning(message);
        } else {
            // 使用父类的ToolCallAgent能力
            return super.processMessage(message);
        }
    }
    
    /**
     * 计划驱动处理
     */
    private Message processWithPlanning(Message message) {
        updateState(AgentState.PLANNING);
        
        String query = message.getContent();
        
        // 创建计划
        Plan plan = createPlanForQuery(query);
        currentPlan = plan;
        totalExecutedPlans++;
        
        // 执行计划
        plan.start();
        StringBuilder result = new StringBuilder();
        result.append("【计划驱动模式】\n");
        result.append("计划：").append(plan.getTitle()).append("\n");
        result.append("目标：").append(plan.getGoal()).append("\n\n");
        
        // 执行任务
        while (!plan.isCompleted() && !plan.isFailed()) {
            Task nextTask = plan.getNextTask();
            if (nextTask == null) break;
            
            result.append("执行任务：").append(nextTask.getDescription()).append("\n");
            
            try {
                Object taskResult = executeTask(nextTask);
                plan.completeCurrentTask(taskResult);
                result.append("任务结果：").append(taskResult).append("\n\n");
            } catch (Exception e) {
                plan.failCurrentTask("任务执行失败：" + e.getMessage());
                result.append("任务失败：").append(e.getMessage()).append("\n\n");
                break;
            }
        }
        
        // 添加执行统计
        Map<String, Object> stats = plan.getStatistics();
        result.append("计划执行统计：").append(stats).append("\n");
        
        return new Message("assistant", result.toString());
    }
    
    /**
     * Flow编排模式处理
     */
    private Message processFlowOrchestrationMode(Message message) {
        updateState(AgentState.ACTING);
        
        String query = message.getContent();
        
        // 选择合适的Flow
        String flowId = selectFlowForQuery(query);
        
        if (flowId != null && registeredFlows.containsKey(flowId)) {
            totalFlowExecutions++;
            return executeFlow(flowId, query);
        } else {
            // 回退到直接Agent模式
            return processDirectAgentMode(message);
        }
    }
    
    /**
     * 为查询创建计划
     */
    private Plan createPlanForQuery(String query) {
        Plan plan = new Plan("处理查询：" + query.substring(0, Math.min(query.length(), 20)) + "...", 
                           "分析并回答用户的查询");
        
        // 分析查询类型并创建相应任务
        if (query.contains("详细") || query.contains("深入") || query.contains("完整")) {
            // 复杂查询需要多步骤处理
            plan.addTask("分析查询意图", "thinking");
            plan.addTask("收集相关信息", "action");
            plan.addTask("整合分析结果", "thinking");
            plan.addTask("生成详细回答", "action");
        } else {
            // 简单查询
            plan.addTask("理解查询", "thinking");
            plan.addTask("执行相应操作", "action");
            plan.addTask("生成回答", "thinking");
        }
        
        return plan;
    }
    
    /**
     * 执行任务
     */
    private Object executeTask(Task task) {
        task.start();
        
        try {
            String taskType = task.getType();
            String description = task.getDescription();
            
            switch (taskType) {
                case "thinking":
                    return executeThinkingTask(description);
                    
                case "action":
                    return executeActionTask(description);
                    
                case "observation":
                    return executeObservationTask(description);
                    
                default:
                    return "任务类型未知：" + taskType;
            }
            
        } catch (Exception e) {
            throw new RuntimeException("任务执行异常：" + e.getMessage(), e);
        }
    }
    
    /**
     * 执行思考任务
     */
    private String executeThinkingTask(String description) {
        if (description.contains("分析查询意图")) {
            return "查询意图已分析，需要使用工具来处理";
        } else if (description.contains("整合分析结果")) {
            return "结果已整合，准备生成回答";
        } else if (description.contains("理解查询")) {
            return "查询已理解，确定处理方案";
        } else if (description.contains("生成回答")) {
            return "基于分析结果生成最终回答";
        } else {
            return "思考完成：" + description;
        }
    }
    
    /**
     * 执行行动任务
     */
    private String executeActionTask(String description) {
        if (description.contains("收集相关信息") || description.contains("执行相应操作")) {
            // 使用工具调用能力
            String lastUserMessage = getLastUserMessage();
            List<String> recommendedTools = getRecommendedTools(lastUserMessage);
            
            if (!recommendedTools.isEmpty()) {
                String toolName = recommendedTools.get(0);
                Map<String, Object> args = prepareToolArgumentsForQuery(toolName, lastUserMessage);
                ToolCall result = callTool(toolName, args);
                
                if (result.isSuccess()) {
                    return "工具执行成功：" + result.getResult();
                } else {
                    return "工具执行失败：" + result.getError();
                }
            } else {
                return "没有找到合适的工具";
            }
        } else {
            return "行动完成：" + description;
        }
    }
    
    /**
     * 执行观察任务
     */
    private String executeObservationTask(String description) {
        return "观察完成：" + description;
    }
    
    /**
     * 为查询选择Flow
     */
    private String selectFlowForQuery(String query) {
        // 计算每个Flow的匹配分数
        Map<String, Integer> flowScores = new HashMap<>();
        
        for (Map.Entry<String, String> mapping : flowMappings.entrySet()) {
            String keyword = mapping.getKey();
            String flowId = mapping.getValue();
            
            if (query.contains(keyword)) {
                flowScores.merge(flowId, 1, Integer::sum);
            }
        }
        
        // 返回分数最高的Flow
        return flowScores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }
    
    /**
     * 执行Flow
     */
    private Message executeFlow(String flowId, String query) {
        FlowDefinition flow = registeredFlows.get(flowId);
        
        StringBuilder result = new StringBuilder();
        result.append("【Flow编排模式】\n");
        result.append("执行Flow：").append(flow.getName()).append("\n");
        result.append("描述：").append(flow.getDescription()).append("\n\n");
        
        try {
            // 获取Flow节点信息
            Map<String, Object> nodes = flow.getNodes();
            String nodeType = (String) nodes.get("type");
            String nodeName = (String) nodes.get("name");
            
            if ("tool".equals(nodeType)) {
                // 执行工具节点
                Map<String, Object> args = prepareToolArgumentsForQuery(nodeName, query);
                ToolCall toolResult = callTool(nodeName, args);
                
                if (toolResult.isSuccess()) {
                    result.append("Flow执行成功：").append(toolResult.getResult());
                } else {
                    result.append("Flow执行失败：").append(toolResult.getError());
                }
            } else {
                result.append("不支持的节点类型：").append(nodeType);
            }
            
        } catch (Exception e) {
            result.append("Flow执行异常：").append(e.getMessage());
        }
        
        return new Message("assistant", result.toString());
    }
    
    /**
     * 为查询准备工具参数
     */
    private Map<String, Object> prepareToolArgumentsForQuery(String toolName, String query) {
        Map<String, Object> args = new HashMap<>();
        
        switch (toolName) {
            case "calculator":
                String expression = extractMathExpressionFromQuery(query);
                if (expression != null) {
                    args.put("expression", expression);
                }
                break;
                
            case "text_analyzer":
                String textToAnalyze = extractTextForAnalysisFromQuery(query);
                args.put("text", textToAnalyze);
                break;
                
            case "get_time":
                // 时间工具不需要参数
                break;
                
            default:
                args.put("query", query);
                break;
        }
        
        return args;
    }
    
    /**
     * 从查询中提取数学表达式
     */
    private String extractMathExpressionFromQuery(String query) {
        Pattern mathPattern = Pattern.compile("(\\d+(?:\\.\\d+)?\\s*[+\\-*/]\\s*\\d+(?:\\.\\d+)?(?:\\s*[+\\-*/]\\s*\\d+(?:\\.\\d+)?)*)");
        Matcher matcher = mathPattern.matcher(query);
        
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        
        return null;
    }
    
    /**
     * 从查询中提取需要分析的文本
     */
    private String extractTextForAnalysisFromQuery(String query) {
        Pattern quotePattern = Pattern.compile("[\"'](.*?)[\"']");
        Matcher matcher = quotePattern.matcher(query);
        
        if (matcher.find()) {
            return matcher.group(1);
        }
        
        return query;
    }
    
    /**
     * 注册Flow
     */
    public void registerFlow(String flowId, FlowDefinition flow) {
        registeredFlows.put(flowId, flow);
    }
    
    /**
     * 移除Flow
     */
    public boolean removeFlow(String flowId) {
        return registeredFlows.remove(flowId) != null;
    }
    
    /**
     * 获取系统状态
     */
    public Map<String, Object> getSystemStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("name", getName());
        status.put("execution_mode", executionMode.getDescription());
        status.put("planning_enabled", planningEnabled);
        status.put("current_state", getState().getDescription());
        status.put("total_messages", totalProcessedMessages);
        status.put("total_plans", totalExecutedPlans);
        status.put("total_flows", totalFlowExecutions);
        status.put("registered_flows", registeredFlows.size());
        status.put("registered_tools", getToolRegistry().getToolCount());
        status.put("uptime_hours", java.time.Duration.between(systemStartTime, LocalDateTime.now()).toHours());
        
        if (currentPlan != null) {
            status.put("current_plan", currentPlan.getStatistics());
        }
        
        return status;
    }
    
    /**
     * 设置执行模式
     */
    public void setExecutionMode(ExecutionMode mode) {
        this.executionMode = mode;
    }
    
    /**
     * 获取执行模式
     */
    public ExecutionMode getExecutionMode() {
        return executionMode;
    }
    
    /**
     * 启用/禁用计划模式
     */
    public void setPlanningEnabled(boolean enabled) {
        this.planningEnabled = enabled;
    }
    
    /**
     * 判断是否启用计划模式
     */
    public boolean isPlanningEnabled() {
        return planningEnabled;
    }
    
    /**
     * 获取当前计划
     */
    public Plan getCurrentPlan() {
        return currentPlan;
    }
    
    /**
     * 获取注册的Flow列表
     */
    public Map<String, FlowDefinition> getRegisteredFlows() {
        return new HashMap<>(registeredFlows);
    }
    
    /**
     * 获取最后一条用户消息
     */
    private String getLastUserMessage() {
        for (int i = getMessages().size() - 1; i >= 0; i--) {
            Message msg = getMessages().get(i);
            if ("user".equals(msg.getRole())) {
                return msg.getContent();
            }
        }
        return "";
    }
}