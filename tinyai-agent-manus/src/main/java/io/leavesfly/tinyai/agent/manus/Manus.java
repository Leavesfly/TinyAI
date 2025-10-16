package io.leavesfly.tinyai.agent.manus;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.leavesfly.tinyai.agent.context.Message;
import io.leavesfly.tinyai.agent.context.ToolCall;

/**
 * Manus核心系统 - LLM模拟版本
 * OpenManus分层架构的第四层（核心层）
 * 基于LLM模拟实现智能的双执行机制、计划驱动和Flow编排
 * 
 * @author 山泽
 */
public class Manus extends ToolCallAgent {
    
    // 执行模式和配置
    private ExecutionMode executionMode;            // 当前执行模式
    private boolean planningEnabled;                // 是否启用计划模式
    private Plan currentPlan;                       // 当前执行的计划
    private boolean enableLLMDecisionMaking;        // 是否启用LLM决策
    
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
        super(name, true, true); // autoToolSelection=true, enableLLMToolSelection=true
        this.executionMode = ExecutionMode.DIRECT_AGENT;
        this.planningEnabled = false;
        this.enableLLMDecisionMaking = true;
        this.registeredFlows = new ConcurrentHashMap<>();
        this.flowMappings = new HashMap<>();
        this.systemMetrics = new ConcurrentHashMap<>();
        this.systemStartTime = LocalDateTime.now();
        this.totalProcessedMessages = 0;
        this.totalExecutedPlans = 0;
        this.totalFlowExecutions = 0;
        
        // 设置Manus专用的系统提示
        this.systemPrompt = generateManusSystemPrompt();
        
        // 初始化Flow映射
        initializeFlowMappings();
        
        // 注册示例Flow
        registerDefaultFlows();
    }
    
    /**
     * 构造函数 - 支持自定义配置
     */
    public Manus(String name, boolean planningEnabled, boolean enableLLMDecisionMaking) {
        super(name, true, true);
        this.executionMode = ExecutionMode.DIRECT_AGENT;
        this.planningEnabled = planningEnabled;
        this.enableLLMDecisionMaking = enableLLMDecisionMaking;
        this.registeredFlows = new ConcurrentHashMap<>();
        this.flowMappings = new HashMap<>();
        this.systemMetrics = new ConcurrentHashMap<>();
        this.systemStartTime = LocalDateTime.now();
        this.totalProcessedMessages = 0;
        this.totalExecutedPlans = 0;
        this.totalFlowExecutions = 0;
        
        this.systemPrompt = generateManusSystemPrompt();
        initializeFlowMappings();
        registerDefaultFlows();
    }
    
    /**
     * 生成Manus专用的系统提示
     */
    private String generateManusSystemPrompt() {
        return String.format(
            "你是%s，Manus智能编排系统的核心。" +
            "你拥有双执行机制：直接Agent模式和Flow编排模式。" +
            "你可以根据任务复杂度智能选择最适合的执行方式。" +
            "对于简单任务，直接调用工具；对于复杂任务，使用计划分解和Flow编排。" +
            "请始终保持高效、智能、有序的工作方式。",
            name
        );
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
            
            // 使用LLM智能选择执行模式
            if (enableLLMDecisionMaking) {
                ExecutionMode selectedMode = selectExecutionModeWithLLM(message.getContent());
                if (selectedMode != null) {
                    executionMode = selectedMode;
                }
            }
            
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
     * 使用LLM选择执行模式
     */
    private ExecutionMode selectExecutionModeWithLLM(String query) {
        if (!llmEnabled) {
            return null; // 使用默认模式
        }
        
        try {
            String context = String.format(
                "查询内容：%s\n" +
                "执行模式选项：\n" +
                "1. DIRECT_AGENT - 适合简单、直接的工具调用任务\n" +
                "2. FLOW_ORCHESTRATION - 适合复杂、需要多步骤编排的任务\n" +
                "请分析查询并选择最合适的执行模式。",
                query
            );
            
            String llmResponse = generateLLMResponse("请选择最适合的执行模式", context);
            
            if (llmResponse.contains("FLOW_ORCHESTRATION") || llmResponse.contains("编排") || llmResponse.contains("复杂")) {
                return ExecutionMode.FLOW_ORCHESTRATION;
            } else if (llmResponse.contains("DIRECT_AGENT") || llmResponse.contains("直接") || llmResponse.contains("简单")) {
                return ExecutionMode.DIRECT_AGENT;
            }
            
        } catch (Exception e) {
            // LLM选择失败时保持原有模式
        }
        
        return null;
    }
    
    /**
     * 直接Agent模式处理
     */
    private Message processDirectAgentMode(Message message) {
        if (planningEnabled) {
            return processWithLLMPlanning(message);
        } else {
            // 使用父类的ToolCallAgent能力
            return super.processMessage(message);
        }
    }
    
    /**
     * 基于LLM的计划驱动处理
     */
    private Message processWithLLMPlanning(Message message) {
        updateState(AgentState.PLANNING);
        
        String query = message.getContent();
        
        try {
            // 使用LLM创建计划
            Plan plan = createPlanWithLLM(query);
            currentPlan = plan;
            totalExecutedPlans++;
            
            StringBuilder result = new StringBuilder();
            result.append("【LLM计划驱动模式】\n");
            result.append("计划：").append(plan.getTitle()).append("\n");
            result.append("目标：").append(plan.getGoal()).append("\n\n");
            
            // 使用LLM指导执行任务
            String planResult = executePlanWithLLM(plan, query);
            result.append(planResult);
            
            // 添加执行统计
            Map<String, Object> stats = plan.getStatistics();
            result.append("\n\n计划执行统计：").append(stats);
            
            return new Message("assistant", result.toString());
            
        } catch (Exception e) {
            return new Message("assistant", "LLM计划执行失败：" + e.getMessage());
        }
    }
    
    /**
     * 使用LLM创建计划
     */
    private Plan createPlanWithLLM(String query) {
        if (!llmEnabled) {
            return createPlanForQuery(query); // 回退到传统方法
        }
        
        try {
            String context = "需要为以下查询创建执行计划：" + query;
            String planningResponse = generateLLMResponse("请分解这个任务为具体的执行步骤", context);
            
            // 解析LLM生成的计划
            Plan plan = new Plan("LLM生成计划", "基于LLM分析的任务执行计划");
            
            // 简单的计划分解逻辑
            if (planningResponse.contains("工具") || planningResponse.contains("计算") || planningResponse.contains("查询")) {
                plan.addTask("分析需求", "thinking");
                plan.addTask("选择并执行工具", "action");
                plan.addTask("整合结果", "thinking");
            } else {
                plan.addTask("理解任务", "thinking");
                plan.addTask("执行操作", "action");
            }
            
            return plan;
            
        } catch (Exception e) {
            return createPlanForQuery(query);
        }
    }
    
    /**
     * 使用LLM执行计划
     */
    private String executePlanWithLLM(Plan plan, String originalQuery) {
        StringBuilder result = new StringBuilder();
        plan.start();
        
        while (!plan.isCompleted() && !plan.isFailed()) {
            Task nextTask = plan.getNextTask();
            if (nextTask == null) break;
            
            result.append("执行任务：").append(nextTask.getDescription()).append("\n");
            
            try {
                Object taskResult = executeTaskWithLLM(nextTask, originalQuery);
                plan.completeCurrentTask(taskResult);
                result.append("任务结果：").append(taskResult).append("\n\n");
            } catch (Exception e) {
                plan.failCurrentTask("任务执行失败：" + e.getMessage());
                result.append("任务失败：").append(e.getMessage()).append("\n\n");
                break;
            }
        }
        
        return result.toString();
    }
    
    /**
     * 使用LLM执行任务
     */
    private Object executeTaskWithLLM(Task task, String originalQuery) {
        String taskType = task.getType();
        String description = task.getDescription();
        
        if (!llmEnabled) {
            return executeTask(task); // 回退到传统方法
        }
        
        try {
            if ("thinking".equals(taskType)) {
                String context = String.format("任务：%s\n原始查询：%s", description, originalQuery);
                return generateLLMResponse("请进行思考分析", context);
            } else if ("action".equals(taskType)) {
                // 对于行动任务，执行实际的工具调用
                if (description.contains("工具") || description.contains("执行")) {
                    // 使用工具调用能力
                    return executeToolAction(originalQuery);
                } else {
                    String context = String.format("任务：%s\n原始查询：%s", description, originalQuery);
                    return generateLLMResponse("请执行相应的行动", context);
                }
            } else {
                return "任务类型：" + taskType + "，描述：" + description;
            }
        } catch (Exception e) {
            return "LLM任务执行异常：" + e.getMessage();
        }
    }
    
    /**
     * 执行工具行动
     */
    private String executeToolAction(String query) {
        // 获取推荐工具
        Map<String, Integer> toolScores = new HashMap<>();
        
        // 基于关键词推荐工具
        if (query.contains("计算") || query.matches(".*\\d+.*[+\\-*/].*\\d+.*")) {
            toolScores.put("calculator", 3);
        }
        if (query.contains("时间") || query.contains("几点")) {
            toolScores.put("get_time", 3);
        }
        if (query.contains("分析") || query.contains("\"") || query.contains("'")) {
            toolScores.put("text_analyzer", 3);
        }
        
        // 选择最佳工具
        String bestTool = toolScores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
        
        if (bestTool != null) {
            Map<String, Object> args = prepareToolArgumentsForQuery(bestTool, query);
            ToolCall result = callTool(bestTool, args);
            
            if (result.isSuccess()) {
                return "工具执行成功：" + result.getResult();
            } else {
                return "工具执行失败：" + result.getError();
            }
        } else {
            return "未找到合适的工具";
        }
    }
    
    /**
     * Flow编排模式处理
     */
    private Message processFlowOrchestrationMode(Message message) {
        updateState(AgentState.ACTING);
        
        String query = message.getContent();
        
        // 使用LLM选择Flow
        String flowId = enableLLMDecisionMaking ? 
            selectFlowWithLLM(query) : 
            selectFlowForQuery(query);
        
        if (flowId != null && registeredFlows.containsKey(flowId)) {
            totalFlowExecutions++;
            return executeFlowWithLLM(flowId, query);
        } else {
            // 回退到直接Agent模式
            return processDirectAgentMode(message);
        }
    }
    
    /**
     * 使用LLM选择Flow
     */
    private String selectFlowWithLLM(String query) {
        if (!llmEnabled) {
            return selectFlowForQuery(query);
        }
        
        try {
            StringBuilder context = new StringBuilder();
            context.append("查询：").append(query).append("\n");
            context.append("可用Flow：\n");
            for (Map.Entry<String, FlowDefinition> entry : registeredFlows.entrySet()) {
                context.append("- ").append(entry.getKey()).append(": ").append(entry.getValue().getDescription()).append("\n");
            }
            
            String flowDecision = llmSimulator.generateFlowDecision(query, 
                registeredFlows.entrySet().stream()
                    .collect(HashMap::new, 
                        (map, entry) -> map.put(entry.getKey(), entry.getValue().getDescription()),
                        HashMap::putAll));
            
            // 从决策中提取Flow ID
            for (String flowId : registeredFlows.keySet()) {
                if (flowDecision.contains(flowId)) {
                    return flowId;
                }
            }
            
        } catch (Exception e) {
            // LLM选择失败时回退
        }
        
        return selectFlowForQuery(query);
    }
    
    /**
     * 使用LLM执行Flow
     */
    private Message executeFlowWithLLM(String flowId, String query) {
        FlowDefinition flow = registeredFlows.get(flowId);
        
        StringBuilder result = new StringBuilder();
        result.append("【LLM Flow编排模式】\n");
        result.append("执行Flow：").append(flow.getName()).append("\n");
        result.append("描述：").append(flow.getDescription()).append("\n\n");
        
        try {
            if (llmEnabled) {
                // 使用LLM指导Flow执行
                String flowGuidance = generateLLMResponse(
                    "请指导Flow的执行过程", 
                    String.format("Flow：%s\n查询：%s", flow.getName(), query)
                );
                result.append("LLM指导：").append(flowGuidance).append("\n\n");
            }
            
            // 执行Flow中的工具
            Map<String, Object> nodes = flow.getNodes();
            String nodeType = (String) nodes.get("type");
            String nodeName = (String) nodes.get("name");
            
            if ("tool".equals(nodeType)) {
                Map<String, Object> args = prepareToolArgumentsForQuery(nodeName, query);
                ToolCall toolResult = callTool(nodeName, args);
                
                if (toolResult.isSuccess()) {
                    String formattedResult = llmEnabled ? 
                        formatFlowResultWithLLM(flow.getName(), toolResult.getResult(), query) :
                        "Flow执行成功：" + toolResult.getResult();
                    result.append(formattedResult);
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
     * 使用LLM格式化Flow结果
     */
    private String formatFlowResultWithLLM(String flowName, Object result, String originalQuery) {
        try {
            String context = String.format(
                "Flow名称：%s\n执行结果：%s\n原始查询：%s", 
                flowName, result.toString(), originalQuery
            );
            return generateLLMResponse("请将Flow执行结果格式化为用户友好的回复", context);
        } catch (Exception e) {
            return "Flow执行成功：" + result;
        }
    }
    
    // 传统方法作为回退方案
    
    /**
     * 传统的计划创建方法（回退方案）
     */
    private Plan createPlanForQuery(String query) {
        Plan plan = new Plan("处理查询：" + query.substring(0, Math.min(query.length(), 20)) + "...", 
                           "分析并回答用户的查询");
        
        if (query.contains("详细") || query.contains("深入") || query.contains("完整")) {
            plan.addTask("分析查询意图", "thinking");
            plan.addTask("收集相关信息", "action");
            plan.addTask("整合分析结果", "thinking");
            plan.addTask("生成详细回答", "action");
        } else {
            plan.addTask("理解查询", "thinking");
            plan.addTask("执行相应操作", "action");
            plan.addTask("生成回答", "thinking");
        }
        
        return plan;
    }
    
    /**
     * 传统的任务执行方法（回退方案）
     */
    private Object executeTask(Task task) {
        String taskType = task.getType();
        String description = task.getDescription();
        
        switch (taskType) {
            case "thinking":
                return "思考完成：" + description;
            case "action":
                return "行动完成：" + description;
            case "observation":
                return "观察完成：" + description;
            default:
                return "任务类型未知：" + taskType;
        }
    }
    
    /**
     * 传统的Flow选择方法（回退方案）
     */
    private String selectFlowForQuery(String query) {
        Map<String, Integer> flowScores = new HashMap<>();
        
        for (Map.Entry<String, String> mapping : flowMappings.entrySet()) {
            String keyword = mapping.getKey();
            String flowId = mapping.getValue();
            
            if (query.contains(keyword)) {
                flowScores.merge(flowId, 1, Integer::sum);
            }
        }
        
        return flowScores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
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
     * 获取最后一条用户消息
     */
    private String getLastUserMessage() {
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message msg = messages.get(i);
            if ("user".equals(msg.getRole())) {
                return msg.getContent();
            }
        }
        return "";
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
        status.put("llm_decision_making", enableLLMDecisionMaking);
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
    
    // Getter 和 Setter 方法
    public ExecutionMode getExecutionMode() {
        return executionMode;
    }
    
    public void setExecutionMode(ExecutionMode mode) {
        this.executionMode = mode;
    }
    
    public boolean isPlanningEnabled() {
        return planningEnabled;
    }
    
    public void setPlanningEnabled(boolean enabled) {
        this.planningEnabled = enabled;
    }
    
    public boolean isEnableLLMDecisionMaking() {
        return enableLLMDecisionMaking;
    }
    
    public void setEnableLLMDecisionMaking(boolean enableLLMDecisionMaking) {
        this.enableLLMDecisionMaking = enableLLMDecisionMaking;
    }
    
    public Plan getCurrentPlan() {
        return currentPlan;
    }
    
    public Map<String, FlowDefinition> getRegisteredFlows() {
        return new HashMap<>(registeredFlows);
    }
    
    public Map<String, String> getFlowMappings() {
        return new HashMap<>(flowMappings);
    }
    
    public Map<String, Object> getSystemMetrics() {
        return new HashMap<>(systemMetrics);
    }
}