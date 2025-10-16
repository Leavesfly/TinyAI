package io.leavesfly.tinyai.agent.manus;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.leavesfly.tinyai.agent.context.Message;
import io.leavesfly.tinyai.agent.context.ToolCall;

/**
 * ReAct (Reasoning and Acting) Agent - LLM模拟版本
 * OpenManus分层架构的第二层
 * 基于LLM模拟实现智能的推理-行动循环
 * 
 * @author 山泽
 */
public class ReActAgent extends BaseAgent {
    
    private static final int MAX_ITERATIONS = 10;  // 最大推理轮数
    private static final Pattern THOUGHT_PATTERN = Pattern.compile("思考[：:](.+?)(?=行动|观察|结论|$)", Pattern.DOTALL);
    private static final Pattern ACTION_PATTERN = Pattern.compile("行动[：:](.+?)(?=思考|观察|结论|$)", Pattern.DOTALL);
    private static final Pattern OBSERVATION_PATTERN = Pattern.compile("观察[：:](.+?)(?=思考|行动|结论|$)", Pattern.DOTALL);
    
    private boolean verboseMode;                    // 详细模式
    private int maxIterations;                      // 最大迭代次数
    
    /**
     * 构造函数
     */
    public ReActAgent(String name) {
        super(name);
        this.verboseMode = false;
        this.maxIterations = MAX_ITERATIONS;
        // 设置ReAct专用的系统提示
        this.systemPrompt = generateReActSystemPrompt();
    }
    
    public ReActAgent(String name, boolean verboseMode) {
        super(name);
        this.verboseMode = verboseMode;
        this.maxIterations = MAX_ITERATIONS;
        this.systemPrompt = generateReActSystemPrompt();
    }
    
    public ReActAgent(String name, boolean verboseMode, boolean llmEnabled) {
        super(name, null, llmEnabled);
        this.verboseMode = verboseMode;
        this.maxIterations = MAX_ITERATIONS;
        this.systemPrompt = generateReActSystemPrompt();
    }
    
    /**
     * 生成ReAct专用的系统提示
     */
    private String generateReActSystemPrompt() {
        return String.format(
            "你是%s，一个基于ReAct（推理-行动）模式的智能助手。" +
            "你会按照'思考-行动-观察'的循环来解决问题。" +
            "在每轮推理中，你需要：" +
            "1. 思考：分析问题，制定策略 " +
            "2. 行动：选择并执行合适的工具 " +
            "3. 观察：分析执行结果，判断是否达到目标 " +
            "请保持逻辑清晰，推理有序。", 
            name
        );
    }
    
    /**
     * 处理消息 - 基于LLM模拟的ReAct循环
     */
    @Override
    public Message processMessage(Message message) {
        addMessage(message);
        updateState(AgentState.THINKING);
        
        String query = message.getContent();
        StringBuilder response = new StringBuilder();
        
        try {
            // 构建上下文
            String context = buildContext();
            
            // 开始LLM驱动的ReAct循环
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                if (verboseMode) {
                    response.append(String.format("=== 第 %d 轮推理 ===\n", iteration + 1));
                }
                
                // 1. 思考阶段 - 使用LLM生成
                updateState(AgentState.THINKING);
                String thought = generateThought(query, context, iteration);
                
                if (verboseMode) {
                    response.append("思考：").append(thought).append("\n");
                }
                
                // 检查是否需要行动
                if (shouldTakeAction(thought)) {
                    // 2. 行动阶段 - 使用LLM规划行动
                    updateState(AgentState.ACTING);
                    String actionPlan = generateActionPlan(thought, getAvailableToolsInfo());
                    
                    if (verboseMode) {
                        response.append("行动计划：").append(actionPlan).append("\n");
                    }
                    
                    // 执行行动
                    String actionResult = executeActionWithLLM(actionPlan, query);
                    
                    // 3. 观察阶段 - 使用LLM分析结果
                    updateState(AgentState.OBSERVING);
                    String observation = generateObservation(actionResult, query);
                    
                    if (verboseMode) {
                        response.append("行动结果：").append(actionResult).append("\n");
                        response.append("观察分析：").append(observation).append("\n");
                    }
                    
                    // 检查是否达到目标
                    if (isGoalAchieved(observation, query)) {
                        updateState(AgentState.DONE);
                        String conclusion = generateConclusion(observation, query);
                        response.append("结论：").append(conclusion);
                        break;
                    }
                    
                    // 更新上下文信息
                    context = updateContextWithIteration(context, thought, actionPlan, actionResult, observation);
                } else {
                    // 直接给出结论
                    updateState(AgentState.DONE);
                    String conclusion = thought.contains("结论") ? thought : 
                        generateDirectConclusion(thought, query);
                    response.append("结论：").append(conclusion);
                    break;
                }
                
                if (verboseMode) {
                    response.append("\n");
                }
            }
            
            // 如果达到最大迭代次数仍未完成
            if (getState() != AgentState.DONE) {
                updateState(AgentState.ERROR);
                response.append("推理超时，未能得出最终结论。");
            }
            
        } catch (Exception e) {
            updateState(AgentState.ERROR);
            response.append("推理过程中发生错误：").append(e.getMessage());
        }
        
        Message responseMessage = new Message("assistant", response.toString());
        addMessage(responseMessage);
        return responseMessage;
    }
    
    /**
     * 使用LLM生成思考内容
     */
    private String generateThought(String query, String context, int iteration) {
        if (!llmEnabled) {
            return think(query, context); // 回退到原有逻辑
        }
        
        try {
            return llmSimulator.generateThought(query, context, iteration);
        } catch (Exception e) {
            return "思考过程出现问题：" + e.getMessage();
        }
    }
    
    /**
     * 使用LLM生成行动计划
     */
    private String generateActionPlan(String thought, String availableTools) {
        if (!llmEnabled) {
            return planAction(thought); // 回退到原有逻辑
        }
        
        try {
            return llmSimulator.generateAction(thought, availableTools);
        } catch (Exception e) {
            return "生成行动计划出现问题：" + e.getMessage();
        }
    }
    
    /**
     * 使用LLM生成观察分析
     */
    private String generateObservation(String actionResult, String originalQuery) {
        if (!llmEnabled) {
            return actionResult; // 直接返回结果
        }
        
        try {
            return llmSimulator.generateObservation(actionResult, originalQuery);
        } catch (Exception e) {
            return "观察分析出现问题：" + e.getMessage();
        }
    }
    
    /**
     * 获取可用工具信息
     */
    private String getAvailableToolsInfo() {
        Map<String, Integer> toolStats = getToolStats();
        StringBuilder tools = new StringBuilder();
        
        if (toolStats.isEmpty()) {
            tools.append("calculator, get_time, text_analyzer"); // 默认工具
        } else {
            tools.append(String.join(", ", toolStats.keySet()));
        }
        
        return tools.toString();
    }
    
    /**
     * 使用LLM执行行动
     */
    private String executeActionWithLLM(String actionPlan, String query) {
        try {
            // 从行动计划中提取工具名称和参数
            String toolName = extractToolNameFromAction(actionPlan);
            Map<String, Object> args = extractToolArgumentsFromAction(actionPlan, query);
            
            if (toolName != null) {
                ToolCall result = callTool(toolName, args);
                
                if (result.isSuccess()) {
                    return "工具执行成功：" + result.getResult();
                } else {
                    return "工具执行失败：" + result.getError();
                }
            } else {
                return "无法从行动计划中提取工具信息";
            }
        } catch (Exception e) {
            return "行动执行异常：" + e.getMessage();
        }
    }
    
    /**
     * 从行动计划中提取工具名称
     */
    private String extractToolNameFromAction(String actionPlan) {
        if (actionPlan.contains("calculator") || actionPlan.contains("计算")) {
            return "calculator";
        } else if (actionPlan.contains("get_time") || actionPlan.contains("时间")) {
            return "get_time";
        } else if (actionPlan.contains("text_analyzer") || actionPlan.contains("分析")) {
            return "text_analyzer";
        }
        return null;
    }
    
    /**
     * 从行动计划中提取工具参数
     */
    private Map<String, Object> extractToolArgumentsFromAction(String actionPlan, String query) {
        Map<String, Object> args = new HashMap<>();
        
        if (actionPlan.contains("calculator") || actionPlan.contains("计算")) {
            String expression = extractMathExpression(query);
            if (expression != null) {
                args.put("expression", expression);
            }
        } else if (actionPlan.contains("text_analyzer") || actionPlan.contains("分析")) {
            String textToAnalyze = extractTextForAnalysis(query);
            args.put("text", textToAnalyze);
        }
        // get_time 不需要参数
        
        return args;
    }
    
    /**
     * 判断是否需要采取行动
     */
    private boolean shouldTakeAction(String thought) {
        return thought.contains("需要") || thought.contains("使用") || 
               thought.contains("工具") || thought.contains("计算") ||
               thought.contains("获取") || thought.contains("分析") ||
               thought.contains("执行") || thought.contains("调用");
    }
    
    /**
     * 判断是否达到目标
     */
    private boolean isGoalAchieved(String observation, String query) {
        return observation.contains("成功") || observation.contains("结果") ||
               observation.contains("完成") || observation.contains("获得");
    }
    
    /**
     * 生成结论
     */
    private String generateConclusion(String observation, String query) {
        if (!llmEnabled) {
            return observation;
        }
        
        String context = String.format("基于观察结果'%s'和原始查询'%s'", observation, query);
        return generateLLMResponse("请生成最终结论", context);
    }
    
    /**
     * 生成直接结论（不需要行动时）
     */
    private String generateDirectConclusion(String thought, String query) {
        if (!llmEnabled) {
            return thought;
        }
        
        String context = String.format("基于思考结果'%s'和查询'%s'", thought, query);
        return generateLLMResponse("请直接给出结论", context);
    }
    
    /**
     * 更新上下文信息
     */
    private String updateContextWithIteration(String context, String thought, String action, String actionResult, String observation) {
        StringBuilder newContext = new StringBuilder(context);
        newContext.append("\n上一轮的推理：");
        newContext.append("思考: ").append(thought).append("; ");
        newContext.append("行动: ").append(action).append("; ");
        newContext.append("结果: ").append(actionResult).append("; ");
        newContext.append("观察: ").append(observation).append("\n");
        return newContext.toString();
    }
    
    // 原有的逻辑方法，作为回退方案
    
    /**
     * 思考阶段 - 分析问题并规划下一步（回退方案）
     */
    private String think(String query, String context) {
        // 简化的思考逻辑
        if (context.isEmpty()) {
            // 首次思考
            if (query.contains("计算") || query.contains("+") || query.contains("-") || 
                query.contains("*") || query.contains("/")) {
                return "我需要使用计算器工具来解决这个数学问题。";
            } else if (query.contains("时间") || query.contains("几点")) {
                return "用户询问时间，我需要使用时间工具获取当前时间。";
            } else if (query.contains("分析")) {
                return "用户需要文本分析，我应该使用文本分析工具。";
            } else {
                return "我需要仔细分析这个问题，看看是否需要使用工具来解决。";
            }
        } else {
            // 后续思考
            if (context.contains("错误") || context.contains("失败")) {
                return "前一个行动失败了，我需要尝试不同的方法。";
            } else if (context.contains("成功") || context.contains("结果")) {
                return "行动成功，我可以基于结果给出答案。";
            } else {
                return "我需要继续分析问题并采取进一步行动。";
            }
        }
    }
    
    /**
     * 规划行动（回退方案）
     */
    private String planAction(String thought) {
        if (thought.contains("计算器")) {
            return "使用计算器工具";
        } else if (thought.contains("时间")) {
            return "使用时间工具";
        } else if (thought.contains("文本分析")) {
            return "使用文本分析工具";
        } else {
            return "使用合适的工具解决问题";
        }
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
     * 从文本中提取数学表达式
     */
    private String extractMathExpression(String text) {
        // 简单的数学表达式提取
        Pattern mathPattern = Pattern.compile("([0-9+\\-*/\\s.()]+)");
        Matcher matcher = mathPattern.matcher(text);
        
        while (matcher.find()) {
            String expression = matcher.group(1).trim();
            if (expression.matches(".*[+\\-*/].*") && expression.matches(".*\\d.*")) {
                return expression;
            }
        }
        
        return null;
    }
    
    /**
     * 从文本中提取需要分析的内容
     */
    private String extractTextForAnalysis(String text) {
        // 查找引号中的文本或"分析"后的内容
        Pattern quotePattern = Pattern.compile("[\"'](.*?)[\"']");
        Matcher quoteMatcher = quotePattern.matcher(text);
        
        if (quoteMatcher.find()) {
            return quoteMatcher.group(1);
        }
        
        // 查找"分析"后的内容
        Pattern analysisPattern = Pattern.compile("分析[：:](.*?)$");
        Matcher analysisMatcher = analysisPattern.matcher(text);
        
        if (analysisMatcher.find()) {
            return analysisMatcher.group(1).trim();
        }
        
        return text; // 默认分析整个文本
    }
    
    // Getter 和 Setter 方法
    public boolean isVerboseMode() {
        return verboseMode;
    }
    
    public void setVerboseMode(boolean verboseMode) {
        this.verboseMode = verboseMode;
    }
    
    public int getMaxIterations() {
        return maxIterations;
    }
    
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(1, maxIterations);
    }
}