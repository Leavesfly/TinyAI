package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.ToolCall;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * ReAct (Reasoning and Acting) Agent
 * OpenManus分层架构的第二层
 * 实现基础的推理-行动循环
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
    }
    
    public ReActAgent(String name, boolean verboseMode) {
        super(name);
        this.verboseMode = verboseMode;
        this.maxIterations = MAX_ITERATIONS;
    }
    
    /**
     * 处理消息 - 实现ReAct循环
     */
    @Override
    public Message processMessage(Message message) {
        addMessage(message);
        updateState(AgentState.THINKING);
        
        String query = message.getContent();
        StringBuilder response = new StringBuilder();
        
        try {
            // 开始ReAct循环
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                if (verboseMode) {
                    response.append(String.format("=== 第 %d 轮推理 ===\n", iteration + 1));
                }
                
                // 1. 思考阶段
                updateState(AgentState.THINKING);
                String thought = think(query, response.toString());
                
                if (verboseMode) {
                    response.append("思考：").append(thought).append("\n");
                }
                
                // 检查是否需要行动
                if (shouldAct(thought)) {
                    // 2. 行动阶段
                    updateState(AgentState.ACTING);
                    String action = planAction(thought);
                    
                    if (verboseMode) {
                        response.append("行动：").append(action).append("\n");
                    }
                    
                    // 执行行动
                    String observation = executeAction(action);
                    
                    // 3. 观察阶段
                    updateState(AgentState.OBSERVING);
                    if (verboseMode) {
                        response.append("观察：").append(observation).append("\n");
                    }
                    
                    // 检查是否达到目标
                    if (isGoalReached(observation, query)) {
                        updateState(AgentState.DONE);
                        String conclusion = formulate_conclusion(observation, query);
                        response.append("结论：").append(conclusion);
                        break;
                    }
                    
                    // 更新查询上下文
                    query = updateContext(query, thought, action, observation);
                } else {
                    // 直接给出结论
                    updateState(AgentState.DONE);
                    String conclusion = thought.contains("结论") ? thought : 
                        formulate_conclusion(thought, message.getContent());
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
     * 思考阶段 - 分析问题并规划下一步
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
     * 判断是否需要行动
     */
    private boolean shouldAct(String thought) {
        return thought.contains("需要使用") || thought.contains("应该使用") || 
               thought.contains("工具") || thought.contains("计算") ||
               thought.contains("获取") || thought.contains("分析");
    }
    
    /**
     * 规划行动
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
     * 执行行动
     */
    private String executeAction(String action) {
        try {
            if (action.contains("计算器")) {
                return executeCalculatorAction();
            } else if (action.contains("时间")) {
                return executeTimeAction();
            } else if (action.contains("文本分析")) {
                return executeTextAnalysisAction();
            } else {
                return "无法执行指定的行动：" + action;
            }
        } catch (Exception e) {
            return "行动执行失败：" + e.getMessage();
        }
    }
    
    /**
     * 执行计算器行动
     */
    private String executeCalculatorAction() {
        // 从最后一条用户消息中提取数学表达式
        String lastUserMessage = getLastUserMessage();
        String expression = extractMathExpression(lastUserMessage);
        
        if (expression != null) {
            Map<String, Object> args = new HashMap<>();
            args.put("expression", expression);
            ToolCall result = callTool("calculator", args);
            
            if (result.isSuccess()) {
                return "计算结果：" + result.getResult();
            } else {
                return "计算失败：" + result.getError();
            }
        } else {
            return "无法从输入中提取数学表达式";
        }
    }
    
    /**
     * 执行时间查询行动
     */
    private String executeTimeAction() {
        ToolCall result = callTool("get_time", new HashMap<>());
        
        if (result.isSuccess()) {
            return "当前时间：" + result.getResult();
        } else {
            return "获取时间失败：" + result.getError();
        }
    }
    
    /**
     * 执行文本分析行动
     */
    private String executeTextAnalysisAction() {
        String lastUserMessage = getLastUserMessage();
        String textToAnalyze = extractTextForAnalysis(lastUserMessage);
        
        if (textToAnalyze != null) {
            Map<String, Object> args = new HashMap<>();
            args.put("text", textToAnalyze);
            ToolCall result = callTool("text_analyzer", args);
            
            if (result.isSuccess()) {
                return "分析结果：" + result.getResult();
            } else {
                return "分析失败：" + result.getError();
            }
        } else {
            return "无法从输入中提取需要分析的文本";
        }
    }
    
    /**
     * 判断是否达到目标
     */
    private boolean isGoalReached(String observation, String query) {
        return observation.contains("结果") || observation.contains("成功") ||
               observation.contains("时间") || observation.contains("分析结果");
    }
    
    /**
     * 形成结论
     */
    private String formulate_conclusion(String observation, String query) {
        if (observation.contains("计算结果")) {
            return observation;
        } else if (observation.contains("当前时间")) {
            return observation;
        } else if (observation.contains("分析结果")) {
            return observation;
        } else {
            return "基于分析，" + observation;
        }
    }
    
    /**
     * 更新上下文
     */
    private String updateContext(String originalQuery, String thought, String action, String observation) {
        return originalQuery + " [上一轮：" + thought + " -> " + action + " -> " + observation + "]";
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