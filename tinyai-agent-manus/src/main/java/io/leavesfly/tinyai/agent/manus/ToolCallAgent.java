package io.leavesfly.tinyai.agent.manus;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.leavesfly.tinyai.agent.context.Message;
import io.leavesfly.tinyai.agent.context.ToolCall;

/**
 * 工具调用Agent - LLM模拟版本
 * OpenManus分层架构的第三层
 * 基于LLM模拟实现智能的工具选择和调用
 * 
 * @author 山泽
 */
public class ToolCallAgent extends ReActAgent {
    
    private Map<String, String> toolMappings;           // 工具映射
    private boolean autoToolSelection;                  // 自动工具选择
    private List<String> preferredTools;                // 优先使用的工具
    private Map<String, Integer> toolUsageCount;       // 工具使用计数
    private boolean enableLLMToolSelection;             // 启用LLM工具选择
    
    /**
     * 构造函数
     */
    public ToolCallAgent(String name) {
        super(name);
        this.toolMappings = new HashMap<>();
        this.autoToolSelection = true;
        this.preferredTools = new ArrayList<>();
        this.toolUsageCount = new HashMap<>();
        this.enableLLMToolSelection = true;
        
        // 设置工具调用专用的系统提示
        this.systemPrompt = generateToolCallSystemPrompt();
        
        // 初始化工具映射
        initializeToolMappings();
    }
    
    public ToolCallAgent(String name, boolean autoToolSelection) {
        super(name);
        this.toolMappings = new HashMap<>();
        this.autoToolSelection = autoToolSelection;
        this.preferredTools = new ArrayList<>();
        this.toolUsageCount = new HashMap<>();
        this.enableLLMToolSelection = true;
        this.systemPrompt = generateToolCallSystemPrompt();
        
        initializeToolMappings();
    }
    
    public ToolCallAgent(String name, boolean autoToolSelection, boolean enableLLMToolSelection) {
        super(name, false, true); // verboseMode=false, llmEnabled=true
        this.toolMappings = new HashMap<>();
        this.autoToolSelection = autoToolSelection;
        this.preferredTools = new ArrayList<>();
        this.toolUsageCount = new HashMap<>();
        this.enableLLMToolSelection = enableLLMToolSelection;
        this.systemPrompt = generateToolCallSystemPrompt();
        
        initializeToolMappings();
    }
    
    /**
     * 生成工具调用专用的系统提示
     */
    private String generateToolCallSystemPrompt() {
        return String.format(
            "你是%s，一个专业的工具调用智能助手。" +
            "你擅长分析用户需求，智能选择最合适的工具来解决问题。" +
            "可用的工具包括：calculator（数学计算）、get_time（时间查询）、text_analyzer（文本分析）。" +
            "请根据用户的具体需求，选择最恰当的工具并正确使用。" +
            "始终以准确、高效的方式完成任务。",
            name
        );
    }
    
    /**
     * 初始化工具映射
     */
    private void initializeToolMappings() {
        // 数学相关关键词映射到计算器
        toolMappings.put("计算", "calculator");
        toolMappings.put("加", "calculator");
        toolMappings.put("减", "calculator");
        toolMappings.put("乘", "calculator");
        toolMappings.put("除", "calculator");
        toolMappings.put("数学", "calculator");
        toolMappings.put("运算", "calculator");
        
        // 时间相关关键词映射到时间工具
        toolMappings.put("时间", "get_time");
        toolMappings.put("几点", "get_time");
        toolMappings.put("现在", "get_time");
        toolMappings.put("当前", "get_time");
        
        // 分析相关关键词映射到文本分析器
        toolMappings.put("分析", "text_analyzer");
        toolMappings.put("解析", "text_analyzer");
        toolMappings.put("统计", "text_analyzer");
        toolMappings.put("检查", "text_analyzer");
    }
    
    /**
     * 处理消息 - 重写父类方法以增强工具调用能力
     */
    @Override
    public Message processMessage(Message message) {
        addMessage(message);
        updateState(AgentState.THINKING);
        
        String query = message.getContent();
        
        try {
            // 1. 使用LLM进行智能工具选择
            List<String> recommendedTools = enableLLMToolSelection ? 
                getRecommendedToolsFromLLM(query) : 
                getRecommendedTools(query);
            
            // 2. 如果有明确的工具需求，直接调用
            if (!recommendedTools.isEmpty() && autoToolSelection) {
                updateState(AgentState.ACTING);
                String result = executeToolChainWithLLM(recommendedTools, query);
                
                updateState(AgentState.DONE);
                Message responseMessage = new Message("assistant", result);
                addMessage(responseMessage);
                return responseMessage;
            } else {
                // 3. 回退到ReAct模式
                return super.processMessage(message);
            }
            
        } catch (Exception e) {
            updateState(AgentState.ERROR);
            String errorResponse = "工具调用过程中发生错误：" + e.getMessage();
            Message responseMessage = new Message("assistant", errorResponse);
            addMessage(responseMessage);
            return responseMessage;
        }
    }
    
    /**
     * 使用LLM获取推荐工具
     */
    private List<String> getRecommendedToolsFromLLM(String query) {
        if (!llmEnabled) {
            return getRecommendedTools(query); // 回退到传统方法
        }
        
        try {
            String context = buildToolSelectionContext(query);
            String toolSuggestion = generateLLMResponse(
                "请分析用户查询并推荐最合适的工具（从calculator, get_time, text_analyzer中选择）", 
                context
            );
            
            return parseToolSuggestionFromLLM(toolSuggestion);
        } catch (Exception e) {
            // 出错时回退到传统方法
            return getRecommendedTools(query);
        }
    }
    
    /**
     * 构建工具选择上下文
     */
    private String buildToolSelectionContext(String query) {
        StringBuilder context = new StringBuilder();
        context.append("用户查询：").append(query).append("\n");
        context.append("可用工具说明：\n");
        context.append("- calculator：用于数学计算，需要数学表达式\n");
        context.append("- get_time：用于获取当前时间，无需参数\n");
        context.append("- text_analyzer：用于文本分析，需要待分析的文本\n");
        context.append("工具使用统计：").append(toolUsageCount.toString()).append("\n");
        return context.toString();
    }
    
    /**
     * 从LLM响应中解析工具建议
     */
    private List<String> parseToolSuggestionFromLLM(String suggestion) {
        List<String> tools = new ArrayList<>();
        
        if (suggestion.contains("calculator") || suggestion.contains("计算")) {
            tools.add("calculator");
        }
        if (suggestion.contains("get_time") || suggestion.contains("时间")) {
            tools.add("get_time");
        }
        if (suggestion.contains("text_analyzer") || suggestion.contains("分析")) {
            tools.add("text_analyzer");
        }
        
        return tools;
    }
    
    /**
     * 传统的工具推荐方法（回退方案）
     */
    private List<String> getRecommendedTools(String query) {
        List<String> recommended = new ArrayList<>();
        Map<String, Integer> toolScores = new HashMap<>();
        
        // 计算每个工具的匹配分数
        for (Map.Entry<String, String> mapping : toolMappings.entrySet()) {
            String keyword = mapping.getKey();
            String tool = mapping.getValue();
            
            if (query.contains(keyword)) {
                toolScores.merge(tool, 1, Integer::sum);
            }
        }
        
        // 特殊模式检测
        if (containsMathExpression(query)) {
            toolScores.merge("calculator", 5, Integer::sum);
        }
        
        if (containsTimeQuery(query)) {
            toolScores.merge("get_time", 3, Integer::sum);
        }
        
        if (containsTextAnalysisRequest(query)) {
            toolScores.merge("text_analyzer", 3, Integer::sum);
        }
        
        // 按分数排序并返回推荐工具
        toolScores.entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .forEach(recommended::add);
        
        return recommended;
    }
    
    /**
     * 使用LLM执行工具链
     */
    private String executeToolChainWithLLM(List<String> tools, String query) {
        if (!llmEnabled) {
            return executeToolChain(tools, query); // 回退到传统方法
        }
        
        StringBuilder result = new StringBuilder();
        boolean hasSuccessfulCall = false;
        
        for (String toolName : tools) {
            try {
                // 使用LLM准备工具参数
                Map<String, Object> arguments = prepareToolArgumentsWithLLM(toolName, query);
                
                ToolCall toolCall = callTool(toolName, arguments);
                toolUsageCount.merge(toolName, 1, Integer::sum);
                
                if (toolCall.isSuccess()) {
                    // 使用LLM格式化结果
                    String formattedResult = formatToolResultWithLLM(toolName, toolCall.getResult(), query);
                    result.append(formattedResult);
                    hasSuccessfulCall = true;
                    break; // 成功后停止尝试其他工具
                } else {
                    result.append("工具 ").append(toolName).append(" 执行失败：").append(toolCall.getError()).append("\n");
                }
            } catch (Exception e) {
                result.append("工具 ").append(toolName).append(" 调用异常：").append(e.getMessage()).append("\n");
            }
        }
        
        if (!hasSuccessfulCall) {
            result.append("所有推荐的工具都无法成功处理该请求。");
        }
        
        return result.toString().trim();
    }
    
    /**
     * 使用LLM准备工具参数
     */
    private Map<String, Object> prepareToolArgumentsWithLLM(String toolName, String query) {
        if (!llmEnabled) {
            return prepareToolArguments(toolName, query); // 回退到传统方法
        }
        
        Map<String, Object> arguments = new HashMap<>();
        
        try {
            switch (toolName) {
                case "calculator":
                    String expression = extractMathExpressionWithLLM(query);
                    if (expression != null) {
                        arguments.put("expression", expression);
                    }
                    break;
                    
                case "text_analyzer":
                    String textToAnalyze = extractTextForAnalysisWithLLM(query);
                    arguments.put("text", textToAnalyze);
                    break;
                    
                case "get_time":
                    // 时间工具不需要参数
                    break;
                    
                default:
                    arguments.put("query", query);
                    break;
            }
        } catch (Exception e) {
            // 出错时回退到传统方法
            return prepareToolArguments(toolName, query);
        }
        
        return arguments;
    }
    
    /**
     * 使用LLM提取数学表达式
     */
    private String extractMathExpressionWithLLM(String query) {
        String context = "从以下查询中提取数学表达式：" + query;
        String llmResponse = generateLLMResponse("请提取其中的数学表达式", context);
        
        // 从LLM响应中提取表达式
        Pattern mathPattern = Pattern.compile("([0-9+\\-*/\\s.()]+)");
        Matcher matcher = mathPattern.matcher(llmResponse);
        
        while (matcher.find()) {
            String expression = matcher.group(1).trim();
            if (expression.matches(".*[+\\-*/].*") && expression.matches(".*\\d.*")) {
                return expression;
            }
        }
        
        // 回退到传统方法
        return extractMathExpression(query);
    }
    
    /**
     * 使用LLM提取分析文本
     */
    private String extractTextForAnalysisWithLLM(String query) {
        String context = "从以下查询中提取需要分析的文本：" + query;
        String llmResponse = generateLLMResponse("请提取需要分析的文本内容", context);
        
        // 简单提取逻辑
        if (llmResponse.contains("\"") || llmResponse.contains("'")) {
            Pattern quotePattern = Pattern.compile("[\"'](.*?)[\"']");
            Matcher matcher = quotePattern.matcher(llmResponse);
            if (matcher.find()) {
                return matcher.group(1);
            }
        }
        
        // 回退到传统方法
        return extractTextForAnalysis(query);
    }
    
    /**
     * 使用LLM格式化工具结果
     */
    private String formatToolResultWithLLM(String toolName, Object result, String originalQuery) {
        if (!llmEnabled) {
            return formatToolResult(toolName, result); // 回退到传统方法
        }
        
        try {
            String context = String.format(
                "工具：%s\n执行结果：%s\n原始查询：%s", 
                toolName, result.toString(), originalQuery
            );
            
            return generateLLMResponse("请将工具执行结果格式化为用户友好的回复", context);
        } catch (Exception e) {
            return formatToolResult(toolName, result);
        }
    }
    
    // 传统方法作为回退方案
    
    /**
     * 传统的工具链执行（回退方案）
     */
    private String executeToolChain(List<String> tools, String query) {
        StringBuilder result = new StringBuilder();
        boolean hasSuccessfulCall = false;
        
        for (String toolName : tools) {
            try {
                ToolCall toolCall = executeSpecificTool(toolName, query);
                toolUsageCount.merge(toolName, 1, Integer::sum);
                
                if (toolCall.isSuccess()) {
                    result.append(formatToolResult(toolName, toolCall.getResult()));
                    hasSuccessfulCall = true;
                    break;
                } else {
                    result.append("工具 ").append(toolName).append(" 执行失败：").append(toolCall.getError()).append("\n");
                }
            } catch (Exception e) {
                result.append("工具 ").append(toolName).append(" 调用异常：").append(e.getMessage()).append("\n");
            }
        }
        
        if (!hasSuccessfulCall) {
            result.append("所有推荐的工具都无法成功处理该请求。");
        }
        
        return result.toString().trim();
    }
    
    /**
     * 执行特定工具（回退方案）
     */
    private ToolCall executeSpecificTool(String toolName, String query) {
        Map<String, Object> arguments = prepareToolArguments(toolName, query);
        return callTool(toolName, arguments);
    }
    
    /**
     * 准备工具参数（回退方案）
     */
    private Map<String, Object> prepareToolArguments(String toolName, String query) {
        Map<String, Object> arguments = new HashMap<>();
        
        switch (toolName) {
            case "calculator":
                String expression = extractMathExpression(query);
                if (expression != null) {
                    arguments.put("expression", expression);
                }
                break;
                
            case "text_analyzer":
                String textToAnalyze = extractTextForAnalysis(query);
                arguments.put("text", textToAnalyze);
                break;
                
            case "get_time":
                // 时间工具不需要参数
                break;
                
            default:
                arguments.put("query", query);
                break;
        }
        
        return arguments;
    }
    
    /**
     * 格式化工具结果（回退方案）
     */
    private String formatToolResult(String toolName, Object result) {
        switch (toolName) {
            case "calculator":
                return "计算结果：" + result;
            case "get_time":
                return "当前时间：" + result;
            case "text_analyzer":
                return "分析结果：" + result;
            default:
                return toolName + " 执行结果：" + result;
        }
    }
    
    /**
     * 检查是否包含数学表达式
     */
    private boolean containsMathExpression(String query) {
        return query.matches(".*\\d+\\s*[+\\-*/]\\s*\\d+.*");
    }
    
    /**
     * 检查是否是时间查询
     */
    private boolean containsTimeQuery(String query) {
        return query.contains("时间") || query.contains("几点") || 
               query.contains("现在") || query.contains("当前");
    }
    
    /**
     * 检查是否是文本分析请求
     */
    private boolean containsTextAnalysisRequest(String query) {
        return query.contains("分析") || query.contains("统计") || 
               query.contains("检查") || query.contains("\"") || query.contains("'");
    }
    
    /**
     * 提取数学表达式（回退方案）
     */
    private String extractMathExpression(String text) {
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
     * 提取分析文本（回退方案）
     */
    private String extractTextForAnalysis(String text) {
        Pattern quotePattern = Pattern.compile("[\"'](.*?)[\"']");
        Matcher quoteMatcher = quotePattern.matcher(text);
        
        if (quoteMatcher.find()) {
            return quoteMatcher.group(1);
        }
        
        Pattern analysisPattern = Pattern.compile("分析[：:](.*?)$");
        Matcher analysisMatcher = analysisPattern.matcher(text);
        
        if (analysisMatcher.find()) {
            return analysisMatcher.group(1).trim();
        }
        
        return text;
    }
    
    // Getter 和 Setter 方法
    public boolean isAutoToolSelection() {
        return autoToolSelection;
    }
    
    public void setAutoToolSelection(boolean autoToolSelection) {
        this.autoToolSelection = autoToolSelection;
    }
    
    public boolean isEnableLLMToolSelection() {
        return enableLLMToolSelection;
    }
    
    public void setEnableLLMToolSelection(boolean enableLLMToolSelection) {
        this.enableLLMToolSelection = enableLLMToolSelection;
    }
    
    public List<String> getPreferredTools() {
        return new ArrayList<>(preferredTools);
    }
    
    public void setPreferredTools(List<String> preferredTools) {
        this.preferredTools = new ArrayList<>(preferredTools);
    }
    
    public Map<String, Integer> getToolUsageCount() {
        return new HashMap<>(toolUsageCount);
    }
    
    public Map<String, String> getToolMappings() {
        return new HashMap<>(toolMappings);
    }
    
    public void addToolMapping(String keyword, String tool) {
        this.toolMappings.put(keyword, tool);
    }
    
    public void removeToolMapping(String keyword) {
        this.toolMappings.remove(keyword);
    }
}