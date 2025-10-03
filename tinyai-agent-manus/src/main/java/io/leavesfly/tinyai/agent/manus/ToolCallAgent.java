package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.ToolCall;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 工具调用Agent
 * OpenManus分层架构的第三层
 * 专门处理工具调用和结果解析
 * 
 * @author 山泽
 */
public class ToolCallAgent extends ReActAgent {
    
    private Map<String, String> toolMappings;           // 工具映射
    private boolean autoToolSelection;                  // 自动工具选择
    private List<String> preferredTools;                // 优先使用的工具
    private Map<String, Integer> toolUsageCount;       // 工具使用计数
    
    /**
     * 构造函数
     */
    public ToolCallAgent(String name) {
        super(name);
        this.toolMappings = new HashMap<>();
        this.autoToolSelection = true;
        this.preferredTools = new ArrayList<>();
        this.toolUsageCount = new HashMap<>();
        
        // 初始化工具映射
        initializeToolMappings();
    }
    
    public ToolCallAgent(String name, boolean autoToolSelection) {
        super(name);
        this.toolMappings = new HashMap<>();
        this.autoToolSelection = autoToolSelection;
        this.preferredTools = new ArrayList<>();
        this.toolUsageCount = new HashMap<>();
        
        initializeToolMappings();
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
            // 1. 智能工具选择
            List<String> recommendedTools = recommendTools(query);
            
            // 2. 如果有明确的工具需求，直接调用
            if (!recommendedTools.isEmpty() && autoToolSelection) {
                updateState(AgentState.ACTING);
                String result = executeToolChain(recommendedTools, query);
                
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
     * 推荐工具
     */
    private List<String> recommendTools(String query) {
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
     * 执行工具链
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
     * 执行特定工具
     */
    private ToolCall executeSpecificTool(String toolName, String query) {
        Map<String, Object> arguments = prepareToolArguments(toolName, query);
        return callTool(toolName, arguments);
    }
    
    /**
     * 准备工具参数
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
                // 对于未知工具，尝试传递查询内容
                arguments.put("query", query);
                break;
        }
        
        return arguments;
    }
    
    /**
     * 格式化工具结果
     */
    private String formatToolResult(String toolName, Object result) {
        switch (toolName) {
            case "calculator":
                return "计算结果：" + result;
                
            case "get_time":
                return "当前时间：" + result;
                
            case "text_analyzer":
                if (result instanceof Map) {
                    Map<?, ?> analysisResult = (Map<?, ?>) result;
                    StringBuilder formatted = new StringBuilder("文本分析结果：\n");
                    for (Map.Entry<?, ?> entry : analysisResult.entrySet()) {
                        formatted.append("- ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
                    }
                    return formatted.toString().trim();
                } else {
                    return "文本分析结果：" + result;
                }
                
            default:
                return toolName + " 执行结果：" + result;
        }
    }
    
    /**
     * 检测是否包含数学表达式
     */
    private boolean containsMathExpression(String text) {
        Pattern mathPattern = Pattern.compile("\\d+\\s*[+\\-*/]\\s*\\d+");
        return mathPattern.matcher(text).find();
    }
    
    /**
     * 检测是否包含时间查询
     */
    private boolean containsTimeQuery(String text) {
        return text.contains("时间") || text.contains("几点") || 
               text.contains("现在") || text.contains("当前时刻");
    }
    
    /**
     * 检测是否包含文本分析请求
     */
    private boolean containsTextAnalysisRequest(String text) {
        return text.contains("分析") && (text.contains("文本") || text.contains("内容") || 
               text.contains("\"") || text.contains("'"));
    }
    
    /**
     * 从文本中提取数学表达式
     */
    private String extractMathExpression(String text) {
        // 查找数学表达式模式
        Pattern patterns[] = {
            Pattern.compile("(\\d+(?:\\.\\d+)?\\s*[+\\-*/]\\s*\\d+(?:\\.\\d+)?(?:\\s*[+\\-*/]\\s*\\d+(?:\\.\\d+)?)*)"),
            Pattern.compile("([0-9+\\-*/\\s.()]+)")
        };
        
        for (Pattern pattern : patterns) {
            Matcher matcher = pattern.matcher(text);
            while (matcher.find()) {
                String expression = matcher.group(1).trim();
                if (expression.matches(".*[+\\-*/].*") && expression.matches(".*\\d.*")) {
                    return expression;
                }
            }
        }
        
        return null;
    }
    
    /**
     * 从文本中提取需要分析的内容
     */
    private String extractTextForAnalysis(String text) {
        // 优先查找引号中的内容
        Pattern quotePatterns[] = {
            Pattern.compile("[\"'](.*?)[\"']"),
            Pattern.compile("['](.*?)[']"),
            Pattern.compile("[\\u201c\\u201d](.*?)[\\u201c\\u201d]")
        };
        
        for (Pattern pattern : quotePatterns) {
            Matcher matcher = pattern.matcher(text);
            if (matcher.find()) {
                return matcher.group(1);
            }
        }
        
        // 查找"分析"关键词后的内容
        Pattern analysisPattern = Pattern.compile("分析[：:]?\\s*(.+?)(?:[。！？]|$)");
        Matcher analysisMatcher = analysisPattern.matcher(text);
        if (analysisMatcher.find()) {
            String content = analysisMatcher.group(1).trim();
            if (!content.isEmpty()) {
                return content;
            }
        }
        
        return text; // 默认分析整个文本
    }
    
    /**
     * 添加工具映射
     */
    public void addToolMapping(String keyword, String toolName) {
        toolMappings.put(keyword, toolName);
    }
    
    /**
     * 移除工具映射
     */
    public void removeToolMapping(String keyword) {
        toolMappings.remove(keyword);
    }
    
    /**
     * 设置优先工具
     */
    public void setPreferredTools(List<String> tools) {
        this.preferredTools = new ArrayList<>(tools);
    }
    
    /**
     * 添加优先工具
     */
    public void addPreferredTool(String toolName) {
        if (!preferredTools.contains(toolName)) {
            preferredTools.add(toolName);
        }
    }
    
    /**
     * 获取工具使用统计
     */
    public Map<String, Integer> getToolUsageStatistics() {
        return new HashMap<>(toolUsageCount);
    }
    
    /**
     * 重置工具使用统计
     */
    public void resetToolUsageStatistics() {
        toolUsageCount.clear();
    }
    
    /**
     * 获取推荐工具（调试用）
     */
    public List<String> getRecommendedTools(String query) {
        return recommendTools(query);
    }
    
    // Getter 和 Setter 方法
    public boolean isAutoToolSelection() {
        return autoToolSelection;
    }
    
    public void setAutoToolSelection(boolean autoToolSelection) {
        this.autoToolSelection = autoToolSelection;
    }
    
    public Map<String, String> getToolMappings() {
        return new HashMap<>(toolMappings);
    }
    
    public List<String> getPreferredTools() {
        return new ArrayList<>(preferredTools);
    }
}