package io.leavesfly.tinyai.agent.manus;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import io.leavesfly.tinyai.agent.context.Message;
import io.leavesfly.tinyai.agent.context.ToolCall;
import io.leavesfly.tinyai.agent.context.ToolRegistry;

/**
 * 基础Agent抽象类 - LLM模拟版本
 * OpenManus分层架构的第一层，集成LLM模拟能力
 * 
 * @author 山泽
 */
public abstract class BaseAgent {
    
    protected String name;                          // Agent名称
    protected AgentState state;                     // 当前状态
    protected List<Message> messages;               // 消息历史
    protected List<ToolCall> toolCallHistory;       // 工具调用历史
    protected ToolRegistry toolRegistry;            // 工具注册表
    protected LocalDateTime createdAt;              // 创建时间
    protected LocalDateTime lastActiveAt;           // 最后活跃时间
    protected Map<String, Object> configuration;    // 配置信息
    
    // LLM模拟相关组件
    protected LLMSimulator llmSimulator;             // LLM模拟器
    protected String systemPrompt;                   // 系统提示
    protected boolean llmEnabled;                    // 是否启用LLM模拟
    
    /**
     * 构造函数
     */
    public BaseAgent(String name) {
        this.name = name;
        this.state = AgentState.IDLE;
        this.messages = new ArrayList<>();
        this.toolCallHistory = new ArrayList<>();
        this.toolRegistry = new ToolRegistry();
        this.createdAt = LocalDateTime.now();
        this.lastActiveAt = LocalDateTime.now();
        this.configuration = new HashMap<>();
        
        // 初始化LLM模拟组件
        this.llmSimulator = new LLMSimulator();
        this.systemPrompt = generateDefaultSystemPrompt();
        this.llmEnabled = true;
        
        // 注册默认工具
        registerDefaultTools();
    }
    
    /**
     * 构造函数 - 支持自定义LLM配置
     */
    public BaseAgent(String name, String systemPrompt, boolean llmEnabled) {
        this(name);
        this.systemPrompt = systemPrompt != null ? systemPrompt : generateDefaultSystemPrompt();
        this.llmEnabled = llmEnabled;
    }
    
    /**
     * 处理消息的抽象方法
     * 子类必须实现此方法
     */
    public abstract Message processMessage(Message message);
    
    /**
     * 生成默认的系统提示
     */
    protected String generateDefaultSystemPrompt() {
        return String.format("你是%s，一个智能AI助手。你擅长分析问题、使用工具和提供准确的回答。请始终保持专业、有帮助的态度。", name);
    }
    
    /**
     * 使用LLM模拟生成响应
     */
    protected String generateLLMResponse(String prompt, String context) {
        if (!llmEnabled) {
            return "LLM模拟未启用，使用默认回复。";
        }
        
        try {
            return llmSimulator.generateResponse(prompt, "base_agent", context);
        } catch (Exception e) {
            return "生成回复时遇到问题：" + e.getMessage();
        }
    }
    
    /**
     * 异步使用LLM模拟生成响应
     */
    protected CompletableFuture<String> generateLLMResponseAsync(String prompt, String context) {
        if (!llmEnabled) {
            return CompletableFuture.completedFuture("LLM模拟未启用，使用默认回复。");
        }
        
        return llmSimulator.generateResponseAsync(prompt, "base_agent", context);
    }
    
    /**
     * 构建上下文信息
     */
    protected String buildContext() {
        StringBuilder context = new StringBuilder();
        
        // 添加系统提示
        if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
            context.append("系统提示：").append(systemPrompt).append("\n\n");
        }
        
        // 添加最近的对话历史
        int recentCount = Math.min(5, messages.size());
        if (recentCount > 0) {
            context.append("最近对话：\n");
            for (int i = messages.size() - recentCount; i < messages.size(); i++) {
                Message msg = messages.get(i);
                context.append(msg.getRole()).append(": ").append(msg.getContent()).append("\n");
            }
            context.append("\n");
        }
        
        // 添加可用工具信息
        if (toolRegistry.getToolCount() > 0) {
            context.append("可用工具：");
            Map<String, Integer> toolStats = getToolStats();
            context.append(String.join(", ", toolStats.keySet())).append("\n");
        }
        
        return context.toString();
    }
    
    /**
     * 获取Agent状态信息
     */
    public Map<String, Object> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("name", name);
        status.put("state", state.toString());
        status.put("message_count", messages.size());
        status.put("tool_call_count", toolCallHistory.size());
        status.put("created_at", createdAt.toString());
        status.put("last_active_at", lastActiveAt.toString());
        status.put("llm_enabled", llmEnabled);
        status.put("llm_model", llmSimulator.getModelName());
        status.put("system_prompt_length", systemPrompt.length());
        return status;
    }
    
    /**
     * 注册默认工具
     */
    protected void registerDefaultTools() {
        // 计算器工具
        toolRegistry.register("calculator", args -> {
            try {
                String expression = (String) args.get("expression");
                if (expression == null) {
                    return "错误：表达式不能为空";
                }
                // 简单的计算实现
                return evaluateExpression(expression);
            } catch (Exception e) {
                return "计算错误：" + e.getMessage();
            }
        }, "计算器工具，可以计算数学表达式");
        
        // 时间工具
        toolRegistry.register("get_time", args -> {
            return LocalDateTime.now().toString();
        }, "获取当前时间");
        
        // 文本分析工具
        toolRegistry.register("text_analyzer", args -> {
            String text = (String) args.get("text");
            if (text == null) {
                return "错误：文本不能为空";
            }
            Map<String, Object> result = new HashMap<>();
            result.put("length", text.length());
            result.put("word_count", text.split("\\s+").length);
            result.put("char_count", text.length());
            return result;
        }, "文本分析工具");
    }
    
    /**
     * 简单的表达式计算
     */
    private Object evaluateExpression(String expression) {
        try {
            // 简单的四则运算计算
            expression = expression.replaceAll("\\s", "");
            
            // 处理乘法和除法
            while (expression.contains("*") || expression.contains("/")) {
                for (int i = 0; i < expression.length(); i++) {
                    char c = expression.charAt(i);
                    if (c == '*' || c == '/') {
                        // 找到操作符前后的数字
                        int start = i - 1;
                        while (start >= 0 && (Character.isDigit(expression.charAt(start)) || expression.charAt(start) == '.')) {
                            start--;
                        }
                        start++;
                        
                        int end = i + 1;
                        while (end < expression.length() && (Character.isDigit(expression.charAt(end)) || expression.charAt(end) == '.')) {
                            end++;
                        }
                        
                        double left = Double.parseDouble(expression.substring(start, i));
                        double right = Double.parseDouble(expression.substring(i + 1, end));
                        double result = c == '*' ? left * right : left / right;
                        
                        expression = expression.substring(0, start) + result + expression.substring(end);
                        break;
                    }
                }
            }
            
            // 处理加法和减法
            double result = 0;
            boolean isPositive = true;
            StringBuilder numberBuilder = new StringBuilder();
            
            for (int i = 0; i < expression.length(); i++) {
                char c = expression.charAt(i);
                if (c == '+' || c == '-') {
                    if (numberBuilder.length() > 0) {
                        double number = Double.parseDouble(numberBuilder.toString());
                        result += isPositive ? number : -number;
                        numberBuilder = new StringBuilder();
                    }
                    isPositive = (c == '+');
                } else {
                    numberBuilder.append(c);
                }
            }
            
            if (numberBuilder.length() > 0) {
                double number = Double.parseDouble(numberBuilder.toString());
                result += isPositive ? number : -number;
            }
            
            return result;
        } catch (Exception e) {
            return "表达式格式错误";
        }
    }
    
    /**
     * 调用工具
     */
    protected ToolCall callTool(String toolName, Map<String, Object> arguments) {
        this.lastActiveAt = LocalDateTime.now();
        ToolCall toolCall = toolRegistry.callTool(toolName, arguments);
        toolCallHistory.add(toolCall);
        return toolCall;
    }
    
    /**
     * 添加消息到历史
     */
    protected void addMessage(Message message) {
        this.messages.add(message);
        this.lastActiveAt = LocalDateTime.now();
    }
    
    /**
     * 更新状态
     */
    protected void updateState(AgentState newState) {
        this.state = newState;
        this.lastActiveAt = LocalDateTime.now();
    }
    
    /**
     * 注册自定义工具
     */
    public void registerCustomTool(String name, java.util.function.Function<Map<String, Object>, Object> function, String description) {
        toolRegistry.register(name, function, description);
    }
    
    /**
     * 获取工具统计信息
     */
    public Map<String, Integer> getToolStats() {
        Map<String, Integer> stats = new HashMap<>();
        for (ToolCall toolCall : toolCallHistory) {
            stats.merge(toolCall.getName(), 1, Integer::sum);
        }
        return stats;
    }
    
    // LLM相关的Getter和Setter方法
    public LLMSimulator getLLMSimulator() {
        return llmSimulator;
    }
    
    public void setLLMSimulator(LLMSimulator llmSimulator) {
        this.llmSimulator = llmSimulator;
    }
    
    public String getSystemPrompt() {
        return systemPrompt;
    }
    
    public void setSystemPrompt(String systemPrompt) {
        this.systemPrompt = systemPrompt;
    }
    
    public boolean isLLMEnabled() {
        return llmEnabled;
    }
    
    public void setLLMEnabled(boolean llmEnabled) {
        this.llmEnabled = llmEnabled;
    }
    
    // 原有的Getter和Setter方法
    public String getName() {
        return name;
    }
    
    public AgentState getState() {
        return state;
    }
    
    public List<Message> getMessages() {
        return new ArrayList<>(messages);
    }
    
    public List<ToolCall> getToolCallHistory() {
        return new ArrayList<>(toolCallHistory);
    }
    
    public ToolRegistry getToolRegistry() {
        return toolRegistry;
    }
    
    public Map<String, Object> getConfiguration() {
        return new HashMap<>(configuration);
    }
    
    public void setConfiguration(Map<String, Object> configuration) {
        this.configuration = configuration != null ? new HashMap<>(configuration) : new HashMap<>();
    }
}