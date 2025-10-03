package io.leavesfly.tinyai.agent.manus;

import io.leavesfly.tinyai.agent.Message;
import io.leavesfly.tinyai.agent.ToolCall;
import io.leavesfly.tinyai.agent.ToolRegistry;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 基础Agent抽象类
 * OpenManus分层架构的第一层
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
        
        // 注册默认工具
        registerDefaultTools();
    }
    
    /**
     * 处理消息的抽象方法
     * 子类必须实现此方法
     */
    public abstract Message processMessage(Message message);
    
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
    
    // Getter 和 Setter 方法
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