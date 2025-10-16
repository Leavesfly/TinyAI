package io.leavesfly.tinyai.agent.context;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 工具调用结构
 * 
 * @author 山泽
 */
public class ToolCall {
    
    private String id;                      // 工具调用ID
    private String name;                    // 工具名称
    private Map<String, Object> arguments;  // 工具参数
    private Object result;                  // 执行结果
    private String error;                   // 错误信息
    private LocalDateTime timestamp;        // 时间戳
    
    // 构造函数
    public ToolCall() {
        this.timestamp = LocalDateTime.now();
        this.arguments = new HashMap<>();
    }
    
    public ToolCall(String id, String name) {
        this();
        this.id = id;
        this.name = name;
    }
    
    public ToolCall(String id, String name, Map<String, Object> arguments) {
        this(id, name);
        this.arguments = arguments != null ? arguments : new HashMap<>();
    }
    
    // Getter 和 Setter 方法
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public Map<String, Object> getArguments() {
        return arguments;
    }
    
    public void setArguments(Map<String, Object> arguments) {
        this.arguments = arguments;
    }
    
    public Object getResult() {
        return result;
    }
    
    public void setResult(Object result) {
        this.result = result;
    }
    
    public String getError() {
        return error;
    }
    
    public void setError(String error) {
        this.error = error;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * 判断调用是否成功
     */
    public boolean isSuccess() {
        return error == null || error.isEmpty();
    }
    
    /**
     * 添加参数
     */
    public void addArgument(String key, Object value) {
        this.arguments.put(key, value);
    }
    
    /**
     * 获取指定参数
     */
    public Object getArgument(String key) {
        return this.arguments.get(key);
    }
    
    /**
     * 获取指定参数的字符串值
     */
    public String getArgumentAsString(String key) {
        Object value = this.arguments.get(key);
        return value != null ? value.toString() : null;
    }
    
    /**
     * 获取指定参数的数值
     */
    public Double getArgumentAsDouble(String key) {
        Object value = this.arguments.get(key);
        if (value == null) {
            return null;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            return null;
        }
    }
    
    /**
     * 获取指定参数的整数值
     */
    public Integer getArgumentAsInteger(String key) {
        Object value = this.arguments.get(key);
        if (value == null) {
            return null;
        }
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        try {
            return Integer.parseInt(value.toString());
        } catch (NumberFormatException e) {
            return null;
        }
    }
    
    @Override
    public String toString() {
        String status = isSuccess() ? "SUCCESS" : "ERROR";
        return String.format("ToolCall{id='%s', name='%s', status='%s', timestamp=%s}", 
                           id, name, status, timestamp);
    }
}