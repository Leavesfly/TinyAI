package io.leavesfly.tinyai.agent.pattern;

import java.util.HashMap;
import java.util.Map;

/**
 * 动作结构类
 * 表示Agent要执行的动作
 * @author 山泽
 */
public class Action {
    /** 动作名称 */
    private final String name;
    
    /** 动作参数 */
    private final Map<String, Object> arguments;
    
    /** 执行结果 */
    private Object result;
    
    /** 错误信息 */
    private String error;
    
    public Action(String name, Map<String, Object> arguments) {
        this.name = name;
        this.arguments = new HashMap<>(arguments);
    }
    
    public Action(String name) {
        this(name, new HashMap<>());
    }
    
    public String getName() {
        return name;
    }
    
    public Map<String, Object> getArguments() {
        return new HashMap<>(arguments);
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
    
    public boolean hasResult() {
        return result != null;
    }
    
    public boolean hasError() {
        return error != null && !error.isEmpty();
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(name).append("(");
        
        if (!arguments.isEmpty()) {
            arguments.forEach((key, value) -> 
                sb.append(key).append("=").append(value).append(", "));
            sb.setLength(sb.length() - 2); // 移除最后的逗号和空格
        }
        
        sb.append(")");
        return sb.toString();
    }
}