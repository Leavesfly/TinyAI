package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具调用数据结构
 * 表示LLM请求调用的工具及参数
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolCall {
    
    /**
     * 工具调用唯一ID
     */
    private String id;
    
    /**
     * 工具类型（默认为function）
     */
    private String type = "function";
    
    /**
     * 函数调用信息
     */
    private FunctionCall function;
    
    public ToolCall() {
    }
    
    public ToolCall(String id, String functionName, Map<String, Object> arguments) {
        this.id = id;
        this.function = new FunctionCall(functionName, arguments);
    }
    
    /**
     * 创建工具调用
     */
    public static ToolCall create(String id, String functionName, Map<String, Object> arguments) {
        return new ToolCall(id, functionName, arguments);
    }
    
    // Getters and Setters
    
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
    }
    
    public FunctionCall getFunction() {
        return function;
    }
    
    public void setFunction(FunctionCall function) {
        this.function = function;
    }
    
    /**
     * 函数调用信息
     */
    public static class FunctionCall {
        /**
         * 函数名称
         */
        private String name;
        
        /**
         * 函数参数（JSON格式的Map）
         */
        private Map<String, Object> arguments;
        
        public FunctionCall() {
        }
        
        public FunctionCall(String name, Map<String, Object> arguments) {
            this.name = name;
            this.arguments = arguments != null ? arguments : new HashMap<>();
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
        
        @Override
        public String toString() {
            return "FunctionCall{" +
                    "name='" + name + '\'' +
                    ", arguments=" + arguments +
                    '}';
        }
    }
    
    @Override
    public String toString() {
        return "ToolCall{" +
                "id='" + id + '\'' +
                ", type='" + type + '\'' +
                ", function=" + function +
                '}';
    }
}
