package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具定义数据结构
 * 描述可供LLM调用的工具及其参数规范
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolDefinition {
    
    /**
     * 工具类型（默认为function）
     */
    private String type = "function";
    
    /**
     * 函数定义
     */
    private FunctionDefinition function;
    
    public ToolDefinition() {
    }
    
    public ToolDefinition(String name, String description, Map<String, Object> parameters) {
        this.function = new FunctionDefinition(name, description, parameters);
    }
    
    /**
     * 创建工具定义
     */
    public static ToolDefinition create(String name, String description, Map<String, Object> parameters) {
        return new ToolDefinition(name, description, parameters);
    }
    
    /**
     * 创建简单工具定义（无参数）
     */
    public static ToolDefinition create(String name, String description) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", new HashMap<>());
        return new ToolDefinition(name, description, parameters);
    }
    
    // Getters and Setters
    
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
    }
    
    public FunctionDefinition getFunction() {
        return function;
    }
    
    public void setFunction(FunctionDefinition function) {
        this.function = function;
    }
    
    /**
     * 函数定义
     */
    public static class FunctionDefinition {
        /**
         * 函数名称
         */
        private String name;
        
        /**
         * 函数描述
         */
        private String description;
        
        /**
         * 参数定义（JSON Schema格式）
         */
        private Map<String, Object> parameters;
        
        public FunctionDefinition() {
        }
        
        public FunctionDefinition(String name, String description, Map<String, Object> parameters) {
            this.name = name;
            this.description = description;
            this.parameters = parameters;
        }
        
        public String getName() {
            return name;
        }
        
        public void setName(String name) {
            this.name = name;
        }
        
        public String getDescription() {
            return description;
        }
        
        public void setDescription(String description) {
            this.description = description;
        }
        
        public Map<String, Object> getParameters() {
            return parameters;
        }
        
        public void setParameters(Map<String, Object> parameters) {
            this.parameters = parameters;
        }
        
        @Override
        public String toString() {
            return "FunctionDefinition{" +
                    "name='" + name + '\'' +
                    ", description='" + description + '\'' +
                    ", parameters=" + parameters +
                    '}';
        }
    }
    
    @Override
    public String toString() {
        return "ToolDefinition{" +
                "type='" + type + '\'' +
                ", function=" + function +
                '}';
    }
}
