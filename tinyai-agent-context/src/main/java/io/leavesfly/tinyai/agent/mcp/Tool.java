package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * MCP 工具定义
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class Tool {
    /**
     * 工具名称
     */
    private String name;
    
    /**
     * 工具描述
     */
    private String description;
    
    /**
     * 工具类别
     */
    private ToolCategory category;
    
    /**
     * 输入参数的JSON Schema
     */
    private Map<String, Object> inputSchema;
    
    /**
     * 工具执行函数
     */
    private Function<Map<String, Object>, Object> function;
    
    /**
     * 元数据
     */
    private Map<String, Object> metadata;
    
    public Tool(String name, String description, ToolCategory category, 
                Map<String, Object> inputSchema, Function<Map<String, Object>, Object> function) {
        this.name = name;
        this.description = description;
        this.category = category;
        this.inputSchema = inputSchema;
        this.function = function;
        this.metadata = new HashMap<>();
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("name", name);
        map.put("description", description);
        map.put("category", category.getValue());
        map.put("inputSchema", inputSchema);
        map.put("metadata", metadata);
        return map;
    }
    
    // Getters and Setters
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
    
    public ToolCategory getCategory() {
        return category;
    }
    
    public void setCategory(ToolCategory category) {
        this.category = category;
    }
    
    public Map<String, Object> getInputSchema() {
        return inputSchema;
    }
    
    public void setInputSchema(Map<String, Object> inputSchema) {
        this.inputSchema = inputSchema;
    }
    
    public Function<Map<String, Object>, Object> getFunction() {
        return function;
    }
    
    public void setFunction(Function<Map<String, Object>, Object> function) {
        this.function = function;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
}
