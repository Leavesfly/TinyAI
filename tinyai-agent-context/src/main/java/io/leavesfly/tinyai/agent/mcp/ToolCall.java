package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * 工具调用请求
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class ToolCall {
    /**
     * 调用ID
     */
    private String id;
    
    /**
     * 工具名称
     */
    private String name;
    
    /**
     * 调用参数
     */
    private Map<String, Object> arguments;
    
    public ToolCall() {
        this.id = UUID.randomUUID().toString();
        this.name = "";
        this.arguments = new HashMap<>();
    }
    
    public ToolCall(String name, Map<String, Object> arguments) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.arguments = arguments;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("id", id);
        map.put("name", name);
        map.put("arguments", arguments);
        return map;
    }
    
    // Getters and Setters
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
}
