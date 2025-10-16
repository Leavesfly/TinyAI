package io.leavesfly.tinyai.agent.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * MCP 提示词模板
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class Prompt {
    /**
     * 提示词名称
     */
    private String name;
    
    /**
     * 提示词描述
     */
    private String description;
    
    /**
     * 提示词模板
     */
    private String template;
    
    /**
     * 参数定义
     */
    private List<Map<String, Object>> arguments;
    
    /**
     * 元数据
     */
    private Map<String, Object> metadata;
    
    public Prompt(String name, String description, String template) {
        this.name = name;
        this.description = description;
        this.template = template;
        this.arguments = new ArrayList<>();
        this.metadata = new HashMap<>();
    }
    
    public Prompt(String name, String description, String template, List<Map<String, Object>> arguments) {
        this.name = name;
        this.description = description;
        this.template = template;
        this.arguments = arguments;
        this.metadata = new HashMap<>();
    }
    
    /**
     * 渲染提示词模板
     * 
     * @param params 参数Map
     * @return 渲染后的提示词
     */
    public String render(Map<String, Object> params) {
        String result = template;
        
        // 替换模板中的占位符 {key}
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            String placeholder = "{" + entry.getKey() + "}";
            String value = entry.getValue() != null ? entry.getValue().toString() : "";
            result = result.replace(placeholder, value);
        }
        
        return result;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("name", name);
        map.put("description", description);
        map.put("arguments", arguments);
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
    
    public String getTemplate() {
        return template;
    }
    
    public void setTemplate(String template) {
        this.template = template;
    }
    
    public List<Map<String, Object>> getArguments() {
        return arguments;
    }
    
    public void setArguments(List<Map<String, Object>> arguments) {
        this.arguments = arguments;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
}
