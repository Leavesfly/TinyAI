package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具注册表
 * 管理所有可用工具的注册、查询和执行
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolRegistry {
    
    /**
     * 工具存储 (toolName -> Tool)
     */
    private final Map<String, Tool> tools;
    
    /**
     * 工具分类索引 (category -> List<toolName>)
     */
    private final Map<String, List<String>> categoryIndex;
    
    public ToolRegistry() {
        this.tools = new ConcurrentHashMap<>();
        this.categoryIndex = new ConcurrentHashMap<>();
        
        // 注册内置工具
        registerBuiltinTools();
    }
    
    /**
     * 注册工具
     */
    public void registerTool(Tool tool) {
        if (tool == null || tool.getName() == null) {
            throw new IllegalArgumentException("Tool and tool name cannot be null");
        }
        
        tools.put(tool.getName(), tool);
        
        // 更新分类索引
        String category = tool.getCategory();
        categoryIndex.computeIfAbsent(category, k -> new ArrayList<>()).add(tool.getName());
        
        System.out.println("[ToolRegistry] 注册工具: " + tool.getName() + " (类别: " + category + ")");
    }
    
    /**
     * 获取工具
     */
    public Tool getTool(String toolName) {
        return tools.get(toolName);
    }
    
    /**
     * 获取所有工具信息
     */
    public List<ToolInfo> listTools() {
        List<ToolInfo> toolInfos = new ArrayList<>();
        for (Tool tool : tools.values()) {
            toolInfos.add(new ToolInfo(
                tool.getName(),
                tool.getDescription(),
                tool.getCategory(),
                tool.getParameters()
            ));
        }
        return toolInfos;
    }
    
    /**
     * 按分类获取工具
     */
    public List<Tool> getToolsByCategory(String category) {
        List<String> toolNames = categoryIndex.get(category);
        if (toolNames == null) {
            return new ArrayList<>();
        }
        
        List<Tool> result = new ArrayList<>();
        for (String name : toolNames) {
            Tool tool = tools.get(name);
            if (tool != null) {
                result.add(tool);
            }
        }
        return result;
    }
    
    /**
     * 执行工具
     */
    public ToolResult executeTool(String toolName, Map<String, Object> parameters) {
        Tool tool = tools.get(toolName);
        if (tool == null) {
            return ToolResult.error("工具不存在: " + toolName);
        }
        
        try {
            // 参数验证
            if (!tool.validate(parameters)) {
                return ToolResult.error("参数验证失败");
            }
            
            // 执行工具
            return tool.execute(parameters);
            
        } catch (Exception e) {
            return ToolResult.error("工具执行失败: " + e.getMessage());
        }
    }
    
    /**
     * 注册内置工具
     */
    private void registerBuiltinTools() {
        registerTool(new WebSearchTool());
        registerTool(new DocumentReaderTool());
        registerTool(new CodeAnalyzerTool());
    }
    
    /**
     * 工具信息类
     */
    public static class ToolInfo {
        private final String name;
        private final String description;
        private final String category;
        private final ToolParameters parameters;
        
        public ToolInfo(String name, String description, String category, ToolParameters parameters) {
            this.name = name;
            this.description = description;
            this.category = category;
            this.parameters = parameters;
        }
        
        public String getName() { return name; }
        public String getDescription() { return description; }
        public String getCategory() { return category; }
        public ToolParameters getParameters() { return parameters; }
    }
}
