package io.leavesfly.tinyai.agent.cursor.v2.tool;

import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具注册表
 * 管理所有可用工具的注册、查询和调用
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolRegistry {
    
    /**
     * 工具存储（toolName -> Tool）
     */
    private final Map<String, Tool> tools;
    
    /**
     * 类别索引（category -> List<toolName>）
     */
    private final Map<Tool.ToolCategory, List<String>> categoryIndex;
    
    public ToolRegistry() {
        this.tools = new ConcurrentHashMap<>();
        this.categoryIndex = new ConcurrentHashMap<>();
    }
    
    /**
     * 注册工具
     * 
     * @param tool 工具实例
     */
    public void register(Tool tool) {
        if (tool == null || tool.getName() == null) {
            throw new IllegalArgumentException("Tool and tool name cannot be null");
        }
        
        String toolName = tool.getName();
        
        // 检查是否已注册
        if (tools.containsKey(toolName)) {
            throw new IllegalStateException("Tool already registered: " + toolName);
        }
        
        // 注册工具
        tools.put(toolName, tool);
        
        // 更新类别索引
        Tool.ToolCategory category = tool.getCategory();
        categoryIndex.computeIfAbsent(category, k -> new ArrayList<>()).add(toolName);
    }
    
    /**
     * 注销工具
     * 
     * @param toolName 工具名称
     */
    public void unregister(String toolName) {
        Tool tool = tools.remove(toolName);
        if (tool != null) {
            // 从类别索引中移除
            List<String> categoryTools = categoryIndex.get(tool.getCategory());
            if (categoryTools != null) {
                categoryTools.remove(toolName);
            }
        }
    }
    
    /**
     * 获取工具
     * 
     * @param toolName 工具名称
     * @return 工具实例，如果不存在返回null
     */
    public Tool getTool(String toolName) {
        return tools.get(toolName);
    }
    
    /**
     * 检查工具是否存在
     * 
     * @param toolName 工具名称
     * @return 是否存在
     */
    public boolean hasTool(String toolName) {
        return tools.containsKey(toolName);
    }
    
    /**
     * 获取所有工具名称
     * 
     * @return 工具名称列表
     */
    public List<String> getAllToolNames() {
        return new ArrayList<>(tools.keySet());
    }
    
    /**
     * 获取所有工具
     * 
     * @return 工具列表
     */
    public List<Tool> getAllTools() {
        return new ArrayList<>(tools.values());
    }
    
    /**
     * 按类别获取工具
     * 
     * @param category 工具类别
     * @return 工具列表
     */
    public List<Tool> getToolsByCategory(Tool.ToolCategory category) {
        List<String> toolNames = categoryIndex.getOrDefault(category, new ArrayList<>());
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
     * 获取所有工具定义（用于传递给LLM）
     * 
     * @return 工具定义列表
     */
    public List<ToolDefinition> getAllToolDefinitions() {
        List<ToolDefinition> definitions = new ArrayList<>();
        
        for (Tool tool : tools.values()) {
            definitions.add(tool.getDefinition());
        }
        
        return definitions;
    }
    
    /**
     * 按类别获取工具定义
     * 
     * @param category 工具类别
     * @return 工具定义列表
     */
    public List<ToolDefinition> getToolDefinitionsByCategory(Tool.ToolCategory category) {
        List<Tool> categoryTools = getToolsByCategory(category);
        List<ToolDefinition> definitions = new ArrayList<>();
        
        for (Tool tool : categoryTools) {
            definitions.add(tool.getDefinition());
        }
        
        return definitions;
    }
    
    /**
     * 获取指定工具名称的定义
     * 
     * @param toolNames 工具名称列表
     * @return 工具定义列表
     */
    public List<ToolDefinition> getToolDefinitions(List<String> toolNames) {
        List<ToolDefinition> definitions = new ArrayList<>();
        
        for (String name : toolNames) {
            Tool tool = tools.get(name);
            if (tool != null) {
                definitions.add(tool.getDefinition());
            }
        }
        
        return definitions;
    }
    
    /**
     * 获取统计信息
     * 
     * @return 注册表统计
     */
    public RegistryStats getStats() {
        RegistryStats stats = new RegistryStats();
        stats.totalTools = tools.size();
        
        for (Tool.ToolCategory category : Tool.ToolCategory.values()) {
            List<String> categoryTools = categoryIndex.getOrDefault(category, new ArrayList<>());
            stats.toolsByCategory.put(category, categoryTools.size());
        }
        
        return stats;
    }
    
    /**
     * 注册表统计信息
     */
    public static class RegistryStats {
        public int totalTools;
        public Map<Tool.ToolCategory, Integer> toolsByCategory = new HashMap<>();
        
        @Override
        public String toString() {
            return "RegistryStats{" +
                    "total=" + totalTools +
                    ", byCategory=" + toolsByCategory +
                    '}';
        }
    }
}
