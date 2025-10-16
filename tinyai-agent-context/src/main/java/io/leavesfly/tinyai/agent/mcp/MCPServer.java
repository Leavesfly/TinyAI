package io.leavesfly.tinyai.agent.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MCP 服务器 - 提供资源、工具和提示词
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPServer {
    /**
     * 服务器名称
     */
    protected String name;
    
    /**
     * 服务器版本
     */
    protected String version;
    
    /**
     * 资源注册表
     */
    protected Map<String, Resource> resources;
    
    /**
     * 工具注册表
     */
    protected Map<String, Tool> tools;
    
    /**
     * 提示词注册表
     */
    protected Map<String, Prompt> prompts;
    
    /**
     * 资源内容缓存
     */
    protected Map<String, Object> resourceContentCache;
    
    public MCPServer(String name) {
        this(name, "1.0.0");
    }
    
    public MCPServer(String name, String version) {
        this.name = name;
        this.version = version;
        this.resources = new HashMap<>();
        this.tools = new HashMap<>();
        this.prompts = new HashMap<>();
        this.resourceContentCache = new HashMap<>();
        
        System.out.println("✅ MCP Server '" + name + "' v" + version + " 初始化完成");
    }
    
    // ========== Resource 管理 ==========
    
    /**
     * 注册资源
     */
    public void registerResource(Resource resource) {
        resources.put(resource.getUri(), resource);
        System.out.println("📦 资源已注册: " + resource.getName() + " (" + resource.getUri() + ")");
    }
    
    /**
     * 设置资源内容
     */
    public void setResourceContent(String uri, Object content) {
        resourceContentCache.put(uri, content);
    }
    
    /**
     * 列出所有资源
     */
    public List<Map<String, Object>> listResources() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Resource resource : resources.values()) {
            result.add(resource.toMap());
        }
        return result;
    }
    
    /**
     * 获取资源内容
     */
    public ResourceContent getResource(String uri) {
        if (!resources.containsKey(uri)) {
            return null;
        }
        
        Resource resource = resources.get(uri);
        Object content = resourceContentCache.get(uri);
        
        if (content == null) {
            // 如果没有缓存，尝试动态加载
            content = loadResourceContent(uri);
        }
        
        return new ResourceContent(uri, content, resource.getMimeType());
    }
    
    /**
     * 动态加载资源内容（子类可重写）
     */
    protected Object loadResourceContent(String uri) {
        return "资源 " + uri + " 的内容";
    }
    
    // ========== Tool 管理 ==========
    
    /**
     * 注册工具
     */
    public void registerTool(Tool tool) {
        tools.put(tool.getName(), tool);
        System.out.println("🔧 工具已注册: " + tool.getName() + " - " + tool.getDescription());
    }
    
    /**
     * 列出所有工具
     */
    public List<Map<String, Object>> listTools() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Tool tool : tools.values()) {
            result.add(tool.toMap());
        }
        return result;
    }
    
    /**
     * 调用工具
     */
    public ToolResult callTool(ToolCall toolCall) {
        long startTime = System.currentTimeMillis();
        
        if (!tools.containsKey(toolCall.getName())) {
            return new ToolResult(
                toolCall.getId(),
                null,
                true,
                "工具 '" + toolCall.getName() + "' 不存在",
                0.0
            );
        }
        
        Tool tool = tools.get(toolCall.getName());
        
        if (tool.getFunction() == null) {
            return new ToolResult(
                toolCall.getId(),
                null,
                true,
                "工具 '" + toolCall.getName() + "' 没有关联函数",
                0.0
            );
        }
        
        try {
            Object result = tool.getFunction().apply(toolCall.getArguments());
            double executionTime = (System.currentTimeMillis() - startTime) / 1000.0;
            
            return new ToolResult(
                toolCall.getId(),
                result,
                false,
                null,
                executionTime
            );
        } catch (Exception e) {
            double executionTime = (System.currentTimeMillis() - startTime) / 1000.0;
            return new ToolResult(
                toolCall.getId(),
                null,
                true,
                e.getMessage(),
                executionTime
            );
        }
    }
    
    // ========== Prompt 管理 ==========
    
    /**
     * 注册提示词模板
     */
    public void registerPrompt(Prompt prompt) {
        prompts.put(prompt.getName(), prompt);
        System.out.println("📝 提示词已注册: " + prompt.getName() + " - " + prompt.getDescription());
    }
    
    /**
     * 列出所有提示词模板
     */
    public List<Map<String, Object>> listPrompts() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Prompt prompt : prompts.values()) {
            result.add(prompt.toMap());
        }
        return result;
    }
    
    /**
     * 获取并渲染提示词
     */
    public String getPrompt(String name, Map<String, Object> params) {
        if (!prompts.containsKey(name)) {
            return null;
        }
        
        Prompt prompt = prompts.get(name);
        return prompt.render(params);
    }
    
    // ========== RPC 处理 ==========
    
    /**
     * 处理 MCP 请求
     */
    public MCPResponse handleRequest(MCPRequest request) {
        String method = request.getMethod();
        Map<String, Object> params = request.getParams();
        
        try {
            Object result = null;
            
            switch (method) {
                case "resources/list":
                    result = listResources();
                    break;
                    
                case "resources/read":
                    String uri = (String) params.get("uri");
                    ResourceContent resourceContent = getResource(uri);
                    result = resourceContent != null ? resourceContent.toMap() : null;
                    break;
                    
                case "tools/list":
                    result = listTools();
                    break;
                    
                case "tools/call":
                    ToolCall toolCall = new ToolCall();
                    toolCall.setId((String) params.getOrDefault("id", toolCall.getId()));
                    toolCall.setName((String) params.getOrDefault("name", ""));
                    @SuppressWarnings("unchecked")
                    Map<String, Object> args = (Map<String, Object>) params.getOrDefault("arguments", new HashMap<>());
                    toolCall.setArguments(args);
                    
                    ToolResult toolResult = callTool(toolCall);
                    result = toolResult.toMap();
                    break;
                    
                case "prompts/list":
                    result = listPrompts();
                    break;
                    
                case "prompts/get":
                    String promptName = (String) params.get("name");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> promptArgs = (Map<String, Object>) params.getOrDefault("arguments", new HashMap<>());
                    String promptText = getPrompt(promptName, promptArgs);
                    
                    Map<String, Object> promptResult = new HashMap<>();
                    promptResult.put("prompt", promptText);
                    result = promptResult;
                    break;
                    
                default:
                    return MCPResponse.createErrorResponse(
                        request.getId(),
                        -32601,
                        "方法不存在: " + method
                    );
            }
            
            return new MCPResponse(request.getId(), result);
            
        } catch (Exception e) {
            return MCPResponse.createErrorResponse(
                request.getId(),
                -32603,
                "内部错误: " + e.getMessage()
            );
        }
    }
    
    /**
     * 获取服务器信息
     */
    public Map<String, Object> getServerInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("name", name);
        info.put("version", version);
        
        Map<String, Object> capabilities = new HashMap<>();
        capabilities.put("resources", resources.size());
        capabilities.put("tools", tools.size());
        capabilities.put("prompts", prompts.size());
        info.put("capabilities", capabilities);
        
        return info;
    }
    
    // Getters
    public String getName() {
        return name;
    }
    
    public String getVersion() {
        return version;
    }
    
    public Map<String, Resource> getResources() {
        return resources;
    }
    
    public Map<String, Tool> getTools() {
        return tools;
    }
    
    public Map<String, Prompt> getPrompts() {
        return prompts;
    }
}
