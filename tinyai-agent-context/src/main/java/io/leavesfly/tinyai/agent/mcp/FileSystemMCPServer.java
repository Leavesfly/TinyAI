package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

/**
 * 文件系统 MCP Server 示例
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class FileSystemMCPServer extends MCPServer {
    
    public FileSystemMCPServer() {
        super("FileSystem Server", "1.0.0");
        setupResources();
        setupTools();
        setupPrompts();
    }
    
    /**
     * 设置文件系统资源
     */
    private void setupResources() {
        // 模拟文件资源
        Map<String, String> files = new HashMap<>();
        files.put("file:///docs/readme.md", "# 项目说明\n\n这是一个使用 MCP 的示例项目。");
        files.put("file:///docs/api.md", "# API 文档\n\n## 端点\n- GET /api/data");
        files.put("file:///config/settings.json", "{\"debug\": true, \"port\": 8080}");
        
        for (Map.Entry<String, String> entry : files.entrySet()) {
            String uri = entry.getKey();
            String content = entry.getValue();
            String filename = uri.substring(uri.lastIndexOf("/") + 1);
            
            Resource resource = new Resource(uri, filename, ResourceType.FILE);
            resource.setDescription("文件: " + filename);
            resource.setMimeType(uri.endsWith(".md") ? "text/plain" : "application/json");
            
            registerResource(resource);
            setResourceContent(uri, content);
        }
    }
    
    /**
     * 设置文件系统工具
     */
    private void setupTools() {
        // 搜索文件工具
        registerTool(new Tool(
            "search_files",
            "在文件中搜索关键词",
            ToolCategory.SEARCH,
            MCPUtils.createJsonSchema(
                new HashMap<String, Map<String, Object>>() {{
                    put("keyword", MCPUtils.createProperty("string", "搜索关键词"));
                }},
                Arrays.asList("keyword")
            ),
            args -> searchFiles((String) args.get("keyword"))
        ));
        
        // 列出目录工具
        registerTool(new Tool(
            "list_directory",
            "列出目录中的文件",
            ToolCategory.DATA_ACCESS,
            MCPUtils.createJsonSchema(
                new HashMap<String, Map<String, Object>>() {{
                    put("path", MCPUtils.createProperty("string", "目录路径", "/"));
                }},
                new ArrayList<>()
            ),
            args -> listDirectory((String) args.getOrDefault("path", "/"))
        ));
    }
    
    /**
     * 设置提示词模板
     */
    private void setupPrompts() {
        String template = "请分析以下文件内容：\n\n" +
                         "文件：{filename}\n" +
                         "内容：\n{content}\n\n" +
                         "请提供：\n" +
                         "1. 文件类型和格式\n" +
                         "2. 主要内容摘要\n" +
                         "3. 关键信息提取\n" +
                         "4. 建议的改进点";
        
        List<Map<String, Object>> arguments = new ArrayList<>();
        Map<String, Object> arg1 = new HashMap<>();
        arg1.put("name", "filename");
        arg1.put("type", "string");
        arg1.put("required", true);
        arguments.add(arg1);
        
        Map<String, Object> arg2 = new HashMap<>();
        arg2.put("name", "content");
        arg2.put("type", "string");
        arg2.put("required", true);
        arguments.add(arg2);
        
        registerPrompt(new Prompt(
            "analyze_file",
            "分析文件内容的提示词模板",
            template,
            arguments
        ));
    }
    
    /**
     * 搜索文件内容
     */
    private Map<String, Object> searchFiles(String keyword) {
        List<Map<String, Object>> results = new ArrayList<>();
        
        for (Map.Entry<String, Object> entry : resourceContentCache.entrySet()) {
            String uri = entry.getKey();
            String content = entry.getValue().toString();
            
            if (content.toLowerCase().contains(keyword.toLowerCase())) {
                Map<String, Object> result = new HashMap<>();
                result.put("uri", uri);
                String preview = content.length() > 100 ? content.substring(0, 100) + "..." : content;
                result.put("preview", preview);
                results.add(result);
            }
        }
        
        Map<String, Object> response = new HashMap<>();
        response.put("found", results.size());
        response.put("results", results);
        return response;
    }
    
    /**
     * 列出目录内容
     */
    private Map<String, Object> listDirectory(String path) {
        List<String> files = new ArrayList<>();
        String prefix = "file://" + path;
        
        for (String uri : resources.keySet()) {
            if (uri.startsWith(prefix)) {
                files.add(uri);
            }
        }
        
        Map<String, Object> response = new HashMap<>();
        response.put("path", path);
        response.put("files", files);
        response.put("count", files.size());
        return response;
    }
}
