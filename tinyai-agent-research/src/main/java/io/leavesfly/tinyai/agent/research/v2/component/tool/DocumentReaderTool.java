package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

/**
 * 文档读取工具
 * 安全地读取文档内容
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class DocumentReaderTool implements Tool {
    
    /**
     * 允许的最大文件大小（10MB）
     */
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024;
    
    @Override
    public String getName() {
        return "document_read";
    }
    
    @Override
    public String getDescription() {
        return "安全地读取文档内容";
    }
    
    @Override
    public String getCategory() {
        return "document";
    }
    
    @Override
    public ToolParameters getParameters() {
        ToolParameters params = new ToolParameters();
        params.addParameter(new ToolParameters.Parameter(
            "path", "string", "文档路径", true
        ));
        params.addParameter(new ToolParameters.Parameter(
            "encoding", "string", "文件编码", false, "UTF-8"
        ));
        return params;
    }
    
    @Override
    public boolean validate(Map<String, Object> parameters) {
        if (parameters == null || !parameters.containsKey("path")) {
            return false;
        }
        
        String path = parameters.get("path").toString();
        return path != null && !path.trim().isEmpty();
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            String path = parameters.get("path").toString();
            String encoding = parameters.containsKey("encoding") 
                ? parameters.get("encoding").toString() 
                : "UTF-8";
            
            // 安全检查
            File file = new File(path);
            if (!file.exists()) {
                return ToolResult.error("文件不存在: " + path);
            }
            
            if (!file.isFile()) {
                return ToolResult.error("不是有效的文件: " + path);
            }
            
            if (file.length() > MAX_FILE_SIZE) {
                return ToolResult.error("文件过大，超过限制: " + MAX_FILE_SIZE);
            }
            
            // 读取文件内容
            String content = new String(Files.readAllBytes(Paths.get(path)), encoding);
            
            ToolResult result = ToolResult.success("文档读取完成");
            result.putData("path", path);
            result.putData("content", content);
            result.putData("size", file.length());
            result.putData("encoding", encoding);
            result.setExecutionTimeMs(System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (IOException e) {
            return ToolResult.error("文件读取失败: " + e.getMessage());
        } catch (Exception e) {
            return ToolResult.error("执行失败: " + e.getMessage());
        }
    }
}
