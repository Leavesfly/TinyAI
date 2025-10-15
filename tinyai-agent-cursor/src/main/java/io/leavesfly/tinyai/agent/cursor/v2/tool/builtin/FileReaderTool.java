package io.leavesfly.tinyai.agent.cursor.v2.tool.builtin;

import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * 文件读取工具
 * 读取文件内容（需在沙箱中执行以确保安全）
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class FileReaderTool implements Tool {
    
    private static final String NAME = "file_reader";
    private static final String DESCRIPTION = "Read file content from the file system";
    
    /**
     * 允许的文件扩展名
     */
    private static final String[] ALLOWED_EXTENSIONS = {
        ".java", ".kt", ".py", ".js", ".ts", ".go", ".rs",
        ".c", ".cpp", ".h", ".cs", ".rb", ".php",
        ".json", ".xml", ".yaml", ".yml", ".toml",
        ".md", ".txt", ".properties", ".conf"
    };
    
    @Override
    public String getName() {
        return NAME;
    }
    
    @Override
    public String getDescription() {
        return DESCRIPTION;
    }
    
    @Override
    public ToolDefinition getDefinition() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        
        // filePath参数
        Map<String, Object> filePathParam = new HashMap<>();
        filePathParam.put("type", "string");
        filePathParam.put("description", "Path to the file to read");
        properties.put("filePath", filePathParam);
        
        // maxLines参数
        Map<String, Object> maxLinesParam = new HashMap<>();
        maxLinesParam.put("type", "integer");
        maxLinesParam.put("description", "Maximum number of lines to read (default: no limit)");
        properties.put("maxLines", maxLinesParam);
        
        parameters.put("properties", properties);
        parameters.put("required", new String[]{"filePath"});
        
        return ToolDefinition.create(NAME, DESCRIPTION, parameters);
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            if (!validateParameters(parameters)) {
                ToolResult result = ToolResult.failure(NAME, "Invalid parameters: filePath is required");
                result.setExecutionTime(System.currentTimeMillis() - startTime);
                return result;
            }
            
            String filePath = (String) parameters.get("filePath");
            Integer maxLines = parameters.containsKey("maxLines") ? 
                              ((Number) parameters.get("maxLines")).intValue() : null;
            
            // 验证文件扩展名
            if (!isAllowedFile(filePath)) {
                ToolResult result = ToolResult.failure(NAME, 
                    "File type not allowed. Only source code and text files are supported.");
                result.setExecutionTime(System.currentTimeMillis() - startTime);
                return result;
            }
            
            // 读取文件
            String content = readFile(filePath, maxLines);
            
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.success(NAME, content);
            result.setExecutionTime(executionTime);
            result.putMetadata("filePath", filePath);
            result.putMetadata("fileSize", content.length());
            
            return result;
            
        } catch (IOException e) {
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.failure(NAME, "Failed to read file: " + e.getMessage());
            result.setExecutionTime(executionTime);
            return result;
        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.failure(NAME, "Error: " + e.getMessage());
            result.setExecutionTime(executionTime);
            return result;
        }
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null || !parameters.containsKey("filePath")) {
            return false;
        }
        
        Object filePath = parameters.get("filePath");
        return filePath instanceof String && !((String) filePath).isEmpty();
    }
    
    @Override
    public boolean requiresSandbox() {
        return true; // 文件操作需要沙箱隔离
    }
    
    @Override
    public ToolCategory getCategory() {
        return ToolCategory.FILE_OPERATION;
    }
    
    /**
     * 检查文件是否允许读取
     */
    private boolean isAllowedFile(String filePath) {
        String lowerPath = filePath.toLowerCase();
        
        for (String ext : ALLOWED_EXTENSIONS) {
            if (lowerPath.endsWith(ext)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 读取文件内容
     */
    private String readFile(String filePath, Integer maxLines) throws IOException {
        Path path = Paths.get(filePath);
        
        if (!Files.exists(path)) {
            throw new IOException("File not found: " + filePath);
        }
        
        if (!Files.isReadable(path)) {
            throw new IOException("File not readable: " + filePath);
        }
        
        if (maxLines == null) {
            // 读取整个文件
            return Files.readString(path);
        } else {
            // 读取指定行数
            StringBuilder content = new StringBuilder();
            int lineCount = 0;
            
            for (String line : Files.readAllLines(path)) {
                if (lineCount >= maxLines) {
                    content.append("\n... (truncated, ").append(maxLines).append(" lines shown)");
                    break;
                }
                content.append(line).append("\n");
                lineCount++;
            }
            
            return content.toString();
        }
    }
}
