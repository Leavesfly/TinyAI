package io.leavesfly.tinyai.agent.cursor.v2.tool.builtin;

import io.leavesfly.tinyai.agent.cursor.v2.model.ToolDefinition;
import io.leavesfly.tinyai.agent.cursor.v2.model.ToolResult;
import io.leavesfly.tinyai.agent.cursor.v2.tool.Tool;

import java.util.HashMap;
import java.util.Map;

/**
 * 代码分析工具
 * 分析代码的结构、复杂度、潜在问题等
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CodeAnalyzerTool implements Tool {
    
    private static final String NAME = "code_analyzer";
    private static final String DESCRIPTION = "Analyze code structure, complexity, and potential issues";
    
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
        // 定义参数Schema
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        
        // code参数
        Map<String, Object> codeParam = new HashMap<>();
        codeParam.put("type", "string");
        codeParam.put("description", "The source code to analyze");
        properties.put("code", codeParam);
        
        // analysisType参数
        Map<String, Object> typeParam = new HashMap<>();
        typeParam.put("type", "string");
        typeParam.put("description", "Type of analysis: structure, complexity, quality, or all");
        typeParam.put("enum", new String[]{"structure", "complexity", "quality", "all"});
        properties.put("analysisType", typeParam);
        
        parameters.put("properties", properties);
        parameters.put("required", new String[]{"code"});
        
        return ToolDefinition.create(NAME, DESCRIPTION, parameters);
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 验证参数
            if (!validateParameters(parameters)) {
                ToolResult result = ToolResult.failure(NAME, "Invalid parameters");
                result.setExecutionTime(System.currentTimeMillis() - startTime);
                return result;
            }
            
            String code = (String) parameters.get("code");
            String analysisType = (String) parameters.getOrDefault("analysisType", "all");
            
            // 执行分析
            Map<String, Object> analysisResult = new HashMap<>();
            
            if ("structure".equals(analysisType) || "all".equals(analysisType)) {
                analysisResult.put("structure", analyzeStructure(code));
            }
            
            if ("complexity".equals(analysisType) || "all".equals(analysisType)) {
                analysisResult.put("complexity", analyzeComplexity(code));
            }
            
            if ("quality".equals(analysisType) || "all".equals(analysisType)) {
                analysisResult.put("quality", analyzeQuality(code));
            }
            
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.success(NAME, formatResult(analysisResult));
            result.setExecutionTime(executionTime);
            return result;
            
        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            ToolResult result = ToolResult.failure(NAME, "Analysis failed: " + e.getMessage());
            result.setExecutionTime(executionTime);
            return result;
        }
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null || !parameters.containsKey("code")) {
            return false;
        }
        
        Object code = parameters.get("code");
        return code instanceof String && !((String) code).isEmpty();
    }
    
    @Override
    public ToolCategory getCategory() {
        return ToolCategory.CODE_ANALYSIS;
    }
    
    /**
     * 分析代码结构
     */
    private Map<String, Object> analyzeStructure(String code) {
        Map<String, Object> result = new HashMap<>();
        
        // 统计类、方法、字段数量
        int classCount = countOccurrences(code, "class ");
        int methodCount = countOccurrences(code, "public ") + 
                         countOccurrences(code, "private ") + 
                         countOccurrences(code, "protected ");
        int lineCount = code.split("\n").length;
        
        result.put("classes", classCount);
        result.put("methods", methodCount);
        result.put("lines", lineCount);
        result.put("isEmpty", code.trim().isEmpty());
        
        return result;
    }
    
    /**
     * 分析代码复杂度
     */
    private Map<String, Object> analyzeComplexity(String code) {
        Map<String, Object> result = new HashMap<>();
        
        // 简单的圈复杂度估算
        int ifCount = countOccurrences(code, "if ");
        int forCount = countOccurrences(code, "for ");
        int whileCount = countOccurrences(code, "while ");
        int switchCount = countOccurrences(code, "switch ");
        int catchCount = countOccurrences(code, "catch ");
        
        int cyclomaticComplexity = 1 + ifCount + forCount + whileCount + switchCount + catchCount;
        
        result.put("cyclomaticComplexity", cyclomaticComplexity);
        result.put("conditionals", ifCount);
        result.put("loops", forCount + whileCount);
        result.put("switches", switchCount);
        result.put("exceptionHandlers", catchCount);
        
        // 复杂度评级
        String rating;
        if (cyclomaticComplexity <= 10) {
            rating = "Low";
        } else if (cyclomaticComplexity <= 20) {
            rating = "Moderate";
        } else if (cyclomaticComplexity <= 50) {
            rating = "High";
        } else {
            rating = "Very High";
        }
        result.put("complexityRating", rating);
        
        return result;
    }
    
    /**
     * 分析代码质量
     */
    private Map<String, Object> analyzeQuality(String code) {
        Map<String, Object> result = new HashMap<>();
        
        // 注释率
        int totalLines = code.split("\n").length;
        int commentLines = countOccurrences(code, "//") + 
                          countOccurrences(code, "/*") +
                          countOccurrences(code, "*");
        double commentRatio = totalLines > 0 ? (double) commentLines / totalLines : 0;
        
        result.put("commentRatio", String.format("%.2f%%", commentRatio * 100));
        result.put("hasComments", commentLines > 0);
        
        // 命名规范检查（简单检查）
        boolean hasGoodNaming = code.matches(".*[A-Z][a-z]+.*"); // 驼峰命名
        result.put("followsNamingConvention", hasGoodNaming);
        
        // 代码异味检测
        boolean hasLongMethod = code.split("\n").length > 100;
        boolean hasMagicNumbers = code.matches(".*\\d{2,}.*");
        
        result.put("hasLongMethod", hasLongMethod);
        result.put("hasPotentialMagicNumbers", hasMagicNumbers);
        
        return result;
    }
    
    /**
     * 格式化分析结果
     */
    private String formatResult(Map<String, Object> analysisResult) {
        StringBuilder sb = new StringBuilder();
        sb.append("Code Analysis Results:\n");
        
        for (Map.Entry<String, Object> entry : analysisResult.entrySet()) {
            sb.append("\n").append(entry.getKey().toUpperCase()).append(":\n");
            
            @SuppressWarnings("unchecked")
            Map<String, Object> section = (Map<String, Object>) entry.getValue();
            for (Map.Entry<String, Object> item : section.entrySet()) {
                sb.append("  ").append(item.getKey()).append(": ").append(item.getValue()).append("\n");
            }
        }
        
        return sb.toString();
    }
    
    /**
     * 统计字符串出现次数
     */
    private int countOccurrences(String text, String pattern) {
        int count = 0;
        int index = 0;
        
        while ((index = text.indexOf(pattern, index)) != -1) {
            count++;
            index += pattern.length();
        }
        
        return count;
    }
}
