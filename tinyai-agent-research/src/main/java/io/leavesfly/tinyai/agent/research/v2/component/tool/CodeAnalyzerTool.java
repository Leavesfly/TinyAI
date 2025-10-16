package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 代码分析工具
 * 分析代码结构和特征
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class CodeAnalyzerTool implements Tool {
    
    @Override
    public String getName() {
        return "code_analyze";
    }
    
    @Override
    public String getDescription() {
        return "分析代码结构和特征";
    }
    
    @Override
    public String getCategory() {
        return "code";
    }
    
    @Override
    public ToolParameters getParameters() {
        ToolParameters params = new ToolParameters();
        params.addParameter(new ToolParameters.Parameter(
            "code", "string", "要分析的代码", true
        ));
        params.addParameter(new ToolParameters.Parameter(
            "language", "string", "编程语言", false, "java"
        ));
        return params;
    }
    
    @Override
    public boolean validate(Map<String, Object> parameters) {
        if (parameters == null || !parameters.containsKey("code")) {
            return false;
        }
        
        String code = parameters.get("code").toString();
        return code != null && !code.trim().isEmpty();
    }
    
    @Override
    public ToolResult execute(Map<String, Object> parameters) {
        long startTime = System.currentTimeMillis();
        
        try {
            String code = parameters.get("code").toString();
            String language = parameters.containsKey("language") 
                ? parameters.get("language").toString() 
                : "java";
            
            // 基础代码分析
            CodeAnalysis analysis = analyzeCode(code, language);
            
            ToolResult result = ToolResult.success("代码分析完成");
            result.putData("language", language);
            result.putData("lineCount", analysis.lineCount);
            result.putData("classCount", analysis.classCount);
            result.putData("methodCount", analysis.methodCount);
            result.putData("commentLines", analysis.commentLines);
            result.putData("complexity", analysis.complexity);
            result.setExecutionTimeMs(System.currentTimeMillis() - startTime);
            
            return result;
            
        } catch (Exception e) {
            return ToolResult.error("代码分析失败: " + e.getMessage());
        }
    }
    
    /**
     * 分析代码
     */
    private CodeAnalysis analyzeCode(String code, String language) {
        CodeAnalysis analysis = new CodeAnalysis();
        
        String[] lines = code.split("\n");
        analysis.lineCount = lines.length;
        
        // Java代码分析
        if ("java".equalsIgnoreCase(language)) {
            // 统计类数量
            Pattern classPattern = Pattern.compile("\\b(class|interface|enum)\\s+\\w+");
            Matcher classMatcher = classPattern.matcher(code);
            while (classMatcher.find()) {
                analysis.classCount++;
            }
            
            // 统计方法数量
            Pattern methodPattern = Pattern.compile("\\b(public|private|protected)?\\s*\\w+\\s+\\w+\\s*\\(");
            Matcher methodMatcher = methodPattern.matcher(code);
            while (methodMatcher.find()) {
                analysis.methodCount++;
            }
            
            // 统计注释行
            for (String line : lines) {
                String trimmed = line.trim();
                if (trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith("*")) {
                    analysis.commentLines++;
                }
            }
            
            // 简单复杂度估算（基于分支语句）
            Pattern branchPattern = Pattern.compile("\\b(if|else|for|while|switch|case)\\b");
            Matcher branchMatcher = branchPattern.matcher(code);
            while (branchMatcher.find()) {
                analysis.complexity++;
            }
        }
        
        return analysis;
    }
    
    /**
     * 代码分析结果
     */
    private static class CodeAnalysis {
        int lineCount = 0;
        int classCount = 0;
        int methodCount = 0;
        int commentLines = 0;
        int complexity = 1; // 基础复杂度为1
    }
}
