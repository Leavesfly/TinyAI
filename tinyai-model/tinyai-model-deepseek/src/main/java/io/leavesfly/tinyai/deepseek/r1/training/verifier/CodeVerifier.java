package io.leavesfly.tinyai.deepseek.r1.training.verifier;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 代码验证器
 * 
 * 支持验证类型：
 * 1. 代码输出验证（通过预期输出比较）
 * 2. 代码逻辑验证（通过测试用例）
 * 3. 代码格式验证（基础语法检查）
 * 
 * 注意：由于安全考虑，本实现不实际执行代码
 * 而是通过模式匹配和规则验证来判断代码正确性
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CodeVerifier implements Verifier {
    
    // 匹配代码块的正则表达式
    private static final Pattern CODE_BLOCK_PATTERN = Pattern.compile(
        "```(?:java|python|javascript)?\\s*\\n(.*?)\\n```",
        Pattern.DOTALL
    );
    
    // 匹配变量名的正则表达式
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\b[a-zA-Z_][a-zA-Z0-9_]*\\b");
    
    @Override
    public String getVerifierType() {
        return "code";
    }
    
    /**
     * 验证代码输出
     * 
     * @param modelOutput 模型输出，包含代码和推理过程
     * @param groundTruth 期望的代码或输出结果
     * @return 验证结果
     */
    @Override
    public VerificationResult verify(String modelOutput, String groundTruth) {
        try {
            // 1. 提取代码
            String extractedCode = extractAnswer(modelOutput);
            
            // 2. 规范化代码（去除空白字符差异）
            String normalizedExtracted = normalizeCode(extractedCode);
            String normalizedExpected = normalizeCode(groundTruth);
            
            // 3. 比较代码
            boolean isCorrect = normalizedExtracted.equals(normalizedExpected);
            
            // 4. 如果完全匹配失败，尝试关键特征匹配
            if (!isCorrect) {
                isCorrect = matchKeyFeatures(normalizedExtracted, normalizedExpected);
            }
            
            String details = isCorrect 
                ? "代码验证通过"
                : String.format("代码不匹配\n期望:\n%s\n实际:\n%s", 
                    normalizedExpected, normalizedExtracted);
            
            return new VerificationResult(
                isCorrect,
                extractedCode,
                groundTruth,
                details
            );
            
        } catch (Exception e) {
            return new VerificationResult(
                false,
                "提取失败",
                groundTruth,
                "无法从输出中提取有效代码: " + e.getMessage()
            );
        }
    }
    
    /**
     * 从模型输出中提取代码
     * 
     * 策略：
     * 1. 优先提取代码块（```code```）
     * 2. 否则查找关键代码模式
     * 
     * @param modelOutput 模型输出
     * @return 提取的代码
     */
    @Override
    public String extractAnswer(String modelOutput) {
        if (modelOutput == null || modelOutput.trim().isEmpty()) {
            throw new IllegalArgumentException("模型输出为空");
        }
        
        // 策略1: 提取代码块
        Matcher matcher = CODE_BLOCK_PATTERN.matcher(modelOutput);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        
        // 策略2: 查找可能的代码片段（包含关键字）
        String[] codeKeywords = {"class", "def", "function", "public", "private", "return"};
        for (String keyword : codeKeywords) {
            if (modelOutput.contains(keyword)) {
                // 提取包含关键字的行及其上下文
                return extractCodeContext(modelOutput, keyword);
            }
        }
        
        throw new IllegalArgumentException("未找到有效代码");
    }
    
    /**
     * 规范化代码
     * 
     * 去除空白字符、注释等差异，便于比较
     * 
     * @param code 原始代码
     * @return 规范化后的代码
     */
    private String normalizeCode(String code) {
        if (code == null) return "";
        
        return code
            .replaceAll("//.*", "")           // 去除单行注释
            .replaceAll("/\\*.*?\\*/", "")    // 去除多行注释
            .replaceAll("\\s+", " ")          // 合并空白字符
            .trim();
    }
    
    /**
     * 提取代码上下文
     * 
     * @param text 文本
     * @param keyword 关键字
     * @return 代码片段
     */
    private String extractCodeContext(String text, String keyword) {
        String[] lines = text.split("\n");
        StringBuilder codeBuilder = new StringBuilder();
        boolean inCodeBlock = false;
        
        for (String line : lines) {
            if (line.contains(keyword)) {
                inCodeBlock = true;
            }
            if (inCodeBlock) {
                codeBuilder.append(line).append("\n");
                // 简单启发式：遇到空行可能结束
                if (line.trim().isEmpty() && codeBuilder.length() > 50) {
                    break;
                }
            }
        }
        
        return codeBuilder.toString().trim();
    }
    
    /**
     * 匹配关键特征
     * 
     * 当完全匹配失败时，检查关键代码特征是否匹配
     * 
     * @param extracted 提取的代码
     * @param expected 期望的代码
     * @return 是否匹配关键特征
     */
    private boolean matchKeyFeatures(String extracted, String expected) {
        // 特征1: 包含相同的关键字
        String[] keywords = {"class", "def", "function", "return", "if", "for", "while"};
        for (String keyword : keywords) {
            boolean extractedHas = extracted.contains(keyword);
            boolean expectedHas = expected.contains(keyword);
            if (extractedHas != expectedHas) {
                return false;
            }
        }
        
        // 特征2: 包含相同的变量名（简化版）
        Matcher extractedMatcher = VARIABLE_PATTERN.matcher(extracted);
        Matcher expectedMatcher = VARIABLE_PATTERN.matcher(expected);
        
        int extractedVarCount = 0;
        int expectedVarCount = 0;
        while (extractedMatcher.find()) extractedVarCount++;
        while (expectedMatcher.find()) expectedVarCount++;
        
        // 变量数量相近（±20%）
        return Math.abs(extractedVarCount - expectedVarCount) <= expectedVarCount * 0.2;
    }
    
    /**
     * 通过测试用例验证代码
     * 
     * @param code 代码
     * @param testInput 测试输入
     * @param expectedOutput 期望输出
     * @return 是否通过测试
     */
    public boolean verifyWithTestCase(String code, String testInput, String expectedOutput) {
        // 注意：实际执行代码存在安全风险，这里仅做模拟验证
        // 生产环境应使用沙箱环境或Docker容器
        
        // 简化版本：检查代码是否包含关键逻辑
        String normalizedCode = normalizeCode(code);
        
        // 启发式检查
        boolean hasInput = normalizedCode.contains(testInput) || 
                          normalizedCode.contains("input") ||
                          normalizedCode.contains("read");
        
        boolean hasOutput = normalizedCode.contains(expectedOutput) ||
                           normalizedCode.contains("print") ||
                           normalizedCode.contains("return");
        
        return hasInput && hasOutput;
    }
}