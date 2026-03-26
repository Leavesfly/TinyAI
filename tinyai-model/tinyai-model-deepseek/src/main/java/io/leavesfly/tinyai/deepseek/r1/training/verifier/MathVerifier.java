package io.leavesfly.tinyai.deepseek.r1.training.verifier;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 数学验证器
 * 
 * 支持验证类型：
 * 1. 算术运算（加减乘除）
 * 2. 代数方程（简单一元方程）
 * 3. 数值比较（带容差）
 * 
 * 验证策略：
 * - 从模型输出中提取数值答案
 * - 与标准答案进行数值比较
 * - 允许小误差（默认1e-6）
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MathVerifier implements Verifier {
    
    private static final double TOLERANCE = 1e-6;
    
    // 匹配数字的正则表达式（支持整数、小数、负数、科学计数法）
    private static final Pattern NUMBER_PATTERN = Pattern.compile(
        "-?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?"
    );
    
    // 匹配答案模式的正则表达式
    private static final Pattern ANSWER_PATTERN = Pattern.compile(
        "(?:answer|答案|结果)(?:\\s+is)?\\s*[:=]?\\s*(-?\\d+\\.?\\d*)",
        Pattern.CASE_INSENSITIVE
    );
    
    // 匹配简单方程的正则表达式
    private static final Pattern EQUATION_PATTERN = Pattern.compile(
        "(-?\\d+\\.?\\d*)\\s*\\*\\s*x\\s*([+-])\\s*(-?\\d+\\.?\\d*)\\s*=\\s*(-?\\d+\\.?\\d*)"
    );
    
    @Override
    public String getVerifierType() {
        return "math";
    }
    
    /**
     * 验证数学输出
     * 
     * @param modelOutput 模型输出，例如："Let me solve this step by step... The answer is 42."
     * @param groundTruth 标准答案，例如："42" 或 "42.0"
     * @return 验证结果
     */
    @Override
    public VerificationResult verify(String modelOutput, String groundTruth) {
        try {
            // 1. 提取模型预测的答案
            String predictedAnswer = extractAnswer(modelOutput);
            
            // 2. 转换为数值
            double predicted = parseNumber(predictedAnswer);
            double expected = parseNumber(groundTruth);
            
            // 3. 数值比较（带容差）
            boolean isCorrect = Math.abs(predicted - expected) < TOLERANCE;
            
            // 4. 构建验证详情
            String details = String.format(
                "数值比较: |%.6f - %.6f| = %.6e %s %.6e",
                predicted, expected, 
                Math.abs(predicted - expected),
                isCorrect ? "<" : ">=",
                TOLERANCE
            );
            
            return new VerificationResult(
                isCorrect,
                String.valueOf(predicted),
                String.valueOf(expected),
                details
            );
            
        } catch (NumberFormatException e) {
            return new VerificationResult(
                false,
                "解析失败",
                groundTruth,
                "无法从输出中提取有效数值: " + e.getMessage()
            );
        }
    }
    
    /**
     * 从模型输出中提取答案
     * 
     * 策略：
     * 1. 优先查找 "answer is X" 或 "答案是 X" 模式
     * 2. 否则提取最后一个出现的数字
     * 
     * @param modelOutput 模型输出
     * @return 提取的答案字符串
     */
    @Override
    public String extractAnswer(String modelOutput) {
        if (modelOutput == null || modelOutput.trim().isEmpty()) {
            throw new NumberFormatException("模型输出为空");
        }
        
        // 策略1: 查找 "answer is X" 模式
        Matcher answerMatcher = ANSWER_PATTERN.matcher(modelOutput);
        if (answerMatcher.find()) {
            return answerMatcher.group(1);
        }
        
        // 策略2: 提取最后一个数字
        Matcher numberMatcher = NUMBER_PATTERN.matcher(modelOutput);
        String lastNumber = null;
        while (numberMatcher.find()) {
            lastNumber = numberMatcher.group();
        }
        
        if (lastNumber != null) {
            return lastNumber;
        }
        
        throw new NumberFormatException("未找到有效数字");
    }
    
    /**
     * 解析数字字符串
     * 
     * @param numberStr 数字字符串
     * @return double值
     * @throws NumberFormatException 如果无法解析
     */
    private double parseNumber(String numberStr) throws NumberFormatException {
        if (numberStr == null || numberStr.trim().isEmpty()) {
            throw new NumberFormatException("数字字符串为空");
        }
        String str = numberStr.trim().toLowerCase();
        // 支持布尔值
        if (str.equals("true") || str.equals("yes")) {
            return 1.0;
        }
        if (str.equals("false") || str.equals("no")) {
            return 0.0;
        }
        return Double.parseDouble(str);
    }
    
    /**
     * 验证代数方程解
     * 
     * 通过代入法验证解是否满足原方程
     * 例如：方程 2x + 5 = 13，解 x = 4，验证 2*4 + 5 == 13
     * 
     * @param equation 方程字符串，例如 "2*x + 5 = 13"
     * @param solution 解，例如 "4"
     * @return 是否正确
     */
    public boolean verifyEquationSolution(String equation, String solution) {
        try {
            double x = parseNumber(solution);
            
            // 简单的代入验证（仅支持形如 "a*x + b = c" 的方程）
            Matcher matcher = EQUATION_PATTERN.matcher(equation.replace(" ", ""));
            
            if (matcher.find()) {
                double a = parseNumber(matcher.group(1));
                String op = matcher.group(2);
                double b = parseNumber(matcher.group(3));
                double c = parseNumber(matcher.group(4));
                
                double leftSide = a * x + (op.equals("+") ? b : -b);
                return Math.abs(leftSide - c) < TOLERANCE;
            }
            
            return false;
        } catch (Exception e) {
            return false;
        }
    }
}