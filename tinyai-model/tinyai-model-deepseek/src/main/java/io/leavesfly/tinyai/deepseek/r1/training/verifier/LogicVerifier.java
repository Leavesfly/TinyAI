package io.leavesfly.tinyai.deepseek.r1.training.verifier;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.HashSet;
import java.util.Set;

/**
 * 逻辑验证器
 * 
 * 支持验证类型：
 * 1. 逻辑推理链验证（前提 -> 结论）
 * 2. 布尔逻辑验证（True/False判断）
 * 3. 推理步骤完整性验证
 * 
 * 验证策略：
 * - 提取逻辑结论
 * - 验证推理步骤的连贯性
 * - 检查是否存在逻辑跳跃或矛盾
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LogicVerifier implements Verifier {
    
    // 逻辑关键词
    private static final String[] LOGIC_KEYWORDS = {
        "therefore", "因此", "thus", "hence", "so",
        "conclude", "conclusion", "结论", "得出",
        "implies", "推出", "导致"
    };
    
    // 布尔值模式
    private static final Pattern BOOL_PATTERN = Pattern.compile(
        "\\b(true|false|yes|no|correct|incorrect|对|错|是|否)\\b",
        Pattern.CASE_INSENSITIVE
    );
    
    // 匹配单词的正则表达式
    private static final Pattern WORD_PATTERN = Pattern.compile("\\b[a-zA-Z]+\\b");
    
    @Override
    public String getVerifierType() {
        return "logic";
    }
    
    /**
     * 验证逻辑推理
     * 
     * @param modelOutput 模型输出，包含推理过程和结论
     * @param groundTruth 正确的结论，例如 "true" 或 "A > B"
     * @return 验证结果
     */
    @Override
    public VerificationResult verify(String modelOutput, String groundTruth) {
        try {
            // 1. 提取模型的结论
            String conclusion = extractAnswer(modelOutput);
            
            // 2. 规范化结论
            String normalizedConclusion = normalizeLogicStatement(conclusion);
            String normalizedTruth = normalizeLogicStatement(groundTruth);
            
            // 3. 比较结论
            boolean isCorrect = normalizedConclusion.equals(normalizedTruth);
            
            // 4. 如果直接匹配失败，尝试语义匹配
            if (!isCorrect) {
                isCorrect = semanticMatch(normalizedConclusion, normalizedTruth);
            }
            
            // 5. 验证推理完整性（额外检查）
            boolean hasValidReasoning = checkReasoningValidity(modelOutput);
            
            String details = String.format(
                "结论匹配: %s | 推理有效性: %s\n提取结论: %s\n期望结论: %s",
                isCorrect ? "✓" : "✗",
                hasValidReasoning ? "✓" : "✗",
                normalizedConclusion,
                normalizedTruth
            );
            
            return new VerificationResult(
                isCorrect && hasValidReasoning,
                normalizedConclusion,
                normalizedTruth,
                details
            );
            
        } catch (Exception e) {
            return new VerificationResult(
                false,
                "提取失败",
                groundTruth,
                "无法从输出中提取有效逻辑结论: " + e.getMessage()
            );
        }
    }
    
    /**
     * 从模型输出中提取答案（逻辑结论）
     * 
     * 策略：
     * 1. 查找逻辑关键词后的内容
     * 2. 提取布尔值
     * 3. 提取最后的陈述句
     * 
     * @param modelOutput 模型输出
     * @return 提取的结论
     */
    @Override
    public String extractAnswer(String modelOutput) {
        if (modelOutput == null || modelOutput.trim().isEmpty()) {
            throw new IllegalArgumentException("模型输出为空");
        }
        
        String lowerOutput = modelOutput.toLowerCase();
        
        // 策略1: 查找逻辑关键词后的内容
        for (String keyword : LOGIC_KEYWORDS) {
            int pos = lowerOutput.lastIndexOf(keyword.toLowerCase());
            if (pos != -1) {
                String afterKeyword = modelOutput.substring(pos + keyword.length()).trim();
                // 提取第一句话（到句号或换行）
                String[] parts = afterKeyword.split("[.。\n]");
                if (parts.length > 0 && !parts[0].trim().isEmpty()) {
                    return parts[0].trim();
                }
            }
        }
        
        // 策略2: 提取布尔值
        Matcher boolMatcher = BOOL_PATTERN.matcher(modelOutput);
        String lastBool = null;
        while (boolMatcher.find()) {
            lastBool = boolMatcher.group(1);
        }
        if (lastBool != null) {
            return lastBool;
        }
        
        // 策略3: 提取最后一句话
        String[] sentences = modelOutput.split("[.。!！]");
        if (sentences.length > 0) {
            return sentences[sentences.length - 1].trim();
        }
        
        throw new IllegalArgumentException("未找到有效逻辑结论");
    }
    
    /**
     * 规范化逻辑语句
     * 
     * @param statement 原始语句
     * @return 规范化后的语句
     */
    private String normalizeLogicStatement(String statement) {
        if (statement == null) return "";
        
        String normalized = statement.toLowerCase().trim();
        
        // 布尔值规范化
        normalized = normalized.replaceAll("\\b(yes|correct|对|是)\\b", "true");
        normalized = normalized.replaceAll("\\b(no|incorrect|错|否)\\b", "false");
        
        // 去除多余空白
        normalized = normalized.replaceAll("\\s+", " ");
        
        return normalized;
    }
    
    /**
     * 语义匹配
     * 
     * 当字面匹配失败时，尝试语义级别的匹配
     * 
     * @param conclusion 提取的结论
     * @param truth 标准答案
     * @return 是否语义匹配
     */
    private boolean semanticMatch(String conclusion, String truth) {
        // 移除连接词和冠词
        Set<String> stopWords = new HashSet<>();
        stopWords.add("the");
        stopWords.add("a");
        stopWords.add("an");
        stopWords.add("is");
        stopWords.add("are");
        
        String[] conclusionWords = conclusion.split("\\s+");
        String[] truthWords = truth.split("\\s+");
        
        Set<String> conclusionSet = new HashSet<>();
        Set<String> truthSet = new HashSet<>();
        
        for (String word : conclusionWords) {
            if (!stopWords.contains(word) && !word.isEmpty()) {
                conclusionSet.add(word);
            }
        }
        
        for (String word : truthWords) {
            if (!stopWords.contains(word) && !word.isEmpty()) {
                truthSet.add(word);
            }
        }
        
        // 计算Jaccard相似度
        Set<String> intersection = new HashSet<>(conclusionSet);
        intersection.retainAll(truthSet);
        
        Set<String> union = new HashSet<>(conclusionSet);
        union.addAll(truthSet);
        
        if (union.isEmpty()) return false;
        
        double similarity = (double) intersection.size() / union.size();
        return similarity >= 0.7; // 70% 相似度阈值
    }
    
    /**
     * 检查推理有效性
     * 
     * 验证推理过程是否包含必要的步骤和逻辑连接
     * 
     * @param reasoning 推理过程
     * @return 是否有效
     */
    private boolean checkReasoningValidity(String reasoning) {
        if (reasoning == null || reasoning.trim().isEmpty()) {
            return false;
        }
        
        // 检查1: 是否包含推理关键词
        boolean hasLogicKeywords = false;
        for (String keyword : LOGIC_KEYWORDS) {
            if (reasoning.toLowerCase().contains(keyword.toLowerCase())) {
                hasLogicKeywords = true;
                break;
            }
        }
        
        // 检查2: 是否有足够的推理步骤（至少2句话）
        String[] sentences = reasoning.split("[.。!！\n]");
        boolean hasSufficientSteps = sentences.length >= 2;
        
        // 检查3: 是否包含因果连接词
        String[] causalWords = {"because", "since", "as", "因为", "由于", "所以"};
        boolean hasCausalConnection = false;
        for (String word : causalWords) {
            if (reasoning.toLowerCase().contains(word)) {
                hasCausalConnection = true;
                break;
            }
        }
        
        // 至少满足2个条件
        int validCount = 0;
        if (hasLogicKeywords) validCount++;
        if (hasSufficientSteps) validCount++;
        if (hasCausalConnection) validCount++;
        
        return validCount >= 2;
    }
    
    /**
     * 验证三段论推理
     * 
     * @param premise1 前提1
     * @param premise2 前提2
     * @param conclusion 结论
     * @return 是否有效
     */
    public boolean verifySyllogism(String premise1, String premise2, String conclusion) {
        // 简化版三段论验证
        // 实际应用中需要更复杂的逻辑推理引擎
        
        String normalized1 = normalizeLogicStatement(premise1);
        String normalized2 = normalizeLogicStatement(premise2);
        String normalizedConc = normalizeLogicStatement(conclusion);
        
        // 检查是否存在共同元素（中项）
        String[] words1 = normalized1.split("\\s+");
        String[] words2 = normalized2.split("\\s+");
        String[] wordsConc = normalizedConc.split("\\s+");
        
        Set<String> common12 = new HashSet<>();
        for (String w1 : words1) {
            for (String w2 : words2) {
                if (w1.equals(w2) && w1.length() > 2) {
                    common12.add(w1);
                }
            }
        }
        
        // 结论应包含前提的部分元素
        boolean conclusionRelated = false;
        for (String word : wordsConc) {
            if (word.length() > 2) {
                for (String w1 : words1) {
                    if (word.equals(w1)) {
                        conclusionRelated = true;
                        break;
                    }
                }
            }
        }
        
        return !common12.isEmpty() && conclusionRelated;
    }
}