package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 思维链提示处理器
 * 
 * 专门处理思维链(Chain-of-Thought)提示，能够将复杂问题
 * 分解为多个推理步骤，并生成结构化的推理提示。
 * 
 * 功能特性：
 * 1. 自动生成思维链提示模板
 * 2. 问题分解和步骤规划
 * 3. 推理过程引导
 * 4. 多种推理策略支持
 * 5. 自适应提示优化
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ChainOfThoughtPrompting {
    
    private DeepSeekR1Model model;         // DeepSeek R1模型
    private Map<String, String> templates; // 提示模板
    private SimpleTokenizer tokenizer;     // 简单分词器
    private int maxPromptLength;           // 最大提示长度
    
    // 推理策略
    public enum ReasoningStrategy {
        STEP_BY_STEP,      // 逐步推理
        PROBLEM_SOLVING,   // 问题解决
        ANALYTICAL,        // 分析式推理
        COMPARATIVE,       // 比较式推理
        DEDUCTIVE,         // 演绎推理
        INDUCTIVE          // 归纳推理
    }
    
    /**
     * 构造思维链提示处理器
     * 
     * @param model DeepSeek R1模型
     */
    public ChainOfThoughtPrompting(DeepSeekR1Model model) {
        this.model = model;
        this.templates = new HashMap<>();
        this.tokenizer = new SimpleTokenizer();
        this.maxPromptLength = 512;
        
        initializeTemplates();
    }
    
    /**
     * 设置最大提示长度
     */
    public void setMaxPromptLength(int maxPromptLength) {
        this.maxPromptLength = maxPromptLength;
    }
    
    /**
     * 初始化提示模板
     */
    private void initializeTemplates() {
        // 逐步推理模板
        templates.put("STEP_BY_STEP", 
            "问题: {question}\n\n" +
            "让我逐步分析这个问题：\n\n" +
            "步骤 1: 理解问题\n" +
            "步骤 2: 分析关键信息\n" +
            "步骤 3: 制定解决方案\n" +
            "步骤 4: 验证答案\n" +
            "步骤 5: 得出结论\n\n" +
            "详细推理过程：\n"
        );
        
        // 问题解决模板
        templates.put("PROBLEM_SOLVING",
            "问题: {question}\n\n" +
            "解题思路：\n" +
            "1. 问题分析：这是什么类型的问题？\n" +
            "2. 信息收集：已知条件有哪些？\n" +
            "3. 方法选择：应该用什么方法解决？\n" +
            "4. 解题过程：按步骤执行\n" +
            "5. 结果验证：检查答案是否合理\n\n" +
            "开始解答：\n"
        );
        
        // 分析式推理模板
        templates.put("ANALYTICAL",
            "分析问题: {question}\n\n" +
            "分析框架：\n" +
            "• 现象观察：观察到什么？\n" +
            "• 原因分析：为什么会这样？\n" +
            "• 影响评估：会产生什么影响？\n" +
            "• 解决方案：如何解决或改进？\n" +
            "• 总结结论：得出什么结论？\n\n" +
            "详细分析：\n"
        );
        
        // 比较式推理模板
        templates.put("COMPARATIVE",
            "比较分析: {question}\n\n" +
            "比较维度：\n" +
            "1. 相似点：有什么共同特征？\n" +
            "2. 差异点：主要区别是什么？\n" +
            "3. 优劣分析：各有什么优缺点？\n" +
            "4. 适用场景：分别适用于什么情况？\n" +
            "5. 综合评价：总体如何评价？\n\n" +
            "比较分析：\n"
        );
        
        // 演绎推理模板
        templates.put("DEDUCTIVE",
            "演绎推理: {question}\n\n" +
            "推理结构：\n" +
            "大前提：普遍规律或原理\n" +
            "小前提：具体情况或条件\n" +
            "结论：逻辑推导的结果\n\n" +
            "推理过程：\n" +
            "1. 确定适用的普遍规律\n" +
            "2. 分析具体情况\n" +
            "3. 应用逻辑推理\n" +
            "4. 得出结论\n\n" +
            "开始推理：\n"
        );
        
        // 归纳推理模板
        templates.put("INDUCTIVE",
            "归纳推理: {question}\n\n" +
            "归纳步骤：\n" +
            "1. 观察现象：收集具体事例\n" +
            "2. 寻找规律：识别共同特征\n" +
            "3. 形成假设：提出一般性结论\n" +
            "4. 验证假设：检验结论的有效性\n" +
            "5. 完善理论：修正和完善结论\n\n" +
            "归纳分析：\n"
        );
    }
    
    /**
     * 生成思维链提示
     * 
     * @param question 问题
     * @param strategy 推理策略
     * @return 生成的提示
     */
    public String generateCoTPrompt(String question, ReasoningStrategy strategy) {
        String template = templates.get(strategy.name());
        if (template == null) {
            template = templates.get("STEP_BY_STEP"); // 默认策略
        }
        
        return template.replace("{question}", question);
    }
    
    /**
     * 自动选择推理策略
     * 
     * @param question 问题
     * @return 推荐的推理策略
     */
    public ReasoningStrategy selectStrategy(String question) {
        String lowerQuestion = question.toLowerCase();
        
        // 基于关键词的简单策略选择
        if (lowerQuestion.contains("比较") || lowerQuestion.contains("对比") || 
            lowerQuestion.contains("区别") || lowerQuestion.contains("相同")) {
            return ReasoningStrategy.COMPARATIVE;
        }
        
        if (lowerQuestion.contains("分析") || lowerQuestion.contains("原因") || 
            lowerQuestion.contains("影响") || lowerQuestion.contains("为什么")) {
            return ReasoningStrategy.ANALYTICAL;
        }
        
        if (lowerQuestion.contains("解决") || lowerQuestion.contains("方法") || 
            lowerQuestion.contains("如何") || lowerQuestion.contains("怎么")) {
            return ReasoningStrategy.PROBLEM_SOLVING;
        }
        
        if (lowerQuestion.contains("规律") || lowerQuestion.contains("总结") || 
            lowerQuestion.contains("归纳") || lowerQuestion.contains("概括")) {
            return ReasoningStrategy.INDUCTIVE;
        }
        
        if (lowerQuestion.contains("推导") || lowerQuestion.contains("证明") || 
            lowerQuestion.contains("依据") || lowerQuestion.contains("根据")) {
            return ReasoningStrategy.DEDUCTIVE;
        }
        
        // 默认使用逐步推理
        return ReasoningStrategy.STEP_BY_STEP;
    }
    
    /**
     * 使用思维链处理问题
     * 
     * @param question 问题
     * @return 处理结果
     */
    public CoTProcessingResult processWithCoT(String question) {
        return processWithCoT(question, null);
    }
    
    /**
     * 使用指定策略处理问题
     * 
     * @param question 问题
     * @param strategy 推理策略（可选）
     * @return 处理结果
     */
    public CoTProcessingResult processWithCoT(String question, ReasoningStrategy strategy) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 1. 选择推理策略
            if (strategy == null) {
                strategy = selectStrategy(question);
            }
            
            // 2. 生成思维链提示
            String cotPrompt = generateCoTPrompt(question, strategy);
            
            // 3. 将提示转换为输入序列
            NdArray inputIds = tokenizePrompt(cotPrompt);
            
            // 4. 使用模型进行推理
            DeepSeekR1Result modelResult = model.performReasoning(inputIds, question);
            
            // 5. 构建处理结果
            CoTProcessingResult result = new CoTProcessingResult();
            result.setOriginalQuestion(question);
            result.setStrategy(strategy);
            result.setGeneratedPrompt(cotPrompt);
            result.setModelResult(modelResult);
            result.setProcessingTime(System.currentTimeMillis() - startTime);
            
            // 6. 分析推理质量
            analyzeReasoningQuality(result);
            
            return result;
            
        } catch (Exception e) {
            System.err.println("思维链处理出错: " + e.getMessage());
            
            CoTProcessingResult errorResult = new CoTProcessingResult();
            errorResult.setOriginalQuestion(question);
            errorResult.setStrategy(strategy);
            errorResult.setError(true);
            errorResult.setErrorMessage(e.getMessage());
            errorResult.setProcessingTime(System.currentTimeMillis() - startTime);
            
            return errorResult;
        }
    }
    
    /**
     * 分析推理质量
     */
    private void analyzeReasoningQuality(CoTProcessingResult result) {
        if (result.getModelResult() == null || 
            result.getModelResult().getReasoningChain() == null) {
            return;
        }
        
        ReasoningChain chain = result.getModelResult().getReasoningChain();
        
        // 评估推理深度
        int stepCount = chain.getStepCount();
        String depthLevel;
        if (stepCount >= 7) {
            depthLevel = "深度";
        } else if (stepCount >= 4) {
            depthLevel = "中等";
        } else {
            depthLevel = "浅层";
        }
        result.addAnalysis("推理深度", depthLevel);
        
        // 评估一致性
        double avgConfidence = chain.getAverageConfidence();
        double confidenceVariance = calculateConfidenceVariance(chain);
        
        String consistencyLevel;
        if (confidenceVariance < 0.1) {
            consistencyLevel = "高度一致";
        } else if (confidenceVariance < 0.2) {
            consistencyLevel = "基本一致";
        } else {
            consistencyLevel = "存在波动";
        }
        result.addAnalysis("一致性", consistencyLevel);
        
        // 评估效率
        long reasoningTime = chain.getDuration();
        String efficiencyLevel;
        if (reasoningTime < 100) {
            efficiencyLevel = "高效";
        } else if (reasoningTime < 500) {
            efficiencyLevel = "适中";
        } else {
            efficiencyLevel = "需要优化";
        }
        result.addAnalysis("推理效率", efficiencyLevel);
    }
    
    /**
     * 计算置信度方差
     */
    private double calculateConfidenceVariance(ReasoningChain chain) {
        List<ReasoningStep> steps = chain.getSteps();
        if (steps.size() < 2) {
            return 0.0;
        }
        
        double mean = chain.getAverageConfidence();
        double variance = 0.0;
        
        for (ReasoningStep step : steps) {
            double diff = step.getConfidence() - mean;
            variance += diff * diff;
        }
        
        return variance / steps.size();
    }
    
    /**
     * 将提示转换为token序列
     */
    private NdArray tokenizePrompt(String prompt) {
        // 简化的分词实现
        String[] tokens = prompt.split("\\s+");
        
        // 限制长度
        int maxTokens = Math.min(tokens.length, maxPromptLength);
        
        NdArray inputIds = NdArray.of(Shape.of(1, maxTokens));
        
        for (int i = 0; i < maxTokens; i++) {
            // 简化的token编码，实际应该使用词汇表
            int tokenId = Math.abs(tokens[i].hashCode()) % model.getVocabSize();
            inputIds.set(tokenId, 0, i);
        }
        
        return inputIds;
    }
    
    /**
     * 批量处理问题
     * 
     * @param questions 问题列表
     * @return 处理结果列表
     */
    public List<CoTProcessingResult> batchProcess(List<String> questions) {
        List<CoTProcessingResult> results = new ArrayList<>();
        
        System.out.printf("开始批量处理 %d 个问题...\n", questions.size());
        
        for (int i = 0; i < questions.size(); i++) {
            String question = questions.get(i);
            System.out.printf("处理问题 %d/%d: %s\n", 
                            i + 1, questions.size(), 
                            question.length() > 50 ? question.substring(0, 50) + "..." : question);
            
            CoTProcessingResult result = processWithCoT(question);
            results.add(result);
            
            System.out.printf("  结果: %s (用时: %d ms)\n", 
                            result.isError() ? "失败" : "成功", result.getProcessingTime());
        }
        
        System.out.println("批量处理完成!");
        printBatchSummary(results);
        
        return results;
    }
    
    /**
     * 打印批量处理摘要
     */
    private void printBatchSummary(List<CoTProcessingResult> results) {
        int successCount = 0;
        long totalTime = 0;
        Map<ReasoningStrategy, Integer> strategyCount = new HashMap<>();
        
        for (CoTProcessingResult result : results) {
            if (!result.isError()) {
                successCount++;
            }
            totalTime += result.getProcessingTime();
            
            strategyCount.put(result.getStrategy(), 
                            strategyCount.getOrDefault(result.getStrategy(), 0) + 1);
        }
        
        System.out.println("\n=== 批量处理摘要 ===");
        System.out.printf("总问题数: %d\n", results.size());
        System.out.printf("成功处理: %d (%.1f%%)\n", 
                        successCount, 100.0 * successCount / results.size());
        System.out.printf("总用时: %d ms\n", totalTime);
        System.out.printf("平均用时: %.1f ms\n", (double) totalTime / results.size());
        
        System.out.println("\n策略使用统计:");
        for (Map.Entry<ReasoningStrategy, Integer> entry : strategyCount.entrySet()) {
            System.out.printf("  %s: %d 次\n", entry.getKey(), entry.getValue());
        }
        System.out.println("===================");
    }
    
    /**
     * 优化提示模板
     * 
     * @param strategy 策略
     * @param newTemplate 新模板
     */
    public void updateTemplate(ReasoningStrategy strategy, String newTemplate) {
        templates.put(strategy.name(), newTemplate);
    }
    
    /**
     * 获取所有支持的策略
     * 
     * @return 策略列表
     */
    public List<ReasoningStrategy> getSupportedStrategies() {
        return Arrays.asList(ReasoningStrategy.values());
    }
    
    // Getter方法
    public DeepSeekR1Model getModel() { return model; }
    public int getMaxPromptLength() { return maxPromptLength; }
    
    @Override
    public String toString() {
        return String.format("ChainOfThoughtPrompting(strategies=%d, maxLength=%d)",
                           templates.size(), maxPromptLength);
    }
}

/**
 * 简单分词器
 */
class SimpleTokenizer {
    
    public String[] tokenize(String text) {
        // 简化的分词实现
        return text.toLowerCase()
                  .replaceAll("[^a-zA-Z0-9\\s\\u4e00-\\u9fa5]", " ")
                  .split("\\s+");
    }
    
    public String detokenize(String[] tokens) {
        return String.join(" ", tokens);
    }
}