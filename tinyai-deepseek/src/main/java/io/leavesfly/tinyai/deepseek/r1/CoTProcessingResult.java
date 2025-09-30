package io.leavesfly.tinyai.deepseek.r1;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * 思维链处理结果类
 * 
 * 封装思维链提示处理的完整结果，包括原始问题、
 * 使用的策略、生成的提示、模型结果和分析信息。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CoTProcessingResult implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private String originalQuestion;                             // 原始问题
    private ChainOfThoughtPrompting.ReasoningStrategy strategy;  // 使用的推理策略
    private String generatedPrompt;                              // 生成的提示
    private DeepSeekR1Result modelResult;                        // 模型推理结果
    private long processingTime;                                 // 处理时间（毫秒）
    private Map<String, String> analysis;                        // 分析结果
    private boolean isError;                                     // 是否出错
    private String errorMessage;                                 // 错误信息
    
    /**
     * 默认构造函数
     */
    public CoTProcessingResult() {
        this.analysis = new HashMap<>();
        this.isError = false;
    }
    
    /**
     * 添加分析信息
     * 
     * @param key 分析项
     * @param value 分析结果
     */
    public void addAnalysis(String key, String value) {
        analysis.put(key, value);
    }
    
    /**
     * 设置错误状态
     * 
     * @param errorMessage 错误信息
     */
    public void setError(String errorMessage) {
        this.isError = true;
        this.errorMessage = errorMessage;
    }
    
    /**
     * 获取详细报告
     * 
     * @return 格式化的处理报告
     */
    public String getDetailedReport() {
        StringBuilder report = new StringBuilder();
        
        report.append("=== 思维链处理报告 ===\n\n");
        
        // 基本信息
        report.append("原始问题: ").append(originalQuestion).append("\n");
        report.append("推理策略: ").append(strategy).append("\n");
        report.append("处理状态: ").append(isError ? "失败" : "成功").append("\n");
        report.append("处理时间: ").append(processingTime).append(" ms\n\n");
        
        // 错误信息
        if (isError && errorMessage != null) {
            report.append("错误信息: ").append(errorMessage).append("\n\n");
        }
        
        // 生成的提示
        if (generatedPrompt != null) {
            report.append("生成的提示:\n");
            report.append("```\n").append(generatedPrompt).append("\n```\n\n");
        }
        
        // 模型结果
        if (modelResult != null && !isError) {
            report.append("模型推理结果:\n");
            report.append(modelResult.getSummary()).append("\n\n");
            
            if (modelResult.getReasoningChain() != null) {
                report.append("推理步骤:\n");
                report.append(modelResult.getReasoningStepsSummary()).append("\n\n");
            }
            
            if (modelResult.getReflectionResult() != null) {
                report.append("质量评估:\n");
                ReflectionModule.ReflectionResult reflection = modelResult.getReflectionResult();
                report.append(String.format("  质量分数: %.3f\n", reflection.getQualityScore()));
                report.append(String.format("  一致性分数: %.3f\n", reflection.getConsistencyScore()));
                report.append(String.format("  风险分数: %.3f\n", reflection.getRiskScore()));
                report.append(String.format("  需要改进: %s\n", reflection.isNeedsRefinement() ? "是" : "否"));
                report.append("\n");
            }
        }
        
        // 分析结果
        if (!analysis.isEmpty()) {
            report.append("分析结果:\n");
            for (Map.Entry<String, String> entry : analysis.entrySet()) {
                report.append(String.format("  %s: %s\n", entry.getKey(), entry.getValue()));
            }
            report.append("\n");
        }
        
        report.append("========================");
        
        return report.toString();
    }
    
    /**
     * 获取简要摘要
     * 
     * @return 结果摘要
     */
    public String getSummary() {
        if (isError) {
            return String.format("CoT处理失败: %s (策略: %s, 用时: %d ms)", 
                               errorMessage, strategy, processingTime);
        }
        
        String qualityLevel = "未知";
        double confidence = 0.0;
        
        if (modelResult != null) {
            confidence = modelResult.getFinalConfidence();
            qualityLevel = modelResult.getQualityLevel();
        }
        
        return String.format("CoT处理成功: 策略=%s, 质量=%s, 置信度=%.3f, 用时=%d ms",
                           strategy, qualityLevel, confidence, processingTime);
    }
    
    /**
     * 获取推理效果评级
     * 
     * @return 效果评级
     */
    public String getEffectivenessRating() {
        if (isError || modelResult == null) {
            return "失败";
        }
        
        double confidence = modelResult.getFinalConfidence();
        boolean needsImprovement = modelResult.needsImprovement();
        
        if (confidence >= 0.9 && !needsImprovement) {
            return "优秀";
        } else if (confidence >= 0.7 && !needsImprovement) {
            return "良好";
        } else if (confidence >= 0.5) {
            return "一般";
        } else {
            return "较差";
        }
    }
    
    /**
     * 判断是否成功且高质量
     * 
     * @return 如果成功且高质量则返回true
     */
    public boolean isHighQuality() {
        return !isError && 
               modelResult != null && 
               modelResult.getFinalConfidence() >= 0.8 && 
               !modelResult.needsImprovement();
    }
    
    /**
     * 获取推理步骤数量
     * 
     * @return 推理步骤数，如果无效则返回0
     */
    public int getReasoningStepCount() {
        if (modelResult == null || modelResult.getReasoningChain() == null) {
            return 0;
        }
        return modelResult.getReasoningChain().getStepCount();
    }
    
    /**
     * 获取最终置信度
     * 
     * @return 最终置信度，如果无效则返回0
     */
    public double getFinalConfidence() {
        if (modelResult == null) {
            return 0.0;
        }
        return modelResult.getFinalConfidence();
    }
    
    /**
     * 获取改进建议
     * 
     * @return 改进建议文本
     */
    public String getImprovementSuggestions() {
        if (isError) {
            return "处理失败，建议检查输入问题或模型配置";
        }
        
        if (modelResult == null) {
            return "无可用结果";
        }
        
        StringBuilder suggestions = new StringBuilder();
        
        // 基于置信度的建议
        double confidence = modelResult.getFinalConfidence();
        if (confidence < 0.5) {
            suggestions.append("置信度较低，建议：1) 尝试不同的推理策略；2) 增加问题上下文信息；");
        }
        
        // 基于推理步骤的建议
        int stepCount = getReasoningStepCount();
        if (stepCount < 3) {
            suggestions.append("推理步骤较少，建议：1) 使用更详细的提示模板；2) 鼓励模型深入思考；");
        } else if (stepCount > 10) {
            suggestions.append("推理步骤过多，建议：1) 优化提示以提高效率；2) 设置步骤数量限制；");
        }
        
        // 基于反思结果的建议
        if (modelResult.needsImprovement()) {
            suggestions.append("模型建议改进，具体：").append(modelResult.getImprovementSuggestions()).append("；");
        }
        
        if (suggestions.length() == 0) {
            return "推理质量良好，继续保持当前策略";
        }
        
        return suggestions.toString();
    }
    
    /**
     * 导出为JSON格式（简化版）
     * 
     * @return JSON字符串
     */
    public String toJsonString() {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"originalQuestion\": \"").append(escapeJson(originalQuestion)).append("\",\n");
        json.append("  \"strategy\": \"").append(strategy).append("\",\n");
        json.append("  \"processingTime\": ").append(processingTime).append(",\n");
        json.append("  \"isError\": ").append(isError).append(",\n");
        
        if (isError && errorMessage != null) {
            json.append("  \"errorMessage\": \"").append(escapeJson(errorMessage)).append("\",\n");
        }
        
        if (modelResult != null && !isError) {
            json.append("  \"finalConfidence\": ").append(modelResult.getFinalConfidence()).append(",\n");
            json.append("  \"stepCount\": ").append(getReasoningStepCount()).append(",\n");
            json.append("  \"qualityLevel\": \"").append(modelResult.getQualityLevel()).append("\",\n");
            json.append("  \"needsImprovement\": ").append(modelResult.needsImprovement()).append(",\n");
        }
        
        json.append("  \"effectivenessRating\": \"").append(getEffectivenessRating()).append("\"\n");
        json.append("}");
        
        return json.toString();
    }
    
    /**
     * 转义JSON字符串
     */
    private String escapeJson(String input) {
        if (input == null) {
            return "";
        }
        return input.replace("\"", "\\\"")
                   .replace("\n", "\\n")
                   .replace("\r", "\\r")
                   .replace("\t", "\\t");
    }
    
    // Getter和Setter方法
    
    public String getOriginalQuestion() {
        return originalQuestion;
    }
    
    public void setOriginalQuestion(String originalQuestion) {
        this.originalQuestion = originalQuestion;
    }
    
    public ChainOfThoughtPrompting.ReasoningStrategy getStrategy() {
        return strategy;
    }
    
    public void setStrategy(ChainOfThoughtPrompting.ReasoningStrategy strategy) {
        this.strategy = strategy;
    }
    
    public String getGeneratedPrompt() {
        return generatedPrompt;
    }
    
    public void setGeneratedPrompt(String generatedPrompt) {
        this.generatedPrompt = generatedPrompt;
    }
    
    public DeepSeekR1Result getModelResult() {
        return modelResult;
    }
    
    public void setModelResult(DeepSeekR1Result modelResult) {
        this.modelResult = modelResult;
    }
    
    public long getProcessingTime() {
        return processingTime;
    }
    
    public void setProcessingTime(long processingTime) {
        this.processingTime = processingTime;
    }
    
    public Map<String, String> getAnalysis() {
        return new HashMap<>(analysis);
    }
    
    public void setAnalysis(Map<String, String> analysis) {
        this.analysis = new HashMap<>(analysis);
    }
    
    public boolean isError() {
        return isError;
    }
    
    public void setError(boolean error) {
        isError = error;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    @Override
    public String toString() {
        return String.format("CoTProcessingResult[%s, strategy=%s, steps=%d, conf=%.3f, time=%dms]",
                           isError ? "ERROR" : "SUCCESS",
                           strategy,
                           getReasoningStepCount(),
                           getFinalConfidence(),
                           processingTime);
    }
}