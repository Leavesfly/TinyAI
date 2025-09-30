package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * DeepSeek R1 推理结果类
 * 
 * 封装DeepSeek R1模型的完整输出，包括模型预测、
 * 推理链、反思结果和相关统计信息。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Result implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private NdArray modelOutput;                           // 模型输出
    private ReasoningChain reasoningChain;                 // 推理链
    private ReflectionModule.ReflectionResult reflectionResult; // 反思结果
    private double finalConfidence;                        // 最终置信度
    private String finalAnswer;                            // 最终答案
    private long processingTime;                           // 处理时间（毫秒）
    private Map<String, Object> metadata;                  // 元数据
    private boolean isSuccessful;                          // 是否成功
    private String errorMessage;                           // 错误信息（如果有）
    
    /**
     * 默认构造函数
     */
    public DeepSeekR1Result() {
        this.metadata = new HashMap<>();
        this.isSuccessful = true;
        this.processingTime = System.currentTimeMillis();
    }
    
    /**
     * 完整构造函数
     * 
     * @param modelOutput 模型输出
     * @param reasoningChain 推理链
     * @param reflectionResult 反思结果
     * @param finalConfidence 最终置信度
     */
    public DeepSeekR1Result(NdArray modelOutput, ReasoningChain reasoningChain,
                           ReflectionModule.ReflectionResult reflectionResult,
                           double finalConfidence) {
        this();
        this.modelOutput = modelOutput;
        this.reasoningChain = reasoningChain;
        this.reflectionResult = reflectionResult;
        this.finalConfidence = finalConfidence;
        
        if (reasoningChain != null) {
            this.finalAnswer = reasoningChain.getFinalAnswer();
        }
    }
    
    /**
     * 完成处理并记录时间
     */
    public void completeProcessing() {
        this.processingTime = System.currentTimeMillis() - this.processingTime;
    }
    
    /**
     * 设置错误状态
     * 
     * @param errorMessage 错误信息
     */
    public void setError(String errorMessage) {
        this.isSuccessful = false;
        this.errorMessage = errorMessage;
    }
    
    /**
     * 获取详细的推理报告
     * 
     * @return 格式化的推理报告
     */
    public String getDetailedReport() {
        StringBuilder report = new StringBuilder();
        
        report.append("=== DeepSeek R1 推理结果报告 ===\n\n");
        
        // 基本信息
        report.append("处理状态: ").append(isSuccessful ? "成功" : "失败").append("\n");
        if (!isSuccessful && errorMessage != null) {
            report.append("错误信息: ").append(errorMessage).append("\n");
        }
        report.append("处理时间: ").append(processingTime).append(" ms\n");
        report.append("最终置信度: ").append(String.format("%.3f", finalConfidence)).append("\n\n");
        
        // 最终答案
        if (finalAnswer != null) {
            report.append("最终答案:\n").append(finalAnswer).append("\n\n");
        }
        
        // 推理链详情
        if (reasoningChain != null) {
            report.append("推理链详情:\n");
            report.append(reasoningChain.getDetailedDescription()).append("\n");
        }
        
        // 反思结果
        if (reflectionResult != null) {
            report.append("自我反思:\n");
            report.append(reflectionResult.getReflectionReport()).append("\n");
        }
        
        // 统计信息
        report.append("统计信息:\n");
        if (reasoningChain != null) {
            report.append(String.format("  推理步骤数: %d\n", reasoningChain.getStepCount()));
            report.append(String.format("  平均置信度: %.3f\n", reasoningChain.getAverageConfidence()));
            report.append(String.format("  最低置信度: %.3f\n", reasoningChain.getMinConfidence()));
            report.append(String.format("  最高置信度: %.3f\n", reasoningChain.getMaxConfidence()));
            report.append(String.format("  推理时长: %d ms\n", reasoningChain.getDuration()));
        }
        
        // 元数据
        if (!metadata.isEmpty()) {
            report.append("\n元数据:\n");
            for (Map.Entry<String, Object> entry : metadata.entrySet()) {
                report.append(String.format("  %s: %s\n", entry.getKey(), entry.getValue()));
            }
        }
        
        report.append("\n================================");
        
        return report.toString();
    }
    
    /**
     * 获取简化摘要
     * 
     * @return 结果摘要
     */
    public String getSummary() {
        if (!isSuccessful) {
            return String.format("推理失败: %s", errorMessage);
        }
        
        return String.format(
            "推理成功 | 置信度: %.3f | 步骤: %d | 用时: %d ms | 质量: %s",
            finalConfidence,
            reasoningChain != null ? reasoningChain.getStepCount() : 0,
            processingTime,
            getQualityLevel()
        );
    }
    
    /**
     * 获取质量等级
     * 
     * @return 质量等级描述
     */
    public String getQualityLevel() {
        if (reflectionResult == null) {
            return "未知";
        }
        
        double qualityScore = reflectionResult.getQualityScore();
        if (qualityScore >= 0.9) {
            return "优秀";
        } else if (qualityScore >= 0.7) {
            return "良好";
        } else if (qualityScore >= 0.5) {
            return "一般";
        } else {
            return "较差";
        }
    }
    
    /**
     * 判断是否需要改进
     * 
     * @return 如果需要改进则返回true
     */
    public boolean needsImprovement() {
        if (reflectionResult == null) {
            return finalConfidence < 0.7;
        }
        return reflectionResult.isNeedsRefinement();
    }
    
    /**
     * 获取改进建议
     * 
     * @return 改进建议文本
     */
    public String getImprovementSuggestions() {
        if (reflectionResult == null) {
            return "无可用反思结果";
        }
        
        Map<String, String> analysis = reflectionResult.getAnalysis();
        return analysis.getOrDefault("改进建议", "无特殊建议");
    }
    
    /**
     * 获取推理步骤摘要
     * 
     * @return 推理步骤的简化描述
     */
    public String getReasoningStepsSummary() {
        if (reasoningChain == null || reasoningChain.getSteps().isEmpty()) {
            return "无推理步骤";
        }
        
        StringBuilder summary = new StringBuilder();
        for (ReasoningStep step : reasoningChain.getSteps()) {
            summary.append(String.format("步骤%d(%.3f) ", 
                         step.getStepIndex() + 1, step.getConfidence()));
        }
        
        return summary.toString().trim();
    }
    
    /**
     * 添加元数据
     * 
     * @param key 键
     * @param value 值
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    /**
     * 验证结果的完整性
     * 
     * @return 如果结果完整则返回true
     */
    public boolean isComplete() {
        return isSuccessful && 
               modelOutput != null && 
               reasoningChain != null && 
               reasoningChain.isCompleted();
    }
    
    /**
     * 导出为JSON格式（简化版）
     * 
     * @return JSON字符串
     */
    public String toJsonString() {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"successful\": ").append(isSuccessful).append(",\n");
        json.append("  \"finalConfidence\": ").append(finalConfidence).append(",\n");
        json.append("  \"processingTime\": ").append(processingTime).append(",\n");
        
        if (finalAnswer != null) {
            json.append("  \"finalAnswer\": \"").append(finalAnswer.replace("\"", "\\\"")).append("\",\n");
        }
        
        if (reasoningChain != null) {
            json.append("  \"stepCount\": ").append(reasoningChain.getStepCount()).append(",\n");
            json.append("  \"averageConfidence\": ").append(reasoningChain.getAverageConfidence()).append(",\n");
        }
        
        if (reflectionResult != null) {
            json.append("  \"qualityScore\": ").append(reflectionResult.getQualityScore()).append(",\n");
            json.append("  \"needsRefinement\": ").append(reflectionResult.isNeedsRefinement()).append(",\n");
        }
        
        if (!isSuccessful && errorMessage != null) {
            json.append("  \"errorMessage\": \"").append(errorMessage.replace("\"", "\\\"")).append("\",\n");
        }
        
        // 移除最后的逗号
        String jsonStr = json.toString();
        if (jsonStr.endsWith(",\n")) {
            jsonStr = jsonStr.substring(0, jsonStr.length() - 2) + "\n";
        }
        
        return jsonStr + "}";
    }
    
    // Getter和Setter方法
    
    public NdArray getModelOutput() {
        return modelOutput;
    }
    
    public void setModelOutput(NdArray modelOutput) {
        this.modelOutput = modelOutput;
    }
    
    public ReasoningChain getReasoningChain() {
        return reasoningChain;
    }
    
    public void setReasoningChain(ReasoningChain reasoningChain) {
        this.reasoningChain = reasoningChain;
        if (reasoningChain != null && reasoningChain.getFinalAnswer() != null) {
            this.finalAnswer = reasoningChain.getFinalAnswer();
        }
    }
    
    public ReflectionModule.ReflectionResult getReflectionResult() {
        return reflectionResult;
    }
    
    public void setReflectionResult(ReflectionModule.ReflectionResult reflectionResult) {
        this.reflectionResult = reflectionResult;
    }
    
    public double getFinalConfidence() {
        return finalConfidence;
    }
    
    public void setFinalConfidence(double finalConfidence) {
        this.finalConfidence = Math.max(0.0, Math.min(1.0, finalConfidence));
    }
    
    public String getFinalAnswer() {
        return finalAnswer;
    }
    
    public void setFinalAnswer(String finalAnswer) {
        this.finalAnswer = finalAnswer;
    }
    
    public long getProcessingTime() {
        return processingTime;
    }
    
    public void setProcessingTime(long processingTime) {
        this.processingTime = processingTime;
    }
    
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = new HashMap<>(metadata);
    }
    
    public boolean isSuccessful() {
        return isSuccessful;
    }
    
    public void setSuccessful(boolean successful) {
        isSuccessful = successful;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    @Override
    public String toString() {
        return String.format("DeepSeekR1Result[%s, conf=%.3f, steps=%d, time=%dms]",
                           isSuccessful ? "SUCCESS" : "FAILED",
                           finalConfidence,
                           reasoningChain != null ? reasoningChain.getStepCount() : 0,
                           processingTime);
    }
}