package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.ndarr.NdArray;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * 推理结果类
 * 
 * 封装DeepSeek-R1模型的推理结果，包含：
 * 1. 最终答案
 * 2. 思维链推理步骤
 * 3. 置信度分数
 * 4. 推理上下文信息
 * 5. 错误信息（如果有）
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReasoningResult {
    
    // 最终答案
    private NdArray finalAnswer;
    
    // 思维链推理步骤
    private List<String> reasoningSteps;
    
    // 置信度分数
    private double confidenceScore;
    
    // 推理上下文信息
    private Map<String, Object> reasoningContext;
    
    // 是否有错误
    private boolean error;
    
    // 错误信息
    private String errorMessage;
    
    // 推理耗时（毫秒）
    private long reasoningTimeMs;
    
    // 推理步骤数量
    private int numReasoningSteps;
    
    /**
     * 默认构造函数
     */
    public ReasoningResult() {
        this.reasoningSteps = new ArrayList<>();
        this.reasoningContext = new HashMap<>();
        this.error = false;
        this.confidenceScore = 0.0;
        this.reasoningTimeMs = 0L;
        this.numReasoningSteps = 0;
    }
    
    /**
     * 完整构造函数
     * 
     * @param finalAnswer 最终答案
     * @param reasoningSteps 推理步骤
     * @param confidenceScore 置信度分数
     * @param reasoningContext 推理上下文
     */
    public ReasoningResult(NdArray finalAnswer, List<String> reasoningSteps, 
                          double confidenceScore, Map<String, Object> reasoningContext) {
        this.finalAnswer = finalAnswer;
        this.reasoningSteps = new ArrayList<>(reasoningSteps);
        this.confidenceScore = confidenceScore;
        this.reasoningContext = new HashMap<>(reasoningContext);
        this.error = false;
        this.numReasoningSteps = reasoningSteps.size();
    }
    
    /**
     * 添加推理步骤
     * 
     * @param step 推理步骤描述
     */
    public void addReasoningStep(String step) {
        reasoningSteps.add(step);
        numReasoningSteps = reasoningSteps.size();
    }
    
    /**
     * 添加推理上下文信息
     * 
     * @param key 上下文键
     * @param value 上下文值
     */
    public void addContextInfo(String key, Object value) {
        reasoningContext.put(key, value);
    }
    
    /**
     * 检查推理是否成功
     * 
     * @return 如果推理成功且无错误返回true
     */
    public boolean isSuccess() {
        return !error && finalAnswer != null;
    }
    
    /**
     * 获取推理步骤的可读格式
     * 
     * @return 格式化的推理步骤字符串
     */
    public String getFormattedReasoningSteps() {
        if (reasoningSteps.isEmpty()) {
            return "无推理步骤记录";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("推理过程:\n");
        for (int i = 0; i < reasoningSteps.size(); i++) {
            sb.append(String.format("步骤 %d: %s\n", i + 1, reasoningSteps.get(i)));
        }
        return sb.toString();
    }
    
    /**
     * 获取详细的推理报告
     * 
     * @return 完整的推理报告字符串
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DeepSeek-R1 推理报告 ===\n");
        
        if (error) {
            sb.append("状态: 推理失败\n");
            sb.append("错误信息: ").append(errorMessage).append("\n");
        } else {
            sb.append("状态: 推理成功\n");
            sb.append("置信度分数: ").append(String.format("%.4f", confidenceScore)).append("\n");
            sb.append("推理步骤数: ").append(numReasoningSteps).append("\n");
            sb.append("推理耗时: ").append(reasoningTimeMs).append(" ms\n");
        }
        
        sb.append("\n").append(getFormattedReasoningSteps());
        
        if (!reasoningContext.isEmpty()) {
            sb.append("\n推理上下文信息:\n");
            for (Map.Entry<String, Object> entry : reasoningContext.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }
        
        sb.append("==========================\n");
        return sb.toString();
    }
    
    // Getter和Setter方法
    public NdArray getFinalAnswer() {
        return finalAnswer;
    }
    
    public void setFinalAnswer(NdArray finalAnswer) {
        this.finalAnswer = finalAnswer;
    }
    
    public List<String> getReasoningSteps() {
        return new ArrayList<>(reasoningSteps);
    }
    
    public void setReasoningSteps(List<String> reasoningSteps) {
        this.reasoningSteps = new ArrayList<>(reasoningSteps);
        this.numReasoningSteps = this.reasoningSteps.size();
    }
    
    public double getConfidenceScore() {
        return confidenceScore;
    }
    
    public void setConfidenceScore(double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }
    
    public Map<String, Object> getReasoningContext() {
        return new HashMap<>(reasoningContext);
    }
    
    public void setReasoningContext(Map<String, Object> reasoningContext) {
        this.reasoningContext = new HashMap<>(reasoningContext);
    }
    
    public boolean isError() {
        return error;
    }
    
    public void setError(boolean error) {
        this.error = error;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    public long getReasoningTimeMs() {
        return reasoningTimeMs;
    }
    
    public void setReasoningTimeMs(long reasoningTimeMs) {
        this.reasoningTimeMs = reasoningTimeMs;
    }
    
    public int getNumReasoningSteps() {
        return numReasoningSteps;
    }
    
    public void setNumReasoningSteps(int numReasoningSteps) {
        this.numReasoningSteps = numReasoningSteps;
    }
    
    @Override
    public String toString() {
        return getDetailedReport();
    }
}