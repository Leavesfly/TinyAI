package io.leavesfly.tinyai.deepseek.r1;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 推理链数据类
 * 
 * 包含完整的推理过程，由多个推理步骤组成，最终得出答案，
 * 并包含整体的置信度评估和自我反思结果。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReasoningChain implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private List<ReasoningStep> steps;           // 推理步骤列表
    private String finalAnswer;                  // 最终答案
    private double totalConfidence;              // 总体置信度
    private String reflection;                   // 自我反思结果
    private String originalQuestion;             // 原始问题
    private Map<String, Object> metadata;       // 元数据
    private long creationTime;                   // 创建时间
    private long completionTime;                 // 完成时间
    private boolean isCompleted;                 // 是否完成
    
    /**
     * 默认构造函数
     */
    public ReasoningChain() {
        this.steps = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.creationTime = System.currentTimeMillis();
        this.isCompleted = false;
    }
    
    /**
     * 完整构造函数
     * 
     * @param steps 推理步骤列表
     * @param finalAnswer 最终答案
     * @param totalConfidence 总体置信度
     * @param reflection 自我反思结果
     */
    public ReasoningChain(List<ReasoningStep> steps, String finalAnswer, 
                         double totalConfidence, String reflection) {
        this();
        this.steps = new ArrayList<>(steps);
        this.finalAnswer = finalAnswer;
        this.totalConfidence = Math.max(0.0, Math.min(1.0, totalConfidence));
        this.reflection = reflection;
    }
    
    /**
     * 创建推理链的工厂方法
     * 
     * @param originalQuestion 原始问题
     * @return 新的推理链实例
     */
    public static ReasoningChain create(String originalQuestion) {
        ReasoningChain chain = new ReasoningChain();
        chain.setOriginalQuestion(originalQuestion);
        return chain;
    }
    
    /**
     * 添加推理步骤
     * 
     * @param step 推理步骤
     */
    public void addStep(ReasoningStep step) {
        if (step != null && step.isValid()) {
            step.setStepIndex(steps.size());
            steps.add(step);
        }
    }
    
    /**
     * 添加推理步骤（便捷方法）
     * 
     * @param thought 思考内容
     * @param action 行动描述
     * @param confidence 置信度
     * @param verification 验证结果
     */
    public void addStep(String thought, String action, double confidence, String verification) {
        ReasoningStep step = new ReasoningStep(thought, action, confidence, verification);
        addStep(step);
    }
    
    /**
     * 完成推理链
     * 
     * @param finalAnswer 最终答案
     * @param reflection 自我反思结果
     */
    public void complete(String finalAnswer, String reflection) {
        this.finalAnswer = finalAnswer;
        this.reflection = reflection;
        this.totalConfidence = calculateTotalConfidence();
        this.completionTime = System.currentTimeMillis();
        this.isCompleted = true;
    }
    
    /**
     * 计算总体置信度
     * 使用加权平均，最后的步骤权重更高
     * 
     * @return 总体置信度
     */
    private double calculateTotalConfidence() {
        if (steps.isEmpty()) {
            return 0.0;
        }
        
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        
        for (int i = 0; i < steps.size(); i++) {
            // 后面的步骤权重更高
            double weight = Math.pow(1.2, i);
            weightedSum += steps.get(i).getConfidence() * weight;
            totalWeight += weight;
        }
        
        return weightedSum / totalWeight;
    }
    
    /**
     * 获取推理链的详细描述
     * 
     * @return 格式化的推理链描述
     */
    public String getDetailedDescription() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 推理链详情 ===\n");
        sb.append("原始问题: ").append(originalQuestion).append("\n\n");
        
        sb.append("推理过程:\n");
        for (ReasoningStep step : steps) {
            sb.append(step.getDetailedDescription()).append("\n\n");
        }
        
        sb.append("最终答案: ").append(finalAnswer).append("\n");
        sb.append("总体置信度: ").append(String.format("%.3f", totalConfidence)).append("\n");
        sb.append("自我反思: ").append(reflection).append("\n");
        
        if (isCompleted) {
            long duration = completionTime - creationTime;
            sb.append("完成时间: ").append(duration).append(" ms\n");
        }
        
        return sb.toString();
    }
    
    /**
     * 获取推理链摘要
     * 
     * @return 简化的推理链摘要
     */
    public String getSummary() {
        return String.format(
            "推理链摘要: %d步 | 置信度: %.3f | 状态: %s",
            steps.size(), totalConfidence, isCompleted ? "已完成" : "进行中"
        );
    }
    
    /**
     * 判断推理链是否有效
     * 
     * @return 如果推理链有效则返回true
     */
    public boolean isValid() {
        return !steps.isEmpty() && 
               steps.stream().allMatch(ReasoningStep::isValid) &&
               finalAnswer != null && !finalAnswer.trim().isEmpty();
    }
    
    /**
     * 获取高置信度步骤
     * 
     * @param threshold 置信度阈值
     * @return 高置信度步骤列表
     */
    public List<ReasoningStep> getHighConfidenceSteps(double threshold) {
        List<ReasoningStep> highConfSteps = new ArrayList<>();
        for (ReasoningStep step : steps) {
            if (step.isHighConfidence(threshold)) {
                highConfSteps.add(step);
            }
        }
        return highConfSteps;
    }
    
    /**
     * 获取平均置信度
     * 
     * @return 平均置信度
     */
    public double getAverageConfidence() {
        if (steps.isEmpty()) {
            return 0.0;
        }
        
        double sum = steps.stream().mapToDouble(ReasoningStep::getConfidence).sum();
        return sum / steps.size();
    }
    
    /**
     * 获取最低置信度
     * 
     * @return 最低置信度
     */
    public double getMinConfidence() {
        return steps.stream().mapToDouble(ReasoningStep::getConfidence).min().orElse(0.0);
    }
    
    /**
     * 获取最高置信度
     * 
     * @return 最高置信度
     */
    public double getMaxConfidence() {
        return steps.stream().mapToDouble(ReasoningStep::getConfidence).max().orElse(0.0);
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
     * 获取推理时长（毫秒）
     * 
     * @return 推理时长，如果未完成则返回当前耗时
     */
    public long getDuration() {
        long endTime = isCompleted ? completionTime : System.currentTimeMillis();
        return endTime - creationTime;
    }
    
    // Getter和Setter方法
    
    public List<ReasoningStep> getSteps() {
        return new ArrayList<>(steps);
    }
    
    public void setSteps(List<ReasoningStep> steps) {
        this.steps = new ArrayList<>(steps);
    }
    
    public String getFinalAnswer() {
        return finalAnswer;
    }
    
    public void setFinalAnswer(String finalAnswer) {
        this.finalAnswer = finalAnswer;
    }
    
    public double getTotalConfidence() {
        return totalConfidence;
    }
    
    public void setTotalConfidence(double totalConfidence) {
        this.totalConfidence = Math.max(0.0, Math.min(1.0, totalConfidence));
    }
    
    public String getReflection() {
        return reflection;
    }
    
    public void setReflection(String reflection) {
        this.reflection = reflection;
    }
    
    public String getOriginalQuestion() {
        return originalQuestion;
    }
    
    public void setOriginalQuestion(String originalQuestion) {
        this.originalQuestion = originalQuestion;
    }
    
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = new HashMap<>(metadata);
    }
    
    public long getCreationTime() {
        return creationTime;
    }
    
    public void setCreationTime(long creationTime) {
        this.creationTime = creationTime;
    }
    
    public long getCompletionTime() {
        return completionTime;
    }
    
    public void setCompletionTime(long completionTime) {
        this.completionTime = completionTime;
    }
    
    public boolean isCompleted() {
        return isCompleted;
    }
    
    public void setCompleted(boolean completed) {
        isCompleted = completed;
    }
    
    public int getStepCount() {
        return steps.size();
    }
    
    @Override
    public String toString() {
        return String.format("ReasoningChain[%d steps, conf=%.3f, %s]", 
                           steps.size(), totalConfidence, 
                           isCompleted ? "completed" : "in-progress");
    }
}