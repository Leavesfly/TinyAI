package io.leavesfly.tinyai.deepseek.v3;

import java.util.Map;

/**
 * V3增强的推理步骤
 * 
 * 表示DeepSeek V3模型中的一个推理步骤，包含思考过程、行动、置信度等信息
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3ReasoningStep {
    
    /**
     * 思考内容
     */
    private final String thought;
    
    /**
     * 执行的行动
     */
    private final String action;
    
    /**
     * 置信度评分 (0.0 - 1.0)
     */
    private final float confidence;
    
    /**
     * 验证结果
     */
    private final String verification;
    
    /**
     * 任务类型
     */
    private final TaskType taskType;
    
    /**
     * 各专家的建议权重
     */
    private final Map<String, Float> expertAdvice;
    
    /**
     * 自我纠错信息
     */
    private final String selfCorrection;
    
    /**
     * 构造函数
     */
    public V3ReasoningStep(String thought, String action, float confidence, 
                          String verification, TaskType taskType, 
                          Map<String, Float> expertAdvice, String selfCorrection) {
        this.thought = thought;
        this.action = action;
        this.confidence = confidence;
        this.verification = verification;
        this.taskType = taskType;
        this.expertAdvice = expertAdvice;
        this.selfCorrection = selfCorrection;
    }
    
    // Getters
    public String getThought() {
        return thought;
    }
    
    public String getAction() {
        return action;
    }
    
    public float getConfidence() {
        return confidence;
    }
    
    public String getVerification() {
        return verification;
    }
    
    public TaskType getTaskType() {
        return taskType;
    }
    
    public Map<String, Float> getExpertAdvice() {
        return expertAdvice;
    }
    
    public String getSelfCorrection() {
        return selfCorrection;
    }
    
    @Override
    public String toString() {
        return String.format("V3ReasoningStep{task=%s, confidence=%.3f, thought='%s', action='%s'}", 
                           taskType, confidence, thought, action);
    }
}