package io.leavesfly.tinyai.deepseek.r1;

import java.io.Serializable;

/**
 * 推理步骤数据类
 * 
 * 记录单个推理步骤的详细信息，包括思考内容、采取的行动、
 * 置信度评估和验证结果。这是DeepSeek R1思维链推理的基础单元。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ReasoningStep implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private String thought;      // 思考内容
    private String action;       // 采取的行动
    private double confidence;   // 置信度(0.0-1.0)
    private String verification; // 验证结果
    private int stepIndex;       // 步骤索引
    private long timestamp;      // 时间戳
    
    /**
     * 默认构造函数
     */
    public ReasoningStep() {
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 完整构造函数
     * 
     * @param thought 思考内容
     * @param action 采取的行动
     * @param confidence 置信度
     * @param verification 验证结果
     */
    public ReasoningStep(String thought, String action, double confidence, String verification) {
        this();
        this.thought = thought;
        this.action = action;
        this.confidence = Math.max(0.0, Math.min(1.0, confidence)); // 确保在[0,1]范围内
        this.verification = verification;
    }
    
    /**
     * 创建推理步骤的工厂方法
     * 
     * @param stepIndex 步骤索引
     * @param thought 思考内容
     * @param action 行动描述
     * @param confidence 置信度
     * @return 新的推理步骤实例
     */
    public static ReasoningStep create(int stepIndex, String thought, String action, double confidence) {
        ReasoningStep step = new ReasoningStep(thought, action, confidence, "");
        step.setStepIndex(stepIndex);
        return step;
    }
    
    /**
     * 验证推理步骤的有效性
     * 
     * @return 如果步骤有效则返回true
     */
    public boolean isValid() {
        return thought != null && !thought.trim().isEmpty() &&
               action != null && !action.trim().isEmpty() &&
               confidence >= 0.0 && confidence <= 1.0;
    }
    
    /**
     * 获取推理步骤的详细描述
     * 
     * @return 格式化的步骤描述
     */
    public String getDetailedDescription() {
        return String.format(
            "步骤 %d:\n" +
            "  思考: %s\n" +
            "  行动: %s\n" +
            "  置信度: %.3f\n" +
            "  验证: %s",
            stepIndex, thought, action, confidence, verification
        );
    }
    
    /**
     * 判断步骤是否达到高置信度
     * 
     * @param threshold 置信度阈值
     * @return 如果置信度超过阈值则返回true
     */
    public boolean isHighConfidence(double threshold) {
        return confidence >= threshold;
    }
    
    // Getter和Setter方法
    
    public String getThought() {
        return thought;
    }
    
    public void setThought(String thought) {
        this.thought = thought;
    }
    
    public String getAction() {
        return action;
    }
    
    public void setAction(String action) {
        this.action = action;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = Math.max(0.0, Math.min(1.0, confidence));
    }
    
    public String getVerification() {
        return verification;
    }
    
    public void setVerification(String verification) {
        this.verification = verification;
    }
    
    public int getStepIndex() {
        return stepIndex;
    }
    
    public void setStepIndex(int stepIndex) {
        this.stepIndex = stepIndex;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    @Override
    public String toString() {
        return String.format("ReasoningStep[%d]: %s -> %s (conf=%.3f)", 
                           stepIndex, thought, action, confidence);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        ReasoningStep other = (ReasoningStep) obj;
        return stepIndex == other.stepIndex &&
               Double.compare(other.confidence, confidence) == 0 &&
               thought.equals(other.thought) &&
               action.equals(other.action);
    }
    
    @Override
    public int hashCode() {
        int result = thought != null ? thought.hashCode() : 0;
        result = 31 * result + (action != null ? action.hashCode() : 0);
        result = 31 * result + stepIndex;
        long temp = Double.doubleToLongBits(confidence);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}