package io.leavesfly.tinyai.deepseek.v3;

import java.util.Map;

/**
 * DeepSeek V3 增强推理步骤
 * 
 * 记录V3模型推理过程中的每个步骤信息，包括思考过程、
 * 执行动作、置信度评估、验证结果以及自我纠错信息。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3ReasoningStep {
    
    /**
     * 思考过程
     * 记录模型在该步骤的思维过程和内在推理
     */
    private String thought;
    
    /**
     * 执行动作
     * 记录模型在该步骤采取的具体行动
     */
    private String action;
    
    /**
     * 置信度
     * 模型对当前步骤结果的信心程度 [0.0, 1.0]
     */
    private double confidence;
    
    /**
     * 验证信息
     * 对当前步骤结果的验证和检查信息
     */
    private String verification;
    
    /**
     * 任务类型
     * 当前步骤针对的任务类型
     */
    private TaskType taskType;
    
    /**
     * 专家建议
     * 各个专家对当前步骤的建议权重
     */
    private Map<String, Double> expertAdvice;
    
    /**
     * 自我纠错信息
     * 模型的自我反思和纠错信息
     */
    private String selfCorrection;
    
    /**
     * 步骤序号
     * 在整个推理过程中的步骤编号
     */
    private int stepNumber;
    
    /**
     * 执行时间(毫秒)
     * 该步骤的执行时长
     */
    private long executionTimeMs;
    
    /**
     * 中间状态
     * 记录该步骤的中间计算状态(可选)
     */
    private Object intermediateState;
    
    /**
     * 构造函数
     * 
     * @param thought 思考过程
     * @param action 执行动作
     * @param confidence 置信度
     * @param verification 验证信息
     * @param taskType 任务类型
     * @param expertAdvice 专家建议
     * @param selfCorrection 自我纠错信息
     */
    public V3ReasoningStep(String thought, String action, double confidence,
                          String verification, TaskType taskType,
                          Map<String, Double> expertAdvice, String selfCorrection) {
        this.thought = thought;
        this.action = action;
        this.confidence = Math.max(0.0, Math.min(1.0, confidence)); // 确保在[0,1]范围内
        this.verification = verification;
        this.taskType = taskType;
        this.expertAdvice = expertAdvice;
        this.selfCorrection = selfCorrection;
        this.stepNumber = -1; // 默认未设置
        this.executionTimeMs = 0L;
    }
    
    /**
     * 完整构造函数
     * 
     * @param thought 思考过程
     * @param action 执行动作
     * @param confidence 置信度
     * @param verification 验证信息
     * @param taskType 任务类型
     * @param expertAdvice 专家建议
     * @param selfCorrection 自我纠错信息
     * @param stepNumber 步骤序号
     * @param executionTimeMs 执行时间
     */
    public V3ReasoningStep(String thought, String action, double confidence,
                          String verification, TaskType taskType,
                          Map<String, Double> expertAdvice, String selfCorrection,
                          int stepNumber, long executionTimeMs) {
        this(thought, action, confidence, verification, taskType, expertAdvice, selfCorrection);
        this.stepNumber = stepNumber;
        this.executionTimeMs = executionTimeMs;
    }
    
    /**
     * 判断是否为高置信度步骤
     * 
     * @param threshold 置信度阈值
     * @return 是否为高置信度
     */
    public boolean isHighConfidence(double threshold) {
        return confidence >= threshold;
    }
    
    /**
     * 判断是否需要进一步验证
     * 
     * @return 是否需要验证
     */
    public boolean needsVerification() {
        return confidence < 0.8 || (verification != null && verification.contains("验证失败"));
    }
    
    /**
     * 获取专家建议的总和
     * 
     * @return 专家建议权重总和
     */
    public double getTotalExpertAdvice() {
        if (expertAdvice == null || expertAdvice.isEmpty()) {
            return 0.0;
        }
        return expertAdvice.values().stream().mapToDouble(Double::doubleValue).sum();
    }
    
    /**
     * 获取主要专家类型
     * 
     * @return 权重最高的专家类型
     */
    public String getDominantExpert() {
        if (expertAdvice == null || expertAdvice.isEmpty()) {
            return "unknown";
        }
        
        return expertAdvice.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("unknown");
    }
    
    /**
     * 计算步骤质量分数
     * 综合置信度、专家一致性等因素
     * 
     * @return 质量分数 [0.0, 1.0]
     */
    public double calculateQualityScore() {
        double baseScore = confidence;
        
        // 专家一致性加分
        if (expertAdvice != null && !expertAdvice.isEmpty()) {
            double maxAdvice = expertAdvice.values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            double totalAdvice = getTotalExpertAdvice();
            if (totalAdvice > 0) {
                double consistency = maxAdvice / totalAdvice;
                baseScore += consistency * 0.1; // 最多加10%
            }
        }
        
        // 自我纠错减分
        if (selfCorrection != null && selfCorrection.contains("纠错")) {
            baseScore -= 0.05; // 减5%
        }
        
        return Math.max(0.0, Math.min(1.0, baseScore));
    }
    
    /**
     * 生成步骤摘要
     * 
     * @return 步骤的文字描述
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        if (stepNumber >= 0) {
            sb.append(String.format("步骤 %d: ", stepNumber));
        }
        sb.append(String.format("[%s] ", taskType != null ? taskType.getValue() : "unknown"));
        sb.append(thought);
        sb.append(String.format(" (置信度: %.2f", confidence));
        if (executionTimeMs > 0) {
            sb.append(String.format(", 耗时: %dms", executionTimeMs));
        }
        sb.append(")");
        return sb.toString();
    }
    
    /**
     * 生成详细报告
     * 
     * @return 步骤的详细信息
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 推理步骤详细报告 ===\n");
        if (stepNumber >= 0) {
            sb.append(String.format("步骤编号: %d\n", stepNumber));
        }
        sb.append(String.format("任务类型: %s\n", taskType != null ? taskType.getValue() : "未知"));
        sb.append(String.format("思考过程: %s\n", thought));
        sb.append(String.format("执行动作: %s\n", action));
        sb.append(String.format("置信度: %.3f\n", confidence));
        sb.append(String.format("验证结果: %s\n", verification));
        
        if (expertAdvice != null && !expertAdvice.isEmpty()) {
            sb.append("专家建议:\n");
            expertAdvice.forEach((expert, weight) ->
                sb.append(String.format("  - %s: %.3f\n", expert, weight)));
            sb.append(String.format("主要专家: %s\n", getDominantExpert()));
        }
        
        if (selfCorrection != null) {
            sb.append(String.format("自我纠错: %s\n", selfCorrection));
        }
        
        if (executionTimeMs > 0) {
            sb.append(String.format("执行时间: %d毫秒\n", executionTimeMs));
        }
        
        sb.append(String.format("质量分数: %.3f\n", calculateQualityScore()));
        sb.append("=======================");
        
        return sb.toString();
    }
    
    // ========== Getter and Setter Methods ==========
    
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
    
    public TaskType getTaskType() {
        return taskType;
    }
    
    public void setTaskType(TaskType taskType) {
        this.taskType = taskType;
    }
    
    public Map<String, Double> getExpertAdvice() {
        return expertAdvice;
    }
    
    public void setExpertAdvice(Map<String, Double> expertAdvice) {
        this.expertAdvice = expertAdvice;
    }
    
    public String getSelfCorrection() {
        return selfCorrection;
    }
    
    public void setSelfCorrection(String selfCorrection) {
        this.selfCorrection = selfCorrection;
    }
    
    public int getStepNumber() {
        return stepNumber;
    }
    
    public void setStepNumber(int stepNumber) {
        this.stepNumber = stepNumber;
    }
    
    public long getExecutionTimeMs() {
        return executionTimeMs;
    }
    
    public void setExecutionTimeMs(long executionTimeMs) {
        this.executionTimeMs = executionTimeMs;
    }
    
    public Object getIntermediateState() {
        return intermediateState;
    }
    
    public void setIntermediateState(Object intermediateState) {
        this.intermediateState = intermediateState;
    }
}