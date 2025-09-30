package io.leavesfly.tinyai.deepseek.v3;

/**
 * DeepSeek V3 任务类型枚举
 * 
 * 用于标识不同类型的任务，以便专家混合模型(MoE)能够
 * 根据任务类型选择最适合的专家进行处理。
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum TaskType {
    
    /**
     * 推理任务
     * 需要复杂的逻辑推理和多步思考过程
     */
    REASONING("reasoning"),
    
    /**
     * 代码生成任务
     * 包括代码编写、调试、优化等编程相关任务
     */
    CODING("coding"),
    
    /**
     * 数学计算任务
     * 数学推理、计算、证明等数学相关任务
     */
    MATH("math"),
    
    /**
     * 通用任务
     * 常规的语言理解、文本生成等通用NLP任务
     */
    GENERAL("general"),
    
    /**
     * 多模态任务
     * 涉及文本、图像、音频等多种模态的综合处理任务
     */
    MULTIMODAL("multimodal");
    
    private final String value;
    
    TaskType(String value) {
        this.value = value;
    }
    
    /**
     * 获取任务类型的字符串值
     * 
     * @return 任务类型字符串
     */
    public String getValue() {
        return value;
    }
    
    /**
     * 从字符串值解析任务类型
     * 
     * @param value 任务类型字符串
     * @return 对应的TaskType枚举值
     * @throws IllegalArgumentException 如果字符串值无效
     */
    public static TaskType fromValue(String value) {
        for (TaskType type : TaskType.values()) {
            if (type.value.equalsIgnoreCase(value)) {
                return type;
            }
        }
        throw new IllegalArgumentException("未知的任务类型: " + value);
    }
    
    /**
     * 判断是否为复杂推理任务
     * 
     * @return 如果是需要复杂推理的任务返回true
     */
    public boolean isComplexReasoning() {
        return this == REASONING || this == MATH || this == CODING;
    }
    
    /**
     * 判断是否为创造性任务
     * 
     * @return 如果是需要创造性思维的任务返回true
     */
    public boolean isCreative() {
        return this == CODING || this == MULTIMODAL;
    }
    
    /**
     * 获取任务的推荐专家数量
     * 
     * @return 推荐使用的专家数量
     */
    public int getRecommendedExpertCount() {
        switch (this) {
            case REASONING:
            case MATH:
                return 3; // 复杂推理任务需要更多专家
            case CODING:
                return 2; // 代码任务需要中等数量专家
            case MULTIMODAL:
                return 4; // 多模态任务需要最多专家
            case GENERAL:
            default:
                return 2; // 通用任务使用默认数量
        }
    }
    
    @Override
    public String toString() {
        return value;
    }
}