package io.leavesfly.tinyai.deepseek.v3;

/**
 * 任务类型枚举
 * 
 * 定义DeepSeek V3支持的不同任务类型，用于专家路由和任务特化处理
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum TaskType {
    /**
     * 推理任务 - 逻辑推理、数学推理等
     */
    REASONING("reasoning"),
    
    /**
     * 代码生成任务 - 编程、代码补全等
     */
    CODING("coding"),
    
    /**
     * 数学计算任务 - 数学公式、计算等
     */
    MATH("math"),
    
    /**
     * 通用任务 - 一般对话、文本生成等
     */
    GENERAL("general"),
    
    /**
     * 多模态任务 - 图文理解、多模态生成等
     */
    MULTIMODAL("multimodal");
    
    private final String value;
    
    TaskType(String value) {
        this.value = value;
    }
    
    /**
     * 获取任务类型的字符串值
     */
    public String getValue() {
        return value;
    }
    
    /**
     * 根据字符串值获取任务类型
     */
    public static TaskType fromValue(String value) {
        for (TaskType type : values()) {
            if (type.value.equals(value)) {
                return type;
            }
        }
        return GENERAL; // 默认为通用任务
    }
}