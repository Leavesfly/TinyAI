package io.leavesfly.tinyai.deepseek.base;

/**
 * 任务类型枚举
 * 
 * 定义DeepSeek系列模型支持的5种任务类型，用于任务感知路由和专家选择。
 * R1和V3共享相同的任务类型定义。
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum TaskType {
    /**
     * 推理任务 - 逻辑推理、数学证明、因果分析等
     * 特征：包含"如果"、"那么"、"证明"、"推理"等关键词
     */
    REASONING(0, "推理任务"),
    
    /**
     * 代码生成任务 - 编写代码、算法实现、代码调试等
     * 特征：包含"实现"、"代码"、"函数"、"class"、"def"等关键词
     */
    CODING(1, "代码生成"),
    
    /**
     * 数学计算任务 - 方程求解、数值计算、公式推导等
     * 特征：包含"计算"、"求解"、"方程"、数学符号等
     */
    MATH(2, "数学计算"),
    
    /**
     * 通用对话任务 - 问答、聊天、信息检索等
     * 特征：日常对话、描述性问题
     */
    GENERAL(3, "通用对话"),
    
    /**
     * 多模态任务 - 图像描述、跨模态推理等
     * 特征：涉及图像、音频等非文本模态
     */
    MULTIMODAL(4, "多模态");
    
    private final int id;
    private final String description;
    
    TaskType(int id, String description) {
        this.id = id;
        this.description = description;
    }
    
    public int getId() {
        return id;
    }
    
    public String getDescription() {
        return description;
    }
    
    /**
     * 根据ID获取任务类型
     */
    public static TaskType fromId(int id) {
        for (TaskType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        throw new IllegalArgumentException("未知的任务类型ID: " + id);
    }
    
    @Override
    public String toString() {
        return String.format("TaskType{id=%d, description='%s'}", id, description);
    }
}
