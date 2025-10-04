package io.leavesfly.tinyai.agent.research;

/**
 * 推理模式枚举
 * 定义深度研究Agent支持的五种推理策略
 * 
 * @author 山泽
 */
public enum ReasoningMode {
    /** 快速推理模式 - 适用于简单问题和紧急响应 */
    QUICK("quick", "快速推理", "简单问题，基础概念分析"),
    
    /** 彻底推理模式 - 适用于复杂问题和深度要求高的场景 */
    THOROUGH("thorough", "彻底推理", "复杂问题，全面深入分析"),
    
    /** 创意推理模式 - 适用于创新问题和发散思维 */
    CREATIVE("creative", "创意推理", "创新问题，发散思维"),
    
    /** 分析推理模式 - 适用于数据分析和比较研究 */
    ANALYTICAL("analytical", "分析推理", "数据驱动，系统分析"),
    
    /** 系统推理模式 - 适用于结构化研究和系统梳理 */
    SYSTEMATIC("systematic", "系统推理", "结构化研究，系统梳理");
    
    private final String code;
    private final String name;
    private final String description;
    
    ReasoningMode(String code, String name, String description) {
        this.code = code;
        this.name = name;
        this.description = description;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getName() {
        return name;
    }
    
    public String getDescription() {
        return description;
    }
    
    public static ReasoningMode fromCode(String code) {
        for (ReasoningMode mode : values()) {
            if (mode.code.equals(code)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("未知的推理模式代码: " + code);
    }
    
    @Override
    public String toString() {
        return String.format("%s(%s)", name, code);
    }
}