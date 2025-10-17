package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 安全等级枚举
 * 定义安全约束的严格程度
 *
 * @author TinyAI Team
 */
public enum SafetyLevel {
    /**
     * 宽松模式 - 允许探索性行为
     */
    RELAXED("宽松", 0.8),

    /**
     * 标准模式 - 平衡安全与性能
     */
    STANDARD("标准", 0.9),

    /**
     * 严格模式 - 高度保守的安全策略
     */
    STRICT("严格", 0.95),

    /**
     * 紧急模式 - 最高安全优先级
     */
    EMERGENCY("紧急", 1.0);

    private final String name;
    private final double safetyThreshold;

    SafetyLevel(String name, double safetyThreshold) {
        this.name = name;
        this.safetyThreshold = safetyThreshold;
    }

    public String getName() {
        return name;
    }

    /**
     * 获取安全阈值（0-1，值越大越保守）
     */
    public double getSafetyThreshold() {
        return safetyThreshold;
    }
}
