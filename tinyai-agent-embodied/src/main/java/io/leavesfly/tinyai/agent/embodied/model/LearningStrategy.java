package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 学习策略枚举
 * 定义支持的学习算法类型
 *
 * @author TinyAI Team
 */
public enum LearningStrategy {
    /**
     * DQN强化学习
     */
    DQN("DQN强化学习", "离散动作空间"),

    /**
     * 端到端学习
     */
    END_TO_END("端到端学习", "连续动作空间"),

    /**
     * 模仿学习
     */
    IMITATION("模仿学习", "监督学习"),

    /**
     * 混合策略
     */
    HYBRID("混合策略", "多策略融合");

    private final String name;
    private final String description;

    LearningStrategy(String name, String description) {
        this.name = name;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }
}
