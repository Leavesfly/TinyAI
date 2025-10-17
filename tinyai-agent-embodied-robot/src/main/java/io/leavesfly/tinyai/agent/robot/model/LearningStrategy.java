package io.leavesfly.tinyai.agent.robot.model;

/**
 * 学习策略枚举
 * 
 * <p>定义智能体可使用的学习策略类型。</p>
 * 
 * @author TinyAI Team
 */
public enum LearningStrategy {
    /**
     * 深度Q网络 - 离散动作空间
     */
    DQN("深度Q网络"),
    
    /**
     * 深度确定性策略梯度 - 连续动作空间
     */
    DDPG("深度确定性策略梯度"),
    
    /**
     * 近端策略优化 - 稳定训练
     */
    PPO("近端策略优化"),
    
    /**
     * 端到端学习 - 直接感知到动作映射
     */
    END_TO_END("端到端学习");
    
    private final String displayName;
    
    LearningStrategy(String displayName) {
        this.displayName = displayName;
    }
    
    /**
     * 获取策略显示名称
     * 
     * @return 策略名称
     */
    public String getDisplayName() {
        return displayName;
    }
}
