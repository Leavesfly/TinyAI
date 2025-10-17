package io.leavesfly.tinyai.agent.robot.model;

/**
 * 动作类型枚举
 * 
 * <p>定义扫地机器人可以执行的基本动作类型。</p>
 * 
 * @author TinyAI Team
 */
public enum ActionType {
    /**
     * 前进
     */
    MOVE_FORWARD("前进"),
    
    /**
     * 左转
     */
    TURN_LEFT("左转"),
    
    /**
     * 右转
     */
    TURN_RIGHT("右转"),
    
    /**
     * 定点清扫
     */
    CLEAN_SPOT("定点清扫"),
    
    /**
     * 返回充电站
     */
    RETURN_HOME("返回充电站"),
    
    /**
     * 停止
     */
    STOP("停止");
    
    private final String displayName;
    
    ActionType(String displayName) {
        this.displayName = displayName;
    }
    
    /**
     * 获取动作显示名称
     * 
     * @return 动作名称
     */
    public String getDisplayName() {
        return displayName;
    }
}
