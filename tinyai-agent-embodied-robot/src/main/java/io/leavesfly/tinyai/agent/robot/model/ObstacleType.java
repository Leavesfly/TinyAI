package io.leavesfly.tinyai.agent.robot.model;

/**
 * 障碍物类型枚举
 * 
 * <p>定义环境中可能出现的各种障碍物类型。</p>
 * 
 * @author TinyAI Team
 */
public enum ObstacleType {
    /**
     * 墙壁 - 房间边界
     */
    WALL("墙壁"),
    
    /**
     * 家具 - 桌椅、沙发等
     */
    FURNITURE("家具"),
    
    /**
     * 楼梯 - 高度落差
     */
    STAIRS("楼梯"),
    
    /**
     * 宠物 - 移动障碍物
     */
    PET("宠物"),
    
    /**
     * 小物体 - 玩具、鞋子等
     */
    SMALL_OBJECT("小物体"),
    
    /**
     * 充电站 - 特殊标记
     */
    CHARGING_STATION("充电站");
    
    private final String displayName;
    
    ObstacleType(String displayName) {
        this.displayName = displayName;
    }
    
    /**
     * 获取障碍物显示名称
     * 
     * @return 障碍物名称
     */
    public String getDisplayName() {
        return displayName;
    }
}
