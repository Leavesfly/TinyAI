package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 障碍物类型枚举
 * 定义环境中可能出现的障碍物类型
 *
 * @author TinyAI Team
 */
public enum ObstacleType {
    /**
     * 车辆 - 动态障碍物
     */
    VEHICLE("车辆", true),

    /**
     * 行人 - 动态障碍物
     */
    PEDESTRIAN("行人", true),

    /**
     * 自行车/摩托车 - 动态障碍物
     */
    BICYCLE("自行车", true),

    /**
     * 静态障碍物（锥桶、路障等）
     */
    STATIC_OBJECT("静态物体", false),

    /**
     * 交通设施（交通灯、标志牌等）
     */
    TRAFFIC_FACILITY("交通设施", false),

    /**
     * 未知障碍物
     */
    UNKNOWN("未知", false);

    private final String name;
    private final boolean dynamic;

    ObstacleType(String name, boolean dynamic) {
        this.name = name;
        this.dynamic = dynamic;
    }

    public String getName() {
        return name;
    }

    public boolean isDynamic() {
        return dynamic;
    }
}
