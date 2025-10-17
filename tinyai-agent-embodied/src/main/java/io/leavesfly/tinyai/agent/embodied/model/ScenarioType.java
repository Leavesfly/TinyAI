package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 场景类型枚举
 * 定义不同的驾驶场景类型
 *
 * @author TinyAI Team
 */
public enum ScenarioType {
    /**
     * 高速公路场景 - 结构化道路
     */
    HIGHWAY("高速公路", 1),

    /**
     * 城市道路场景 - 复杂交互
     */
    URBAN("城市道路", 3),

    /**
     * 乡村道路场景 - 简单道路
     */
    RURAL("乡村道路", 2),

    /**
     * 停车场场景 - 低速精细控制
     */
    PARKING_LOT("停车场", 2),

    /**
     * 路口场景 - 交通规则重点
     */
    INTERSECTION("路口", 4),

    /**
     * 测试场景 - 用于简单测试
     */
    TEST("测试场景", 1);

    private final String name;
    private final int complexityLevel;

    ScenarioType(String name, int complexityLevel) {
        this.name = name;
        this.complexityLevel = complexityLevel;
    }

    public String getName() {
        return name;
    }

    /**
     * 获取场景复杂度等级（1-5，数值越大越复杂）
     */
    public int getComplexityLevel() {
        return complexityLevel;
    }
}
