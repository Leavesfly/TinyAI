package io.leavesfly.tinyai.agent.robot.model;

/**
 * 清扫场景类型枚举
 * 
 * <p>定义不同难度级别的清扫场景类型，从简单的空房间到复杂的多房间布局。</p>
 * 
 * @author TinyAI Team
 */
public enum ScenarioType {
    /**
     * 简单房间 - 空旷环境，障碍物少
     * 难度：★☆☆☆☆
     */
    SIMPLE_ROOM("简单房间", 1),
    
    /**
     * 客厅 - 中等家具密度
     * 难度：★★☆☆☆
     */
    LIVING_ROOM("客厅", 2),
    
    /**
     * 卧室 - 家具密集
     * 难度：★★★☆☆
     */
    BEDROOM("卧室", 3),
    
    /**
     * 厨房 - 障碍复杂
     * 难度：★★★☆☆
     */
    KITCHEN("厨房", 3),
    
    /**
     * 多房间 - 大面积环境
     * 难度：★★★★☆
     */
    MULTI_ROOM("多房间", 4),
    
    /**
     * 复杂布局 - 混合场景
     * 难度：★★★★★
     */
    COMPLEX_LAYOUT("复杂布局", 5);
    
    private final String displayName;
    private final int difficultyLevel;
    
    ScenarioType(String displayName, int difficultyLevel) {
        this.displayName = displayName;
        this.difficultyLevel = difficultyLevel;
    }
    
    /**
     * 获取场景显示名称
     * 
     * @return 场景名称
     */
    public String getDisplayName() {
        return displayName;
    }
    
    /**
     * 获取难度等级
     * 
     * @return 难度等级（1-5）
     */
    public int getDifficultyLevel() {
        return difficultyLevel;
    }
}
