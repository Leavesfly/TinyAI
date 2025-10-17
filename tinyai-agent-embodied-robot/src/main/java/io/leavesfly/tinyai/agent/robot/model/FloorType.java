package io.leavesfly.tinyai.agent.robot.model;

/**
 * 地面类型枚举
 * 
 * <p>定义不同的地面材质类型，影响清扫效率。</p>
 * 
 * @author TinyAI Team
 */
public enum FloorType {
    /**
     * 瓷砖 - 易清扫
     */
    TILE("瓷砖", 1.0),
    
    /**
     * 木地板 - 标准清扫
     */
    WOOD("木地板", 0.9),
    
    /**
     * 地毯 - 难清扫
     */
    CARPET("地毯", 0.7),
    
    /**
     * 大理石 - 易清扫
     */
    MARBLE("大理石", 1.0);
    
    private final String displayName;
    private final double cleaningEfficiency;
    
    FloorType(String displayName, double cleaningEfficiency) {
        this.displayName = displayName;
        this.cleaningEfficiency = cleaningEfficiency;
    }
    
    /**
     * 获取地面类型显示名称
     * 
     * @return 地面类型名称
     */
    public String getDisplayName() {
        return displayName;
    }
    
    /**
     * 获取清扫效率系数
     * 
     * @return 清扫效率（0-1）
     */
    public double getCleaningEfficiency() {
        return cleaningEfficiency;
    }
}
