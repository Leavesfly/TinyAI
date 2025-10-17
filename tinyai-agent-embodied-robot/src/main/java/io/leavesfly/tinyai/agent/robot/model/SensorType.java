package io.leavesfly.tinyai.agent.robot.model;

/**
 * 传感器类型枚举
 * 
 * <p>定义扫地机器人使用的各种传感器类型。</p>
 * 
 * @author TinyAI Team
 */
public enum SensorType {
    /**
     * 摄像头 - 用于视觉识别
     */
    CAMERA("摄像头", 256),
    
    /**
     * 激光雷达 - 用于距离测量
     */
    LIDAR("激光雷达", 128),
    
    /**
     * 悬崖传感器 - 防止跌落
     */
    CLIFF_SENSOR("悬崖传感器", 4),
    
    /**
     * 碰撞传感器 - 碰撞检测
     */
    BUMP_SENSOR("碰撞传感器", 8),
    
    /**
     * 灰尘传感器 - 灰尘密度检测
     */
    DIRT_SENSOR("灰尘传感器", 1),
    
    /**
     * 里程计 - 位移测量
     */
    ODOMETER("里程计", 3);
    
    private final String displayName;
    private final int outputDimension;
    
    SensorType(String displayName, int outputDimension) {
        this.displayName = displayName;
        this.outputDimension = outputDimension;
    }
    
    /**
     * 获取传感器显示名称
     * 
     * @return 传感器名称
     */
    public String getDisplayName() {
        return displayName;
    }
    
    /**
     * 获取传感器输出维度
     * 
     * @return 输出维度
     */
    public int getOutputDimension() {
        return outputDimension;
    }
}
