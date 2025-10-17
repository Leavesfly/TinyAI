package io.leavesfly.tinyai.agent.robot.env;

import io.leavesfly.tinyai.agent.robot.model.ScenarioType;

/**
 * 环境配置类
 * 
 * <p>定义清扫环境的配置参数。</p>
 * 
 * @author TinyAI Team
 */
public class EnvironmentConfig {
    /**
     * 场景类型
     */
    private ScenarioType scenarioType;
    
    /**
     * 房间宽度（米）
     */
    private double roomWidth;
    
    /**
     * 房间高度（米）
     */
    private double roomHeight;
    
    /**
     * 网格大小（米）
     */
    private double gridSize;
    
    /**
     * 障碍物数量
     */
    private int obstacleCount;
    
    /**
     * 初始电量（百分比）
     */
    private double initialBattery;
    
    /**
     * 目标覆盖率
     */
    private double targetCoverage;
    
    /**
     * 时间步长（秒）
     */
    private double timeStep;
    
    /**
     * 最大步数
     */
    private int maxSteps;
    
    /**
     * 是否启用可视化
     */
    private boolean enableVisualization;
    
    /**
     * 默认构造函数
     */
    public EnvironmentConfig() {
        this.scenarioType = ScenarioType.SIMPLE_ROOM;
        this.roomWidth = 5.0;
        this.roomHeight = 5.0;
        this.gridSize = 0.1;
        this.obstacleCount = 5;
        this.initialBattery = 100.0;
        this.targetCoverage = 0.95;
        this.timeStep = 0.1;
        this.maxSteps = 2000;
        this.enableVisualization = false;
    }
    
    /**
     * 创建简单房间配置
     */
    public static EnvironmentConfig createSimpleRoomConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setScenarioType(ScenarioType.SIMPLE_ROOM);
        config.setRoomWidth(5.0);
        config.setRoomHeight(5.0);
        config.setObstacleCount(3);
        config.setTargetCoverage(0.95);
        return config;
    }
    
    /**
     * 创建客厅配置
     */
    public static EnvironmentConfig createLivingRoomConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setScenarioType(ScenarioType.LIVING_ROOM);
        config.setRoomWidth(8.0);
        config.setRoomHeight(6.0);
        config.setObstacleCount(10);
        config.setTargetCoverage(0.90);
        return config;
    }
    
    /**
     * 创建卧室配置
     */
    public static EnvironmentConfig createBedroomConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setScenarioType(ScenarioType.BEDROOM);
        config.setRoomWidth(6.0);
        config.setRoomHeight(5.0);
        config.setObstacleCount(15);
        config.setTargetCoverage(0.85);
        return config;
    }
    
    /**
     * 创建自定义配置
     */
    public static EnvironmentConfig createCustomConfig(ScenarioType type, double width, 
                                                       double height, int obstacles) {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setScenarioType(type);
        config.setRoomWidth(width);
        config.setRoomHeight(height);
        config.setObstacleCount(obstacles);
        return config;
    }
    
    // Getters and Setters
    public ScenarioType getScenarioType() {
        return scenarioType;
    }
    
    public void setScenarioType(ScenarioType scenarioType) {
        this.scenarioType = scenarioType;
    }
    
    public double getRoomWidth() {
        return roomWidth;
    }
    
    public void setRoomWidth(double roomWidth) {
        this.roomWidth = roomWidth;
    }
    
    public double getRoomHeight() {
        return roomHeight;
    }
    
    public void setRoomHeight(double roomHeight) {
        this.roomHeight = roomHeight;
    }
    
    public double getGridSize() {
        return gridSize;
    }
    
    public void setGridSize(double gridSize) {
        this.gridSize = gridSize;
    }
    
    public int getObstacleCount() {
        return obstacleCount;
    }
    
    public void setObstacleCount(int obstacleCount) {
        this.obstacleCount = obstacleCount;
    }
    
    public double getInitialBattery() {
        return initialBattery;
    }
    
    public void setInitialBattery(double initialBattery) {
        this.initialBattery = initialBattery;
    }
    
    public double getTargetCoverage() {
        return targetCoverage;
    }
    
    public void setTargetCoverage(double targetCoverage) {
        this.targetCoverage = targetCoverage;
    }
    
    public double getTimeStep() {
        return timeStep;
    }
    
    public void setTimeStep(double timeStep) {
        this.timeStep = timeStep;
    }
    
    public int getMaxSteps() {
        return maxSteps;
    }
    
    public void setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
    }
    
    public boolean isEnableVisualization() {
        return enableVisualization;
    }
    
    public void setEnableVisualization(boolean enableVisualization) {
        this.enableVisualization = enableVisualization;
    }
    
    @Override
    public String toString() {
        return String.format("EnvironmentConfig(type=%s, room=%.1fx%.1f, obstacles=%d, target=%.1f%%)",
                             scenarioType, roomWidth, roomHeight, obstacleCount, targetCoverage * 100);
    }
}
