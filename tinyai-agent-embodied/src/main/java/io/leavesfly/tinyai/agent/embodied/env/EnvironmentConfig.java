package io.leavesfly.tinyai.agent.embodied.env;

import io.leavesfly.tinyai.agent.embodied.model.ScenarioType;

/**
 * 环境配置类
 * 定义驾驶环境的各项参数
 *
 * @author TinyAI Team
 */
public class EnvironmentConfig {
    // 道路参数
    private int laneCount = 3;              // 车道数量
    private double laneWidth = 3.5;         // 车道宽度（米）
    private double roadLength = 1000.0;     // 道路长度（米）
    private double curvatureRadius = 500.0; // 曲率半径（米，0表示直道）

    // 交通参数
    private int vehicleDensity = 20;        // 车辆密度（辆/公里）
    private double speedLimit = 120.0;      // 速度限制（km/h）
    private double targetSpeed = 80.0;      // 目标速度（km/h）

    // 天气参数
    private double visibility = 1000.0;     // 能见度（米）
    private double frictionCoeff = 0.8;     // 路面摩擦系数

    // 仿真参数
    private double timeStep = 0.05;         // 时间步长（秒）
    private int maxSteps = 2000;            // 最大步数
    private ScenarioType scenarioType = ScenarioType.HIGHWAY;

    // 奖励函数权重
    private double rewardSpeedWeight = 0.3;
    private double rewardLaneWeight = 0.4;
    private double rewardCollisionWeight = 1.0;
    private double rewardComfortWeight = 0.1;

    public EnvironmentConfig() {
    }

    /**
     * 创建测试场景配置
     */
    public static EnvironmentConfig createTestConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setLaneCount(2);
        config.setVehicleDensity(5);
        config.setRoadLength(500.0);
        config.setMaxSteps(1000);
        config.setScenarioType(ScenarioType.TEST);
        return config;
    }

    /**
     * 创建高速公路配置
     */
    public static EnvironmentConfig createHighwayConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setLaneCount(3);
        config.setVehicleDensity(20);
        config.setSpeedLimit(120.0);
        config.setScenarioType(ScenarioType.HIGHWAY);
        return config;
    }

    /**
     * 创建城市道路配置
     */
    public static EnvironmentConfig createUrbanConfig() {
        EnvironmentConfig config = new EnvironmentConfig();
        config.setLaneCount(2);
        config.setVehicleDensity(40);
        config.setSpeedLimit(60.0);
        config.setCurvatureRadius(100.0);
        config.setScenarioType(ScenarioType.URBAN);
        return config;
    }

    // Getters and Setters
    public int getLaneCount() {
        return laneCount;
    }

    public void setLaneCount(int laneCount) {
        this.laneCount = laneCount;
    }

    public double getLaneWidth() {
        return laneWidth;
    }

    public void setLaneWidth(double laneWidth) {
        this.laneWidth = laneWidth;
    }

    public double getRoadLength() {
        return roadLength;
    }

    public void setRoadLength(double roadLength) {
        this.roadLength = roadLength;
    }

    public double getCurvatureRadius() {
        return curvatureRadius;
    }

    public void setCurvatureRadius(double curvatureRadius) {
        this.curvatureRadius = curvatureRadius;
    }

    public int getVehicleDensity() {
        return vehicleDensity;
    }

    public void setVehicleDensity(int vehicleDensity) {
        this.vehicleDensity = vehicleDensity;
    }

    public double getSpeedLimit() {
        return speedLimit;
    }

    public void setSpeedLimit(double speedLimit) {
        this.speedLimit = speedLimit;
    }

    public double getTargetSpeed() {
        return targetSpeed;
    }

    public void setTargetSpeed(double targetSpeed) {
        this.targetSpeed = targetSpeed;
    }

    public double getVisibility() {
        return visibility;
    }

    public void setVisibility(double visibility) {
        this.visibility = visibility;
    }

    public double getFrictionCoeff() {
        return frictionCoeff;
    }

    public void setFrictionCoeff(double frictionCoeff) {
        this.frictionCoeff = frictionCoeff;
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

    public ScenarioType getScenarioType() {
        return scenarioType;
    }

    public void setScenarioType(ScenarioType scenarioType) {
        this.scenarioType = scenarioType;
    }

    public double getRewardSpeedWeight() {
        return rewardSpeedWeight;
    }

    public void setRewardSpeedWeight(double rewardSpeedWeight) {
        this.rewardSpeedWeight = rewardSpeedWeight;
    }

    public double getRewardLaneWeight() {
        return rewardLaneWeight;
    }

    public void setRewardLaneWeight(double rewardLaneWeight) {
        this.rewardLaneWeight = rewardLaneWeight;
    }

    public double getRewardCollisionWeight() {
        return rewardCollisionWeight;
    }

    public void setRewardCollisionWeight(double rewardCollisionWeight) {
        this.rewardCollisionWeight = rewardCollisionWeight;
    }

    public double getRewardComfortWeight() {
        return rewardComfortWeight;
    }

    public void setRewardComfortWeight(double rewardComfortWeight) {
        this.rewardComfortWeight = rewardComfortWeight;
    }

    @Override
    public String toString() {
        return String.format("EnvConfig[type=%s, lanes=%d, speedLimit=%.0f km/h]",
                scenarioType.getName(), laneCount, speedLimit);
    }
}
