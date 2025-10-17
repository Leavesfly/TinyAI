package io.leavesfly.tinyai.agent.robot.env;

import io.leavesfly.tinyai.agent.robot.model.*;
import io.leavesfly.tinyai.agent.robot.dynamics.RobotDynamics;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 简单清扫环境实现
 * 
 * <p>提供基本的清扫环境仿真功能。</p>
 * 
 * @author TinyAI Team
 */
public class SimpleCleaningEnv implements CleaningEnvironment {
    private EnvironmentConfig config;
    private RobotDynamics dynamics;
    private CleaningState currentState;
    private List<ObstacleInfo> obstacles;
    private ChargingStationInfo chargingStation;
    private FloorMap floorMap;
    private Random random;
    
    private int stepCount;
    private double totalReward;
    private boolean terminated;
    
    /**
     * 构造函数
     * 
     * @param config 环境配置
     */
    public SimpleCleaningEnv(EnvironmentConfig config) {
        this.config = config;
        this.dynamics = new RobotDynamics();
        this.random = new Random();
        this.obstacles = new ArrayList<>();
        
        // 初始化地图
        int mapWidth = (int) (config.getRoomWidth() / config.getGridSize());
        int mapHeight = (int) (config.getRoomHeight() / config.getGridSize());
        this.floorMap = new FloorMap(mapWidth, mapHeight, config.getGridSize());
    }
    
    @Override
    public CleaningState reset() {
        // 重置计数器
        stepCount = 0;
        totalReward = 0.0;
        terminated = false;
        
        // 重置地图
        floorMap.reset();
        floorMap.setDustDistribution("uniform");
        
        // 生成障碍物
        generateObstacles();
        
        // 设置充电站（房间角落）
        chargingStation = new ChargingStationInfo(
            new Vector2D(0.5, 0.5)
        );
        
        // 初始化机器人状态（从充电站开始）
        RobotState robotState = new RobotState(
            new Vector2D(0.5, 0.5), 
            0.0
        );
        robotState.setBatteryLevel(config.getInitialBattery());
        robotState.setDustCapacity(0.0);
        
        // 创建初始观测
        currentState = new CleaningState();
        currentState.setRobotState(robotState);
        currentState.setObstacleMap(obstacles);
        currentState.setFloorMap(floorMap);
        currentState.setChargingStationInfo(chargingStation);
        
        // 更新传感器数据
        updateSensorData();
        
        return currentState;
    }
    
    @Override
    public StepResult step(CleaningAction action) {
        if (terminated) {
            throw new IllegalStateException("Environment is terminated. Call reset() first.");
        }
        
        stepCount++;
        
        // 执行动作，更新机器人状态
        RobotState oldState = currentState.getRobotState();
        RobotState newState = dynamics.update(oldState, action, config.getTimeStep());
        
        // 碰撞检测
        boolean collision = checkCollision(newState.getPosition());
        if (collision) {
            // 碰撞后恢复到原位置
            newState.setPosition(oldState.getPosition());
            newState.setLinearSpeed(0.0);
            newState.setAngularSpeed(0.0);
        }
        
        // 更新地图（标记已清扫区域）
        if (newState.isCleaning() && !collision) {
            double efficiency = dynamics.computeCleaningEfficiency(
                FloorType.TILE, action
            );
            floorMap.markCleaned(newState.getPosition(), efficiency);
        }
        
        // 更新充电站相对信息
        chargingStation.updateRelativeInfo(
            newState.getPosition(), 
            newState.getHeading()
        );
        
        // 更新障碍物相对信息
        for (ObstacleInfo obstacle : obstacles) {
            obstacle.updateRelativeInfo(
                newState.getPosition(), 
                newState.getHeading()
            );
        }
        
        // 创建新观测
        CleaningState newObservation = new CleaningState();
        newObservation.setRobotState(newState);
        newObservation.setObstacleMap(obstacles);
        newObservation.setFloorMap(floorMap);
        newObservation.setChargingStationInfo(chargingStation);
        
        currentState = newObservation;
        updateSensorData();
        
        // 计算奖励
        double reward = computeReward(oldState, newState, action, collision);
        totalReward += reward;
        
        // 检查终止条件
        terminated = checkTermination(newState);
        
        // 创建结果
        StepResult result = new StepResult(newObservation, reward, terminated);
        result.addInfo("step", stepCount);
        result.addInfo("coverage", floorMap.getCoverageRate());
        result.addInfo("battery", newState.getBatteryLevel());
        result.addInfo("collision", collision);
        
        return result;
    }
    
    @Override
    public CleaningState getObservation() {
        return currentState;
    }
    
    @Override
    public boolean isTerminated() {
        return terminated;
    }
    
    @Override
    public NdArray getSensorData(SensorType type) {
        // 简化实现：返回模拟传感器数据
        switch (type) {
            case CAMERA:
                return NdArray.zeros(Shape.of(256));
            case LIDAR:
                return generateLidarData();
            case CLIFF_SENSOR:
                return NdArray.zeros(Shape.of(4));
            case BUMP_SENSOR:
                return NdArray.zeros(Shape.of(8));
            case DIRT_SENSOR:
                return NdArray.of(new float[]{
                    (float) floorMap.getDustAt(currentState.getRobotState().getPosition())
                }, Shape.of(1));
            case ODOMETER:
                return NdArray.of(new float[]{
                    (float) currentState.getRobotState().getLinearSpeed(),
                    (float) currentState.getRobotState().getAngularSpeed(),
                    (float) currentState.getRobotState().getHeading()
                }, Shape.of(3));
            default:
                return NdArray.zeros(Shape.of(1));
        }
    }
    
    @Override
    public Object render() {
        // 可选：实现可视化
        return null;
    }
    
    @Override
    public void close() {
        // 清理资源
        obstacles.clear();
    }
    
    @Override
    public ScenarioType getScenarioType() {
        return config.getScenarioType();
    }
    
    @Override
    public EnvironmentConfig getConfig() {
        return config;
    }
    
    /**
     * 生成障碍物
     */
    private void generateObstacles() {
        obstacles.clear();
        int count = config.getObstacleCount();
        
        for (int i = 0; i < count; i++) {
            // 随机位置
            double x = random.nextDouble() * config.getRoomWidth();
            double y = random.nextDouble() * config.getRoomHeight();
            
            // 随机大小
            double width = 0.3 + random.nextDouble() * 0.5;
            double length = 0.3 + random.nextDouble() * 0.5;
            double height = 0.5 + random.nextDouble() * 1.0;
            
            ObstacleInfo obstacle = new ObstacleInfo(
                ObstacleType.FURNITURE,
                new Vector2D(x, y),
                new BoundingBox(width, length, height)
            );
            
            obstacles.add(obstacle);
        }
    }
    
    /**
     * 检查碰撞
     */
    private boolean checkCollision(Vector2D position) {
        double radius = RobotDynamics.getRobotRadius();
        
        // 检查墙壁碰撞
        if (position.getX() < radius || position.getX() > config.getRoomWidth() - radius ||
            position.getY() < radius || position.getY() > config.getRoomHeight() - radius) {
            return true;
        }
        
        // 检查障碍物碰撞
        for (ObstacleInfo obstacle : obstacles) {
            double distance = position.distanceTo(obstacle.getPosition());
            double minDist = radius + obstacle.getBoundingBox().getWidth() / 2.0;
            if (distance < minDist) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 计算奖励
     */
    private double computeReward(RobotState oldState, RobotState newState, 
                                 CleaningAction action, boolean collision) {
        double reward = 0.0;
        
        // 覆盖奖励（主要奖励）
        double oldCoverage = 0.0; // 简化：可以记录上一步覆盖率
        double newCoverage = floorMap.getCoverageRate();
        double coverageIncrease = newCoverage - oldCoverage;
        reward += coverageIncrease * 10.0;
        
        // 碰撞惩罚
        if (collision) {
            reward -= 10.0;
        }
        
        // 能量惩罚
        double energyUsed = oldState.getBatteryLevel() - newState.getBatteryLevel();
        reward -= energyUsed * 0.2;
        
        // 时间惩罚（鼓励快速完成）
        reward -= 0.01;
        
        // 达到目标奖励
        if (newCoverage >= config.getTargetCoverage()) {
            reward += 100.0;
        }
        
        return reward;
    }
    
    /**
     * 检查终止条件
     */
    private boolean checkTermination(RobotState state) {
        // 达到目标覆盖率
        if (floorMap.getCoverageRate() >= config.getTargetCoverage()) {
            return true;
        }
        
        // 电量耗尽
        if (state.getBatteryLevel() <= 0.0) {
            return true;
        }
        
        // 超过最大步数
        if (stepCount >= config.getMaxSteps()) {
            return true;
        }
        
        return false;
    }
    
    /**
     * 更新传感器数据
     */
    private void updateSensorData() {
        // 更新视觉特征（简化：使用随机噪声）
        NdArray visualFeatures = NdArray.likeRandom(0.0f, 0.1f, Shape.of(256));
        currentState.setVisualFeatures(visualFeatures);
        
        // 更新雷达特征
        currentState.setLidarFeatures(generateLidarData());
    }
    
    /**
     * 生成激光雷达数据
     */
    private NdArray generateLidarData() {
        float[] lidarData = new float[128];
        Vector2D position = currentState.getRobotState().getPosition();
        double heading = currentState.getRobotState().getHeading();
        
        // 每2.8度一个测量点
        for (int i = 0; i < 128; i++) {
            double angle = heading + (i * 2.8 * Math.PI / 180.0);
            double distance = raycast(position, angle);
            lidarData[i] = (float) Math.min(distance / 5.0, 1.0); // 归一化到[0,1]
        }
        
        return NdArray.of(lidarData, Shape.of(128));
    }
    
    /**
     * 射线检测
     */
    private double raycast(Vector2D position, double angle) {
        double maxDistance = 5.0;
        double step = 0.1;
        
        for (double d = 0; d < maxDistance; d += step) {
            double x = position.getX() + d * Math.cos(angle);
            double y = position.getY() + d * Math.sin(angle);
            
            // 检查墙壁
            if (x < 0 || x > config.getRoomWidth() || 
                y < 0 || y > config.getRoomHeight()) {
                return d;
            }
            
            // 检查障碍物
            Vector2D testPoint = new Vector2D(x, y);
            for (ObstacleInfo obstacle : obstacles) {
                if (testPoint.distanceTo(obstacle.getPosition()) < 
                    obstacle.getBoundingBox().getWidth() / 2.0) {
                    return d;
                }
            }
        }
        
        return maxDistance;
    }
}
