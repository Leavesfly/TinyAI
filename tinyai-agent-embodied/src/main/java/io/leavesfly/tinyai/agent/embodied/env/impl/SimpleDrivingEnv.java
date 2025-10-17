package io.leavesfly.tinyai.agent.embodied.env.impl;

import io.leavesfly.tinyai.agent.embodied.dynamics.VehicleDynamics;
import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 简单驾驶环境实现
 * 提供基础的直道和简单障碍物场景
 *
 * @author TinyAI Team
 */
public class SimpleDrivingEnv implements DrivingEnvironment {
    private EnvironmentConfig config;
    private VehicleDynamics dynamics;
    private VehicleState egoVehicle;  // 自车状态
    private List<ObstacleInfo> obstacles;  // 障碍物列表
    private LaneGeometry laneInfo;
    
    private int currentStep;
    private boolean terminated;
    private Random random;
    
    // 环境状态
    private double roadProgress;  // 道路前进距离
    private int collisionCount;
    private int laneViolationCount;

    public SimpleDrivingEnv() {
        this(EnvironmentConfig.createTestConfig());
    }

    public SimpleDrivingEnv(EnvironmentConfig config) {
        this.config = config;
        this.dynamics = new VehicleDynamics();
        this.dynamics.setFrictionCoeff(config.getFrictionCoeff());
        this.obstacles = new ArrayList<>();
        this.random = new Random();
        reset();
    }

    @Override
    public PerceptionState reset() {
        // 重置车辆状态
        egoVehicle = new VehicleState();
        egoVehicle.setPosition(new Vector3D(0, config.getLaneWidth() * config.getLaneCount() / 2.0, 0));
        egoVehicle.setHeading(0.0);  // 朝向正东
        egoVehicle.setSpeed(config.getTargetSpeed() / 3.6);  // 转换为m/s
        
        // 重置车道信息
        laneInfo = new LaneGeometry(config.getLaneCount() / 2, config.getLaneWidth());
        laneInfo.setLateralDeviation(0.0);
        laneInfo.setCurvature(0.0);
        laneInfo.setLeftLaneAvailable(true);
        laneInfo.setRightLaneAvailable(true);
        
        // 生成障碍物
        generateObstacles();
        
        // 重置计数器
        currentStep = 0;
        terminated = false;
        roadProgress = 0.0;
        collisionCount = 0;
        laneViolationCount = 0;
        
        return getObservation();
    }

    @Override
    public StepResult step(DrivingAction action) {
        currentStep++;
        
        // 限制动作范围
        action.clip();
        
        // 更新车辆状态
        egoVehicle = dynamics.update(egoVehicle, action, config.getTimeStep());
        
        // 更新道路进度
        roadProgress = egoVehicle.getPosition().getX();
        
        // 更新车道偏离
        updateLaneDeviation();
        
        // 更新障碍物相对位置
        updateObstacles();
        
        // 计算奖励
        double reward = calculateReward(action);
        
        // 检查终止条件
        checkTermination();
        
        // 构建结果
        StepResult result = new StepResult(getObservation(), reward, terminated);
        result.addInfo("step", currentStep);
        result.addInfo("speed", egoVehicle.getSpeed());
        result.addInfo("road_progress", roadProgress);
        result.addInfo("collision_count", collisionCount);
        
        return result;
    }

    @Override
    public PerceptionState getObservation() {
        PerceptionState state = new PerceptionState();
        state.setVehicleState(egoVehicle);
        state.setLaneInfo(laneInfo);
        state.setObstacleMap(new ArrayList<>(obstacles));
        
        // 简化的特征表示（实际应用中需要更复杂的处理）
        // 视觉特征：简单编码车道和障碍物信息
        float[] visualFeatures = new float[256];
        visualFeatures[0] = (float) laneInfo.getLateralDeviation();
        visualFeatures[1] = (float) laneInfo.getCurvature();
        state.setVisualFeatures(NdArray.of(visualFeatures));
        
        // 激光雷达特征：距离信息
        float[] lidarFeatures = new float[128];
        if (!obstacles.isEmpty()) {
            ObstacleInfo nearest = state.getNearestObstacle();
            if (nearest != null) {
                lidarFeatures[0] = (float) nearest.getDistance();
            }
        }
        state.setLidarFeatures(NdArray.of(lidarFeatures));
        
        return state;
    }

    @Override
    public boolean isTerminated() {
        return terminated;
    }

    @Override
    public NdArray getSensorData(SensorType sensorType) {
        // 简化实现，返回模拟数据
        switch (sensorType) {
            case CAMERA:
                return NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(224, 224, 3));
            case LIDAR:
                return NdArray.zeros(io.leavesfly.tinyai.ndarr.Shape.of(360, 3));
            case IMU:
                return NdArray.of(new float[]{
                    (float) egoVehicle.getAcceleration(),
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    (float) egoVehicle.getAngularVelocity()
                });
            case GPS:
                return NdArray.of(new float[]{
                    (float) egoVehicle.getPosition().getX(),
                    (float) egoVehicle.getPosition().getY()
                });
            case SPEEDOMETER:
                return NdArray.of((float) egoVehicle.getSpeed());
            default:
                return NdArray.of(0.0f);
        }
    }

    @Override
    public Object render() {
        // 简单的文本渲染
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("=== Step %d ===\n", currentStep));
        sb.append(String.format("Vehicle: pos=(%.1f, %.1f), speed=%.1f m/s, heading=%.2f rad\n",
                egoVehicle.getPosition().getX(), egoVehicle.getPosition().getY(),
                egoVehicle.getSpeed(), egoVehicle.getHeading()));
        sb.append(String.format("Lane: deviation=%.2f m\n", laneInfo.getLateralDeviation()));
        sb.append(String.format("Obstacles: %d\n", obstacles.size()));
        sb.append(String.format("Progress: %.1f m\n", roadProgress));
        return sb.toString();
    }

    @Override
    public EnvironmentConfig getConfig() {
        return config;
    }

    @Override
    public void setConfig(EnvironmentConfig config) {
        this.config = config;
        this.dynamics.setFrictionCoeff(config.getFrictionCoeff());
    }

    @Override
    public ScenarioType getScenarioType() {
        return config.getScenarioType();
    }

    @Override
    public void close() {
        // 清理资源
        obstacles.clear();
    }

    /**
     * 生成障碍物
     */
    private void generateObstacles() {
        obstacles.clear();
        
        int numObstacles = config.getVehicleDensity() * (int)(config.getRoadLength() / 1000.0);
        
        for (int i = 0; i < numObstacles; i++) {
            // 随机位置
            double x = 50 + random.nextDouble() * (config.getRoadLength() - 50);
            double y = random.nextInt(config.getLaneCount()) * config.getLaneWidth() 
                      + config.getLaneWidth() / 2.0;
            
            Vector3D position = new Vector3D(x - egoVehicle.getPosition().getX(), 
                                            y - egoVehicle.getPosition().getY(), 0);
            
            // 随机速度（相对速度）
            double speedDiff = (random.nextDouble() - 0.5) * 10.0;  // -5 到 +5 m/s
            Vector3D velocity = new Vector3D(speedDiff, 0, 0);
            
            // 车辆包围盒
            BoundingBox bbox = new BoundingBox(4.5, 1.8, 1.5);
            
            ObstacleInfo obstacle = new ObstacleInfo(
                ObstacleType.VEHICLE,
                position,
                velocity,
                bbox,
                0.95
            );
            
            obstacles.add(obstacle);
        }
    }

    /**
     * 更新车道偏离
     */
    private void updateLaneDeviation() {
        double currentY = egoVehicle.getPosition().getY();
        int currentLane = (int)(currentY / config.getLaneWidth());
        double laneCenter = (currentLane + 0.5) * config.getLaneWidth();
        
        double deviation = currentY - laneCenter;
        laneInfo.setLateralDeviation(deviation);
        laneInfo.setLaneId(currentLane);
        
        // 检查车道偏离
        if (Math.abs(deviation) > config.getLaneWidth() / 2.0) {
            laneViolationCount++;
        }
    }

    /**
     * 更新障碍物相对位置
     */
    private void updateObstacles() {
        for (ObstacleInfo obstacle : obstacles) {
            // 更新相对位置
            Vector3D relPos = obstacle.getPosition();
            relPos.setX(relPos.getX() - egoVehicle.getSpeed() * config.getTimeStep());
            
            // 移除超出范围的障碍物
            obstacles.removeIf(obs -> obs.getPosition().getX() < -50);
        }
    }

    /**
     * 计算奖励
     */
    private double calculateReward(DrivingAction action) {
        double reward = 0.0;
        
        // 1. 速度奖励：鼓励接近目标速度
        double targetSpeedMs = config.getTargetSpeed() / 3.6;
        double speedReward = 1.0 - Math.abs(egoVehicle.getSpeed() - targetSpeedMs) / targetSpeedMs;
        reward += config.getRewardSpeedWeight() * speedReward;
        
        // 2. 车道保持奖励
        double laneReward = Math.exp(-Math.pow(laneInfo.getLateralDeviation(), 2));
        reward += config.getRewardLaneWeight() * laneReward;
        
        // 3. 碰撞惩罚
        if (checkCollision()) {
            reward -= config.getRewardCollisionWeight() * 100.0;
            collisionCount++;
        }
        
        // 4. 舒适性奖励：惩罚急加速、急转向
        double comfortPenalty = Math.abs(egoVehicle.getAcceleration()) 
                              + Math.abs(egoVehicle.getSteeringAngle()) * 2.0;
        reward -= config.getRewardComfortWeight() * comfortPenalty;
        
        // 5. 进度奖励：鼓励前进
        reward += 0.1;
        
        return reward;
    }

    /**
     * 检查碰撞
     */
    private boolean checkCollision() {
        for (ObstacleInfo obstacle : obstacles) {
            double distance = obstacle.getDistance();
            
            // 简化的碰撞检测：基于距离
            if (distance < 3.0) {  // 碰撞阈值3米
                return true;
            }
        }
        
        // 检查是否驶出道路
        double y = egoVehicle.getPosition().getY();
        if (y < 0 || y > config.getLaneCount() * config.getLaneWidth()) {
            return true;
        }
        
        return false;
    }

    /**
     * 检查终止条件
     */
    private void checkTermination() {
        // 1. 达到最大步数
        if (currentStep >= config.getMaxSteps()) {
            terminated = true;
            return;
        }
        
        // 2. 发生碰撞
        if (collisionCount > 0) {
            terminated = true;
            return;
        }
        
        // 3. 完成道路
        if (roadProgress >= config.getRoadLength()) {
            terminated = true;
            return;
        }
        
        // 4. 速度过低（停车）
        if (egoVehicle.getSpeed() < 0.1 && currentStep > 10) {
            terminated = true;
            return;
        }
    }

    // Getters
    public VehicleState getEgoVehicle() {
        return egoVehicle;
    }

    public int getCurrentStep() {
        return currentStep;
    }

    public double getRoadProgress() {
        return roadProgress;
    }

    public int getCollisionCount() {
        return collisionCount;
    }

    public int getLaneViolationCount() {
        return laneViolationCount;
    }
}
