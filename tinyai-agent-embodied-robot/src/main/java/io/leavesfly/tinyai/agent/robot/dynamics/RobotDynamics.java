package io.leavesfly.tinyai.agent.robot.dynamics;

import io.leavesfly.tinyai.agent.robot.model.*;

/**
 * 机器人动力学模型
 * 
 * <p>实现差分驱动机器人的运动学模型和能量消耗模型。</p>
 * 
 * @author TinyAI Team
 */
public class RobotDynamics {
    /**
     * 轮距（米）
     */
    private static final double WHEEL_BASE = 0.25;
    
    /**
     * 最大线速度（m/s）
     */
    private static final double MAX_LINEAR_SPEED = 0.5;
    
    /**
     * 最大角速度（rad/s）
     */
    private static final double MAX_ANGULAR_SPEED = Math.PI / 2;
    
    /**
     * 最大加速度（m/s²）
     */
    private static final double MAX_ACCELERATION = 0.5;
    
    /**
     * 机器人半径（米）
     */
    private static final double ROBOT_RADIUS = 0.175;
    
    /**
     * 电池容量（mAh）
     */
    private static final double BATTERY_CAPACITY = 3000.0;
    
    /**
     * 尘盒容量（升）
     */
    private static final double DUST_CAPACITY = 0.6;
    
    /**
     * 最大刷速（RPM）
     */
    private static final double MAX_BRUSH_SPEED = 5000.0;
    
    /**
     * 基础功耗（%/秒）
     */
    private static final double BASE_CONSUMPTION = 0.1;
    
    /**
     * 更新机器人状态
     * 
     * @param current 当前状态
     * @param action 控制动作
     * @param dt 时间步长（秒）
     * @return 新状态
     */
    public RobotState update(RobotState current, CleaningAction action, double dt) {
        RobotState newState = new RobotState(current);
        
        // 计算实际速度（考虑加速度限制）
        double targetLinearSpeed = action.getLinearVelocity() * MAX_LINEAR_SPEED;
        double targetAngularSpeed = action.getAngularVelocity() * MAX_ANGULAR_SPEED;
        
        double currentLinearSpeed = current.getLinearSpeed();
        double currentAngularSpeed = current.getAngularSpeed();
        
        // 速度平滑过渡（限制加速度）
        double newLinearSpeed = smoothTransition(currentLinearSpeed, targetLinearSpeed, 
                                                 MAX_ACCELERATION * dt);
        double newAngularSpeed = smoothTransition(currentAngularSpeed, targetAngularSpeed, 
                                                  MAX_ANGULAR_SPEED * dt);
        
        newState.setLinearSpeed(newLinearSpeed);
        newState.setAngularSpeed(newAngularSpeed);
        
        // 更新位置和朝向（差分驱动运动学模型）
        double heading = current.getHeading();
        Vector2D position = current.getPosition();
        
        double newX = position.getX() + newLinearSpeed * Math.cos(heading) * dt;
        double newY = position.getY() + newLinearSpeed * Math.sin(heading) * dt;
        double newHeading = heading + newAngularSpeed * dt;
        
        newState.setPosition(new Vector2D(newX, newY));
        newState.setHeading(newHeading);
        
        // 更新刷子状态
        double targetBrushSpeed = action.getBrushPower() * MAX_BRUSH_SPEED;
        newState.setBrushSpeed(targetBrushSpeed);
        newState.setCleaning(action.getBrushPower() > 0.1 || action.getSuctionPower() > 0.1);
        
        // 计算能量消耗
        double energyConsumed = computeEnergyConsumption(action, newLinearSpeed, 
                                                         newAngularSpeed, dt);
        double newBattery = current.getBatteryLevel() - energyConsumed;
        newState.setBatteryLevel(newBattery);
        
        // 更新尘盒（简化模型：按清扫功率累积）
        if (newState.isCleaning()) {
            double dustCollected = (action.getBrushPower() + action.getSuctionPower()) * 0.01 * dt;
            double newDustCapacity = current.getDustCapacity() + dustCollected;
            newState.setDustCapacity(newDustCapacity);
        }
        
        return newState;
    }
    
    /**
     * 计算能量消耗
     * 
     * @param action 动作
     * @param linearSpeed 实际线速度
     * @param angularSpeed 实际角速度
     * @param dt 时间步长
     * @return 消耗的电量（百分比）
     */
    public double computeEnergyConsumption(CleaningAction action, double linearSpeed, 
                                           double angularSpeed, double dt) {
        // 基础功耗
        double consumption = BASE_CONSUMPTION * dt;
        
        // 运动功耗
        consumption += 0.3 * Math.abs(linearSpeed) * dt;
        consumption += 0.2 * Math.abs(angularSpeed) * dt;
        
        // 刷子功耗
        consumption += 0.4 * action.getBrushPower() * dt;
        
        // 吸尘功耗
        consumption += 0.5 * action.getSuctionPower() * dt;
        
        return consumption;
    }
    
    /**
     * 计算清扫效率
     * 
     * @param floorType 地面类型
     * @param action 清扫动作
     * @return 清扫效率（0-1）
     */
    public double computeCleaningEfficiency(FloorType floorType, CleaningAction action) {
        // 基础效率由地面类型决定
        double baseEfficiency = floorType.getCleaningEfficiency();
        
        // 刷子和吸力的综合影响
        double powerFactor = (action.getBrushPower() + action.getSuctionPower()) / 2.0;
        
        // 速度影响（速度太快效率降低）
        double speedPenalty = 1.0;
        if (Math.abs(action.getLinearVelocity()) > 0.5) {
            speedPenalty = 0.7;
        }
        
        return baseEfficiency * powerFactor * speedPenalty;
    }
    
    /**
     * 平滑过渡函数（限制变化率）
     * 
     * @param current 当前值
     * @param target 目标值
     * @param maxChange 最大变化量
     * @return 新值
     */
    private double smoothTransition(double current, double target, double maxChange) {
        double diff = target - current;
        if (Math.abs(diff) <= maxChange) {
            return target;
        }
        return current + Math.signum(diff) * maxChange;
    }
    
    // Getters for constants
    public static double getWheelBase() {
        return WHEEL_BASE;
    }
    
    public static double getMaxLinearSpeed() {
        return MAX_LINEAR_SPEED;
    }
    
    public static double getMaxAngularSpeed() {
        return MAX_ANGULAR_SPEED;
    }
    
    public static double getRobotRadius() {
        return ROBOT_RADIUS;
    }
    
    public static double getBatteryCapacity() {
        return BATTERY_CAPACITY;
    }
    
    public static double getDustCapacity() {
        return DUST_CAPACITY;
    }
    
    public static double getMaxBrushSpeed() {
        return MAX_BRUSH_SPEED;
    }
}
