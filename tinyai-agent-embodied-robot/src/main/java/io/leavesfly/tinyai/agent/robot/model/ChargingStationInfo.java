package io.leavesfly.tinyai.agent.robot.model;

/**
 * 充电站信息类
 * 
 * <p>表示充电站的位置和状态信息。</p>
 * 
 * @author TinyAI Team
 */
public class ChargingStationInfo {
    /**
     * 充电站位置
     */
    private Vector2D position;
    
    /**
     * 与机器人的距离
     */
    private double distance;
    
    /**
     * 方向角度（弧度）
     */
    private double direction;
    
    /**
     * 是否被占用
     */
    private boolean isOccupied;
    
    /**
     * 充电速率（%/秒）
     */
    private double chargingRate;
    
    /**
     * 构造函数
     * 
     * @param position 充电站位置
     */
    public ChargingStationInfo(Vector2D position) {
        this.position = position;
        this.distance = 0.0;
        this.direction = 0.0;
        this.isOccupied = false;
        this.chargingRate = 10.0; // 默认充电速率：10%/秒
    }
    
    /**
     * 更新相对于机器人的信息
     * 
     * @param robotPosition 机器人位置
     * @param robotHeading 机器人朝向
     */
    public void updateRelativeInfo(Vector2D robotPosition, double robotHeading) {
        // 计算距离
        this.distance = position.distanceTo(robotPosition);
        
        // 计算方向角度
        Vector2D diff = position.subtract(robotPosition);
        double absoluteAngle = diff.angle();
        this.direction = absoluteAngle - robotHeading;
        
        // 归一化到 [-π, π]
        while (this.direction > Math.PI) {
            this.direction -= 2 * Math.PI;
        }
        while (this.direction < -Math.PI) {
            this.direction += 2 * Math.PI;
        }
    }
    
    /**
     * 判断机器人是否到达充电站
     * 
     * @param robotPosition 机器人位置
     * @return 是否到达
     */
    public boolean isReached(Vector2D robotPosition) {
        return distance < 0.3; // 30cm以内认为到达
    }
    
    // Getters and Setters
    public Vector2D getPosition() {
        return position;
    }
    
    public void setPosition(Vector2D position) {
        this.position = position;
    }
    
    public double getDistance() {
        return distance;
    }
    
    public void setDistance(double distance) {
        this.distance = distance;
    }
    
    public double getDirection() {
        return direction;
    }
    
    public void setDirection(double direction) {
        this.direction = direction;
    }
    
    public boolean isOccupied() {
        return isOccupied;
    }
    
    public void setOccupied(boolean occupied) {
        isOccupied = occupied;
    }
    
    public double getChargingRate() {
        return chargingRate;
    }
    
    public void setChargingRate(double chargingRate) {
        this.chargingRate = chargingRate;
    }
    
    @Override
    public String toString() {
        return String.format("ChargingStationInfo(pos=%s, distance=%.2f, direction=%.2f)",
                             position, distance, direction);
    }
}
