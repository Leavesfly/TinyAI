package io.leavesfly.tinyai.agent.robot.model;

/**
 * 机器人状态类
 * 
 * <p>表示扫地机器人的完整状态，包括位置、朝向、速度、电量等。</p>
 * 
 * @author TinyAI Team
 */
public class RobotState {
    /**
     * 机器人位置（米）
     */
    private Vector2D position;
    
    /**
     * 朝向角度（弧度，范围[0, 2π)）
     */
    private double heading;
    
    /**
     * 线速度（m/s，范围[0, 0.5]）
     */
    private double linearSpeed;
    
    /**
     * 角速度（rad/s，范围[-π/2, π/2]）
     */
    private double angularSpeed;
    
    /**
     * 电池电量（百分比，范围[0, 100]）
     */
    private double batteryLevel;
    
    /**
     * 尘盒容量（百分比，范围[0, 100]）
     */
    private double dustCapacity;
    
    /**
     * 刷子转速（RPM，范围[0, 5000]）
     */
    private double brushSpeed;
    
    /**
     * 是否正在清扫
     */
    private boolean isCleaning;
    
    /**
     * 构造函数
     * 
     * @param position 位置
     * @param heading 朝向
     */
    public RobotState(Vector2D position, double heading) {
        this.position = new Vector2D(position);
        this.heading = heading;
        this.linearSpeed = 0.0;
        this.angularSpeed = 0.0;
        this.batteryLevel = 100.0;
        this.dustCapacity = 0.0;
        this.brushSpeed = 0.0;
        this.isCleaning = false;
    }
    
    /**
     * 默认构造函数
     */
    public RobotState() {
        this(new Vector2D(0, 0), 0.0);
    }
    
    /**
     * 拷贝构造函数
     * 
     * @param other 要拷贝的状态
     */
    public RobotState(RobotState other) {
        this.position = new Vector2D(other.position);
        this.heading = other.heading;
        this.linearSpeed = other.linearSpeed;
        this.angularSpeed = other.angularSpeed;
        this.batteryLevel = other.batteryLevel;
        this.dustCapacity = other.dustCapacity;
        this.brushSpeed = other.brushSpeed;
        this.isCleaning = other.isCleaning;
    }
    
    /**
     * 判断是否需要充电
     * 
     * @return 是否需要充电
     */
    public boolean needsCharging() {
        return batteryLevel < 20.0;
    }
    
    /**
     * 判断尘盒是否需要清空
     * 
     * @return 是否需要清空
     */
    public boolean needsEmptying() {
        return dustCapacity > 90.0;
    }
    
    /**
     * 判断是否可正常工作
     * 
     * @return 是否可正常工作
     */
    public boolean isOperational() {
        return batteryLevel > 5.0 && dustCapacity < 100.0;
    }
    
    // Getters and Setters
    public Vector2D getPosition() {
        return position;
    }
    
    public void setPosition(Vector2D position) {
        this.position = position;
    }
    
    public double getHeading() {
        return heading;
    }
    
    public void setHeading(double heading) {
        // 归一化到 [0, 2π)
        this.heading = (heading % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
    }
    
    public double getLinearSpeed() {
        return linearSpeed;
    }
    
    public void setLinearSpeed(double linearSpeed) {
        this.linearSpeed = Math.max(0.0, Math.min(0.5, linearSpeed));
    }
    
    public double getAngularSpeed() {
        return angularSpeed;
    }
    
    public void setAngularSpeed(double angularSpeed) {
        this.angularSpeed = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, angularSpeed));
    }
    
    public double getBatteryLevel() {
        return batteryLevel;
    }
    
    public void setBatteryLevel(double batteryLevel) {
        this.batteryLevel = Math.max(0.0, Math.min(100.0, batteryLevel));
    }
    
    public double getDustCapacity() {
        return dustCapacity;
    }
    
    public void setDustCapacity(double dustCapacity) {
        this.dustCapacity = Math.max(0.0, Math.min(100.0, dustCapacity));
    }
    
    public double getBrushSpeed() {
        return brushSpeed;
    }
    
    public void setBrushSpeed(double brushSpeed) {
        this.brushSpeed = Math.max(0.0, Math.min(5000.0, brushSpeed));
    }
    
    public boolean isCleaning() {
        return isCleaning;
    }
    
    public void setCleaning(boolean cleaning) {
        isCleaning = cleaning;
    }
    
    @Override
    public String toString() {
        return String.format("RobotState(pos=%s, heading=%.2f, battery=%.1f%%, dust=%.1f%%, cleaning=%b)",
                             position, heading, batteryLevel, dustCapacity, isCleaning);
    }
}
