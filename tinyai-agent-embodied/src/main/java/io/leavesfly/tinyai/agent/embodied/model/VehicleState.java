package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 车辆状态类
 * 描述车辆的运动状态
 *
 * @author TinyAI Team
 */
public class VehicleState {
    private Vector3D position;      // 位置 (x, y, z)
    private double heading;         // 航向角（弧度）
    private double speed;           // 速度 (m/s)
    private double acceleration;    // 加速度 (m/s²)
    private double steeringAngle;   // 转向角（弧度）
    private double angularVelocity; // 角速度 (rad/s)

    public VehicleState() {
        this.position = new Vector3D();
        this.heading = 0.0;
        this.speed = 0.0;
        this.acceleration = 0.0;
        this.steeringAngle = 0.0;
        this.angularVelocity = 0.0;
    }

    /**
     * 复制构造函数
     */
    public VehicleState(VehicleState other) {
        this.position = new Vector3D(other.position.getX(), other.position.getY(), other.position.getZ());
        this.heading = other.heading;
        this.speed = other.speed;
        this.acceleration = other.acceleration;
        this.steeringAngle = other.steeringAngle;
        this.angularVelocity = other.angularVelocity;
    }

    /**
     * 获取速度的x和y分量
     */
    public Vector3D getVelocityVector() {
        double vx = speed * Math.cos(heading);
        double vy = speed * Math.sin(heading);
        return new Vector3D(vx, vy, 0.0);
    }

    // Getters and Setters
    public Vector3D getPosition() {
        return position;
    }

    public void setPosition(Vector3D position) {
        this.position = position;
    }

    public double getHeading() {
        return heading;
    }

    public void setHeading(double heading) {
        this.heading = heading;
    }

    public double getSpeed() {
        return speed;
    }

    public void setSpeed(double speed) {
        this.speed = speed;
    }

    public double getAcceleration() {
        return acceleration;
    }

    public void setAcceleration(double acceleration) {
        this.acceleration = acceleration;
    }

    public double getSteeringAngle() {
        return steeringAngle;
    }

    public void setSteeringAngle(double steeringAngle) {
        this.steeringAngle = steeringAngle;
    }

    public double getAngularVelocity() {
        return angularVelocity;
    }

    public void setAngularVelocity(double angularVelocity) {
        this.angularVelocity = angularVelocity;
    }

    @Override
    public String toString() {
        return String.format("VehicleState[pos=%s, heading=%.2f, speed=%.2f, accel=%.2f]",
                position, heading, speed, acceleration);
    }
}
