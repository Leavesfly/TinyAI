package io.leavesfly.tinyai.agent.embodied.dynamics;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.VehicleState;

/**
 * 车辆动力学模型
 * 采用简化的自行车模型（Bicycle Model）描述车辆运动
 *
 * @author TinyAI Team
 */
public class VehicleDynamics {
    // 车辆物理参数
    private double wheelbase = 2.7;           // 轴距（米）
    private double maxSteeringAngle = 0.6;    // 最大转向角（弧度，约34度）
    private double maxAcceleration = 3.0;     // 最大加速度（m/s²）
    private double maxDeceleration = 8.0;     // 最大减速度（m/s²）
    private double frictionCoeff = 0.8;       // 路面摩擦系数

    // 控制参数
    private double throttleGain = 3.0;        // 油门增益
    private double brakeGain = 8.0;           // 刹车增益
    private double steeringGain = 0.6;        // 转向增益

    public VehicleDynamics() {
    }

    public VehicleDynamics(double wheelbase, double frictionCoeff) {
        this.wheelbase = wheelbase;
        this.frictionCoeff = frictionCoeff;
    }

    /**
     * 更新车辆状态
     * 根据当前状态和控制指令，计算下一时刻的车辆状态
     *
     * @param currentState 当前车辆状态
     * @param action       控制动作
     * @param dt           时间步长（秒）
     * @return 更新后的车辆状态
     */
    public VehicleState update(VehicleState currentState, DrivingAction action, double dt) {
        // 创建新状态
        VehicleState newState = new VehicleState(currentState);

        // 1. 计算加速度
        double acceleration = computeAcceleration(action, currentState.getSpeed());

        // 2. 更新速度
        double newSpeed = currentState.getSpeed() + acceleration * dt;
        newSpeed = Math.max(0.0, newSpeed); // 速度不能为负
        newState.setSpeed(newSpeed);
        newState.setAcceleration(acceleration);

        // 3. 计算转向角
        double steeringAngle = action.getSteering() * steeringGain;
        steeringAngle = Math.max(-maxSteeringAngle, Math.min(maxSteeringAngle, steeringAngle));
        newState.setSteeringAngle(steeringAngle);

        // 4. 计算角速度（使用自行车模型）
        double angularVelocity = 0.0;
        if (Math.abs(newSpeed) > 1e-3) {
            angularVelocity = newSpeed * Math.tan(steeringAngle) / wheelbase;
        }
        newState.setAngularVelocity(angularVelocity);

        // 5. 更新航向角
        double newHeading = currentState.getHeading() + angularVelocity * dt;
        newHeading = normalizeAngle(newHeading);
        newState.setHeading(newHeading);

        // 6. 更新位置
        double vx = newSpeed * Math.cos(newHeading);
        double vy = newSpeed * Math.sin(newHeading);
        
        double newX = currentState.getPosition().getX() + vx * dt;
        double newY = currentState.getPosition().getY() + vy * dt;
        
        newState.getPosition().setX(newX);
        newState.getPosition().setY(newY);

        return newState;
    }

    /**
     * 计算加速度
     * 基于油门和刹车输入，考虑摩擦系数
     */
    private double computeAcceleration(DrivingAction action, double currentSpeed) {
        double acceleration = 0.0;

        // 油门和刹车互斥，刹车优先
        if (action.getBrake() > 0.01) {
            // 刹车
            acceleration = -action.getBrake() * brakeGain * frictionCoeff;
            acceleration = Math.max(-maxDeceleration, acceleration);
        } else if (action.getThrottle() > 0.01) {
            // 加速
            acceleration = action.getThrottle() * throttleGain;
            acceleration = Math.min(maxAcceleration, acceleration);
        } else {
            // 自然减速（空气阻力和滚动阻力）
            double dragForce = 0.1 + 0.01 * currentSpeed * currentSpeed;
            acceleration = -dragForce;
        }

        return acceleration;
    }

    /**
     * 归一化角度到 [-π, π]
     */
    private double normalizeAngle(double angle) {
        while (angle > Math.PI) {
            angle -= 2 * Math.PI;
        }
        while (angle < -Math.PI) {
            angle += 2 * Math.PI;
        }
        return angle;
    }

    /**
     * 计算转弯半径
     */
    public double getTurningRadius(double steeringAngle) {
        if (Math.abs(steeringAngle) < 1e-6) {
            return Double.POSITIVE_INFINITY;
        }
        return wheelbase / Math.tan(Math.abs(steeringAngle));
    }

    /**
     * 估计停车距离
     */
    public double getStoppingDistance(double currentSpeed) {
        if (currentSpeed < 1e-3) {
            return 0.0;
        }
        // 使用最大制动减速度
        return (currentSpeed * currentSpeed) / (2 * maxDeceleration * frictionCoeff);
    }

    // Getters and Setters
    public double getWheelbase() {
        return wheelbase;
    }

    public void setWheelbase(double wheelbase) {
        this.wheelbase = wheelbase;
    }

    public double getMaxSteeringAngle() {
        return maxSteeringAngle;
    }

    public void setMaxSteeringAngle(double maxSteeringAngle) {
        this.maxSteeringAngle = maxSteeringAngle;
    }

    public double getMaxAcceleration() {
        return maxAcceleration;
    }

    public void setMaxAcceleration(double maxAcceleration) {
        this.maxAcceleration = maxAcceleration;
    }

    public double getMaxDeceleration() {
        return maxDeceleration;
    }

    public void setMaxDeceleration(double maxDeceleration) {
        this.maxDeceleration = maxDeceleration;
    }

    public double getFrictionCoeff() {
        return frictionCoeff;
    }

    public void setFrictionCoeff(double frictionCoeff) {
        this.frictionCoeff = frictionCoeff;
        // 摩擦系数影响最大减速度
        this.maxDeceleration = 8.0 * frictionCoeff;
    }

    public double getThrottleGain() {
        return throttleGain;
    }

    public void setThrottleGain(double throttleGain) {
        this.throttleGain = throttleGain;
    }

    public double getBrakeGain() {
        return brakeGain;
    }

    public void setBrakeGain(double brakeGain) {
        this.brakeGain = brakeGain;
    }

    public double getSteeringGain() {
        return steeringGain;
    }

    public void setSteeringGain(double steeringGain) {
        this.steeringGain = steeringGain;
    }
}
