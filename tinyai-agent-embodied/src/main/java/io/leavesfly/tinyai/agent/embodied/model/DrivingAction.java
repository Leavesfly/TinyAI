package io.leavesfly.tinyai.agent.embodied.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 驾驶动作类
 * 定义车辆的控制指令
 *
 * @author TinyAI Team
 */
public class DrivingAction {
    private double steering;   // 转向角 [-1.0, 1.0]
    private double throttle;   // 油门 [0.0, 1.0]
    private double brake;      // 刹车 [0.0, 1.0]

    public DrivingAction() {
        this.steering = 0.0;
        this.throttle = 0.0;
        this.brake = 0.0;
    }

    public DrivingAction(double steering, double throttle, double brake) {
        this.steering = steering;
        this.throttle = throttle;
        this.brake = brake;
    }

    /**
     * 从NdArray创建动作
     * @param actionArray 包含[steering, throttle, brake]的数组
     */
    public static DrivingAction fromArray(NdArray actionArray) {
        Shape shape = actionArray.getShape();
        if (shape.size() != 3) {
            throw new IllegalArgumentException("Action array must have 3 elements");
        }

        float steering = actionArray.get(0);
        float throttle = actionArray.get(1);
        float brake = actionArray.get(2);
        return new DrivingAction(steering, throttle, brake);
    }

    /**
     * 转换为NdArray
     */
    public NdArray toArray() {
        return NdArray.of(new float[]{(float)steering, (float)throttle, (float)brake}, 
                         Shape.of(3));
    }

    /**
     * 限制动作范围到有效区间
     */
    public void clip() {
        steering = Math.max(-1.0, Math.min(1.0, steering));
        throttle = Math.max(0.0, Math.min(1.0, throttle));
        brake = Math.max(0.0, Math.min(1.0, brake));
    }

    /**
     * 判断是否为空动作（无操作）
     */
    public boolean isNullAction() {
        return Math.abs(steering) < 1e-6 && Math.abs(throttle) < 1e-6 && Math.abs(brake) < 1e-6;
    }

    /**
     * 判断是否为紧急制动
     */
    public boolean isEmergencyBrake() {
        return brake > 0.8;
    }

    // Getters and Setters
    public double getSteering() {
        return steering;
    }

    public void setSteering(double steering) {
        this.steering = steering;
    }

    public double getThrottle() {
        return throttle;
    }

    public void setThrottle(double throttle) {
        this.throttle = throttle;
    }

    public double getBrake() {
        return brake;
    }

    public void setBrake(double brake) {
        this.brake = brake;
    }

    @Override
    public String toString() {
        return String.format("Action[steer=%.3f, throttle=%.3f, brake=%.3f]",
                steering, throttle, brake);
    }
}
