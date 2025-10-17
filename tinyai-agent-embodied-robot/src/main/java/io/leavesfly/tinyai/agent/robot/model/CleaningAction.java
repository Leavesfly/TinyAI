package io.leavesfly.tinyai.agent.robot.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 清扫动作类
 * 
 * <p>表示扫地机器人的控制动作，包括运动控制和清扫控制。</p>
 * 
 * @author TinyAI Team
 */
public class CleaningAction {
    /**
     * 线速度控制（范围[-1.0, 1.0]）
     */
    private double linearVelocity;
    
    /**
     * 角速度控制（范围[-1.0, 1.0]）
     */
    private double angularVelocity;
    
    /**
     * 刷子功率（范围[0.0, 1.0]）
     */
    private double brushPower;
    
    /**
     * 吸力大小（范围[0.0, 1.0]）
     */
    private double suctionPower;
    
    /**
     * 动作类型
     */
    private ActionType actionType;
    
    /**
     * 构造函数
     * 
     * @param linearVelocity 线速度控制
     * @param angularVelocity 角速度控制
     * @param brushPower 刷子功率
     * @param suctionPower 吸力大小
     */
    public CleaningAction(double linearVelocity, double angularVelocity, 
                          double brushPower, double suctionPower) {
        this.linearVelocity = linearVelocity;
        this.angularVelocity = angularVelocity;
        this.brushPower = brushPower;
        this.suctionPower = suctionPower;
        this.actionType = determineActionType();
        clip();
    }
    
    /**
     * 默认构造函数（停止状态）
     */
    public CleaningAction() {
        this(0.0, 0.0, 0.0, 0.0);
        this.actionType = ActionType.STOP;
    }
    
    /**
     * 从数组创建动作
     * 
     * @param array 动作数组（4维）
     * @return 清扫动作
     */
    public static CleaningAction fromArray(NdArray array) {
        if (array.getShape().size() != 4) {
            throw new IllegalArgumentException("Action array must have 4 dimensions");
        }
        float v0 = array.get(0);
        float v1 = array.get(1);
        float v2 = array.get(2);
        float v3 = array.get(3);
        return new CleaningAction(v0, v1, v2, v3);
    }
    
    /**
     * 创建前进动作
     * 
     * @param speed 前进速度（0-1）
     * @return 清扫动作
     */
    public static CleaningAction moveForward(double speed) {
        CleaningAction action = new CleaningAction(speed, 0.0, 0.5, 0.5);
        action.actionType = ActionType.MOVE_FORWARD;
        return action;
    }
    
    /**
     * 创建左转动作
     * 
     * @param turnRate 转向速率（0-1）
     * @return 清扫动作
     */
    public static CleaningAction turnLeft(double turnRate) {
        CleaningAction action = new CleaningAction(0.0, turnRate, 0.3, 0.3);
        action.actionType = ActionType.TURN_LEFT;
        return action;
    }
    
    /**
     * 创建右转动作
     * 
     * @param turnRate 转向速率（0-1）
     * @return 清扫动作
     */
    public static CleaningAction turnRight(double turnRate) {
        CleaningAction action = new CleaningAction(0.0, -turnRate, 0.3, 0.3);
        action.actionType = ActionType.TURN_RIGHT;
        return action;
    }
    
    /**
     * 创建定点清扫动作
     * 
     * @return 清扫动作
     */
    public static CleaningAction cleanSpot() {
        CleaningAction action = new CleaningAction(0.0, 0.0, 1.0, 1.0);
        action.actionType = ActionType.CLEAN_SPOT;
        return action;
    }
    
    /**
     * 限制到有效范围
     */
    public void clip() {
        linearVelocity = Math.max(-1.0, Math.min(1.0, linearVelocity));
        angularVelocity = Math.max(-1.0, Math.min(1.0, angularVelocity));
        brushPower = Math.max(0.0, Math.min(1.0, brushPower));
        suctionPower = Math.max(0.0, Math.min(1.0, suctionPower));
    }
    
    /**
     * 判断是否在移动
     * 
     * @return 是否在移动
     */
    public boolean isMoving() {
        return Math.abs(linearVelocity) > 0.01 || Math.abs(angularVelocity) > 0.01;
    }
    
    /**
     * 转换为数组表示
     * 
     * @return 动作数组
     */
    public NdArray toArray() {
        return NdArray.of(new float[]{
            (float) linearVelocity, (float) angularVelocity, 
            (float) brushPower, (float) suctionPower
        }, Shape.of(4));
    }
    
    /**
     * 确定动作类型
     */
    private ActionType determineActionType() {
        if (Math.abs(linearVelocity) < 0.01 && Math.abs(angularVelocity) < 0.01) {
            if (brushPower > 0.5 && suctionPower > 0.5) {
                return ActionType.CLEAN_SPOT;
            }
            return ActionType.STOP;
        }
        if (Math.abs(angularVelocity) < 0.01) {
            return ActionType.MOVE_FORWARD;
        }
        if (angularVelocity > 0) {
            return ActionType.TURN_LEFT;
        }
        return ActionType.TURN_RIGHT;
    }
    
    // Getters and Setters
    public double getLinearVelocity() {
        return linearVelocity;
    }
    
    public void setLinearVelocity(double linearVelocity) {
        this.linearVelocity = linearVelocity;
        clip();
    }
    
    public double getAngularVelocity() {
        return angularVelocity;
    }
    
    public void setAngularVelocity(double angularVelocity) {
        this.angularVelocity = angularVelocity;
        clip();
    }
    
    public double getBrushPower() {
        return brushPower;
    }
    
    public void setBrushPower(double brushPower) {
        this.brushPower = brushPower;
        clip();
    }
    
    public double getSuctionPower() {
        return suctionPower;
    }
    
    public void setSuctionPower(double suctionPower) {
        this.suctionPower = suctionPower;
        clip();
    }
    
    public ActionType getActionType() {
        return actionType;
    }
    
    public void setActionType(ActionType actionType) {
        this.actionType = actionType;
    }
    
    @Override
    public String toString() {
        return String.format("CleaningAction(type=%s, linear=%.2f, angular=%.2f, brush=%.2f, suction=%.2f)",
                             actionType, linearVelocity, angularVelocity, brushPower, suctionPower);
    }
}
