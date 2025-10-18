package io.leavesfly.tinyai.agent.vla.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 本体感知输入数据模型
 * 封装VLA智能体的本体感知模态输入（机器人自身状态）
 *
 * @author TinyAI
 */
public class ProprioceptionInput {

    /**
     * 关节位置，维度 [n_joints]
     */
    private NdArray jointPositions;

    /**
     * 关节速度，维度 [n_joints]
     */
    private NdArray jointVelocities;

    /**
     * 末端执行器姿态（位置+四元数），维度 [7]
     */
    private NdArray endEffectorPose;

    /**
     * 夹爪状态，维度 [1]，范围 [0, 1]
     */
    private double gripperState;

    /**
     * 构造函数 - 基本关节状态
     */
    public ProprioceptionInput(NdArray jointPositions, NdArray jointVelocities) {
        this.jointPositions = jointPositions;
        this.jointVelocities = jointVelocities;
        this.gripperState = 0.0;
    }

    /**
     * 完整构造函数
     */
    public ProprioceptionInput(NdArray jointPositions, NdArray jointVelocities,
                               NdArray endEffectorPose, double gripperState) {
        this.jointPositions = jointPositions;
        this.jointVelocities = jointVelocities;
        this.endEffectorPose = endEffectorPose;
        this.gripperState = gripperState;
    }

    // Getters and Setters
    public NdArray getJointPositions() {
        return jointPositions;
    }

    public void setJointPositions(NdArray jointPositions) {
        this.jointPositions = jointPositions;
    }

    public NdArray getJointVelocities() {
        return jointVelocities;
    }

    public void setJointVelocities(NdArray jointVelocities) {
        this.jointVelocities = jointVelocities;
    }

    public NdArray getEndEffectorPose() {
        return endEffectorPose;
    }

    public void setEndEffectorPose(NdArray endEffectorPose) {
        this.endEffectorPose = endEffectorPose;
    }

    public double getGripperState() {
        return gripperState;
    }

    public void setGripperState(double gripperState) {
        this.gripperState = gripperState;
    }

    @Override
    public String toString() {
        return "ProprioceptionInput{" +
                "jointPositionsShape=" + (jointPositions != null ? jointPositions.getShape() : "null") +
                ", jointVelocitiesShape=" + (jointVelocities != null ? jointVelocities.getShape() : "null") +
                ", hasEndEffectorPose=" + (endEffectorPose != null) +
                ", gripperState=" + gripperState +
                '}';
    }
}
