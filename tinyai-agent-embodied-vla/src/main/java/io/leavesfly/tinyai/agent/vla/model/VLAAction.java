package io.leavesfly.tinyai.agent.vla.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * VLA动作表示
 * 封装VLA智能体的输出动作（连续+离散）
 *
 * @author TinyAI
 */
public class VLAAction {

    /**
     * 连续动作向量，维度 [action_dim]，例如末端执行器增量
     */
    private NdArray continuousAction;

    /**
     * 离散动作索引，范围 [0, num_actions)
     */
    private int discreteAction;

    /**
     * 动作类型枚举
     */
    private ActionType actionType;

    /**
     * 动作置信度，范围 [0.0, 1.0]
     */
    private double confidence;

    /**
     * 自然语言反馈
     */
    private String languageFeedback;

    /**
     * 构造函数 - 连续动作
     */
    public VLAAction(NdArray continuousAction) {
        this.continuousAction = continuousAction;
        this.discreteAction = -1;
        this.actionType = ActionType.MOVE_END_EFFECTOR;
        this.confidence = 1.0;
    }

    /**
     * 构造函数 - 连续 + 离散动作
     */
    public VLAAction(NdArray continuousAction, int discreteAction, ActionType actionType) {
        this.continuousAction = continuousAction;
        this.discreteAction = discreteAction;
        this.actionType = actionType;
        this.confidence = 1.0;
    }

    /**
     * 完整构造函数
     */
    public VLAAction(NdArray continuousAction, int discreteAction, ActionType actionType,
                     double confidence, String languageFeedback) {
        this.continuousAction = continuousAction;
        this.discreteAction = discreteAction;
        this.actionType = actionType;
        this.confidence = confidence;
        this.languageFeedback = languageFeedback;
    }

    // Getters and Setters
    public NdArray getContinuousAction() {
        return continuousAction;
    }

    public void setContinuousAction(NdArray continuousAction) {
        this.continuousAction = continuousAction;
    }

    public int getDiscreteAction() {
        return discreteAction;
    }

    public void setDiscreteAction(int discreteAction) {
        this.discreteAction = discreteAction;
    }

    public ActionType getActionType() {
        return actionType;
    }

    public void setActionType(ActionType actionType) {
        this.actionType = actionType;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public String getLanguageFeedback() {
        return languageFeedback;
    }

    public void setLanguageFeedback(String languageFeedback) {
        this.languageFeedback = languageFeedback;
    }

    @Override
    public String toString() {
        return "VLAAction{" +
                "continuousActionShape=" + (continuousAction != null ? continuousAction.getShape() : "null") +
                ", discreteAction=" + discreteAction +
                ", actionType=" + actionType +
                ", confidence=" + confidence +
                ", languageFeedback='" + languageFeedback + '\'' +
                '}';
    }
}
