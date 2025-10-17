package io.leavesfly.tinyai.agent.embodied.decision;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;

/**
 * 决策模块
 * 基于感知状态生成驾驶动作
 *
 * @author TinyAI Team
 */
public class DecisionModule {
    private PolicyNetwork policyNetwork;
    private SafetyConstraint safetyConstraint;

    public DecisionModule() {
        this.policyNetwork = new SimplePolicy();
        this.safetyConstraint = new SafetyConstraint();
    }

    /**
     * 基于感知状态做出决策
     */
    public DrivingAction decide(PerceptionState state) {
        // 1. 策略网络生成原始动作
        DrivingAction rawAction = policyNetwork.predict(state);
        
        // 2. 安全约束检查和修正
        DrivingAction safeAction = safetyConstraint.check(rawAction, state);
        
        return safeAction;
    }

    public void setPolicyNetwork(PolicyNetwork network) {
        this.policyNetwork = network;
    }
}
