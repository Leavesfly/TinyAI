package io.leavesfly.tinyai.agent.robot.decision;

import io.leavesfly.tinyai.agent.robot.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 决策模块
 * 
 * <p>基于感知状态做出决策。</p>
 * 
 * @author TinyAI Team
 */
public class DecisionModule {
    private DecisionStrategy strategy;
    
    public DecisionModule() {
        this.strategy = DecisionStrategy.RULE_BASED;
    }
    
    public CleaningAction decide(CleaningState state) {
        RobotState robotState = state.getRobotState();
        
        // 简单规则策略
        if (robotState.needsCharging()) {
            return CleaningAction.moveForward(0.3);
        }
        
        if (robotState.needsEmptying()) {
            return new CleaningAction();
        }
        
        // 默认：前进并清扫
        return CleaningAction.moveForward(0.5);
    }
    
    public CleaningAction selectAction(NdArray features) {
        // 简化实现
        return CleaningAction.moveForward(0.5);
    }
    
    public void setStrategy(DecisionStrategy strategy) {
        this.strategy = strategy;
    }
    
    public enum DecisionStrategy {
        RULE_BASED,
        NEURAL_NETWORK,
        HYBRID
    }
}
