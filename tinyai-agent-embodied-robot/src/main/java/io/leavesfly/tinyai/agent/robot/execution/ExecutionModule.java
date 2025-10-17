package io.leavesfly.tinyai.agent.robot.execution;

import io.leavesfly.tinyai.agent.robot.model.*;
import io.leavesfly.tinyai.agent.robot.env.CleaningEnvironment;

/**
 * 执行模块
 * 
 * <p>执行决策动作并收集反馈。</p>
 * 
 * @author TinyAI Team
 */
public class ExecutionModule {
    private CleaningEnvironment environment;
    
    public ExecutionModule(CleaningEnvironment environment) {
        this.environment = environment;
    }
    
    public ExecutionFeedback execute(CleaningAction action) {
        // 安全检查
        action.clip();
        
        // 执行动作
        StepResult result = environment.step(action);
        
        // 创建反馈
        ExecutionFeedback feedback = new ExecutionFeedback(
            result.getObservation(),
            result.getReward(),
            result.isDone()
        );
        
        // 设置额外信息
        if (result.getInfo("collision") != null) {
            feedback.setCollisionOccurred((Boolean) result.getInfo("collision"));
        }
        
        return feedback;
    }
    
    public ExecutionStatus getExecutionStatus() {
        return environment.isTerminated() ? 
            ExecutionStatus.TERMINATED : ExecutionStatus.RUNNING;
    }
    
    public enum ExecutionStatus {
        IDLE,
        RUNNING,
        PAUSED,
        TERMINATED,
        ERROR
    }
}
