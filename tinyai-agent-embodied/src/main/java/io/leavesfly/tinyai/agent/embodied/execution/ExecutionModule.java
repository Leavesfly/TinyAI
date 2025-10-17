package io.leavesfly.tinyai.agent.embodied.execution;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.ExecutionFeedback;
import io.leavesfly.tinyai.agent.embodied.model.StepResult;
import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;

/**
 * 执行模块
 * 将决策动作应用到环境并收集反馈
 *
 * @author TinyAI Team
 */
public class ExecutionModule {
    private DrivingEnvironment environment;

    public ExecutionModule(DrivingEnvironment environment) {
        this.environment = environment;
    }

    /**
     * 执行动作并收集反馈
     */
    public ExecutionFeedback execute(DrivingAction action) {
        // 执行动作
        StepResult result = environment.step(action);
        
        // 构建反馈
        ExecutionFeedback feedback = new ExecutionFeedback();
        feedback.setSuccess(true);
        feedback.setActualAction(action.toArray());
        feedback.setNextState(result.getObservation());
        feedback.setReward(result.getReward());
        feedback.setDone(result.isDone());
        feedback.setInfo(result.getInfo());
        
        return feedback;
    }

    public DrivingEnvironment getEnvironment() {
        return environment;
    }
}
