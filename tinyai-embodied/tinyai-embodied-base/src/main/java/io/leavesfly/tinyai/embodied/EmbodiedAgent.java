package io.leavesfly.tinyai.embodied;

import io.leavesfly.tinyai.embodied.decision.DecisionModule;
import io.leavesfly.tinyai.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.embodied.env.impl.SimpleDrivingEnv;
import io.leavesfly.tinyai.embodied.execution.ExecutionModule;
import io.leavesfly.tinyai.embodied.perception.PerceptionModule;
import io.leavesfly.tinyai.embodied.sensor.SensorSuite;
import io.leavesfly.tinyai.embodied.model.*;

/**
 * 具身智能体，整合感知-决策-执行的完整闭环。
 *
 * 这是具身智能系统的核心类，展示智能体与物理环境交互的完整流程。
 *
 * 架构设计：
 *   EmbodiedAgent
 *   ├── Perception (感知) → Decision (决策) → Execution (执行)
 *   └── Environment (环境) ← 执行反馈
 *
 * 感知-决策-执行闭环：
 *   1. 感知：通过传感器获取环境信息，处理成内部状态表示
 *   2. 决策：基于当前状态，决定下一步动作
 *   3. 执行：将动作发送到环境，获取反馈
 *   4. 学习：(可选) 根据经验更新决策策略
 *
 * 使用示例：
 *   EnvironmentConfig config = EnvironmentConfig.createHighwayConfig();
 *   EmbodiedAgent agent = new EmbodiedAgent(config);
 *   Episode episode = agent.runEpisode(200);
 *   agent.close();
 *
 * @author TinyAI Team
 */
public class EmbodiedAgent {
    private DrivingEnvironment environment;
    private PerceptionModule perceptionModule;
    private DecisionModule decisionModule;
    private ExecutionModule executionModule;
    
    private PerceptionState currentState;
    private int episodeSteps;
    private double totalReward;
    private boolean initialized;

    public EmbodiedAgent(EnvironmentConfig config) {
        // 创建环境
        this.environment = new SimpleDrivingEnv(config);
        
        // 创建传感器套件
        SensorSuite sensorSuite = new SensorSuite(environment);
        
        // 创建各模块
        this.perceptionModule = new PerceptionModule(sensorSuite);
        this.decisionModule = new DecisionModule();
        this.executionModule = new ExecutionModule(environment);
        
        this.initialized = false;
    }

    /**
     * 初始化智能体
     */
    public void initialize() {
        perceptionModule.initialize();
        initialized = true;
    }

    /**
     * 重置智能体到初始状态
     */
    public PerceptionState reset() {
        if (!initialized) {
            initialize();
        }
        
        // 重置环境
        currentState = environment.reset();
        
        // 重置统计
        episodeSteps = 0;
        totalReward = 0.0;
        
        // 处理初始感知
        currentState = perceptionModule.process(currentState);
        
        return currentState;
    }

    /**
     * 执行一步：决策 → 执行 → 更新状态。
     */
    public StepResult step() {
        if (!initialized) {
            throw new IllegalStateException("Agent not initialized. Call reset() first.");
        }
        DrivingAction action = decisionModule.decide(currentState);
        return step(action);
    }

    /**
     * 使用指定动作执行一步（用于 runEpisode 等场景，避免重复决策）。
     */
    public StepResult step(DrivingAction action) {
        if (!initialized) {
            throw new IllegalStateException("Agent not initialized. Call reset() first.");
        }

        ExecutionFeedback feedback = executionModule.execute(action);
        currentState = perceptionModule.process(feedback.getNextState());

        episodeSteps++;
        totalReward += feedback.getReward();

        StepResult result = new StepResult(currentState, feedback.getReward(), feedback.isDone());
        result.addInfo("total_reward", totalReward);
        result.addInfo("episode_steps", episodeSteps);

        return result;
    }

    /**
     * 运行完整的情景。
     */
    public Episode runEpisode(int maxSteps) {
        Episode episode = new Episode("episode_" + System.currentTimeMillis(),
                environment.getScenarioType());
        PerceptionState state = reset();

        for (int step = 0; step < maxSteps; step++) {
            DrivingAction action = decisionModule.decide(state);
            StepResult result = step(action);

            episode.addTransition(new Transition(
                    state, action, result.getReward(),
                    result.getObservation(), result.isDone()));

            state = result.getObservation();
            if (result.isDone()) {
                break;
            }
        }

        episode.finish();
        return episode;
    }

    /**
     * 关闭智能体
     */
    public void close() {
        environment.close();
    }

    public PerceptionState getCurrentState() {
        return currentState;
    }

    public int getEpisodeSteps() {
        return episodeSteps;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public DrivingEnvironment getEnvironment() {
        return environment;
    }
}
