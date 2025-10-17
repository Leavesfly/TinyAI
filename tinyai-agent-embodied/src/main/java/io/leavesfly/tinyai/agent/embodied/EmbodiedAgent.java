package io.leavesfly.tinyai.agent.embodied;

import io.leavesfly.tinyai.agent.embodied.decision.DecisionModule;
import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.env.impl.SimpleDrivingEnv;
import io.leavesfly.tinyai.agent.embodied.execution.ExecutionModule;
import io.leavesfly.tinyai.agent.embodied.model.*;
import io.leavesfly.tinyai.agent.embodied.perception.PerceptionModule;
import io.leavesfly.tinyai.agent.embodied.sensor.SensorSuite;

/**
 * 具身智能体
 * 整合感知-决策-执行的完整闭环
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
     * 执行一步
     */
    public StepResult step() {
        if (!initialized) {
            throw new IllegalStateException("Agent not initialized. Call reset() first.");
        }
        
        // 1. 决策
        DrivingAction action = decisionModule.decide(currentState);
        
        // 2. 执行
        ExecutionFeedback feedback = executionModule.execute(action);
        
        // 3. 更新状态
        currentState = feedback.getNextState();
        currentState = perceptionModule.process(currentState);
        
        // 4. 更新统计
        episodeSteps++;
        totalReward += feedback.getReward();
        
        // 5. 构建返回结果
        StepResult result = new StepResult(currentState, feedback.getReward(), feedback.isDone());
        result.addInfo("total_reward", totalReward);
        result.addInfo("episode_steps", episodeSteps);
        
        return result;
    }

    /**
     * 运行完整的情景
     */
    public Episode runEpisode(int maxSteps) {
        Episode episode = new Episode("episode_" + System.currentTimeMillis(), 
                                     environment.getScenarioType());
        
        // 重置
        PerceptionState state = reset();
        
        // 运行
        for (int step = 0; step < maxSteps; step++) {
            // 决策
            DrivingAction action = decisionModule.decide(state);
            
            // 执行
            StepResult result = step();
            
            // 记录转移
            Transition transition = new Transition(
                state,
                action,
                result.getReward(),
                result.getObservation(),
                result.isDone()
            );
            episode.addTransition(transition);
            
            // 更新状态
            state = result.getObservation();
            
            // 检查终止
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

    // Getters
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
