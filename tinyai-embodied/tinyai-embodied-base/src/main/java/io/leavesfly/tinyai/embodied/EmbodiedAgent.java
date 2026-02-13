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
 * 具身智能体
 * 整合感知-决策-执行的完整闭环
 * 
 * <p>这是具身智能系统的核心类，展示了智能体与物理环境交互的完整流程。
 * 通过这个类，可以理解具身智能的基本工作原理。
 * 
 * <h2>架构设计</h2>
 * <pre>
 * ┌─────────────────────────────────────┐
 * │           EmbodiedAgent              │
 * ├─────────────────────────────────────┤
 * │  ┌─────────────┐  ┌──────────────┐ │
 * │  │  Perception │→ │  Decision    │→│
 * │  │  (感知模块)  │  │  (决策模块)  │ │
 * │  └─────────────┘  └──────────────┘ │
 * │                        ↓             │
 * │  ┌─────────────┐  ┌──────────────┐ │
 * │  │ Environment │← │  Execution   │ │
 * │  │  (环境)     │  │  (执行模块)  │ │
 * │  └─────────────┘  └──────────────┘ │
 * └─────────────────────────────────────┘
 * </pre>
 * 
 * <h2>感知-决策-执行闭环</h2>
 * <ol>
 *   <li><b>感知(Perception)</b>: 通过传感器获取环境信息，处理成内部状态表示</li>
 *   <li><b>决策(Decision)</b>: 基于当前状态，决定下一步动作</li>
 *   <li><b>执行(Execution)</b>: 将动作发送到环境，获取反馈</li>
 *   <li><b>学习(Learning)</b>: (可选)根据经验更新决策策略</li>
 * </ol>
 * 
 * <h2>使用示例</h2>
 * <pre>{@code
 * // 1. 创建配置
 * EnvironmentConfig config = EnvironmentConfig.createHighwayConfig();
 * 
 * // 2. 创建智能体
 * EmbodiedAgent agent = new EmbodiedAgent(config);
 * 
 * // 3. 运行一个回合
 * Episode episode = agent.runEpisode(200);
 * System.out.println("总奖励: " + episode.getTotalReward());
 * 
 * // 4. 清理资源
 * agent.close();
 * }</pre>
 * 
 * <h2>学习要点</h2>
 * <ul>
 *   <li>理解为什么需要感知-决策-执行的闭环结构</li>
 *   <li>观察各个模块是如何解耦的</li>
 *   <li>注意状态是如何在不同模块之间传递的</li>
 * </ul>
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
