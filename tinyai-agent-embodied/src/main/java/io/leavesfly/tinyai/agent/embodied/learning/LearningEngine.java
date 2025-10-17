package io.leavesfly.tinyai.agent.embodied.learning;

import io.leavesfly.tinyai.agent.embodied.memory.EpisodicMemory;
import io.leavesfly.tinyai.agent.embodied.model.Episode;
import io.leavesfly.tinyai.agent.embodied.model.LearningStrategy;
import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;

/**
 * 学习引擎
 * 管理和协调不同的学习策略
 *
 * @author TinyAI Team
 */
public class LearningEngine {
    private LearningStrategy currentStrategy;
    private EpisodicMemory memory;
    private boolean trainingMode;
    
    // 具体学习器
    private DQNLearner dqnLearner;
    private EndToEndLearner endToEndLearner;

    public LearningEngine() {
        this.memory = new EpisodicMemory();
        this.currentStrategy = LearningStrategy.END_TO_END;
        this.trainingMode = false;
        
        // 初始化学习器
        this.dqnLearner = new DQNLearner();
        this.endToEndLearner = new EndToEndLearner();
    }

    /**
     * 从情景中学习
     */
    public void learnFromEpisode(Episode episode) {
        // 存储情景到记忆
        memory.storeEpisode(episode);
        
        if (trainingMode) {
            switch (currentStrategy) {
                case DQN:
                    // DQN强化学习
                    dqnLearner.learn(memory);
                    System.out.println("DQN学习：情景 " + episode.getEpisodeId() + 
                                     "，探索率=" + String.format("%.4f", dqnLearner.getEpsilon()));
                    break;
                case END_TO_END:
                    // 端到端学习
                    endToEndLearner.learn(memory);
                    System.out.println("端到端学习：情景 " + episode.getEpisodeId() + 
                                     "，训练步骤=" + endToEndLearner.getTrainingSteps());
                    break;
                case IMITATION:
                    // 模仿学习逻辑（预留接口）
                    System.out.println("模仿学习：情景 " + episode.getEpisodeId());
                    break;
                default:
                    break;
            }
        }
    }
    
    /**
     * 根据当前策略选择动作
     */
    public DrivingAction selectAction(PerceptionState state) {
        switch (currentStrategy) {
            case DQN:
                return dqnLearner.selectAction(state);
            case END_TO_END:
                return endToEndLearner.predict(state);
            default:
                // 默认返回简单动作
                return new DrivingAction(0.0, 0.3, 0.0);
        }
    }

    /**
     * 设置学习策略
     */
    public void setStrategy(LearningStrategy strategy) {
        this.currentStrategy = strategy;
    }

    /**
     * 开启训练模式
     */
    public void enableTraining() {
        this.trainingMode = true;
    }

    /**
     * 关闭训练模式
     */
    public void disableTraining() {
        this.trainingMode = false;
    }

    public EpisodicMemory getMemory() {
        return memory;
    }

    public boolean isTrainingMode() {
        return trainingMode;
    }

    public LearningStrategy getCurrentStrategy() {
        return currentStrategy;
    }
    
    public DQNLearner getDqnLearner() {
        return dqnLearner;
    }
    
    public EndToEndLearner getEndToEndLearner() {
        return endToEndLearner;
    }
}
