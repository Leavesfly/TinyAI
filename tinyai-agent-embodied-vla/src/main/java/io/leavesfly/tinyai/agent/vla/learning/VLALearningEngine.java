package io.leavesfly.tinyai.agent.vla.learning;

import io.leavesfly.tinyai.agent.vla.VLAAgent;
import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.model.VLAState;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;

/**
 * VLA学习引擎接口
 * 定义VLA智能体的学习方法
 * 
 * @author TinyAI
 */
public interface VLALearningEngine {
    
    /**
     * 训练VLA智能体
     * 
     * @param agent VLA智能体
     * @param env 训练环境
     * @param numEpisodes 训练回合数
     */
    void train(VLAAgent agent, RobotEnvironment env, int numEpisodes);
    
    /**
     * 评估VLA智能体
     * 
     * @param agent VLA智能体
     * @param env 评估环境
     * @param numEpisodes 评估回合数
     * @return 平均回报
     */
    double evaluate(VLAAgent agent, RobotEnvironment env, int numEpisodes);
    
    /**
     * 训练单个回合
     * 
     * @param agent VLA智能体
     * @param env 训练环境
     * @return 回合总奖励
     */
    default double trainEpisode(VLAAgent agent, RobotEnvironment env) {
        VLAState state = env.reset();
        double episodeReward = 0.0;
        
        while (true) {
            VLAAction action = agent.predict(state);
            RobotEnvironment.EnvironmentStep step = env.step(action);
            
            episodeReward += step.getReward();
            
            if (step.isDone()) {
                break;
            }
            
            state = step.getNextState();
        }
        
        return episodeReward;
    }
    
    /**
     * 从演示数据预训练
     * 
     * @param agent VLA智能体
     * @param demonstrations 演示数据列表
     */
    default void pretrainFromDemonstrations(VLAAgent agent, java.util.List<?> demonstrations) {
        System.out.println("Pre-training from " + demonstrations.size() + " demonstrations...");
        // 默认实现：简化处理
    }
    
    /**
     * 保存学习状态
     */
    void saveCheckpoint(String path);
    
    /**
     * 加载学习状态
     */
    void loadCheckpoint(String path);
}
