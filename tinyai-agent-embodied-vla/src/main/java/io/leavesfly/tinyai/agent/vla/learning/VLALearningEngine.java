package io.leavesfly.tinyai.agent.vla.learning;

import io.leavesfly.tinyai.agent.vla.VLAAgent;
import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;

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
     * 保存学习状态
     */
    void saveCheckpoint(String path);
    
    /**
     * 加载学习状态
     */
    void loadCheckpoint(String path);
}
