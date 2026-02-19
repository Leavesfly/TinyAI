package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.policy.EpsilonGreedyPolicy;
import io.leavesfly.tinyai.rl.util.ModelUtil;
import io.leavesfly.tinyai.rl.util.QValueComputer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Deep Q-Network (DQN) 智能体实现
 *
 * @author leavesfly
 * @version 0.01
 * 
 * 【DQN算法】
 * DQNAgent实现了深度Q网络算法,这是第一个成功将深度学习应用于强化学习的算法。
 * 
 * 【核心创新】
 * 1. 神经网络逼近Q函数: 解决高维状态空间问题
 * 2. 经验回放机制: 打破数据相关性,提高样本效率
 * 3. 目标网络: 稳定训练目标,避免自举导致的不稳定
 * 4. ε-贪婪策略: 平衡探索与利用
 * 
 * 【目标Q值计算 - DQN特有】
 * DQN使用目标网络计算最大Q值作为训练目标:
 * y = r + γ * max_a' Q_target(s', a')
 * 
 * 【与DoubleDQN的区别】
 * - DQN: 目标网络同时负责动作选择和评估
 * - DoubleDQN: 在线网络选择动作,目标网络评估Q值
 * 
 * 【继承关系】
 * DQNAgent → ValueBasedAgent → Agent
 * 继承了ValueBasedAgent的统一训练流程,只需实现目标Q值计算
 * 
 * 【代码精简】
 * 相比原始实现,通过继承ValueBasedAgent和使用工具类:
 * - 删除重复的训练统计代码 → 使用TrainingStatistics
 * - 删除重复的Q值计算方法 → 使用QValueComputer
 * - 删除重复的网络创建代码 → 使用ModelUtil
 * - 从459行精简到约200行,代码更清晰
 */
public class DQNAgent extends ValueBasedAgent {

    /**
     * 构造函数
     *
     * @param name             智能体名称
     * @param stateDim         状态空间维度
     * @param actionDim        动作空间维度
     * @param hiddenSizes      隐藏层尺寸数组
     * @param learningRate     学习率
     * @param epsilon          初始探索率
     * @param gamma            折扣因子
     * @param batchSize        批次大小
     * @param bufferSize       经验回放缓冲区大小
     * @param targetUpdateFreq 目标网络更新频率
     */
    public DQNAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
                    float learningRate, float epsilon, float gamma,
                    int batchSize, int bufferSize, int targetUpdateFreq) {
        super(name, stateDim, actionDim, learningRate, epsilon, gamma,
              batchSize, bufferSize, targetUpdateFreq);

        // 创建Q网络(在线网络) - 使用ModelUtil工具类
        this.model = ModelUtil.createQNetwork(name + "_Q", stateDim, actionDim, hiddenSizes);

        // 创建目标网络并复制权重 - 使用ModelUtil工具类
        this.targetModel = ModelUtil.createQNetwork(name + "_Target", stateDim, actionDim, hiddenSizes);
        ModelUtil.copyWeights(model, targetModel);
        
        // 创建优化器
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-3f);
        
        // 创建ε-贪婪策略
        this.policy = new EpsilonGreedyPolicy(stateDim, actionDim, epsilon,
                state -> model.forward(state));
    }

    /**
     * 计算目标Q值 - DQN算法实现
     * 
     * 【DQN目标Q值】
     * y_i = r_i + γ * max_a' Q_target(s'_i, a')
     * 
     * 【关键步骤】
     * 1. 使用目标网络前向传播: Q_target(s'_i, :)
     * 2. 提取最大Q值: max_a' Q_target(s'_i, a')
     * 3. 计算目标: r_i + γ * maxQ (如果done则只有r_i)
     * 4. 堆叠成批次Variable
     * 
     * 【与DoubleDQN的区别】
     * DoubleDQN会先用在线网络选择动作,再用目标网络评估,
     * 而DQN直接用目标网络的max操作。
     */
    @Override
    protected Variable computeTargetQValues(Experience[] experiences) {
        int batchSize = experiences.length;
        Variable[] targetArray = new Variable[batchSize];
        
        for (int i = 0; i < batchSize; i++) {
            Experience exp = experiences[i];
            Variable nextState = exp.getNextState();
            
            if (exp.isDone()) {
                // 终止状态: 目标值就是奖励
                targetArray[i] = new Variable(NdArray.of(exp.getReward()));
            } else {
                // 非终止状态: y = r + γ * max_a' Q_target(s', a')
                Variable nextQValues = targetModel.forward(nextState);
                Variable maxNextQ = QValueComputer.findMaxQValue(nextQValues);  // 使用工具类
                
                Variable rewardVar = new Variable(NdArray.of(exp.getReward()));
                Variable gammaVar = new Variable(NdArray.of(gamma));
                Variable discountedQ = maxNextQ.mul(gammaVar);
                targetArray[i] = rewardVar.add(discountedQ);
            }
        }
        
        // 使用工具类堆叠Variable
        return QValueComputer.stackVariables(targetArray, batchSize);
    }

    /**
     * 计算当前Q值
     * 
     * 【当前Q值】
     * Q_current(s_i, a_i) - 智能体实际选择动作a_i的Q值
     * 
     * 【关键步骤】
     * 1. 使用在线网络前向传播: Q(s_i, :)
     * 2. 提取实际动作的Q值: Q(s_i, a_i)
     * 3. 堆叠成批次Variable
     */
    @Override
    protected Variable computeCurrentQValues(Experience[] experiences) {
        int batchSize = experiences.length;
        Variable[] currentQArray = new Variable[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Experience exp = experiences[i];
            Variable state = exp.getState();
            Variable qValues = model.forward(state);

            // 提取实际动作的Q值 - 使用工具类
            int actionIndex = exp.getAction().getValue().getNumber().intValue();
            currentQArray[i] = QValueComputer.selectActionQValue(qValues, actionIndex);
        }

        // 使用工具类堆叠Variable
        return QValueComputer.stackVariables(currentQArray, batchSize);
    }
}
