package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.policy.EpsilonGreedyPolicy;
import io.leavesfly.tinyai.rl.util.ModelUtil;
import io.leavesfly.tinyai.rl.util.QValueComputer;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * Double DQN (Double Deep Q-Network) 智能体实现
 *
 * @author leavesfly
 * @version 0.01
 * 
 * 【DoubleDQN算法】
 * DoubleDQNAgent实现了Double DQN算法,这是对标准DQN的重要改进。
 * 
 * 【核心创新 - 解决Q值过估计】
 * DQN存在Q值过估计问题,因为max操作会选择噪声导致的高估值。
 * DoubleDQN通过解耦动作选择和评估来缓解这个问题:
 * 
 * DQN目标Q值:
 *   y = r + γ * max_a' Q_target(s', a')
 *   问题: max同时负责选择和评估,倾向于高估
 * 
 * DoubleDQN目标Q值:
 *   a_best = argmax_a' Q_online(s', a')  // 在线网络选择动作
 *   y = r + γ * Q_target(s', a_best)      // 目标网络评估Q值
 *   优势: 分离选择和评估,减少过估计
 * 
 * 【与DQN的对比】
 * - DQN: 目标网络同时负责动作选择和Q值评估
 * - DoubleDQN: 在线网络选择动作,目标网络评估Q值
 * - 代码差异: 仅在computeTargetQValues方法实现上不同
 * 
 * 【继承关系】
 * DoubleDQNAgent → ValueBasedAgent → Agent
 * 继承了ValueBasedAgent的统一训练流程,只需实现目标Q值计算
 * 
 * 【教学价值】
 * 通过对比DQNAgent和DoubleDQNAgent的代码,学习者可以:
 * - 清晰看到两种算法仅在目标Q值计算上的差异
 * - 理解为什么解耦动作选择和评估能减少过估计
 * - 体会到良好架构设计带来的易扩展性
 * - 只需修改约20行代码就能实现新算法
 */
public class DoubleDQNAgent extends ValueBasedAgent {

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
    public DoubleDQNAgent(String name, int stateDim, int actionDim, int[] hiddenSizes,
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
     * 计算目标Q值 - DoubleDQN算法实现
     * 
     * 【DoubleDQN核心 - 解耦动作选择和评估】
     * 
     * 步骤1: 使用在线网络选择最优动作
     *   Q_online(s', :) = model.forward(s')
     *   a_best = argmax_a' Q_online(s', a')
     * 
     * 步骤2: 使用目标网络评估该动作的Q值
     *   Q_target(s', :) = targetModel.forward(s')
     *   Q_value = Q_target(s', a_best)
     * 
     * 步骤3: 计算目标值
     *   y = r + γ * Q_value
     * 
     * 【为什么这样做能减少过估计】
     * - 在线网络的argmax可能选到被噪声高估的动作
     * - 但目标网络对该动作的评估是独立的
     * - 两个网络不太可能同时高估同一个动作
     * - 因此期望的估计偏差更小
     * 
     * 【与DQN的代码对比】
     * DQN: maxQ = QValueComputer.findMaxQValue(targetQValues)
     * DoubleDQN: 
     *   bestAction = QValueComputer.selectBestActionIndex(onlineQValues, actionDim)
     *   Q = QValueComputer.selectActionQValue(targetQValues, bestAction)
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
                // DoubleDQN核心逻辑:
                // 1. 在线网络选择动作
                Variable onlineQValues = model.forward(nextState);
                int bestAction = QValueComputer.selectBestActionIndex(onlineQValues, actionDim);
                
                // 2. 目标网络评估该动作
                Variable targetQValues = targetModel.forward(nextState);
                Variable targetQForBestAction = QValueComputer.selectActionQValue(targetQValues, bestAction);
                
                // 3. 计算目标: y = r + γ * Q_target(s', a_best_online)
                Variable rewardVar = new Variable(NdArray.of(exp.getReward()));
                Variable gammaVar = new Variable(NdArray.of(gamma));
                Variable discountedQ = targetQForBestAction.mul(gammaVar);
                targetArray[i] = rewardVar.add(discountedQ);
            }
        }
        
        // 使用工具类堆叠Variable
        return QValueComputer.stackVariables(targetArray, batchSize);
    }

    /**
     * 计算当前Q值 - 与DQN完全相同
     * 
     * 【当前Q值】
     * Q_current(s_i, a_i) - 智能体实际选择动作a_i的Q值
     * 
     * 【实现】
     * 与DQN完全一致,因为当前Q值的计算方式不影响算法差异
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
