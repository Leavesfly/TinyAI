package io.leavesfly.tinyai.rl.util;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Q值计算工具类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【职责】
 * 提供基于值函数的强化学习算法中常用的Q值相关计算方法。
 * 统一处理Q值选择、最大值提取、批量堆叠等操作,避免代码重复。
 * 
 * 【适用算法】
 * - DQN: Q值贪婪选择、目标Q值计算
 * - DoubleDQN: 双网络Q值提取
 * - DuelingDQN: 优势函数与价值函数合并
 * 
 * 【设计原则】
 * - 静态方法: 无状态工具类,便于复用
 * - 保持计算图: 使用Variable操作保持梯度传播
 * - 异常安全: 参数校验和边界检查
 * 
 * 【消除的重复代码】
 * - DQNAgent: L152-166 (selectGreedyAction), L321-324 (findMaxQValueVariable), L353-359 (stackVariables)
 * - DoubleDQNAgent: L134-148 (selectGreedyAction), L254-268 (selectBestAction), L292-298 (stackVariables)
 */
public class QValueComputer {
    
    /**
     * 私有构造函数 - 工具类不允许实例化
     */
    private QValueComputer() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * 选择贪婪动作(Q值最大的动作)
     * 
     * 【算法逻辑】
     * 遍历所有动作的Q值,返回Q值最大的动作索引。
     * 这是强化学习中最基本的贪婪策略实现。
     * 
     * 【应用场景】
     * - DQN评估模式: 选择最优动作
     * - ε-贪婪策略: 以概率(1-ε)选择贪婪动作
     * - 目标网络更新: 计算 max_a Q(s',a)
     * 
     * @param qValues Q值向量 (1, actionDim)
     * @param actionDim 动作空间维度
     * @return 贪婪动作的Variable表示
     */
    public static Variable selectGreedyAction(Variable qValues, int actionDim) {
        NdArray qArray = qValues.getValue();
        int bestAction = 0;
        float maxQ = qArray.get(0, 0);
        
        // 遍历找到Q值最大的动作
        for (int i = 1; i < actionDim; i++) {
            float q = qArray.get(0, i);
            if (q > maxQ) {
                maxQ = q;
                bestAction = i;
            }
        }
        
        return new Variable(NdArray.of(bestAction));
    }
    
    /**
     * 选择最优动作的索引(返回int)
     * 
     * 【与selectGreedyAction的区别】
     * - selectGreedyAction: 返回Variable,用于需要保持计算图的场景
     * - selectBestActionIndex: 返回int,用于纯数值计算场景
     * 
     * @param qValues Q值向量
     * @param actionDim 动作空间维度
     * @return 最优动作索引
     */
    public static int selectBestActionIndex(Variable qValues, int actionDim) {
        NdArray qArray = qValues.getValue();
        int bestAction = 0;
        float maxQ = qArray.get(0, 0);
        
        for (int i = 1; i < actionDim; i++) {
            float q = qArray.get(0, i);
            if (q > maxQ) {
                maxQ = q;
                bestAction = i;
            }
        }
        
        return bestAction;
    }
    
    /**
     * 提取Q值向量的最大值(保持计算图)
     * 
     * 【计算图保持】
     * 使用Variable的max操作而不是直接数值比较,
     * 确保梯度可以反向传播,这对训练至关重要。
     * 
     * 【DQN目标Q值计算】
     * 目标: y = r + γ * max_a' Q_target(s', a')
     * 此方法计算其中的 max_a' Q_target(s', a') 部分
     * 
     * @param qValues Q值向量 (1, actionDim)
     * @return 最大Q值的Variable
     */
    public static Variable findMaxQValue(Variable qValues) {
        // 使用Variable的max操作保持计算图连通性
        // dim=1: 在动作维度上求最大值
        // keepDim=true: 保持维度用于后续计算
        return qValues.max(1, true);
    }
    
    /**
     * 将Variable数组堆叠成批次Variable
     * 
     * 【批量处理】
     * 强化学习通常使用批量更新提高效率和稳定性。
     * 此方法将多个单独的Variable合并成一个批次Variable。
     * 
     * 【使用场景】
     * - 批量经验回放: 将batch_size个Q值合并
     * - 目标Q值计算: 合并batch_size个目标值
     * - 当前Q值计算: 合并batch_size个当前值
     * 
     * 【实现细节】
     * 提取每个Variable的标量值,组成一维数组,
     * 然后reshape为 (batchSize, 1) 的NdArray。
     * 
     * @param variables Variable数组
     * @param batchSize 批次大小
     * @return 堆叠后的批次Variable
     */
    public static Variable stackVariables(Variable[] variables, int batchSize) {
        if (variables == null || variables.length == 0) {
            throw new IllegalArgumentException("Variables array cannot be null or empty");
        }
        
        if (variables.length != batchSize) {
            throw new IllegalArgumentException(
                String.format("Variables length %d does not match batch size %d", 
                             variables.length, batchSize)
            );
        }
        
        float[] values = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            values[i] = variables[i].getValue().getNumber().floatValue();
        }
        
        return new Variable(NdArray.of(values, Shape.of(batchSize, 1)));
    }
    
    /**
     * 计算Q值向量的最大值(数值版本,不保持计算图)
     * 
     * 【使用场景】
     * - 评估模式: 不需要梯度的Q值查询
     * - 统计分析: 监控Q值范围
     * - 调试输出: 打印最大Q值
     * 
     * @param qValues Q值向量
     * @param actionDim 动作空间维度
     * @return 最大Q值(float)
     */
    public static float findMaxQValueFloat(Variable qValues, int actionDim) {
        NdArray qArray = qValues.getValue();
        float maxQ = qArray.get(0, 0);
        
        for (int i = 1; i < actionDim; i++) {
            float q = qArray.get(0, i);
            if (q > maxQ) {
                maxQ = q;
            }
        }
        
        return maxQ;
    }
    
    /**
     * 从Q值向量中提取指定动作的Q值
     * 
     * 【使用场景】
     * DQN当前Q值计算: Q(s,a) 其中a是智能体实际选择的动作
     * 
     * @param qValues Q值向量 (1, actionDim)
     * @param actionIndex 动作索引
     * @return 该动作的Q值Variable
     */
    public static Variable selectActionQValue(Variable qValues, int actionIndex) {
        Variable indexVar = new Variable(NdArray.of(new float[]{actionIndex}));
        return qValues.indexSelect(1, indexVar);
    }
}
