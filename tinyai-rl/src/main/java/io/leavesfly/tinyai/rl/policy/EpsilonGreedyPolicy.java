package io.leavesfly.tinyai.rl.policy;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Policy;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * ε-贪婪策略实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * EpsilonGreedyPolicy实现了ε-贪婪策略，这是强化学习中最常用的探索策略之一。
 * 以1-ε的概率选择当前最优动作（贪婪），以ε的概率随机选择动作（探索）。
 * 支持动态调整ε值，通常在训练过程中逐渐减小ε以减少探索。
 */
public class EpsilonGreedyPolicy extends Policy {

    /**
     * 探索率
     */
    private float epsilon;

    /**
     * Q值函数（用于选择贪婪动作）
     */
    private final QFunction qFunction;

    /**
     * 随机数生成器
     */
    private final Random random;

    /**
     * 构造函数
     *
     * @param stateDim  状态空间维度
     * @param actionDim 动作空间维度
     * @param epsilon   初始探索率
     * @param qFunction Q值函数
     */
    public EpsilonGreedyPolicy(int stateDim, int actionDim, float epsilon, QFunction qFunction) {
        super("EpsilonGreedy", stateDim, actionDim);
        this.epsilon = epsilon;
        this.qFunction = qFunction;
        this.random = new Random();
    }

    @Override
    public Variable selectAction(Variable state) {
        if (random.nextFloat() < epsilon) {
            // 探索：随机选择动作
            return selectRandomAction();
        } else {
            // 利用：选择Q值最大的动作
            return selectGreedyAction(state);
        }
    }

    @Override
    public Variable getActionProbabilities(Variable state) {
        // 计算每个动作的选择概率
        Variable qValues = qFunction.getQValues(state);
        NdArray qArray = qValues.getValue();

        // 找到最优动作
        int bestAction = findBestAction(qArray);

        // 创建概率分布
        float[] probs = new float[actionDim];
        float explorationProb = epsilon / actionDim; // 探索概率平均分配

        for (int i = 0; i < actionDim; i++) {
            if (i == bestAction) {
                probs[i] = (1.0f - epsilon) + explorationProb; // 贪婪概率 + 探索概率
            } else {
                probs[i] = explorationProb; // 仅探索概率
            }
        }

        return new Variable(NdArray.of(probs, Shape.of(1, actionDim)));
    }

    @Override
    public float getActionProbability(Variable state, Variable action) {
        Variable probs = getActionProbabilities(state);
        int actionIndex = getActionIndex(action);
        return probs.getValue().get(0, actionIndex);
    }

    @Override
    public Variable getLogProbability(Variable state, Variable action) {
        float prob = getActionProbability(state, action);
        // 避免log(0)
        prob = Math.max(prob, 1e-8f);
        float logProb = (float) Math.log(prob);
        return new Variable(NdArray.of(logProb));
    }

    /**
     * 选择随机动作
     *
     * @return 随机动作
     */
    private Variable selectRandomAction() {
        int randomAction = random.nextInt(actionDim);
        return createActionVariable(randomAction);
    }

    /**
     * 选择贪婪动作（Q值最大的动作）
     *
     * @param state 当前状态
     * @return 贪婪动作
     */
    private Variable selectGreedyAction(Variable state) {
        Variable qValues = qFunction.getQValues(state);
        int bestAction = findBestAction(qValues.getValue());
        return createActionVariable(bestAction);
    }

    /**
     * 找到Q值最大的动作索引
     *
     * @param qValues Q值数组
     * @return 最优动作索引
     */
    private int findBestAction(NdArray qValues) {
        int bestAction = 0;
        float maxQValue = qValues.get(0, 0);

        for (int i = 1; i < actionDim; i++) {
            float qValue = qValues.get(0, i);
            if (qValue > maxQValue) {
                maxQValue = qValue;
                bestAction = i;
            }
        }

        return bestAction;
    }

    /**
     * 创建动作变量
     *
     * @param actionIndex 动作索引
     * @return 动作变量
     */
    private Variable createActionVariable(int actionIndex) {
        return new Variable(NdArray.of(actionIndex));
    }

    /**
     * 从动作变量中获取动作索引
     *
     * @param action 动作变量
     * @return 动作索引
     */
    private int getActionIndex(Variable action) {
        return (int) action.getValue().getNumber().floatValue();
    }

    /**
     * 获取当前探索率
     *
     * @return 探索率
     */
    public float getEpsilon() {
        return epsilon;
    }

    /**
     * 设置探索率
     *
     * @param epsilon 新的探索率
     */
    public void setEpsilon(float epsilon) {
        this.epsilon = Math.max(0.0f, Math.min(1.0f, epsilon));
    }

    /**
     * 衰减探索率
     *
     * @param decayRate  衰减率
     * @param minEpsilon 最小探索率
     */
    public void decayEpsilon(float decayRate, float minEpsilon) {
        this.epsilon = Math.max(minEpsilon, this.epsilon * decayRate);
    }

    /**
     * Q值函数接口
     */
    public interface QFunction {
        /**
         * 获取状态的所有动作Q值
         *
         * @param state 状态
         * @return Q值向量
         */
        Variable getQValues(Variable state);
    }
}