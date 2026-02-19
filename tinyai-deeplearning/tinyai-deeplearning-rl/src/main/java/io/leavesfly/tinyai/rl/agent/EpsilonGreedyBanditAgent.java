package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * ε-贪心(Epsilon-Greedy)多臂老虎机智能体
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【算法原理】
 * ε-贪心算法是最简单的多臂老虎机算法之一:
 * - 以概率 ε 随机探索(选择随机臂)
 * - 以概率 (1-ε) 利用(选择当前最优臂)
 * 
 * 【探索-利用权衡】
 * - 探索(Exploration): 尝试未知或不确定的选项,获取更多信息
 * - 利用(Exploitation): 选择当前已知的最优选项,最大化即时收益
 * - ε参数控制探索程度: ε越大探索越多,ε越小利用越多
 * 
 * 【ε衰减策略】
 * 通常使用衰减的ε:
 * - 初期: ε较大,多探索获取信息
 * - 后期: ε较小,多利用已知最优
 * - 公式: ε_t = ε_0 * decay^t
 * 
 * 【算法优缺点】
 * 优点:
 * - 简单易懂,易于实现
 * - 计算开销小,O(1)复杂度
 * - 在许多场景下表现良好
 * 
 * 缺点:
 * - 探索是盲目的,不考虑不确定性
 * - 需要手动调节ε参数
 * - 理论遗憾界不如UCB等算法
 * 
 * 【应用场景】
 * - 在线广告推荐(A/B测试)
 * - 强化学习中的动作选择策略
 * - 资源分配和调度问题
 * - 超参数优化
 */
public class EpsilonGreedyBanditAgent extends BanditAgent {
    
    /**
     * 探索率(epsilon)
     * 取值范围: [0, 1]
     * - 0: 完全利用,总是选择最优臂
     * - 1: 完全探索,随机选择
     * - 0.1: 典型值,10%探索,90%利用
     */
    private float epsilon;
    
    /**
     * 随机数生成器(用于探索)
     */
    private final Random random;
    
    /**
     * 构造函数 - 使用默认探索率0.1
     * 
     * @param name 智能体名称
     * @param numArms 臂的数量
     */
    public EpsilonGreedyBanditAgent(String name, int numArms) {
        this(name, numArms, 0.1f);
    }
    
    /**
     * 完整构造函数
     * 
     * @param name 智能体名称
     * @param numArms 臂的数量
     * @param epsilon 探索率
     */
    public EpsilonGreedyBanditAgent(String name, int numArms, float epsilon) {
        super(name, numArms);
        this.epsilon = epsilon;
        this.random = new Random();
    }
    
    /**
     * 选择动作 - ε-贪心策略
     * 
     * 【决策流程】
     * 1. 生成随机数 r ~ Uniform(0,1)
     * 2. if r < ε: 探索(随机选择臂)
     * 3. else: 利用(选择当前估计最优的臂)
     */
    @Override
    public Variable selectAction(Variable state) {
        int armIndex = selectArm();
        return new Variable(NdArray.of(new float[]{armIndex}, Shape.of(1)));
    }
    
    /**
     * 选择臂 - ε-贪心核心逻辑
     * 
     * @return 选择的臂索引
     */
    @Override
    public int selectArm() {
        // 探索: 以概率ε随机选择
        if (random.nextFloat() < epsilon) {
            return random.nextInt(actionDim);
        }
        
        // 利用: 选择当前平均奖励最高的臂
        return selectBestArm();
    }
    
    /**
     * 选择当前最优臂(平均奖励最高)
     * 
     * 【贪婪选择】
     * 选择 argmax_i Q(i), 其中 Q(i) = totalRewards[i] / actionCounts[i]
     * 
     * @return 最优臂索引
     */
    private int selectBestArm() {
        int bestArm = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < actionDim; i++) {
            // 未被选择过的臂,赋予最高值(乐观初始化)
            float value;
            if (actionCounts[i] == 0) {
                value = Float.POSITIVE_INFINITY; // 乐观初始化,鼓励探索
            } else {
                value = estimatedRewards[i];  // 使用基类的estimatedRewards字段
            }
            
            if (value > bestValue) {
                bestValue = value;
                bestArm = i;
            }
        }
        
        return bestArm;
    }
    
    /**
     * 获取当前探索率
     * 
     * @return epsilon值
     */
    public float getEpsilon() {
        return epsilon;
    }
    
    /**
     * 获取当前探索率(别名方法,兼容demo)
     * 
     * @return epsilon值
     */
    public float getCurrentEpsilon() {
        return epsilon;
    }
    
    /**
     * 设置探索率
     * 
     * 【动态调整】
     * 可以在训练过程中动态调整ε:
     * - 初期设置较大的ε(如0.5)多探索
     * - 后期减小ε(如0.01)多利用
     * 
     * @param epsilon 新的探索率
     */
    public void setEpsilon(float epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("Epsilon must be in [0, 1], got: " + epsilon);
        }
        this.epsilon = epsilon;
    }
    
    /**
     * 探索率衰减
     * 
     * 【衰减策略】
     * ε_new = ε_old * decay
     * 
     * @param decay 衰减因子(0 < decay <= 1)
     */
    public void decayEpsilon(float decay) {
        if (decay <= 0 || decay > 1) {
            throw new IllegalArgumentException("Decay must be in (0, 1], got: " + decay);
        }
        this.epsilon *= decay;
    }
    
    @Override
    public String toString() {
        return String.format("EpsilonGreedyBanditAgent{name='%s', numArms=%d, epsilon=%.3f, totalSteps=%d}",
                name, actionDim, epsilon, getTotalSteps());
    }
    
    /**
     * 获取总步数
     * 
     * @return 总的动作选择次数
     */
    private int getTotalSteps() {
        int total = 0;
        for (int count : actionCounts) {
            total += count;
        }
        return total;
    }
}
