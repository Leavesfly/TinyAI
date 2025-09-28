package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * ε-贪心多臂老虎机智能体
 * 
 * ε-贪心算法是最简单的多臂老虎机算法之一：
 * - 以ε的概率随机选择一个臂（探索）
 * - 以(1-ε)的概率选择当前估计奖励最高的臂（利用）
 * 
 * 算法特点：
 * - 实现简单
 * - 平衡探索和利用
 * - 适合奖励分布相对稳定的环境
 * 
 * @author leavesfly
 */
public class EpsilonGreedyBanditAgent extends BanditAgent {
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * ε衰减率
     */
    private float epsilonDecay;
    
    /**
     * 最小ε值
     */
    private float minEpsilon;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param numArms 臂的数量
     * @param epsilon 初始探索率
     */
    public EpsilonGreedyBanditAgent(String name, int numArms, float epsilon) {
        this(name, numArms, epsilon, 0.995f, 0.01f);
    }
    
    /**
     * 完整构造函数
     * 
     * @param name 智能体名称
     * @param numArms 臂的数量
     * @param epsilon 初始探索率
     * @param epsilonDecay ε衰减率
     * @param minEpsilon 最小ε值
     */
    public EpsilonGreedyBanditAgent(String name, int numArms, float epsilon, float epsilonDecay, float minEpsilon) {
        super(name, numArms);
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.minEpsilon = minEpsilon;
        this.random = new Random();
    }
    
    @Override
    public Variable selectAction(Variable state) {
        int armIndex = selectArm();
        return new Variable(NdArray.of(new float[]{armIndex}, Shape.of(1)));
    }
    
    @Override
    public int selectArm() {
        // ε-贪心策略
        if (random.nextFloat() < epsilon) {
            // 探索：随机选择一个臂
            return random.nextInt(actionDim);
        } else {
            // 利用：选择当前估计奖励最高的臂
            return getBestArmIndex();
        }
    }
    
    /**
     * 更新统计信息并衰减ε
     */
    @Override
    protected void updateStatistics(int armIndex, float reward) {
        super.updateStatistics(armIndex, reward);
        
        // 衰减探索率
        epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);
    }
    
    /**
     * 获取当前的探索率
     * 
     * @return 当前ε值
     */
    public float getCurrentEpsilon() {
        return epsilon;
    }
    
    /**
     * 设置探索率
     * 
     * @param epsilon 新的ε值
     */
    public void setCurrentEpsilon(float epsilon) {
        this.epsilon = Math.max(0.0f, Math.min(1.0f, epsilon));
    }
    
    /**
     * 获取ε衰减率
     * 
     * @return ε衰减率
     */
    public float getEpsilonDecay() {
        return epsilonDecay;
    }
    
    /**
     * 设置ε衰减率
     * 
     * @param epsilonDecay 新的ε衰减率
     */
    public void setEpsilonDecay(float epsilonDecay) {
        this.epsilonDecay = Math.max(0.0f, Math.min(1.0f, epsilonDecay));
    }
    
    /**
     * 获取最小ε值
     * 
     * @return 最小ε值
     */
    public float getMinEpsilon() {
        return minEpsilon;
    }
    
    /**
     * 设置最小ε值
     * 
     * @param minEpsilon 新的最小ε值
     */
    public void setMinEpsilon(float minEpsilon) {
        this.minEpsilon = Math.max(0.0f, Math.min(1.0f, minEpsilon));
    }
    
    /**
     * 设置随机种子（用于实验重现）
     * 
     * @param seed 随机种子
     */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }
    
    @Override
    public void reset() {
        super.reset();
        // 可以选择是否重置ε值，通常不重置以保持学习到的探索率
    }
    
    @Override
    public void printStatus() {
        super.printStatus();
        System.out.println("当前ε值: " + String.format("%.4f", epsilon));
        System.out.println("ε衰减率: " + String.format("%.4f", epsilonDecay));
        System.out.println("最小ε值: " + String.format("%.4f", minEpsilon));
    }
    
    /**
     * 获取算法描述信息
     * 
     * @return 算法描述
     */
    public String getAlgorithmDescription() {
        return String.format("ε-贪心算法 (ε=%.4f, 衰减率=%.4f, 最小ε=%.4f)", 
                           epsilon, epsilonDecay, minEpsilon);
    }
}