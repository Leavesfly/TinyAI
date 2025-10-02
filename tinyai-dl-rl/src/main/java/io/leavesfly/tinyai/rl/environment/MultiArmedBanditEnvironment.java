package io.leavesfly.tinyai.rl.environment;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 多臂老虎机环境
 * 
 * 多臂老虎机是强化学习中的经典问题，智能体需要在多个选择（臂）中选择一个，
 * 以最大化累积奖励。每个臂都有一个未知的奖励分布。
 * 
 * @author leavesfly
 */
public class MultiArmedBanditEnvironment extends Environment {
    
    /**
     * 每个臂的真实奖励均值
     */
    private final float[] trueRewards;
    
    /**
     * 每个臂的奖励方差
     */
    private final float[] rewardVariances;
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * 最优臂的索引
     */
    private final int optimalArm;
    
    /**
     * 累积奖励
     */
    private float totalReward;
    
    /**
     * 累积悔恨值
     */
    private float totalRegret;
    
    /**
     * 构造一个具有固定奖励均值的多臂老虎机环境
     * 
     * @param trueRewards 每个臂的真实奖励均值
     * @param rewardVariances 每个臂的奖励方差
     * @param maxSteps 最大步数
     */
    public MultiArmedBanditEnvironment(float[] trueRewards, float[] rewardVariances, int maxSteps) {
        super(1, trueRewards.length, maxSteps); // 状态维度为1（无状态），动作维度为臂的数量
        this.trueRewards = trueRewards.clone();
        this.rewardVariances = rewardVariances.clone();
        this.random = new Random();
        
        // 找到最优臂
        int bestArm = 0;
        for (int i = 1; i < trueRewards.length; i++) {
            if (trueRewards[i] > trueRewards[bestArm]) {
                bestArm = i;
            }
        }
        this.optimalArm = bestArm;
        
        this.totalReward = 0.0f;
        this.totalRegret = 0.0f;
    }
    
    /**
     * 构造一个标准的多臂老虎机环境（方差为1）
     * 
     * @param trueRewards 每个臂的真实奖励均值
     * @param maxSteps 最大步数
     */
    public MultiArmedBanditEnvironment(float[] trueRewards, int maxSteps) {
        this(trueRewards, createUnitVariances(trueRewards.length), maxSteps);
    }
    
    /**
     * 创建单位方差数组
     */
    private static float[] createUnitVariances(int numArms) {
        float[] variances = new float[numArms];
        Arrays.fill(variances, 1.0f);
        return variances;
    }
    
    @Override
    public Variable reset() {
        currentStep = 0;
        done = maxSteps <= 0; // 如果maxSteps为0或负数，直接结束
        totalReward = 0.0f;
        totalRegret = 0.0f;
        
        // 多臂老虎机没有状态，返回一个虚拟状态
        currentState = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        return currentState;
    }
    
    @Override
    public StepResult step(Variable action) {
        if (done) {
            throw new IllegalStateException("环境已结束，请先调用reset()");
        }
        
        // 获取选择的臂
        int armIndex = (int) action.getValue().get(0);
        if (armIndex < 0 || armIndex >= actionDim) {
            throw new IllegalArgumentException("无效的动作: " + armIndex);
        }
        
        // 根据臂的奖励分布生成奖励
        float reward = generateReward(armIndex);
        totalReward += reward;
        
        // 计算悔恨值（与最优选择的差距）
        float regret = trueRewards[optimalArm] - trueRewards[armIndex];
        totalRegret += regret;
        
        currentStep++;
        done = currentStep >= maxSteps;
        
        // 返回结果信息
        Map<String, Object> info = new HashMap<>();
        info.put("selectedArm", armIndex);
        info.put("trueReward", trueRewards[armIndex]);
        info.put("instantRegret", regret);
        info.put("totalReward", totalReward);
        info.put("totalRegret", totalRegret);
        info.put("averageReward", totalReward / currentStep);
        info.put("averageRegret", totalRegret / currentStep);
        info.put("optimalArm", optimalArm);
        info.put("isOptimal", armIndex == optimalArm);
        
        return new StepResult(currentState, reward, done, info);
    }
    
    /**
     * 根据指定臂的奖励分布生成奖励
     * 
     * @param armIndex 臂的索引
     * @return 生成的奖励值
     */
    private float generateReward(int armIndex) {
        float mean = trueRewards[armIndex];
        float variance = rewardVariances[armIndex];
        float stdDev = (float) Math.sqrt(variance);
        
        // 使用正态分布生成奖励
        return mean + stdDev * (float) random.nextGaussian();
    }
    
    @Override
    public Variable sampleAction() {
        // 随机选择一个臂
        int randomArm = random.nextInt(actionDim);
        return new Variable(NdArray.of(new float[]{randomArm}, Shape.of(1)));
    }
    
    @Override
    public boolean isValidAction(Variable action) {
        if (action == null || action.getValue() == null) {
            return false;
        }
        
        int armIndex = (int) action.getValue().get(0);
        return armIndex >= 0 && armIndex < actionDim;
    }
    
    /**
     * 获取每个臂的真实奖励均值
     * 
     * @return 奖励均值数组
     */
    public float[] getTrueRewards() {
        return trueRewards.clone();
    }
    
    /**
     * 获取最优臂的索引
     * 
     * @return 最优臂索引
     */
    public int getOptimalArm() {
        return optimalArm;
    }
    
    /**
     * 获取最优臂的奖励均值
     * 
     * @return 最优奖励均值
     */
    public float getOptimalReward() {
        return trueRewards[optimalArm];
    }
    
    /**
     * 获取累积奖励
     * 
     * @return 累积奖励
     */
    public float getTotalReward() {
        return totalReward;
    }
    
    /**
     * 获取累积悔恨值
     * 
     * @return 累积悔恨值
     */
    public float getTotalRegret() {
        return totalRegret;
    }
    
    /**
     * 获取平均奖励
     * 
     * @return 平均奖励
     */
    public float getAverageReward() {
        return currentStep > 0 ? totalReward / currentStep : 0.0f;
    }
    
    /**
     * 获取平均悔恨值
     * 
     * @return 平均悔恨值
     */
    public float getAverageRegret() {
        return currentStep > 0 ? totalRegret / currentStep : 0.0f;
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
    public void render() {
        System.out.println("=== 多臂老虎机环境状态 ===");
        System.out.println("步数: " + currentStep + "/" + maxSteps);
        System.out.println("累积奖励: " + String.format("%.4f", totalReward));
        System.out.println("平均奖励: " + String.format("%.4f", getAverageReward()));
        System.out.println("累积悔恨: " + String.format("%.4f", totalRegret));
        System.out.println("平均悔恨: " + String.format("%.4f", getAverageRegret()));
        System.out.println("最优臂: " + optimalArm + " (奖励: " + String.format("%.4f", trueRewards[optimalArm]) + ")");
        
        System.out.print("各臂奖励: [");
        for (int i = 0; i < trueRewards.length; i++) {
            System.out.print(String.format("%.4f", trueRewards[i]));
            if (i < trueRewards.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        System.out.println("========================");
    }
}