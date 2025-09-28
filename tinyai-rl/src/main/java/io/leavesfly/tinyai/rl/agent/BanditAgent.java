package io.leavesfly.tinyai.rl.agent;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Experience;

import java.util.Arrays;

/**
 * 多臂老虎机智能体基类
 * 
 * 提供多臂老虎机算法的通用功能，包括统计信息记录、动作选择历史等。
 * 具体的选择策略由子类实现。
 * 
 * @author leavesfly
 */
public abstract class BanditAgent extends Agent {
    
    /**
     * 每个臂被选择的次数
     */
    protected int[] actionCounts;
    
    /**
     * 每个臂的累积奖励
     */
    protected float[] totalRewards;
    
    /**
     * 每个臂的估计平均奖励（Q值）
     */
    protected float[] estimatedRewards;
    
    /**
     * 总的动作选择次数
     */
    protected int totalActions;
    
    /**
     * 构造函数
     * 
     * @param name 智能体名称
     * @param numArms 臂的数量
     */
    public BanditAgent(String name, int numArms) {
        super(name, 1, numArms, 0.0f, 0.0f, 0.0f); // 多臂老虎机不需要学习率、探索率、折扣因子
        
        this.actionCounts = new int[numArms];
        this.totalRewards = new float[numArms];
        this.estimatedRewards = new float[numArms];
        this.totalActions = 0;
        
        // 初始化估计奖励为0
        Arrays.fill(estimatedRewards, 0.0f);
    }
    
    /**
     * 根据当前状态选择动作（多臂老虎机不依赖状态）
     * 
     * @param state 当前状态（在多臂老虎机中会被忽略）
     * @return 选择的动作（臂的索引）
     */
    @Override
    public abstract Variable selectAction(Variable state);
    
    /**
     * 多臂老虎机专用的动作选择方法
     * 
     * @return 选择的臂的索引
     */
    public abstract int selectArm();
    
    /**
     * 从经验中学习更新模型
     * 
     * @param experience 经验数据（包含选择的动作和获得的奖励）
     */
    @Override
    public void learn(Experience experience) {
        int actionIndex = (int) experience.getAction().getValue().get(0);
        float reward = experience.getReward();
        
        updateStatistics(actionIndex, reward);
    }
    
    /**
     * 更新统计信息
     * 
     * @param armIndex 选择的臂索引
     * @param reward 获得的奖励
     */
    protected void updateStatistics(int armIndex, float reward) {
        actionCounts[armIndex]++;
        totalRewards[armIndex] += reward;
        totalActions++;
        
        // 更新估计奖励（增量式平均）
        estimatedRewards[armIndex] = totalRewards[armIndex] / actionCounts[armIndex];
        
        incrementTrainingStep();
    }
    
    /**
     * 获取指定臂的估计奖励
     * 
     * @param armIndex 臂索引
     * @return 估计奖励
     */
    public float getEstimatedReward(int armIndex) {
        return estimatedRewards[armIndex];
    }
    
    /**
     * 获取指定臂被选择的次数
     * 
     * @param armIndex 臂索引
     * @return 选择次数
     */
    public int getActionCount(int armIndex) {
        return actionCounts[armIndex];
    }
    
    /**
     * 获取指定臂的累积奖励
     * 
     * @param armIndex 臂索引
     * @return 累积奖励
     */
    public float getTotalReward(int armIndex) {
        return totalRewards[armIndex];
    }
    
    /**
     * 获取所有臂的估计奖励
     * 
     * @return 估计奖励数组
     */
    public float[] getAllEstimatedRewards() {
        return estimatedRewards.clone();
    }
    
    /**
     * 获取所有臂的选择次数
     * 
     * @return 选择次数数组
     */
    public int[] getAllActionCounts() {
        return actionCounts.clone();
    }
    
    /**
     * 获取总的动作选择次数
     * 
     * @return 总选择次数
     */
    public int getTotalActions() {
        return totalActions;
    }
    
    /**
     * 获取当前最优臂的索引（基于估计奖励）
     * 
     * @return 最优臂索引
     */
    public int getBestArmIndex() {
        int bestArm = 0;
        for (int i = 1; i < actionDim; i++) {
            if (estimatedRewards[i] > estimatedRewards[bestArm]) {
                bestArm = i;
            }
        }
        return bestArm;
    }
    
    /**
     * 获取当前最优臂的估计奖励
     * 
     * @return 最优臂估计奖励
     */
    public float getBestEstimatedReward() {
        return estimatedRewards[getBestArmIndex()];
    }
    
    /**
     * 计算选择概率最高的臂被选择的频率
     * 
     * @return 最优选择频率
     */
    public float getOptimalActionRate() {
        if (totalActions == 0) return 0.0f;
        
        int bestArm = getBestArmIndex();
        return (float) actionCounts[bestArm] / totalActions;
    }
    
    /**
     * 重置智能体的统计信息
     */
    @Override
    public void reset() {
        super.reset();
        Arrays.fill(actionCounts, 0);
        Arrays.fill(totalRewards, 0.0f);
        Arrays.fill(estimatedRewards, 0.0f);
        totalActions = 0;
        trainingStep = 0;
    }
    
    /**
     * 批量学习更新模型（多臂老虎机通常不需要批量学习）
     * 
     * @param experiences 经验批次
     */
    @Override
    public void learnBatch(Experience[] experiences) {
        for (Experience experience : experiences) {
            learn(experience);
        }
    }
    
    /**
     * 存储经验（多臂老虎机通常不需要经验回放）
     * 
     * @param experience 要存储的经验
     */
    @Override
    public void storeExperience(Experience experience) {
        // 多臂老虎机通常不需要存储经验，可以直接学习
        learn(experience);
    }
    
    /**
     * 保存模型参数（多臂老虎机保存统计信息）
     * 
     * @param filepath 保存路径
     */
    @Override
    public void saveModel(String filepath) {
        // 实现统计信息的保存逻辑
        System.out.println("保存多臂老虎机统计信息到: " + filepath);
        // 这里可以实现具体的文件保存逻辑
    }
    
    /**
     * 加载模型参数（多臂老虎机加载统计信息）
     * 
     * @param filepath 加载路径
     */
    @Override
    public void loadModel(String filepath) {
        // 实现统计信息的加载逻辑
        System.out.println("从文件加载多臂老虎机统计信息: " + filepath);
        // 这里可以实现具体的文件加载逻辑
    }
    
    /**
     * 打印智能体当前状态
     */
    public void printStatus() {
        System.out.println("=== " + name + " 状态 ===");
        System.out.println("总动作次数: " + totalActions);
        System.out.println("最优臂: " + getBestArmIndex() + " (估计奖励: " + String.format("%.4f", getBestEstimatedReward()) + ")");
        System.out.println("最优选择率: " + String.format("%.4f", getOptimalActionRate()));
        
        System.out.println("各臂统计:");
        for (int i = 0; i < actionDim; i++) {
            System.out.println("  臂 " + i + ": 选择次数=" + actionCounts[i] + 
                             ", 估计奖励=" + String.format("%.4f", estimatedRewards[i]) + 
                             ", 累积奖励=" + String.format("%.4f", totalRewards[i]));
        }
        System.out.println("===================");
    }
}