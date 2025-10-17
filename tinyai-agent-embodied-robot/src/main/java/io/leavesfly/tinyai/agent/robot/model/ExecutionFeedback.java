package io.leavesfly.tinyai.agent.robot.model;

/**
 * 执行反馈类
 * 
 * <p>表示动作执行后的详细反馈信息。</p>
 * 
 * @author TinyAI Team
 */
public class ExecutionFeedback {
    /**
     * 下一个状态
     */
    private CleaningState nextState;
    
    /**
     * 奖励值
     */
    private double reward;
    
    /**
     * 是否结束
     */
    private boolean done;
    
    /**
     * 消耗的能量（百分比）
     */
    private double energyConsumed;
    
    /**
     * 清扫面积（平方米）
     */
    private double areaCleaned;
    
    /**
     * 是否发生碰撞
     */
    private boolean collisionOccurred;
    
    /**
     * 构造函数
     * 
     * @param nextState 下一个状态
     * @param reward 奖励
     * @param done 是否结束
     */
    public ExecutionFeedback(CleaningState nextState, double reward, boolean done) {
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
        this.energyConsumed = 0.0;
        this.areaCleaned = 0.0;
        this.collisionOccurred = false;
    }
    
    // Getters and Setters
    public CleaningState getNextState() {
        return nextState;
    }
    
    public void setNextState(CleaningState nextState) {
        this.nextState = nextState;
    }
    
    public double getReward() {
        return reward;
    }
    
    public void setReward(double reward) {
        this.reward = reward;
    }
    
    public boolean isDone() {
        return done;
    }
    
    public void setDone(boolean done) {
        this.done = done;
    }
    
    public double getEnergyConsumed() {
        return energyConsumed;
    }
    
    public void setEnergyConsumed(double energyConsumed) {
        this.energyConsumed = energyConsumed;
    }
    
    public double getAreaCleaned() {
        return areaCleaned;
    }
    
    public void setAreaCleaned(double areaCleaned) {
        this.areaCleaned = areaCleaned;
    }
    
    public boolean isCollisionOccurred() {
        return collisionOccurred;
    }
    
    public void setCollisionOccurred(boolean collisionOccurred) {
        this.collisionOccurred = collisionOccurred;
    }
    
    @Override
    public String toString() {
        return String.format("ExecutionFeedback(reward=%.2f, done=%b, energy=%.2f%%, area=%.2f, collision=%b)",
                             reward, done, energyConsumed, areaCleaned, collisionOccurred);
    }
}
