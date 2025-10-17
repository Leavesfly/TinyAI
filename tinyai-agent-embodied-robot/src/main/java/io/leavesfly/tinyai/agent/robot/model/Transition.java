package io.leavesfly.tinyai.agent.robot.model;

/**
 * 状态转移类
 * 
 * <p>表示强化学习中的一次状态转移，包含(s, a, r, s', done)五元组。</p>
 * 
 * @author TinyAI Team
 */
public class Transition {
    /**
     * 当前状态
     */
    private CleaningState state;
    
    /**
     * 执行的动作
     */
    private CleaningAction action;
    
    /**
     * 获得的奖励
     */
    private double reward;
    
    /**
     * 下一个状态
     */
    private CleaningState nextState;
    
    /**
     * 是否终止
     */
    private boolean done;
    
    /**
     * 构造函数
     * 
     * @param state 当前状态
     * @param action 执行动作
     * @param reward 奖励
     * @param nextState 下一状态
     * @param done 是否终止
     */
    public Transition(CleaningState state, CleaningAction action, double reward,
                      CleaningState nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
    
    // Getters and Setters
    public CleaningState getState() {
        return state;
    }
    
    public void setState(CleaningState state) {
        this.state = state;
    }
    
    public CleaningAction getAction() {
        return action;
    }
    
    public void setAction(CleaningAction action) {
        this.action = action;
    }
    
    public double getReward() {
        return reward;
    }
    
    public void setReward(double reward) {
        this.reward = reward;
    }
    
    public CleaningState getNextState() {
        return nextState;
    }
    
    public void setNextState(CleaningState nextState) {
        this.nextState = nextState;
    }
    
    public boolean isDone() {
        return done;
    }
    
    public void setDone(boolean done) {
        this.done = done;
    }
    
    @Override
    public String toString() {
        return String.format("Transition(reward=%.2f, done=%b)", reward, done);
    }
}
