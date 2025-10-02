package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

/**
 * 强化学习经验数据类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Experience类用于存储强化学习中的一步经验，包含状态、动作、奖励、下一状态等信息。
 * 主要用于经验回放(Experience Replay)机制，提高学习效率和稳定性。
 */
public class Experience {
    
    /**
     * 当前状态
     */
    private final Variable state;
    
    /**
     * 执行的动作
     */
    private final Variable action;
    
    /**
     * 获得的奖励
     */
    private final float reward;
    
    /**
     * 下一状态
     */
    private final Variable nextState;
    
    /**
     * 是否为终止状态
     */
    private final boolean done;
    
    /**
     * 时间步索引
     */
    private final int timeStep;
    
    /**
     * 构造函数
     * 
     * @param state 当前状态
     * @param action 执行的动作
     * @param reward 获得的奖励
     * @param nextState 下一状态
     * @param done 是否为终止状态
     */
    public Experience(Variable state, Variable action, float reward, Variable nextState, boolean done) {
        this(state, action, reward, nextState, done, -1);
    }
    
    /**
     * 完整构造函数
     * 
     * @param state 当前状态
     * @param action 执行的动作
     * @param reward 获得的奖励
     * @param nextState 下一状态
     * @param done 是否为终止状态
     * @param timeStep 时间步索引
     */
    public Experience(Variable state, Variable action, float reward, Variable nextState, boolean done, int timeStep) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
        this.timeStep = timeStep;
    }
    
    /**
     * 获取当前状态
     * 
     * @return 当前状态
     */
    public Variable getState() {
        return state;
    }
    
    /**
     * 获取执行的动作
     * 
     * @return 执行的动作
     */
    public Variable getAction() {
        return action;
    }
    
    /**
     * 获取奖励
     * 
     * @return 奖励值
     */
    public float getReward() {
        return reward;
    }
    
    /**
     * 获取下一状态
     * 
     * @return 下一状态
     */
    public Variable getNextState() {
        return nextState;
    }
    
    /**
     * 判断是否为终止状态
     * 
     * @return 是否为终止状态
     */
    public boolean isDone() {
        return done;
    }
    
    /**
     * 获取时间步索引
     * 
     * @return 时间步索引
     */
    public int getTimeStep() {
        return timeStep;
    }
    
    @Override
    public String toString() {
        return String.format("Experience{timeStep=%d, reward=%.4f, done=%s}", 
                           timeStep, reward, done);
    }
}