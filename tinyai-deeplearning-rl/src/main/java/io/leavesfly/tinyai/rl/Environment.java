package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;

import java.util.HashMap;
import java.util.Map;

/**
 * 强化学习环境抽象基类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Environment类定义了强化学习环境的标准接口，所有具体环境都应继承此类。
 * 环境负责维护状态、处理动作、计算奖励，并判断回合是否结束。
 * 遵循OpenAI Gym的标准接口设计。
 */
public abstract class Environment {
    
    /**
     * 状态空间维度
     */
    protected int stateDim;
    
    /**
     * 动作空间维度
     */
    protected int actionDim;
    
    /**
     * 当前状态
     */
    protected Variable currentState;
    
    /**
     * 回合是否结束
     */
    protected boolean done;
    
    /**
     * 当前步数
     */
    protected int currentStep;
    
    /**
     * 最大步数限制
     */
    protected int maxSteps;
    
    /**
     * 构造函数
     * 
     * @param stateDim 状态空间维度
     * @param actionDim 动作空间维度
     * @param maxSteps 最大步数限制
     */
    public Environment(int stateDim, int actionDim, int maxSteps) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.maxSteps = maxSteps;
        this.currentStep = 0;
        this.done = false;
    }
    
    /**
     * 重置环境到初始状态
     * 
     * @return 初始状态
     */
    public abstract Variable reset();
    
    /**
     * 执行动作，环境状态转移
     * 
     * @param action 智能体选择的动作
     * @return StepResult 包含下一状态、奖励、是否结束等信息
     */
    public abstract StepResult step(Variable action);
    
    /**
     * 渲染环境（可选实现）
     */
    public void render() {
        // 默认空实现，子类可选择性重写
    }
    
    /**
     * 获取状态空间维度
     * 
     * @return 状态空间维度
     */
    public int getStateDim() {
        return stateDim;
    }
    
    /**
     * 获取动作空间维度
     * 
     * @return 动作空间维度
     */
    public int getActionDim() {
        return actionDim;
    }
    
    /**
     * 获取当前状态
     * 
     * @return 当前状态
     */
    public Variable getCurrentState() {
        return currentState;
    }
    
    /**
     * 判断回合是否结束
     * 
     * @return 回合是否结束
     */
    public boolean isDone() {
        return done;
    }
    
    /**
     * 获取当前步数
     * 
     * @return 当前步数
     */
    public int getCurrentStep() {
        return currentStep;
    }
    
    /**
     * 获取环境信息（用于调试和监控）
     * 
     * @return 环境信息字典
     */
    public Map<String, Object> getInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("currentStep", currentStep);
        info.put("maxSteps", maxSteps);
        info.put("done", done);
        info.put("stateDim", stateDim);
        info.put("actionDim", actionDim);
        return info;
    }
    
    /**
     * 随机采样一个动作（用于探索）
     * 
     * @return 随机动作
     */
    public abstract Variable sampleAction();
    
    /**
     * 检查动作是否有效
     * 
     * @param action 要检查的动作
     * @return 动作是否有效
     */
    public abstract boolean isValidAction(Variable action);
    
    /**
     * 步骤结果类，封装环境step方法的返回值
     */
    public static class StepResult {
        /** 下一状态 */
        private final Variable nextState;
        /** 奖励 */
        private final float reward;
        /** 是否结束 */
        private final boolean done;
        /** 附加信息 */
        private final Map<String, Object> info;
        
        /**
         * 构造函数
         * 
         * @param nextState 下一状态
         * @param reward 奖励
         * @param done 是否结束
         * @param info 附加信息
         */
        public StepResult(Variable nextState, float reward, boolean done, Map<String, Object> info) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }
        
        public Variable getNextState() { return nextState; }
        public float getReward() { return reward; }
        public boolean isDone() { return done; }
        public Map<String, Object> getInfo() { return info; }
    }
}