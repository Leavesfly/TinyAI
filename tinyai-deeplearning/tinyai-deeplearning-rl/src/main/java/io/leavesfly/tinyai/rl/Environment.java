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
 * 【MDP框架 - Environment】
 * Environment实现马尔可夫决策过程(MDP)的核心组件,定义了智能体交互的世界:
 * - S: 状态空间(State Space) - 所有可能的状态集合
 * - A: 动作空间(Action Space) - 所有可能的动作集合
 * - P: 状态转移概率 P(s'|s,a) - 执行动作a后转移到s'的概率
 * - R: 奖励函数 R(s,a,s') - 状态转移获得的即时奖励
 * - γ: 折扣因子(在Agent中) - 未来奖励的衰减系数
 * 
 * 【MDP的马尔可夫性质】
 * 下一状态只依赖当前状态和动作,与历史无关:
 * P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},...) = P(s_{t+1}|s_t,a_t)
 * 
 * 【标准交互流程】
 * 强化学习的标准Agent-Environment交互循环:
 * ```
 * 1. 初始化: s0 = env.reset()
 * 2. 循环直到done:
 *    a. Agent选择动作: a = agent.selectAction(s)
 *    b. 环境执行动作: (s', r, done) = env.step(a)
 *    c. Agent学习: agent.learn(Experience(s, a, r, s', done))
 *    d. 更新状态: s = s'
 * 3. 回合结束
 * ```
 * 
 * 【OpenAI Gym兼容性】
 * 接口设计遵循OpenAI Gym规范,便于:
 * - Python代码迁移到Java
 * - 算法对比验证
 * - 使用标准benchmark环境
 * 
 * 【核心方法】
 * - reset(): 重置环境到初始状态
 * - step(action): 执行动作,返回(nextState, reward, done, info)
 * - render(): 可视化环境(可选)
 * - sampleAction(): 随机采样动作(用于探索)
 * 
 * 【教学价值】
 * 通过Environment抽象,学习者可以:
 * - 理解MDP的数学定义如何映射到代码
 * - 看到状态转移和奖励设计的实现
 * - 学习如何自定义环境(继承并实现step和reset)
 * - 体会环境设计对学习效果的影响
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
     * 校验Agent与本环境的维度兼容性
     * 
     * 等价于 agent.validateCompatibility(this)，提供从Environment侧调用的便捷方式。
     * 
     * @param agent 要校验的智能体
     * @throws IllegalArgumentException 如果维度不匹配
     */
    public void validateAgent(Agent agent) {
        agent.validateCompatibility(this);
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
     * 
     * 【terminated vs truncated vs done】
     * 遵循OpenAI Gym v26+规范,区分两种结束原因:
     * - terminated: 环境自然结束(如到达目标、倒下等),此时bootstrap target应为0
     * - truncated: 因超时/步数限制被截断,此时仍应使用bootstrap估计未来回报
     * - done: terminated || truncated,向后兼容的便捷属性
     * 
     * 这一区分对值函数估计的正确性至关重要:
     * - terminated时: target = reward (无后续回报)
     * - truncated时: target = reward + γ * V(s') (仍有后续回报,只是被截断)
     */
    public static class StepResult {
        /** 下一状态 */
        private final Variable nextState;
        /** 奖励 */
        private final float reward;
        /** 环境是否自然终止（如到达目标、失败等） */
        private final boolean terminated;
        /** 是否因步数限制等外部原因被截断 */
        private final boolean truncated;
        /** 附加信息 */
        private final Map<String, Object> info;
        
        /**
         * 完整构造函数（推荐使用）
         * 
         * @param nextState 下一状态
         * @param reward 奖励
         * @param terminated 是否自然终止
         * @param truncated 是否被截断
         * @param info 附加信息
         */
        public StepResult(Variable nextState, float reward, boolean terminated, 
                          boolean truncated, Map<String, Object> info) {
            this.nextState = nextState;
            this.reward = reward;
            this.terminated = terminated;
            this.truncated = truncated;
            this.info = info;
        }
        
        /**
         * 向后兼容的构造函数
         * 当done=true时默认视为terminated（保持旧代码行为不变）
         * 
         * @param nextState 下一状态
         * @param reward 奖励
         * @param done 是否结束（等同于terminated）
         * @param info 附加信息
         */
        public StepResult(Variable nextState, float reward, boolean done, Map<String, Object> info) {
            this(nextState, reward, done, false, info);
        }
        
        public Variable getNextState() { return nextState; }
        public float getReward() { return reward; }
        
        /** 是否自然终止 */
        public boolean isTerminated() { return terminated; }
        
        /** 是否被截断 */
        public boolean isTruncated() { return truncated; }
        
        /** 是否结束（terminated || truncated），向后兼容 */
        public boolean isDone() { return terminated || truncated; }
        
        public Map<String, Object> getInfo() { return info; }
    }
}