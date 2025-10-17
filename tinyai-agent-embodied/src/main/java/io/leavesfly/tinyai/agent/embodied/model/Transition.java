package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 状态转移类
 * 记录强化学习中的单步转移 (s, a, r, s', done)
 *
 * @author TinyAI Team
 */
public class Transition {
    private PerceptionState state;      // 当前状态
    private DrivingAction action;       // 执行动作
    private double reward;              // 即时奖励
    private PerceptionState nextState;  // 下一状态
    private boolean done;               // 是否终止

    public Transition() {
    }

    public Transition(PerceptionState state, DrivingAction action, double reward,
                     PerceptionState nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }

    // Getters and Setters
    public PerceptionState getState() {
        return state;
    }

    public void setState(PerceptionState state) {
        this.state = state;
    }

    public DrivingAction getAction() {
        return action;
    }

    public void setAction(DrivingAction action) {
        this.action = action;
    }

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public PerceptionState getNextState() {
        return nextState;
    }

    public void setNextState(PerceptionState nextState) {
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
        return String.format("Transition[reward=%.3f, done=%b]", reward, done);
    }
}
