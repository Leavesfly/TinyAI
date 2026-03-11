package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Random;

/**
 * 离散动作空间实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 离散动作空间包含有限个可选动作,每个动作用整数索引表示。
 * 适用于:
 * - GridWorld: 上下左右4个动作
 * - CartPole: 向左/向右2个动作
 * - Atari游戏: 多个按键组合
 * - 多臂老虎机: N个臂的选择
 */
public class DiscreteActionSpace implements ActionSpace {
    
    /**
     * 动作数量
     */
    private final int numberOfActions;
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * 构造函数
     * 
     * @param numberOfActions 动作数量
     */
    public DiscreteActionSpace(int numberOfActions) {
        if (numberOfActions <= 0) {
            throw new IllegalArgumentException("动作数量必须为正数,got: " + numberOfActions);
        }
        this.numberOfActions = numberOfActions;
        this.random = new Random();
    }
    
    @Override
    public int getDimension() {
        return numberOfActions;
    }
    
    @Override
    public Variable sample() {
        int randomAction = random.nextInt(numberOfActions);
        return new Variable(NdArray.of(randomAction));
    }
    
    @Override
    public boolean contains(Variable action) {
        if (action == null || action.getValue() == null) {
            return false;
        }
        float value = action.getValue().getNumber().floatValue();
        int intValue = (int) value;
        return value == intValue && intValue >= 0 && intValue < numberOfActions;
    }
    
    @Override
    public boolean isDiscrete() {
        return true;
    }
    
    /**
     * 获取动作数量
     * 
     * @return 动作数量
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }
    
    @Override
    public String toString() {
        return String.format("DiscreteActionSpace{n=%d}", numberOfActions);
    }
}
