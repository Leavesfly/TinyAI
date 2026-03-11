package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * 连续动作空间实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 连续动作空间中的动作是实数向量,每个维度有上下界。
 * 适用于:
 * - 机器人控制: 关节角度、力矩
 * - 自动驾驶: 方向盘角度、油门/刹车力度
 * - 连续控制任务: Pendulum、MountainCarContinuous等
 */
public class ContinuousActionSpace implements ActionSpace {
    
    /**
     * 动作向量维度
     */
    private final int dimension;
    
    /**
     * 每个维度的下界
     */
    private final float[] lowerBounds;
    
    /**
     * 每个维度的上界
     */
    private final float[] upperBounds;
    
    /**
     * 随机数生成器
     */
    private final Random random;
    
    /**
     * 构造函数
     * 
     * @param dimension 动作向量维度
     * @param lowerBounds 每个维度的下界
     * @param upperBounds 每个维度的上界
     */
    public ContinuousActionSpace(int dimension, float[] lowerBounds, float[] upperBounds) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("动作维度必须为正数,got: " + dimension);
        }
        if (lowerBounds.length != dimension || upperBounds.length != dimension) {
            throw new IllegalArgumentException("上下界数组长度必须等于维度");
        }
        for (int i = 0; i < dimension; i++) {
            if (lowerBounds[i] >= upperBounds[i]) {
                throw new IllegalArgumentException(
                    String.format("维度%d的下界(%.4f)必须小于上界(%.4f)", i, lowerBounds[i], upperBounds[i]));
            }
        }
        this.dimension = dimension;
        this.lowerBounds = lowerBounds.clone();
        this.upperBounds = upperBounds.clone();
        this.random = new Random();
    }
    
    /**
     * 构造函数（所有维度使用相同的上下界）
     * 
     * @param dimension 动作向量维度
     * @param lowerBound 统一下界
     * @param upperBound 统一上界
     */
    public ContinuousActionSpace(int dimension, float lowerBound, float upperBound) {
        this(dimension, createUniformBounds(dimension, lowerBound), createUniformBounds(dimension, upperBound));
    }
    
    private static float[] createUniformBounds(int dimension, float value) {
        float[] bounds = new float[dimension];
        java.util.Arrays.fill(bounds, value);
        return bounds;
    }
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public Variable sample() {
        float[] actionValues = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            float range = upperBounds[i] - lowerBounds[i];
            actionValues[i] = lowerBounds[i] + random.nextFloat() * range;
        }
        return new Variable(NdArray.of(actionValues, Shape.of(1, dimension)));
    }
    
    @Override
    public boolean contains(Variable action) {
        if (action == null || action.getValue() == null) {
            return false;
        }
        NdArray values = action.getValue();
        for (int i = 0; i < dimension; i++) {
            float value = values.get(0, i);
            if (value < lowerBounds[i] || value > upperBounds[i]) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public boolean isDiscrete() {
        return false;
    }
    
    /**
     * 获取指定维度的下界
     * 
     * @param dimensionIndex 维度索引
     * @return 下界
     */
    public float getLowerBound(int dimensionIndex) {
        return lowerBounds[dimensionIndex];
    }
    
    /**
     * 获取指定维度的上界
     * 
     * @param dimensionIndex 维度索引
     * @return 上界
     */
    public float getUpperBound(int dimensionIndex) {
        return upperBounds[dimensionIndex];
    }
    
    @Override
    public String toString() {
        return String.format("ContinuousActionSpace{dim=%d}", dimension);
    }
}
