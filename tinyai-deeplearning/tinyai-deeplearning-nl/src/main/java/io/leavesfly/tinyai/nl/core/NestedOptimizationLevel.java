package io.leavesfly.tinyai.nl.core;

import io.leavesfly.tinyai.func.Variable;
import java.util.ArrayList;
import java.util.List;

/**
 * 嵌套优化层级（NestedOptimizationLevel）
 * 表示嵌入学习中的单个优化层级
 * 
 * <p>在嵌入学习范式中，模型被视为多个嵌套的优化问题。
 * 每个层级有自己的更新频率、上下文流和参数集合。</p>
 * 
 * @author TinyAI Team
 */
public class NestedOptimizationLevel {
    
    /**
     * 层级索引，从0开始
     */
    private int levelIndex;
    
    /**
     * 更新频率率（0 < frequency ≤ 1）
     * 频率为1表示每步都更新，0.1表示每10步更新一次
     */
    private float updateFrequency;
    
    /**
     * 该层级的上下文流
     */
    private ContextFlow contextFlow;
    
    /**
     * 该层级管理的参数列表
     */
    private List<Variable> parameters;
    
    /**
     * 父层级引用（更低频率的层级）
     */
    private NestedOptimizationLevel parentLevel;
    
    /**
     * 子层级列表（更高频率的层级）
     */
    private List<NestedOptimizationLevel> childLevels;
    
    /**
     * 上次更新的步骤数
     */
    private int lastUpdateStep;
    
    /**
     * 学习率（每个层级可以有不同的学习率）
     */
    private float learningRate;
    
    /**
     * 构造函数
     * 
     * @param levelIndex 层级索引
     * @param updateFrequency 更新频率
     * @param learningRate 学习率
     */
    public NestedOptimizationLevel(int levelIndex, float updateFrequency, float learningRate) {
        this.levelIndex = levelIndex;
        this.updateFrequency = Math.max(0.0001f, Math.min(1.0f, updateFrequency));
        this.learningRate = learningRate;
        this.parameters = new ArrayList<>();
        this.childLevels = new ArrayList<>();
        this.lastUpdateStep = -1;
        this.contextFlow = null;
    }
    
    /**
     * 简化构造函数
     * 
     * @param levelIndex 层级索引
     * @param updateFrequency 更新频率
     */
    public NestedOptimizationLevel(int levelIndex, float updateFrequency) {
        this(levelIndex, updateFrequency, 0.001f);
    }
    
    /**
     * 判断在当前步骤是否应该更新
     * 
     * <p>更新判定规则：当 (currentStep × updateFrequency) 的整数部分
     * 发生变化时，表示应该更新</p>
     * 
     * @param currentStep 当前训练步骤
     * @return 是否应该更新
     */
    public boolean shouldUpdate(int currentStep) {
        if (currentStep <= 0) {
            return true; // 第一步总是更新
        }
        
        // 计算更新间隔
        int updateInterval = (int)(1.0f / updateFrequency);
        
        // 检查是否到达更新间隔
        boolean should = (currentStep % updateInterval == 0);
        
        return should;
    }
    
    /**
     * 计算该层级的局部误差（均方误差）
     * 
     * @param input 输入数据
     * @param target 目标数据
     * @return 局部误差
     */
    public Variable computeLocalError(Variable input, Variable target) {
        if (input == null || target == null) {
            return null;
        }
        
        Variable diff = input.sub(target);
        Variable squared = diff.mul(diff);
        return squared.mean(0, true);
    }
    
    /**
     * 更新该层级的参数
     * 注意：参数通过替换列表中的引用来更新。外部持有参数引用的代码
     * 应通过 getParameters() 重新获取最新引用。
     * 
     * @param gradients 梯度列表
     */
    public void updateParameters(List<Variable> gradients) {
        if (gradients == null || parameters.isEmpty()) {
            return;
        }
        
        int count = Math.min(parameters.size(), gradients.size());
        
        for (int i = 0; i < count; i++) {
            Variable param = parameters.get(i);
            Variable grad = gradients.get(i);
            
            if (param != null && grad != null) {
                // param = param - learningRate * grad
                Variable update = grad.mul(new Variable(learningRate));
                Variable newParam = param.sub(update);
                
                // 通过替换列表中的引用来更新参数
                // 如果原参数有名称，保留到新参数上
                if (param.getName() != null) {
                    newParam.setName(param.getName());
                }
                parameters.set(i, newParam);
            }
        }
        
        lastUpdateStep = Math.max(lastUpdateStep, 0);
    }
    
    /**
     * 向父层级传播上下文信息
     * 
     * @param contextData 上下文数据
     */
    public void propagateToParent(Variable contextData) {
        if (parentLevel == null || contextData == null) {
            return;
        }
        
        // 获取父层级的上下文流
        ContextFlow parentFlow = parentLevel.getContextFlow();
        if (parentFlow != null) {
            // 流动上下文到父层级
            parentFlow.flow(contextData);
        }
    }
    
    /**
     * 向子层级分发上下文信息
     * 
     * @param contextData 上下文数据
     */
    public void propagateToChildren(Variable contextData) {
        if (childLevels.isEmpty() || contextData == null) {
            return;
        }
        
        // 向所有子层级传播上下文
        for (NestedOptimizationLevel child : childLevels) {
            ContextFlow childFlow = child.getContextFlow();
            if (childFlow != null) {
                childFlow.flow(contextData);
            }
        }
    }
    
    /**
     * 添加参数
     * 
     * @param parameter 要添加的参数
     */
    public void addParameter(Variable parameter) {
        if (parameter != null) {
            this.parameters.add(parameter);
        }
    }
    
    /**
     * 添加子层级
     * 
     * @param child 子层级
     */
    public void addChildLevel(NestedOptimizationLevel child) {
        if (child != null && !childLevels.contains(child)) {
            childLevels.add(child);
            child.setParentLevel(this);
        }
    }
    
    // Getters and Setters
    
    public int getLevelIndex() {
        return levelIndex;
    }
    
    public void setLevelIndex(int levelIndex) {
        this.levelIndex = levelIndex;
    }
    
    public float getUpdateFrequency() {
        return updateFrequency;
    }
    
    public void setUpdateFrequency(float updateFrequency) {
        this.updateFrequency = Math.max(0.0001f, Math.min(1.0f, updateFrequency));
    }
    
    public ContextFlow getContextFlow() {
        return contextFlow;
    }
    
    public void setContextFlow(ContextFlow contextFlow) {
        this.contextFlow = contextFlow;
    }
    
    public List<Variable> getParameters() {
        return parameters;
    }
    
    public void setParameters(List<Variable> parameters) {
        this.parameters = parameters != null ? parameters : new ArrayList<>();
    }
    
    public NestedOptimizationLevel getParentLevel() {
        return parentLevel;
    }
    
    public void setParentLevel(NestedOptimizationLevel parentLevel) {
        this.parentLevel = parentLevel;
    }
    
    public List<NestedOptimizationLevel> getChildLevels() {
        return childLevels;
    }
    
    public int getLastUpdateStep() {
        return lastUpdateStep;
    }
    
    public void setLastUpdateStep(int lastUpdateStep) {
        this.lastUpdateStep = lastUpdateStep;
    }
    
    public float getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
}
