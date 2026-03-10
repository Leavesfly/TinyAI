package io.leavesfly.tinyai.nl.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;
import io.leavesfly.tinyai.nl.core.ContextFlow;
import io.leavesfly.tinyai.nl.core.FlowDirection;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import java.util.Collection;

import java.util.ArrayList;
import java.util.List;

/**
 * 嵌套学习块（NestedLearningBlock）
 * 实现支持多层级优化的神经网络块
 * 
 * <p>该块将传统的神经网络层组织成多个嵌套的优化层级，
 * 每个层级有自己的更新频率和学习率，实现多时间尺度的学习。</p>
 * 
 * @author TinyAI Team
 */
public class NestedLearningBlock extends Module {
    
    /**
     * 子模块列表 (替代v1的layers)
     */
    private List<Module> subModules;
    
    /**
     * 嵌套优化层级列表
     */
    private List<NestedOptimizationLevel> optimizationLevels;
    
    /**
     * 层级数量
     */
    private int numLevels;
    
    /**
     * 上下文流列表
     */
    private List<ContextFlow> contextFlows;
    
    /**
     * 当前训练步骤
     */
    private int currentStep;
    
    /**
     * 是否启用层级间上下文传播
     */
    private boolean enableContextFlow;
    
    /**
     * 构造函数
     * 
     * @param name 块名称
     * @param numLevels 层级数量
     * @param inputShape 输入形状
     */
    public NestedLearningBlock(String name, int numLevels, Shape inputShape) {
        super(name);
        this.numLevels = Math.max(1, numLevels);
        this.optimizationLevels = new ArrayList<>();
        this.contextFlows = new ArrayList<>();
        this.subModules = new ArrayList<>();
        this.currentStep = 0;
        this.enableContextFlow = true;
        
        initializeLevels();
    }
    
    /**
     * 简化构造函数
     * 
     * @param name 块名称
     * @param numLevels 层级数量
     */
    public NestedLearningBlock(String name, int numLevels) {
        this(name, numLevels, null);
    }
    
    /**
     * 初始化优化层级
     * 创建具有不同更新频率的层级
     */
    private void initializeLevels() {
        // 创建层级，频率从高到低
        for (int i = 0; i < numLevels; i++) {
            // 计算更新频率：高层级频率高，低层级频率低
            float frequency = (float) Math.pow(0.1, i);
            float learningRate = 0.001f * frequency; // 学习率与频率成正比
            
            NestedOptimizationLevel level = new NestedOptimizationLevel(i, frequency, learningRate);
            optimizationLevels.add(level);
            
            // 创建上下文流
            float compressionRate = 1.0f - (i * 0.2f); // 层级越低，压缩率越高
            ContextFlow flow = new ContextFlow(null, FlowDirection.BIDIRECTIONAL, compressionRate);
            contextFlows.add(flow);
            level.setContextFlow(flow);
        }
        
        // 建立层级间的父子关系
        for (int i = 0; i < numLevels - 1; i++) {
            optimizationLevels.get(i).addChildLevel(optimizationLevels.get(i + 1));
        }
    }
    
    @Override
    public void resetParameters() {
        // 初始化所有子模块
        for (Module module : subModules) {
            module.resetParameters();
        }
        
        // 将子模块的参数分配到不同的优化层级
        distributeParametersToLevels();
    }
    
    /**
     * 将参数分配到不同的优化层级
     * 策略：将子模块均匀分配到各个优化层级，
     * 每个子模块的所有参数归属于同一个层级
     */
    private void distributeParametersToLevels() {
        if (subModules.isEmpty()) {
            return;
        }
        
        // 先清空所有层级的参数
        for (NestedOptimizationLevel level : optimizationLevels) {
            level.getParameters().clear();
        }
        
        int modulesPerLevel = Math.max(1, subModules.size() / numLevels);
        
        for (int i = 0; i < subModules.size(); i++) {
            int levelIndex = Math.min(i / modulesPerLevel, numLevels - 1);
            NestedOptimizationLevel level = optimizationLevels.get(levelIndex);
            
            Module module = subModules.get(i);
            // Parameter 继承自 Variable，可以直接添加到优化层级
            for (Parameter param : module.parameters()) {
                level.addParameter(param);
            }
        }
    }
    
    /**
     * 添加子模块并注册到 Module 系统
     * 
     * @param moduleName 模块名称
     * @param module 子模块
     */
    public void addSubModule(String moduleName, Module module) {
        if (module != null) {
            subModules.add(module);
            registerModule(moduleName, module);
        }
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable x = inputs[0];
        
        // 执行前向传播
        if (!subModules.isEmpty()) {
            Variable y = subModules.get(0).forward(x);
            for (int i = 1; i < subModules.size(); i++) {
                y = subModules.get(i).forward(y);
            }
            
            // 如果启用上下文流，传播上下文信息
            if (enableContextFlow) {
                propagateContext(y);
            }
            
            return y;
        }
        
        return x;
    }
    
    /**
     * 传播上下文信息
     * 
     * @param output 输出数据
     */
    private void propagateContext(Variable output) {
        if (output == null || optimizationLevels.isEmpty()) {
            return;
        }
        
        // 从低层级向高层级传播
        for (int i = optimizationLevels.size() - 1; i >= 0; i--) {
            NestedOptimizationLevel level = optimizationLevels.get(i);
            level.propagateToParent(output);
        }
        
        // 从高层级向低层级传播
        for (int i = 0; i < optimizationLevels.size(); i++) {
            NestedOptimizationLevel level = optimizationLevels.get(i);
            level.propagateToChildren(output);
        }
    }
    
    /**
     * 更新块状态
     * 根据当前步骤更新各个层级
     */
    public void updateLevels() {
        currentStep++;
        
        for (NestedOptimizationLevel level : optimizationLevels) {
            if (level.shouldUpdate(currentStep)) {
                // 层级需要更新
                level.setLastUpdateStep(currentStep);
            }
        }
    }
    
    /**
     * 检查指定层级是否应该更新
     * 
     * @param levelIndex 层级索引
     * @return 是否应该更新
     */
    public boolean shouldUpdateLevel(int levelIndex) {
        if (levelIndex < 0 || levelIndex >= optimizationLevels.size()) {
            return false;
        }
        
        NestedOptimizationLevel level = optimizationLevels.get(levelIndex);
        return level.shouldUpdate(currentStep);
    }
    
    /**
     * 获取层级统计信息
     * 
     * @return 统计信息字符串
     */
    public String getLevelStatistics() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("嵌套学习块 '%s' 统计:\n", name));
        sb.append(String.format("  当前步骤: %d\n", currentStep));
        sb.append(String.format("  层级数量: %d\n", numLevels));
        
        for (int i = 0; i < optimizationLevels.size(); i++) {
            NestedOptimizationLevel level = optimizationLevels.get(i);
            sb.append(String.format("  层级 %d: 频率=%.4f, 学习率=%.6f, 上次更新=%d\n",
                i,
                level.getUpdateFrequency(),
                level.getLearningRate(),
                level.getLastUpdateStep()));
        }
        
        return sb.toString();
    }
    
    /**
     * 重置所有层级状态
     */
    public void resetLevels() {
        currentStep = 0;
        for (NestedOptimizationLevel level : optimizationLevels) {
            level.setLastUpdateStep(-1);
        }
    }
    
    // Getters and Setters
    
    public List<NestedOptimizationLevel> getOptimizationLevels() {
        return new ArrayList<>(optimizationLevels);
    }
    
    public NestedOptimizationLevel getLevel(int index) {
        if (index >= 0 && index < optimizationLevels.size()) {
            return optimizationLevels.get(index);
        }
        return null;
    }
    
    public int getNumLevels() {
        return numLevels;
    }
    
    public int getCurrentStep() {
        return currentStep;
    }
    
    public boolean isEnableContextFlow() {
        return enableContextFlow;
    }
    
    public void setEnableContextFlow(boolean enableContextFlow) {
        this.enableContextFlow = enableContextFlow;
    }
    
    public List<ContextFlow> getContextFlows() {
        return new ArrayList<>(contextFlows);
    }
}
