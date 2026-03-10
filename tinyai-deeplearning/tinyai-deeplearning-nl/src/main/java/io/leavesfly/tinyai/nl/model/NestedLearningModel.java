package io.leavesfly.tinyai.nl.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;
import io.leavesfly.tinyai.nl.memory.ContinuumMemorySystem;
import io.leavesfly.tinyai.nl.memory.MemoryType;
import io.leavesfly.tinyai.nl.optimizer.DeepOptimizer;

import java.util.HashMap;
import java.util.Map;

/**
 * 嵌套学习模型（NestedLearningModel）
 * 实现基于嵌套优化的完整模型，包含前向传播、反向传播和训练循环
 * 
 * @author TinyAI Team
 */
public class NestedLearningModel {
    
    private String name;
    private NestedLearningBlock block;
    private ContinuumMemorySystem memorySystem;
    private DeepOptimizer optimizer;
    private boolean training;
    
    public NestedLearningModel(String name, NestedLearningBlock block) {
        this.name = name;
        this.block = block;
        this.memorySystem = new ContinuumMemorySystem();
        this.training = true;
    }
    
    /**
     * 前向传播
     * 
     * @param inputs 输入变量
     * @return 输出变量
     */
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable output = block.forward(inputs);
        
        // 更新层级状态
        block.updateLevels();
        
        // 更新记忆系统
        if (memorySystem != null) {
            memorySystem.update(block.getCurrentStep());
        }
        
        return output;
    }
    
    /**
     * 执行一步训练
     * 包含前向传播、损失计算、反向传播和参数更新
     * 
     * @param input 输入数据
     * @param target 目标数据
     * @return 损失值
     */
    public Variable trainStep(Variable input, Variable target) {
        if (input == null || target == null || optimizer == null) {
            return null;
        }
        
        // 1. 清零梯度
        optimizer.zeroGrad();
        
        // 2. 前向传播
        Variable output = forward(input);
        if (output == null) {
            return null;
        }
        
        // 3. 计算损失（均方误差）
        Variable diff = output.sub(target);
        Variable loss = diff.mul(diff).mean(0, true);
        
        // 4. 反向传播
        loss.backward();
        
        // 5. 收集梯度并更新参数
        Map<Variable, Variable> gradients = new HashMap<>();
        for (NestedOptimizationLevel level : block.getOptimizationLevels()) {
            for (Variable param : level.getParameters()) {
                if (param.getGrad() != null) {
                    gradients.put(param, new Variable(param.getGrad()));
                }
            }
        }
        optimizer.step(gradients);
        
        // 6. 将输入存入短期记忆
        if (memorySystem != null) {
            memorySystem.store(MemoryType.SHORT_TERM, input, target);
        }
        
        return loss;
    }
    
    /**
     * 执行多步训练
     * 
     * @param inputs 输入数据数组
     * @param targets 目标数据数组
     * @param epochs 训练轮数
     * @return 最终损失值
     */
    public float train(Variable[] inputs, Variable[] targets, int epochs) {
        if (inputs == null || targets == null || inputs.length != targets.length) {
            return Float.NaN;
        }
        
        float lastLoss = Float.NaN;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0.0f;
            int validSteps = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                Variable loss = trainStep(inputs[i], targets[i]);
                if (loss != null) {
                    epochLoss += loss.getValue().get(new int[]{0});
                    validSteps++;
                }
            }
            
            if (validSteps > 0) {
                lastLoss = epochLoss / validSteps;
            }
        }
        
        return lastLoss;
    }
    
    /**
     * 设置训练/推理模式
     * 
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    public boolean isTraining() {
        return training;
    }
    
    public void setOptimizer(DeepOptimizer optimizer) {
        this.optimizer = optimizer;
        
        // 将模型的优化层级注册到优化器
        if (optimizer != null && block != null) {
            for (NestedOptimizationLevel level : block.getOptimizationLevels()) {
                optimizer.addLevel(level);
            }
        }
    }
    
    public DeepOptimizer getOptimizer() {
        return optimizer;
    }
    
    public String getName() {
        return name;
    }
    
    public NestedLearningBlock getBlock() {
        return block;
    }
    
    public ContinuumMemorySystem getMemorySystem() {
        return memorySystem;
    }
}
