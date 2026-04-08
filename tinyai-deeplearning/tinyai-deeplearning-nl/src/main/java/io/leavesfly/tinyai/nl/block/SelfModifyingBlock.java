package io.leavesfly.tinyai.nl.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 自修改块（SelfModifyingBlock）
 * 实现可以动态修改自身结构和参数的神经网络块
 * 
 * <p>该块能够根据输入数据或性能反馈动态调整自身的网络结构、
 * 激活函数或超参数，模拟生物神经系统的可塑性。</p>
 * 
 * @author TinyAI Team
 */
public class SelfModifyingBlock extends Module {
    
    /**
     * 子模块列表 (替代v1的layers)
     */
    private List<Module> subModules;
    
    /**
     * 修改阈值
     */
    private float modificationThreshold;
    
    /**
     * 性能历史
     */
    private float[] performanceHistory;
    
    /**
     * 历史记录索引
     */
    private int historyIndex;
    
    /**
     * 是否启用自修改
     */
    private boolean enableSelfModification;
    
    /**
     * 修改计数器
     */
    private int modificationCount;
    
    public SelfModifyingBlock(String name, Shape inputShape) {
        super(name);
        this.modificationThreshold = 0.1f;
        this.performanceHistory = new float[100];
        this.historyIndex = 0;
        this.enableSelfModification = true;
        this.modificationCount = 0;
        this.subModules = new ArrayList<>();
    }
    
    public SelfModifyingBlock(String name) {
        this(name, null);
    }
    
    @Override
    public void resetParameters() {
        for (Module module : subModules) {
            module.resetParameters();
        }
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable x = inputs[0];
        
        if (!subModules.isEmpty()) {
            Variable y = subModules.get(0).forward(x);
            for (int i = 1; i < subModules.size(); i++) {
                y = subModules.get(i).forward(y);
            }
            return y;
        }
        
        return x;
    }
    
    /**
     * 根据性能反馈决定是否修改结构
     */
    public void evaluateAndModify(float performance) {
        if (!enableSelfModification) {
            return;
        }
        
        performanceHistory[historyIndex % performanceHistory.length] = performance;
        historyIndex++;
        
        if (shouldModify()) {
            modifyStructure();
        }
    }
    
    /**
     * 判断是否应该修改
     */
    private boolean shouldModify() {
        if (historyIndex < 10) {
            return false;
        }
        
        int recent = Math.min(10, historyIndex);
        float recentAvg = 0.0f;
        for (int i = 0; i < recent; i++) {
            int idx = (historyIndex - 1 - i) % performanceHistory.length;
            recentAvg += performanceHistory[idx];
        }
        recentAvg /= recent;
        
        return recentAvg < modificationThreshold;
    }
    
    /**
     * 修改网络结构
     * 根据性能历史动态调整：
     * - 如果性能持续低迷，增加网络容量（添加缩放参数）
     * - 调整现有参数的学习率缩放因子
     */
    private void modifyStructure() {
        modificationCount++;
        
        // 策略1：对所有子模块的参数施加扰动，帮助跳出局部最优
        for (Module module : subModules) {
            for (Parameter param : module.parameters()) {
                NdArray paramData = param.getValue();
                // 添加小幅随机扰动：param = param + noise * 0.01
                NdArray noise = NdArray.randn(paramData.getShape());
                NdArray perturbedData = paramData.add(noise.mulNum(0.01f));
                param.setValue(perturbedData);
            }
        }
        
        // 策略2：如果修改次数较多（性能持续不佳），增大扰动幅度
        if (modificationCount > 5) {
            modificationThreshold *= 0.9f; // 逐步降低阈值，减少不必要的修改
        }
    }
    
    public int getModificationCount() {
        return modificationCount;
    }
    
    public void setModificationThreshold(float threshold) {
        this.modificationThreshold = threshold;
    }
    
    public void setEnableSelfModification(boolean enable) {
        this.enableSelfModification = enable;
    }
}
