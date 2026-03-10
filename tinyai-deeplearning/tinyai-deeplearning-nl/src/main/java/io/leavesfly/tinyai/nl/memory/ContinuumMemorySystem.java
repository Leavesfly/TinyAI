package io.leavesfly.tinyai.nl.memory;

import io.leavesfly.tinyai.func.Variable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 连续体记忆系统（ContinuumMemorySystem）
 * 实现多时间尺度的记忆连续体
 * 
 * <p>连续体记忆理论认为记忆不是离散的类型（短期、长期），
 * 而是一个连续的频谱。该系统实现了从高频到低频的多层级记忆。</p>
 * 
 * @author TinyAI Team
 */
public class ContinuumMemorySystem {
    
    /**
     * 记忆模块列表，按频率从高到低排序
     */
    private List<MemoryModule> memoryModules;
    
    /**
     * 记忆类型到模块的映射
     */
    private Map<MemoryType, MemoryModule> typeToModuleMap;
    
    /**
     * 记忆整合阈值
     * 当短期记忆的惊异度低于阈值时，整合到长期记忆
     */
    private float consolidationThreshold;
    
    /**
     * 是否启用自动整合
     */
    private boolean enableAutoConsolidation;
    
    /**
     * 整合间隔（步数）
     */
    private int consolidationInterval;
    
    /**
     * 上次整合的步骤
     */
    private int lastConsolidationStep;
    
    /**
     * 构造函数
     * 
     * @param memoryCapacities 各类型记忆的容量映射
     * @param consolidationThreshold 整合阈值
     */
    public ContinuumMemorySystem(Map<MemoryType, Integer> memoryCapacities, float consolidationThreshold) {
        this.memoryModules = new ArrayList<>();
        this.typeToModuleMap = new HashMap<>();
        this.consolidationThreshold = consolidationThreshold;
        this.enableAutoConsolidation = true;
        this.consolidationInterval = 100;
        this.lastConsolidationStep = -1;
        
        // 初始化各类型记忆模块
        for (MemoryType type : MemoryType.values()) {
            int capacity = memoryCapacities.getOrDefault(type, 100);
            MemoryModule module = new MemoryModule(type, capacity);
            memoryModules.add(module);
            typeToModuleMap.put(type, module);
        }
        
        // 按更新频率从高到低排序
        memoryModules.sort((m1, m2) -> 
            Float.compare(m2.getUpdateFrequency(), m1.getUpdateFrequency()));
    }
    
    /**
     * 简化构造函数，使用默认容量
     */
    public ContinuumMemorySystem() {
        Map<MemoryType, Integer> defaultCapacities = new HashMap<>();
        defaultCapacities.put(MemoryType.SHORT_TERM, 50);
        defaultCapacities.put(MemoryType.MEDIUM_TERM, 100);
        defaultCapacities.put(MemoryType.LONG_TERM, 200);
        defaultCapacities.put(MemoryType.ULTRA_LONG_TERM, 500);
        
        this.memoryModules = new ArrayList<>();
        this.typeToModuleMap = new HashMap<>();
        this.consolidationThreshold = 0.3f;
        this.enableAutoConsolidation = true;
        this.consolidationInterval = 100;
        this.lastConsolidationStep = -1;
        
        // 初始化各类型记忆模块
        for (MemoryType type : MemoryType.values()) {
            int capacity = defaultCapacities.get(type);
            MemoryModule module = new MemoryModule(type, capacity);
            memoryModules.add(module);
            typeToModuleMap.put(type, module);
        }
        
        // 按更新频率从高到低排序
        memoryModules.sort((m1, m2) -> 
            Float.compare(m2.getUpdateFrequency(), m1.getUpdateFrequency()));
    }
    
    /**
     * 存储记忆到指定类型的模块
     * 
     * @param type 记忆类型
     * @param key 记忆键
     * @param value 记忆值
     */
    public void store(MemoryType type, Variable key, Variable value) {
        MemoryModule module = typeToModuleMap.get(type);
        if (module != null) {
            module.store(key, value);
        }
    }
    
    /**
     * 存储到短期记忆（默认入口）
     * 
     * @param key 记忆键
     * @param value 记忆值
     */
    public void store(Variable key, Variable value) {
        store(MemoryType.SHORT_TERM, key, value);
    }
    
    /**
     * 从所有层级检索记忆
     * 优先从高频记忆检索
     * 
     * @param queryKey 查询键
     * @return 检索到的记忆值
     */
    public Variable retrieve(Variable queryKey) {
        if (queryKey == null) {
            return null;
        }
        
        // 从高频到低频依次检索
        for (MemoryModule module : memoryModules) {
            Variable result = module.retrieve(queryKey);
            if (result != null) {
                return result;
            }
        }
        
        return null;
    }
    
    /**
     * 从指定类型的记忆模块检索
     * 
     * @param type 记忆类型
     * @param queryKey 查询键
     * @return 检索到的记忆值
     */
    public Variable retrieve(MemoryType type, Variable queryKey) {
        MemoryModule module = typeToModuleMap.get(type);
        if (module != null) {
            return module.retrieve(queryKey);
        }
        return null;
    }
    
    /**
     * 更新所有记忆模块
     * 
     * @param currentStep 当前步骤
     */
    public void update(int currentStep) {
        // 更新所有模块
        for (MemoryModule module : memoryModules) {
            module.update(currentStep);
        }
        
        // 执行记忆整合
        if (enableAutoConsolidation && shouldConsolidate(currentStep)) {
            consolidate();
            lastConsolidationStep = currentStep;
        }
    }
    
    /**
     * 判断是否应该执行整合
     * 
     * @param currentStep 当前步骤
     * @return 是否应该整合
     */
    private boolean shouldConsolidate(int currentStep) {
        if (lastConsolidationStep < 0) {
            return true;
        }
        return (currentStep - lastConsolidationStep) >= consolidationInterval;
    }
    
    /**
     * 执行记忆整合
     * 将短期记忆整合到长期记忆
     */
    public void consolidate() {
        MemoryModule shortTerm = typeToModuleMap.get(MemoryType.SHORT_TERM);
        MemoryModule mediumTerm = typeToModuleMap.get(MemoryType.MEDIUM_TERM);
        MemoryModule longTerm = typeToModuleMap.get(MemoryType.LONG_TERM);
        
        if (shortTerm == null || mediumTerm == null || longTerm == null) {
            return;
        }
        
        // 整合短期记忆到中期记忆
        consolidateBetween(shortTerm, mediumTerm);
        
        // 整合中期记忆到长期记忆
        consolidateBetween(mediumTerm, longTerm);
    }
    
    /**
     * 在两个记忆模块间执行整合
     * 将源模块中惊异度低于阈值的稳定记忆迁移到目标模块
     * 
     * @param source 源模块
     * @param target 目标模块
     */
    private void consolidateBetween(MemoryModule source, MemoryModule target) {
        List<Variable> sourceKeys = source.getMemory().getKeys();
        List<Variable> sourceValues = source.getMemory().getValues();
        
        // 收集需要整合（迁移）的记忆索引
        List<Integer> indicesToRemove = new ArrayList<>();
        
        for (int i = 0; i < sourceKeys.size(); i++) {
            Variable key = sourceKeys.get(i);
            Variable value = sourceValues.get(i);
            
            float surprise = source.computeSurprise(key);
            
            // 惊异度低于阈值的记忆是稳定记忆，应整合到更长期的模块
            if (surprise < consolidationThreshold) {
                target.store(key, value);
                indicesToRemove.add(i);
            }
        }
        
        // 从源模块中移除已整合的记忆
        if (!indicesToRemove.isEmpty()) {
            java.util.Set<Integer> removeSet = new java.util.HashSet<>(indicesToRemove);
            List<Variable> keys = sourceKeys;
            List<Variable> values = sourceValues;
            source.getMemory().clear();
            for (int i = 0; i < keys.size(); i++) {
                if (!removeSet.contains(i)) {
                    source.store(keys.get(i), values.get(i));
                }
            }
        }
    }
    
    /**
     * 计算跨所有层级的平均惊异度
     * 
     * @param input 输入数据
     * @return 平均惊异度
     */
    public float computeAverageSurprise(Variable input) {
        if (input == null || memoryModules.isEmpty()) {
            return 1.0f;
        }
        
        float totalSurprise = 0.0f;
        int count = 0;
        
        for (MemoryModule module : memoryModules) {
            if (module.getSize() > 0) {
                totalSurprise += module.computeSurprise(input);
                count++;
            }
        }
        
        return count > 0 ? totalSurprise / count : 1.0f;
    }
    
    /**
     * 清空所有记忆
     */
    public void clear() {
        for (MemoryModule module : memoryModules) {
            module.clear();
        }
        lastConsolidationStep = -1;
    }
    
    /**
     * 获取指定类型的记忆模块
     * 
     * @param type 记忆类型
     * @return 记忆模块
     */
    public MemoryModule getMemoryModule(MemoryType type) {
        return typeToModuleMap.get(type);
    }
    
    /**
     * 获取所有记忆模块
     * 
     * @return 记忆模块列表
     */
    public List<MemoryModule> getAllModules() {
        return new ArrayList<>(memoryModules);
    }
    
    /**
     * 获取记忆系统统计信息
     * 
     * @return 统计信息字符串
     */
    public String getStatistics() {
        StringBuilder sb = new StringBuilder();
        sb.append("记忆系统统计:\n");
        
        for (MemoryModule module : memoryModules) {
            sb.append(String.format("  %s: %d/%d (%.1f%%)\n",
                module.getMemoryType().getDescription(),
                module.getSize(),
                module.getCapacity(),
                (module.getSize() * 100.0 / module.getCapacity())));
        }
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getConsolidationThreshold() {
        return consolidationThreshold;
    }
    
    public void setConsolidationThreshold(float consolidationThreshold) {
        this.consolidationThreshold = consolidationThreshold;
    }
    
    public boolean isEnableAutoConsolidation() {
        return enableAutoConsolidation;
    }
    
    public void setEnableAutoConsolidation(boolean enableAutoConsolidation) {
        this.enableAutoConsolidation = enableAutoConsolidation;
    }
    
    public int getConsolidationInterval() {
        return consolidationInterval;
    }
    
    public void setConsolidationInterval(int consolidationInterval) {
        this.consolidationInterval = Math.max(1, consolidationInterval);
    }
}
