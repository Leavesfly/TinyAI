package io.leavesfly.tinyai.agent.cursor.v2.adapter;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 模型适配器注册表
 * 管理所有已注册的模型适配器
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class AdapterRegistry {
    
    /**
     * 适配器映射（适配器名称 -> 适配器实例）
     */
    private final Map<String, ModelAdapter> adapters;
    
    /**
     * 模型到适配器的映射（模型名称 -> 适配器名称）
     */
    private final Map<String, String> modelToAdapter;
    
    public AdapterRegistry() {
        this.adapters = new ConcurrentHashMap<>();
        this.modelToAdapter = new ConcurrentHashMap<>();
    }
    
    /**
     * 注册适配器
     * 
     * @param adapter 模型适配器
     */
    public void register(ModelAdapter adapter) {
        if (adapter == null) {
            throw new IllegalArgumentException("Adapter cannot be null");
        }
        
        String adapterName = adapter.getName();
        adapters.put(adapterName, adapter);
        
        // 构建模型名称到适配器的映射
        String[] supportedModels = adapter.getSupportedModels();
        if (supportedModels != null) {
            for (String modelName : supportedModels) {
                modelToAdapter.put(modelName, adapterName);
            }
        }
        
        System.out.println("Registered adapter: " + adapterName + 
                          " (supports: " + Arrays.toString(supportedModels) + ")");
    }
    
    /**
     * 注销适配器
     * 
     * @param adapterName 适配器名称
     */
    public void unregister(String adapterName) {
        ModelAdapter adapter = adapters.remove(adapterName);
        if (adapter != null) {
            // 移除模型映射
            String[] supportedModels = adapter.getSupportedModels();
            if (supportedModels != null) {
                for (String modelName : supportedModels) {
                    modelToAdapter.remove(modelName);
                }
            }
        }
    }
    
    /**
     * 根据模型名称获取适配器
     * 
     * @param modelName 模型名称
     * @return 模型适配器
     */
    public ModelAdapter getAdapter(String modelName) {
        // 先查找精确匹配
        String adapterName = modelToAdapter.get(modelName);
        if (adapterName != null) {
            return adapters.get(adapterName);
        }
        
        // 遍历所有适配器查找支持该模型的适配器
        for (ModelAdapter adapter : adapters.values()) {
            if (adapter.supports(modelName)) {
                return adapter;
            }
        }
        
        return null;
    }
    
    /**
     * 根据适配器名称获取适配器
     * 
     * @param adapterName 适配器名称
     * @return 模型适配器
     */
    public ModelAdapter getAdapterByName(String adapterName) {
        return adapters.get(adapterName);
    }
    
    /**
     * 获取所有注册的适配器
     * 
     * @return 适配器列表
     */
    public List<ModelAdapter> getAllAdapters() {
        return new ArrayList<>(adapters.values());
    }
    
    /**
     * 获取所有可用的适配器
     * 
     * @return 可用的适配器列表
     */
    public List<ModelAdapter> getAvailableAdapters() {
        List<ModelAdapter> availableAdapters = new ArrayList<>();
        for (ModelAdapter adapter : adapters.values()) {
            if (adapter.isAvailable()) {
                availableAdapters.add(adapter);
            }
        }
        return availableAdapters;
    }
    
    /**
     * 获取所有支持的模型名称
     * 
     * @return 模型名称列表
     */
    public List<String> getSupportedModels() {
        return new ArrayList<>(modelToAdapter.keySet());
    }
    
    /**
     * 检查模型是否被支持
     * 
     * @param modelName 模型名称
     * @return 是否支持
     */
    public boolean isModelSupported(String modelName) {
        return getAdapter(modelName) != null;
    }
    
    /**
     * 获取适配器数量
     * 
     * @return 适配器数量
     */
    public int size() {
        return adapters.size();
    }
    
    /**
     * 清空所有适配器
     */
    public void clear() {
        adapters.clear();
        modelToAdapter.clear();
    }
}
