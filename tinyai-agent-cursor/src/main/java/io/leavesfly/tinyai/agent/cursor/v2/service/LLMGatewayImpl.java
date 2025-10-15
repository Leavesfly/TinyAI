package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.adapter.AdapterRegistry;
import io.leavesfly.tinyai.agent.cursor.v2.adapter.ModelAdapter;
import io.leavesfly.tinyai.agent.cursor.v2.model.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * LLM网关实现类
 * 提供统一的大语言模型调用接口，支持：
 * - 模型路由：根据模型名称自动选择适配器
 * - 负载均衡：多适配器轮询（预留扩展）
 * - 降级处理：首选模型不可用时自动切换
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class LLMGatewayImpl implements LLMGateway {
    
    /**
     * 适配器注册表
     */
    private final AdapterRegistry adapterRegistry;
    
    /**
     * 首选模型名称
     */
    private String preferredModel;
    
    /**
     * 降级模型列表
     */
    private final List<String> fallbackModels;
    
    /**
     * 是否启用自动降级
     */
    private boolean enableFallback = true;
    
    public LLMGatewayImpl(AdapterRegistry adapterRegistry) {
        this.adapterRegistry = adapterRegistry;
        this.fallbackModels = new ArrayList<>();
    }
    
    @Override
    public ChatResponse chat(ChatRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        
        // 确定使用的模型
        String modelName = determineModel(request);
        
        // 获取适配器
        ModelAdapter adapter = getAdapter(modelName);
        if (adapter == null) {
            throw new RuntimeException("No adapter available for model: " + modelName);
        }
        
        // 尝试调用，支持降级
        Exception lastException = null;
        List<String> modelsToTry = new ArrayList<>();
        modelsToTry.add(modelName);
        
        if (enableFallback) {
            modelsToTry.addAll(fallbackModels);
        }
        
        for (String model : modelsToTry) {
            try {
                adapter = getAdapter(model);
                if (adapter != null && adapter.isAvailable()) {
                    request.setModel(model); // 更新请求的模型名称
                    return adapter.chat(request);
                }
            } catch (Exception e) {
                lastException = e;
                System.err.println("Model " + model + " failed: " + e.getMessage());
                // 继续尝试下一个模型
            }
        }
        
        // 所有模型都失败
        throw new RuntimeException("All models failed", lastException);
    }
    
    @Override
    public void chatStream(ChatRequest request, StreamCallback callback) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        if (callback == null) {
            throw new IllegalArgumentException("Callback cannot be null");
        }
        
        String modelName = determineModel(request);
        ModelAdapter adapter = getAdapter(modelName);
        
        if (adapter == null) {
            callback.onError(new RuntimeException("No adapter available for model: " + modelName));
            return;
        }
        
        if (!adapter.isAvailable()) {
            callback.onError(new RuntimeException("Adapter not available: " + adapter.getName()));
            return;
        }
        
        try {
            adapter.chatStream(request, callback);
        } catch (Exception e) {
            callback.onError(e);
        }
    }
    
    @Override
    public String complete(String prefix, String suffix, String language, int maxTokens) {
        // 构建代码补全请求
        ChatRequest request = ChatRequest.builder()
                .model(preferredModel != null ? preferredModel : "deepseek-coder")
                .maxTokens(maxTokens)
                .temperature(0.2) // 代码补全使用较低温度
                .build();
        
        // 构建提示词
        StringBuilder prompt = new StringBuilder();
        prompt.append("Complete the following ").append(language).append(" code:\n\n");
        prompt.append("```").append(language).append("\n");
        prompt.append(prefix);
        if (suffix != null && !suffix.isEmpty()) {
            prompt.append("\n[COMPLETE HERE]\n");
            prompt.append(suffix);
        }
        prompt.append("\n```\n\n");
        prompt.append("Only provide the code to fill in [COMPLETE HERE]. Do not include explanations.");
        
        request.addUserMessage(prompt.toString());
        
        ChatResponse response = chat(request);
        return response.getContent();
    }
    
    @Override
    public List<double[]> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            throw new IllegalArgumentException("Texts cannot be empty");
        }
        
        String modelName = preferredModel != null ? preferredModel : "deepseek-chat";
        ModelAdapter adapter = getAdapter(modelName);
        
        if (adapter == null) {
            throw new RuntimeException("No adapter available for embedding");
        }
        
        List<double[]> embeddings = new ArrayList<>();
        for (String text : texts) {
            embeddings.add(adapter.embed(text));
        }
        
        return embeddings;
    }
    
    @Override
    public double[] embedSingle(String text) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("Text cannot be empty");
        }
        
        String modelName = preferredModel != null ? preferredModel : "deepseek-chat";
        ModelAdapter adapter = getAdapter(modelName);
        
        if (adapter == null) {
            throw new RuntimeException("No adapter available for embedding");
        }
        
        return adapter.embed(text);
    }
    
    @Override
    public List<ModelInfo> getAvailableModels() {
        List<ModelInfo> models = new ArrayList<>();
        
        for (ModelAdapter adapter : adapterRegistry.getAvailableAdapters()) {
            String[] supportedModels = adapter.getSupportedModels();
            for (String modelName : supportedModels) {
                ModelInfo info = new ModelInfo(modelName, adapter.getName());
                info.setDisplayName(modelName);
                info.setAvailable(adapter.isAvailable());
                info.setSupportsToolCalling(true); // 大部分现代模型支持
                info.setSupportsStreaming(true);
                info.setType("chat");
                models.add(info);
            }
        }
        
        return models;
    }
    
    @Override
    public void setPreferredModel(String modelName) {
        this.preferredModel = modelName;
    }
    
    @Override
    public String getPreferredModel() {
        return preferredModel;
    }
    
    @Override
    public boolean isModelAvailable(String modelName) {
        ModelAdapter adapter = getAdapter(modelName);
        return adapter != null && adapter.isAvailable();
    }
    
    /**
     * 添加降级模型
     */
    public void addFallbackModel(String modelName) {
        if (!fallbackModels.contains(modelName)) {
            fallbackModels.add(modelName);
        }
    }
    
    /**
     * 设置是否启用降级
     */
    public void setEnableFallback(boolean enableFallback) {
        this.enableFallback = enableFallback;
    }
    
    /**
     * 确定使用的模型
     */
    private String determineModel(ChatRequest request) {
        // 优先使用请求中指定的模型
        if (request.getModel() != null && !request.getModel().isEmpty()) {
            return request.getModel();
        }
        
        // 其次使用首选模型
        if (preferredModel != null && !preferredModel.isEmpty()) {
            return preferredModel;
        }
        
        // 最后使用第一个可用的模型
        List<String> supportedModels = adapterRegistry.getSupportedModels();
        if (!supportedModels.isEmpty()) {
            return supportedModels.get(0);
        }
        
        throw new RuntimeException("No model available");
    }
    
    /**
     * 获取适配器
     */
    private ModelAdapter getAdapter(String modelName) {
        return adapterRegistry.getAdapter(modelName);
    }
    
    /**
     * 获取适配器注册表
     */
    public AdapterRegistry getAdapterRegistry() {
        return adapterRegistry;
    }
}
