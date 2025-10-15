package io.leavesfly.tinyai.agent.cursor.v2.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.model.ChatRequest;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;
import io.leavesfly.tinyai.agent.cursor.v2.service.StreamCallback;

/**
 * 模型适配器接口
 * 定义不同LLM提供商的统一适配规范
 * 
 * 职责：
 * - 判断是否支持指定模型
 * - 构建模型专用请求格式
 * - 解析模型响应为统一格式
 * - 处理流式响应
 * - 统一错误处理
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public interface ModelAdapter {
    
    /**
     * 判断是否支持指定模型
     * 
     * @param modelName 模型名称
     * @return 是否支持
     */
    boolean supports(String modelName);
    
    /**
     * 获取适配器名称
     * 
     * @return 适配器名称
     */
    String getName();
    
    /**
     * 同步调用模型
     * 
     * @param request 聊天请求
     * @return 聊天响应
     */
    ChatResponse chat(ChatRequest request);
    
    /**
     * 流式调用模型
     * 
     * @param request 聊天请求
     * @param callback 流式响应回调
     */
    void chatStream(ChatRequest request, StreamCallback callback);
    
    /**
     * 文本向量化
     * 
     * @param text 待向量化的文本
     * @return 向量
     */
    double[] embed(String text);
    
    /**
     * 检查适配器是否可用
     * 
     * @return 是否可用
     */
    boolean isAvailable();
    
    /**
     * 获取支持的模型列表
     * 
     * @return 模型名称列表
     */
    String[] getSupportedModels();
}
