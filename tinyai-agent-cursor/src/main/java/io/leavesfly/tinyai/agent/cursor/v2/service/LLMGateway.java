package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.model.*;

import java.util.List;

/**
 * LLM统一网关接口
 * 提供统一的大语言模型调用接口，屏蔽不同模型的API差异
 * 
 * 核心功能：
 * - 支持多模型统一调用（DeepSeek、Qwen等）
 * - 支持同步和流式响应
 * - 支持代码补全和对话
 * - 支持文本向量化
 * - 支持模型路由和负载均衡
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public interface LLMGateway {
    
    /**
     * 同步对话请求
     * 
     * @param request 聊天请求
     * @return 聊天响应
     */
    ChatResponse chat(ChatRequest request);
    
    /**
     * 流式对话请求
     * 
     * @param request 聊天请求
     * @param callback 流式响应回调
     */
    void chatStream(ChatRequest request, StreamCallback callback);
    
    /**
     * 代码补全请求
     * 
     * @param prefix 光标前的代码
     * @param suffix 光标后的代码（可选）
     * @param language 编程语言
     * @param maxTokens 最大生成token数
     * @return 补全建议
     */
    String complete(String prefix, String suffix, String language, int maxTokens);
    
    /**
     * 文本向量化
     * 
     * @param texts 待向量化的文本列表
     * @return 向量列表
     */
    List<double[]> embed(List<String> texts);
    
    /**
     * 单个文本向量化
     * 
     * @param text 待向量化的文本
     * @return 向量
     */
    double[] embedSingle(String text);
    
    /**
     * 获取可用模型列表
     * 
     * @return 模型信息列表
     */
    List<ModelInfo> getAvailableModels();
    
    /**
     * 设置首选模型
     * 
     * @param modelName 模型名称
     */
    void setPreferredModel(String modelName);
    
    /**
     * 获取当前首选模型
     * 
     * @return 模型名称
     */
    String getPreferredModel();
    
    /**
     * 检查模型是否可用
     * 
     * @param modelName 模型名称
     * @return 是否可用
     */
    boolean isModelAvailable(String modelName);
}
