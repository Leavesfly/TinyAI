package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;

/**
 * 流式响应回调接口
 * 用于处理LLM的流式输出
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public interface StreamCallback {
    
    /**
     * 接收到新的Token
     * 
     * @param token 新生成的Token
     */
    void onToken(String token);
    
    /**
     * 流式响应完成
     * 
     * @param response 完整的响应对象
     */
    void onComplete(ChatResponse response);
    
    /**
     * 发生错误
     * 
     * @param error 错误信息
     */
    void onError(Throwable error);
}
