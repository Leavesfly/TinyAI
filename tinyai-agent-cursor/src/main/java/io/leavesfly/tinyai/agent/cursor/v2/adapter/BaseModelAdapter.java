package io.leavesfly.tinyai.agent.cursor.v2.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.model.ChatRequest;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;
import io.leavesfly.tinyai.agent.cursor.v2.service.StreamCallback;

/**
 * 模型适配器抽象基类
 * 提供通用的错误处理和重试逻辑
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public abstract class BaseModelAdapter implements ModelAdapter {
    
    /**
     * 适配器名称
     */
    protected final String name;
    
    /**
     * 最大重试次数
     */
    protected int maxRetries = 3;
    
    /**
     * 重试延迟（毫秒）
     */
    protected long retryDelay = 1000L;
    
    /**
     * API密钥
     */
    protected String apiKey;
    
    /**
     * API基础URL
     */
    protected String baseUrl;
    
    public BaseModelAdapter(String name) {
        this.name = name;
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public boolean isAvailable() {
        return apiKey != null && !apiKey.isEmpty();
    }
    
    /**
     * 执行带重试的操作
     * 
     * @param operation 操作函数
     * @param <T> 返回类型
     * @return 操作结果
     */
    protected <T> T executeWithRetry(RetryableOperation<T> operation) {
        Exception lastException = null;
        
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return operation.execute();
            } catch (Exception e) {
                lastException = e;
                
                // 判断是否应该重试
                if (attempt < maxRetries && shouldRetry(e)) {
                    try {
                        // 指数退避
                        long delay = retryDelay * (long) Math.pow(2, attempt);
                        Thread.sleep(delay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        throw new RuntimeException("Operation failed after " + (maxRetries + 1) + " attempts", lastException);
    }
    
    /**
     * 判断是否应该重试
     * 
     * @param exception 异常
     * @return 是否应该重试
     */
    protected boolean shouldRetry(Exception exception) {
        // 默认对网络错误和限流错误重试
        String message = exception.getMessage();
        if (message != null) {
            return message.contains("timeout") ||
                   message.contains("429") ||
                   message.contains("503") ||
                   message.contains("connection");
        }
        return false;
    }
    
    /**
     * 创建错误响应
     * 
     * @param errorCode 错误码
     * @param errorMessage 错误消息
     * @return 错误响应
     */
    protected ChatResponse createErrorResponse(String errorCode, String errorMessage) {
        ChatResponse response = new ChatResponse();
        ChatResponse.Error error = new ChatResponse.Error(errorCode, errorMessage);
        response.setError(error);
        return response;
    }
    
    /**
     * 验证请求参数
     * 
     * @param request 聊天请求
     */
    protected void validateRequest(ChatRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        if (request.getMessages() == null || request.getMessages().isEmpty()) {
            throw new IllegalArgumentException("Messages cannot be empty");
        }
    }
    
    /**
     * 获取或设置配置
     */
    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }
    
    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }
    
    public void setMaxRetries(int maxRetries) {
        this.maxRetries = maxRetries;
    }
    
    public void setRetryDelay(long retryDelay) {
        this.retryDelay = retryDelay;
    }
    
    /**
     * 可重试操作接口
     */
    @FunctionalInterface
    protected interface RetryableOperation<T> {
        T execute() throws Exception;
    }
}
