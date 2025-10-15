package io.leavesfly.tinyai.agent.cursor.v2.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.StreamCallback;

import java.util.*;

/**
 * Qwen模型适配器
 * 适配阿里云通义千问系列模型
 * 
 * 支持的模型：
 * - qwen-max: 最强性能模型
 * - qwen-plus: 平衡性能模型
 * - qwen-turbo: 快速响应模型
 * - qwen-coder: 代码专用模型
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class QwenAdapter extends BaseModelAdapter {
    
    private static final String DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1";
    
    private static final String[] SUPPORTED_MODELS = {
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen-coder"
    };
    
    public QwenAdapter() {
        super("Qwen");
        this.baseUrl = DEFAULT_BASE_URL;
    }
    
    public QwenAdapter(String apiKey) {
        this();
        this.apiKey = apiKey;
    }
    
    @Override
    public boolean supports(String modelName) {
        if (modelName == null) {
            return false;
        }
        for (String model : SUPPORTED_MODELS) {
            if (model.equals(modelName) || modelName.startsWith(model) || modelName.startsWith("qwen-")) {
                return true;
            }
        }
        return false;
    }
    
    @Override
    public String[] getSupportedModels() {
        return SUPPORTED_MODELS.clone();
    }
    
    @Override
    public ChatResponse chat(ChatRequest request) {
        validateRequest(request);
        
        return executeWithRetry(() -> {
            // 构建Qwen API请求
            Map<String, Object> apiRequest = buildApiRequest(request);
            
            // TODO: 使用HTTP客户端发送请求
            // 注意：Qwen使用POST /services/aigc/text-generation/generation
            String responseJson = simulateApiResponse(request);
            
            return parseApiResponse(responseJson);
        });
    }
    
    @Override
    public void chatStream(ChatRequest request, StreamCallback callback) {
        validateRequest(request);
        
        if (callback == null) {
            throw new IllegalArgumentException("Callback cannot be null");
        }
        
        try {
            Map<String, Object> apiRequest = buildApiRequest(request);
            apiRequest.put("stream", true);
            
            // TODO: 实现Qwen流式调用
            simulateStreamResponse(request, callback);
            
        } catch (Exception e) {
            callback.onError(e);
        }
    }
    
    @Override
    public double[] embed(String text) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("Text cannot be empty");
        }
        
        return executeWithRetry(() -> {
            // TODO: 调用Qwen Embedding API
            return simulateEmbedding(text);
        });
    }
    
    /**
     * 构建Qwen API请求
     * 注意：Qwen的API格式可能与OpenAI格式略有不同
     */
    private Map<String, Object> buildApiRequest(ChatRequest request) {
        Map<String, Object> apiRequest = new HashMap<>();
        
        // Qwen使用input字段包装messages
        Map<String, Object> input = new HashMap<>();
        List<Map<String, Object>> messages = new ArrayList<>();
        
        for (Message message : request.getMessages()) {
            Map<String, Object> messageMap = new HashMap<>();
            messageMap.put("role", convertRole(message.getRole()));
            messageMap.put("content", message.getContent());
            messages.add(messageMap);
        }
        
        input.put("messages", messages);
        apiRequest.put("model", request.getModel());
        apiRequest.put("input", input);
        
        // 参数配置
        Map<String, Object> parameters = new HashMap<>();
        if (request.getTemperature() != null) {
            parameters.put("temperature", request.getTemperature());
        }
        if (request.getTopP() != null) {
            parameters.put("top_p", request.getTopP());
        }
        if (request.getMaxTokens() != null) {
            parameters.put("max_tokens", request.getMaxTokens());
        }
        
        if (!parameters.isEmpty()) {
            apiRequest.put("parameters", parameters);
        }
        
        return apiRequest;
    }
    
    /**
     * 转换角色名称（Qwen可能使用不同的角色名称）
     */
    private String convertRole(Message.Role role) {
        switch (role) {
            case SYSTEM:
                return "system";
            case USER:
                return "user";
            case ASSISTANT:
                return "assistant";
            case TOOL:
                return "tool";
            default:
                return role.getValue();
        }
    }
    
    /**
     * 解析Qwen API响应
     */
    private ChatResponse parseApiResponse(String responseJson) {
        // TODO: 解析Qwen特定格式的响应
        ChatResponse response = new ChatResponse();
        
        response.setId("qwen-" + UUID.randomUUID().toString());
        response.setObject("chat.completion");
        response.setCreated(System.currentTimeMillis() / 1000);
        response.setModel("qwen-max");
        
        ChatResponse.Choice choice = new ChatResponse.Choice();
        choice.setIndex(0);
        choice.setMessage(Message.assistant("This is a simulated Qwen response"));
        choice.setFinishReason("stop");
        
        response.setChoices(Collections.singletonList(choice));
        
        ChatResponse.Usage usage = new ChatResponse.Usage();
        usage.setPromptTokens(10);
        usage.setCompletionTokens(20);
        usage.setTotalTokens(30);
        response.setUsage(usage);
        
        return response;
    }
    
    // 以下为模拟方法，实际使用时应删除
    
    private String simulateApiResponse(ChatRequest request) {
        return "{\"output\":{\"text\":\"This is a simulated response from Qwen\"}," +
               "\"usage\":{\"input_tokens\":10,\"output_tokens\":20}}";
    }
    
    private void simulateStreamResponse(ChatRequest request, StreamCallback callback) {
        String[] tokens = {"This ", "is ", "Qwen ", "streaming ", "response"};
        for (String token : tokens) {
            callback.onToken(token);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                callback.onError(e);
                return;
            }
        }
        callback.onComplete(new ChatResponse());
    }
    
    private double[] simulateEmbedding(String text) {
        double[] embedding = new double[1536]; // Qwen embedding维度
        Random random = new Random(text.hashCode());
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] = random.nextGaussian();
        }
        return embedding;
    }
}
