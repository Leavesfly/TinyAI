package io.leavesfly.tinyai.agent.cursor.v2.adapter;

import io.leavesfly.tinyai.agent.cursor.v2.model.*;
import io.leavesfly.tinyai.agent.cursor.v2.service.StreamCallback;

import java.util.*;

/**
 * DeepSeek模型适配器
 * 适配DeepSeek系列模型（deepseek-chat, deepseek-coder等）
 * 
 * 支持的模型：
 * - deepseek-chat: 通用对话模型
 * - deepseek-coder: 代码专用模型
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class DeepSeekAdapter extends BaseModelAdapter {
    
    private static final String DEFAULT_BASE_URL = "https://api.deepseek.com/v1";
    
    private static final String[] SUPPORTED_MODELS = {
        "deepseek-chat",
        "deepseek-coder"
    };
    
    public DeepSeekAdapter() {
        super("DeepSeek");
        this.baseUrl = DEFAULT_BASE_URL;
    }
    
    public DeepSeekAdapter(String apiKey) {
        this();
        this.apiKey = apiKey;
    }
    
    @Override
    public boolean supports(String modelName) {
        if (modelName == null) {
            return false;
        }
        for (String model : SUPPORTED_MODELS) {
            if (model.equals(modelName) || modelName.startsWith(model)) {
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
            // 构建DeepSeek API请求
            Map<String, Object> apiRequest = buildApiRequest(request);
            
            // TODO: 使用HTTP客户端发送请求
            // String responseJson = httpClient.post(baseUrl + "/chat/completions", apiRequest, apiKey);
            
            // 模拟响应（实际应从API获取）
            String responseJson = simulateApiResponse(request);
            
            // 解析响应
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
            // 构建DeepSeek API请求（流式）
            Map<String, Object> apiRequest = buildApiRequest(request);
            apiRequest.put("stream", true);
            
            // TODO: 使用HTTP客户端发送SSE流式请求
            // httpClient.streamPost(baseUrl + "/chat/completions", apiRequest, apiKey, 
            //     (chunk) -> handleStreamChunk(chunk, callback));
            
            // 模拟流式响应
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
            // TODO: 调用DeepSeek Embedding API
            // 暂时返回模拟向量
            return simulateEmbedding(text);
        });
    }
    
    /**
     * 构建DeepSeek API请求
     */
    private Map<String, Object> buildApiRequest(ChatRequest request) {
        Map<String, Object> apiRequest = new HashMap<>();
        
        // 模型
        apiRequest.put("model", request.getModel());
        
        // 消息列表
        List<Map<String, Object>> messages = new ArrayList<>();
        for (Message message : request.getMessages()) {
            Map<String, Object> messageMap = new HashMap<>();
            messageMap.put("role", message.getRole().getValue());
            messageMap.put("content", message.getContent());
            
            // 处理工具调用
            if (message.getToolCalls() != null && !message.getToolCalls().isEmpty()) {
                List<Map<String, Object>> toolCalls = new ArrayList<>();
                for (ToolCall toolCall : message.getToolCalls()) {
                    Map<String, Object> toolCallMap = new HashMap<>();
                    toolCallMap.put("id", toolCall.getId());
                    toolCallMap.put("type", toolCall.getType());
                    
                    Map<String, Object> function = new HashMap<>();
                    function.put("name", toolCall.getFunction().getName());
                    function.put("arguments", toolCall.getFunction().getArguments());
                    toolCallMap.put("function", function);
                    
                    toolCalls.add(toolCallMap);
                }
                messageMap.put("tool_calls", toolCalls);
            }
            
            // 处理工具响应
            if (message.getToolCallId() != null) {
                messageMap.put("tool_call_id", message.getToolCallId());
            }
            
            messages.add(messageMap);
        }
        apiRequest.put("messages", messages);
        
        // 工具定义
        if (request.getTools() != null && !request.getTools().isEmpty()) {
            List<Map<String, Object>> tools = new ArrayList<>();
            for (ToolDefinition tool : request.getTools()) {
                Map<String, Object> toolMap = new HashMap<>();
                toolMap.put("type", tool.getType());
                
                Map<String, Object> function = new HashMap<>();
                function.put("name", tool.getFunction().getName());
                function.put("description", tool.getFunction().getDescription());
                function.put("parameters", tool.getFunction().getParameters());
                toolMap.put("function", function);
                
                tools.add(toolMap);
            }
            apiRequest.put("tools", tools);
            
            if (request.getToolChoice() != null) {
                apiRequest.put("tool_choice", request.getToolChoice());
            }
        }
        
        // 可选参数
        if (request.getTemperature() != null) {
            apiRequest.put("temperature", request.getTemperature());
        }
        if (request.getTopP() != null) {
            apiRequest.put("top_p", request.getTopP());
        }
        if (request.getMaxTokens() != null) {
            apiRequest.put("max_tokens", request.getMaxTokens());
        }
        if (request.getStop() != null) {
            apiRequest.put("stop", request.getStop());
        }
        
        return apiRequest;
    }
    
    /**
     * 解析DeepSeek API响应
     */
    private ChatResponse parseApiResponse(String responseJson) {
        // TODO: 使用JSON解析库解析响应
        // 这里提供解析逻辑框架
        ChatResponse response = new ChatResponse();
        
        // 模拟解析
        response.setId("chatcmpl-" + UUID.randomUUID().toString());
        response.setObject("chat.completion");
        response.setCreated(System.currentTimeMillis() / 1000);
        response.setModel("deepseek-chat");
        
        ChatResponse.Choice choice = new ChatResponse.Choice();
        choice.setIndex(0);
        choice.setMessage(Message.assistant("This is a simulated response"));
        choice.setFinishReason("stop");
        
        response.setChoices(Collections.singletonList(choice));
        
        ChatResponse.Usage usage = new ChatResponse.Usage();
        usage.setPromptTokens(10);
        usage.setCompletionTokens(20);
        usage.setTotalTokens(30);
        response.setUsage(usage);
        
        return response;
    }
    
    /**
     * 处理流式响应块
     */
    private void handleStreamChunk(String chunk, StreamCallback callback) {
        // TODO: 解析SSE数据块
        // 典型格式：data: {"choices":[{"delta":{"content":"token"}}]}
        
        if (chunk.startsWith("data: ")) {
            String data = chunk.substring(6).trim();
            if ("[DONE]".equals(data)) {
                callback.onComplete(new ChatResponse());
                return;
            }
            
            // TODO: 解析JSON获取token
            // 暂时提取content
            callback.onToken(data);
        }
    }
    
    /**
     * 模拟API响应（用于演示，实际应删除）
     */
    private String simulateApiResponse(ChatRequest request) {
        return "{\"id\":\"chatcmpl-123\",\"object\":\"chat.completion\",\"created\":1234567890," +
               "\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"," +
               "\"content\":\"This is a simulated response from DeepSeek\"},\"finish_reason\":\"stop\"}]," +
               "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}";
    }
    
    /**
     * 模拟流式响应（用于演示，实际应删除）
     */
    private void simulateStreamResponse(ChatRequest request, StreamCallback callback) {
        String[] tokens = {"This ", "is ", "a ", "simulated ", "stream ", "response"};
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
    
    /**
     * 模拟向量化（用于演示，实际应删除）
     */
    private double[] simulateEmbedding(String text) {
        // 返回768维的模拟向量
        double[] embedding = new double[768];
        Random random = new Random(text.hashCode());
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] = random.nextGaussian();
        }
        return embedding;
    }
}
