package io.leavesfly.tinyai.minimind.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.util.*;

/**
 * Chat Completion API处理器
 * 
 * 实现OpenAI兼容的/v1/chat/completions端点
 * 支持多轮对话功能
 * 
 * API格式:
 * ```json
 * POST /v1/chat/completions
 * {
 *   "model": "minimind",
 *   "messages": [
 *     {"role": "system", "content": "You are a helpful assistant."},
 *     {"role": "user", "content": "Hello!"}
 *   ],
 *   "max_tokens": 100,
 *   "temperature": 0.7,
 *   "top_p": 0.9,
 *   "stream": false
 * }
 * ```
 * 
 * @author leavesfly
 * @since 2024
 */
public class ChatCompletionHandler implements HttpHandler {
    
    /** 最大对话轮数 */
    private static final int MAX_CONVERSATION_ROUNDS = 10;
    
    @Override
    public void handle(HttpExchange exchange) throws IOException {
        // 处理OPTIONS预检请求
        if ("OPTIONS".equals(exchange.getRequestMethod())) {
            handleOptions(exchange);
            return;
        }
        
        // 只支持POST
        if (!"POST".equals(exchange.getRequestMethod())) {
            sendError(exchange, 405, "Method Not Allowed");
            return;
        }
        
        // 认证和速率限制
        if (!MiniMindAPIServer.validateRequest(exchange)) {
            return;
        }
        
        try {
            // 读取请求
            String requestBody = MiniMindAPIServer.readRequestBody(exchange);
            Map<String, Object> request = SimpleJSON.parseJSON(requestBody);
            
            // 解析参数
            String model = (String) request.getOrDefault("model", "minimind");
            Object messagesObj = request.get("messages");
            int maxTokens = getInt(request, "max_tokens", 100);
            double temperature = getDouble(request, "temperature", 0.7);
            double topP = getDouble(request, "top_p", 0.9);
            boolean stream = getBoolean(request, "stream", false);
            
            // 解析消息列表
            if (!(messagesObj instanceof List)) {
                sendError(exchange, 400, "Messages must be an array");
                return;
            }
            
            List<?> messagesList = (List<?>) messagesObj;
            if (messagesList.isEmpty()) {
                sendError(exchange, 400, "Messages array is empty");
                return;
            }
            
            // 转换消息格式
            List<ChatMessage> messages = parseMessages(messagesList);
            
            // 生成回复
            String reply = generateChatReply(messages, maxTokens, temperature, topP);
            
            // 构建响应
            Map<String, Object> response = buildChatResponse(model, messages, reply);
            
            // 发送响应
            String json = SimpleJSON.toJSON(response);
            MiniMindAPIServer.sendJSONResponse(exchange, 200, json);
            
        } catch (Exception e) {
            e.printStackTrace();
            sendError(exchange, 500, "Internal Server Error: " + e.getMessage());
        }
    }
    
    /**
     * 解析消息列表
     */
    private List<ChatMessage> parseMessages(List<?> messagesList) {
        List<ChatMessage> messages = new ArrayList<>();
        
        for (Object msgObj : messagesList) {
            if (msgObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> msgMap = (Map<String, Object>) msgObj;
                String role = msgMap.containsKey("role") ? String.valueOf(msgMap.get("role")) : "user";
                String content = msgMap.containsKey("content") ? String.valueOf(msgMap.get("content")) : "";
                messages.add(new ChatMessage(role, content));
            }
        }
        
        return messages;
    }
    
    /**
     * 生成对话回复（集成实际的MiniMind模型）
     */
    private String generateChatReply(List<ChatMessage> messages, int maxTokens, 
                                    double temperature, double topP) {
        try {
            if (SharedModelHolder.getModel() == null || SharedModelHolder.getTokenizer() == null) {
                return "[Error: Model not initialized]";
            }
            
            // 1. 构建对话上下文
            StringBuilder context = new StringBuilder();
            
            // 保留最近MAX_CONVERSATION_ROUNDS轮对话
            int startIdx = Math.max(0, messages.size() - MAX_CONVERSATION_ROUNDS);
            for (int i = startIdx; i < messages.size(); i++) {
                ChatMessage msg = messages.get(i);
                if ("system".equals(msg.role)) {
                    context.append("系统: ").append(msg.content).append("\n");
                } else if ("user".equals(msg.role)) {
                    context.append("用户: ").append(msg.content).append("\n");
                } else if ("assistant".equals(msg.role)) {
                    context.append("助手: ").append(msg.content).append("\n");
                }
            }
            context.append("助手: ");
            
            // 保存prompt的长度，用于后续提取响应
            String promptText = context.toString();
            
            // 2. 编码输入
            List<Integer> promptIds = SharedModelHolder.getTokenizer().encode(promptText, false, false);
            int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
            
            // 3. 调用模型生成
            int[] generated = SharedModelHolder.getModel().generate(
                promptArray,
                maxTokens,
                (float) temperature,
                0,  // topK
                (float) topP
            );
            
            // 4. 解码输出
            List<Integer> genIds = new ArrayList<>();
            for (int id : generated) {
                genIds.add(id);
            }
            String fullResponse = SharedModelHolder.getTokenizer().decode(genIds, true);
            
            // 5. 提取助手回复部分（更鲁棒的方式：移除prompt部分）
            String response = fullResponse;
            if (fullResponse.length() > promptText.length()) {
                response = fullResponse.substring(promptText.length()).trim();
            }
            
            // 移除可能的其他角色标签
            if (response.contains("\n用户: ")) {
                response = response.substring(0, response.indexOf("\n用户: ")).trim();
            }
            if (response.contains("\n系统: ")) {
                response = response.substring(0, response.indexOf("\n系统: ")).trim();
            }
            
            return response;
            
        } catch (Exception e) {
            e.printStackTrace();
            return "[Error: " + e.getMessage() + "]";
        }
    }
    
    /**
     * 构建OpenAI格式响应
     */
    private Map<String, Object> buildChatResponse(String model, List<ChatMessage> messages, String reply) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "chatcmpl-" + UUID.randomUUID().toString());
        response.put("object", "chat.completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", model);
        
        // Choices数组
        List<Map<String, Object>> choices = new ArrayList<>();
        Map<String, Object> choice = new LinkedHashMap<>();
        
        // Message对象
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", reply);
        
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", "stop");
        choices.add(choice);
        response.put("choices", choices);
        
        // Usage统计
        int promptTokens = messages.stream()
            .mapToInt(m -> estimateTokens(m.content))
            .sum();
        int completionTokens = estimateTokens(reply);
        
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", promptTokens);
        usage.put("completion_tokens", completionTokens);
        usage.put("total_tokens", promptTokens + completionTokens);
        response.put("usage", usage);
        
        return response;
    }
    
    /**
     * 估算Token数量(简化实现)
     */
    private int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        return text.split("\\s+").length;
    }
    
    /**
     * 获取整数参数
     */
    private int getInt(Map<String, Object> map, String key, int defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        if (value instanceof Number) return ((Number) value).intValue();
        return defaultValue;
    }
    
    /**
     * 获取浮点参数
     */
    private double getDouble(Map<String, Object> map, String key, double defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        if (value instanceof Number) return ((Number) value).doubleValue();
        return defaultValue;
    }
    
    /**
     * 获取布尔参数
     */
    private boolean getBoolean(Map<String, Object> map, String key, boolean defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        if (value instanceof Boolean) return (Boolean) value;
        return defaultValue;
    }
    
    /**
     * 处理OPTIONS请求
     */
    private void handleOptions(HttpExchange exchange) throws IOException {
        MiniMindAPIServer.addCORSHeaders(exchange);
        exchange.sendResponseHeaders(204, -1);
    }
    
    /**
     * 发送错误响应
     */
    private void sendError(HttpExchange exchange, int statusCode, String message) throws IOException {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("error", Map.of(
            "message", message,
            "type", "invalid_request_error",
            "code", statusCode
        ));
        
        String json = SimpleJSON.toJSON(error);
        MiniMindAPIServer.sendJSONResponse(exchange, statusCode, json);
    }
    
    /**
     * 聊天消息
     */
    private static class ChatMessage {
        String role;
        String content;
        
        ChatMessage(String role, String content) {
            this.role = role;
            this.content = content;
        }
    }
}