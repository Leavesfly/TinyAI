package io.leavesfly.tinyai.minimind.api;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.util.*;

/**
 * Completion API处理器
 * 
 * 实现OpenAI兼容的/v1/completions端点
 * 支持文本补全功能
 * 
 * API格式:
 * ```json
 * POST /v1/completions
 * {
 *   "model": "minimind",
 *   "prompt": "Hello, world!",
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
public class CompletionHandler implements HttpHandler {
    
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
            Object promptObj = request.get("prompt");
            int maxTokens = getInt(request, "max_tokens", 100);
            double temperature = getDouble(request, "temperature", 0.7);
            double topP = getDouble(request, "top_p", 0.9);
            boolean stream = getBoolean(request, "stream", false);
            
            // 处理prompt(支持字符串或字符串数组)
            String prompt;
            if (promptObj instanceof String) {
                prompt = (String) promptObj;
            } else if (promptObj instanceof List) {
                List<?> prompts = (List<?>) promptObj;
                prompt = prompts.isEmpty() ? "" : prompts.get(0).toString();
            } else {
                sendError(exchange, 400, "Invalid prompt format");
                return;
            }
            
            // 验证参数
            if (prompt == null || prompt.isEmpty()) {
                sendError(exchange, 400, "Prompt is required");
                return;
            }
            
            // 生成文本
            String generatedText = generateText(prompt, maxTokens, temperature, topP);
            
            // 构建响应
            Map<String, Object> response = buildResponse(model, prompt, generatedText, maxTokens);
            
            // 发送响应
            String json = SimpleJSON.toJSON(response);
            MiniMindAPIServer.sendJSONResponse(exchange, 200, json);
            
        } catch (Exception e) {
            e.printStackTrace();
            sendError(exchange, 500, "Internal Server Error: " + e.getMessage());
        }
    }
    
    /**
     * 生成文本（集成实际的MiniMind模型）
     */
    private String generateText(String prompt, int maxTokens, double temperature, double topP) {
        try {
            if (SharedModelHolder.getModel() == null || SharedModelHolder.getTokenizer() == null) {
                return "[Error: Model not initialized]";
            }
                
            // 1. 编码输入
            List<Integer> promptIds = SharedModelHolder.getTokenizer().encode(prompt, false, false);
            int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
                
            // 2. 调用模型生成
            int[] generated = SharedModelHolder.getModel().generate(
                promptArray,
                maxTokens,
                (float) temperature,
                0,  // topK
                (float) topP
            );
                
            // 3. 解码输出
            List<Integer> genIds = new ArrayList<>();
            for (int id : generated) {
                genIds.add(id);
            }
            String fullText = SharedModelHolder.getTokenizer().decode(genIds, true);
                
            // 4. 提取生成部分（移除prompt）
            String generatedPart = fullText;
            if (fullText.length() > prompt.length()) {
                generatedPart = fullText.substring(prompt.length());
            }
                
            return generatedPart;
                
        } catch (Exception e) {
            e.printStackTrace();
            return "[Error: " + e.getMessage() + "]";
        }
    }
    
    /**
     * 构建OpenAI格式响应
     */
    private Map<String, Object> buildResponse(String model, String prompt, String text, int maxTokens) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "cmpl-" + UUID.randomUUID().toString());
        response.put("object", "text_completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", model);
        
        // Choices数组
        List<Map<String, Object>> choices = new ArrayList<>();
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("text", text);
        choice.put("index", 0);
        choice.put("logprobs", null);
        choice.put("finish_reason", "length");
        choices.add(choice);
        response.put("choices", choices);
        
        // Usage统计
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", estimateTokens(prompt));
        usage.put("completion_tokens", estimateTokens(text));
        usage.put("total_tokens", estimateTokens(prompt) + estimateTokens(text));
        response.put("usage", usage);
        
        return response;
    }
    
    /**
     * 估算Token数量(简化实现)
     */
    private int estimateTokens(String text) {
        // 简化估算: 按空格分词
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
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return defaultValue;
    }
    
    /**
     * 获取浮点参数
     */
    private double getDouble(Map<String, Object> map, String key, double defaultValue) {
        Object value = map.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return defaultValue;
    }
    
    /**
     * 获取布尔参数
     */
    private boolean getBoolean(Map<String, Object> map, String key, boolean defaultValue) {
        Object value = map.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
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
}
