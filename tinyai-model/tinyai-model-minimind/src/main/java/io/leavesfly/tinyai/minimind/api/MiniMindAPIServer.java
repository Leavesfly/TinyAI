package io.leavesfly.tinyai.minimind.api;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MiniMind API服务器
 * 
 * 基于Java标准库HttpServer实现的轻量级API服务
 * 兼容OpenAI API格式,零第三方依赖
 * 
 * 使用示例:
 * ```java
 * MiniMindAPIServer server = new MiniMindAPIServer(8080);
 * server.start();
 * ```
 * 
 * @author leavesfly
 * @since 2024
 */
public class MiniMindAPIServer {
    
    /** 最大请求体大小: 1MB */
    private static final int MAX_REQUEST_BODY_SIZE = 1024 * 1024;
    
    /** 速率限制: 每个IP每分钟最大请求数 */
    private static final int MAX_REQUESTS_PER_MINUTE = 60;
    
    /** 可选的 API Key（为空则不启用认证） */
    private static String apiKey = null;
    
    /** 每个IP的请求计数器 */
    private static final ConcurrentHashMap<String, AtomicInteger> requestCounters = new ConcurrentHashMap<>();
    
    /** 计数器重置时间戳 */
    private static volatile long counterResetTime = System.currentTimeMillis();
    
    private final HttpServer server;
    private final int port;
    
    /**
     * 构造函数
     * 
     * @param port 服务端口
     */
    public MiniMindAPIServer(int port) throws IOException {
        this.port = port;
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        
        // 设置线程池
        server.setExecutor(Executors.newFixedThreadPool(10));
        
        // 注册路由
        registerHandlers();
    }
    
    /**
     * 注册处理器
     */
    private void registerHandlers() {
        server.createContext("/v1/completions", new CompletionHandler());
        server.createContext("/v1/chat/completions", new ChatCompletionHandler());
        server.createContext("/v1/models", new ModelsHandler());
        server.createContext("/health", new HealthHandler());
        server.createContext("/", new RootHandler());
    }
    
    /**
     * 启动服务器
     */
    public void start() {
        server.start();
        System.out.println("=".repeat(60));
        System.out.println("MiniMind API Server Started");
        System.out.println("=".repeat(60));
        System.out.println("Port: " + port);
        System.out.println("Endpoints:");
        System.out.println("  POST http://localhost:" + port + "/v1/completions");
        System.out.println("  POST http://localhost:" + port + "/v1/chat/completions");
        System.out.println("  GET  http://localhost:" + port + "/v1/models");
        System.out.println("  GET  http://localhost:" + port + "/health");
        System.out.println("=".repeat(60));
    }
    
    /**
     * 停止服务器
     */
    public void stop() {
        server.stop(0);
        System.out.println("MiniMind API Server Stopped");
    }
    
    /**
     * 根路径处理器
     */
    static class RootHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String response = "MiniMind API Server v1.0.0\n" +
                            "API Documentation: /docs\n";
            
            sendResponse(exchange, 200, response);
        }
    }
    
    /**
     * 健康检查处理器
     */
    static class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            Map<String, Object> health = new LinkedHashMap<>();
            health.put("status", "healthy");
            health.put("timestamp", System.currentTimeMillis());
            
            String json = SimpleJSON.toJSON(health);
            sendJSONResponse(exchange, 200, json);
        }
    }
    
    /**
     * 模型列表处理器
     */
    static class ModelsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            List<Map<String, Object>> models = new ArrayList<>();
            
            Map<String, Object> model = new LinkedHashMap<>();
            model.put("id", "minimind");
            model.put("object", "model");
            model.put("created", System.currentTimeMillis() / 1000);
            model.put("owned_by", "tinyai");
            models.add(model);
            
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("object", "list");
            response.put("data", models);
            
            String json = SimpleJSON.toJSON(response);
            sendJSONResponse(exchange, 200, json);
        }
    }
    
    /**
     * 设置 API Key（设置后所有请求需携带 Authorization: Bearer <key>）
     */
    public static void setApiKey(String key) {
        apiKey = key;
    }
    
    /**
     * 验证请求的认证和速率限制
     * 
     * @return 如果验证通过返回 true，否则已发送错误响应并返回 false
     */
    static boolean validateRequest(HttpExchange exchange) throws IOException {
        // API Key 认证
        if (apiKey != null && !apiKey.isEmpty()) {
            String authHeader = exchange.getRequestHeaders().getFirst("Authorization");
            if (authHeader == null || !authHeader.equals("Bearer " + apiKey)) {
                sendJSONResponse(exchange, 401, 
                    "{\"error\":{\"message\":\"Invalid or missing API key\",\"type\":\"authentication_error\",\"code\":401}}");
                return false;
            }
        }
        
        // 速率限制
        String clientIp = exchange.getRemoteAddress().getAddress().getHostAddress();
        long now = System.currentTimeMillis();
        if (now - counterResetTime > 60_000) {
            requestCounters.clear();
            counterResetTime = now;
        }
        AtomicInteger counter = requestCounters.computeIfAbsent(clientIp, k -> new AtomicInteger(0));
        if (counter.incrementAndGet() > MAX_REQUESTS_PER_MINUTE) {
            sendJSONResponse(exchange, 429, 
                "{\"error\":{\"message\":\"Rate limit exceeded\",\"type\":\"rate_limit_error\",\"code\":429}}");
            return false;
        }
        
        return true;
    }
    
    /**
     * 发送文本响应（使用 try-with-resources 确保资源释放）
     */
    static void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
        byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
        
        exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=UTF-8");
        addCORSHeaders(exchange);
        
        exchange.sendResponseHeaders(statusCode, bytes.length);
        try (OutputStream outputStream = exchange.getResponseBody()) {
            outputStream.write(bytes);
        }
    }
    
    /**
     * 发送JSON响应（使用 try-with-resources 确保资源释放）
     */
    static void sendJSONResponse(HttpExchange exchange, int statusCode, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
        addCORSHeaders(exchange);
        
        exchange.sendResponseHeaders(statusCode, bytes.length);
        try (OutputStream outputStream = exchange.getResponseBody()) {
            outputStream.write(bytes);
        }
    }
    
    /**
     * 添加CORS头
     */
    static void addCORSHeaders(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }
    
    /**
     * 读取请求体（带大小限制）
     */
    static String readRequestBody(HttpExchange exchange) throws IOException {
        byte[] bytes = exchange.getRequestBody().readAllBytes();
        if (bytes.length > MAX_REQUEST_BODY_SIZE) {
            throw new IOException("Request body too large: " + bytes.length + " bytes (max: " + MAX_REQUEST_BODY_SIZE + ")");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    /**
     * 主函数
     */
    public static void main(String[] args) {
        try {
            int port = 8080;
            if (args.length > 0) {
                port = Integer.parseInt(args[0]);
            }
            
            MiniMindAPIServer server = new MiniMindAPIServer(port);
            server.start();
            
            // 优雅关闭
            Runtime.getRuntime().addShutdownHook(new Thread(server::stop));
            
        } catch (IOException e) {
            System.err.println("Failed to start server: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
