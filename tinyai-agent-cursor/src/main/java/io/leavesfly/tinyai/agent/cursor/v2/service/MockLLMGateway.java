package io.leavesfly.tinyai.agent.cursor.v2.service;


import io.leavesfly.tinyai.agent.cursor.v2.model.ChatRequest;
import io.leavesfly.tinyai.agent.cursor.v2.model.ChatResponse;
import io.leavesfly.tinyai.agent.cursor.v2.model.Message;
import io.leavesfly.tinyai.agent.cursor.v2.model.ModelInfo;

import java.util.*;

/**
 * 模拟LLM网关
 * 用于演示和测试
 *
 * @author TinyAI
 * @since 2.0.0
 */
public class MockLLMGateway implements LLMGateway {

    @Override
    public ChatResponse chat(ChatRequest request) {
        ChatResponse response = new ChatResponse();
//        response.setContent(generateMockResponse(request));
//        response.setModel(request.getModel() != null ? request.getModel() : "mock-model");
//        response.setFinishReason("stop");
//
//        Usage usage = new Usage();
//        usage.setPromptTokens(100);
//        usage.setCompletionTokens(200);
//        usage.setTotalTokens(300);
//        response.setUsage(usage);

        return response;
    }

    @Override
    public void chatStream(ChatRequest request, StreamCallback callback) {
        String content = generateMockResponse(request);
        String[] words = content.split(" ");

        for (String word : words) {
            callback.onToken(word + " ");
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

//        callback.onComplete();
    }

    @Override
    public String complete(String prefix, String suffix, String language, int maxTokens) {
        return null;
    }


    @Override
    public List<double[]> embed(List<String> texts) {
        List<double[]> embeddings = new ArrayList<>();
        for (String text : texts) {
            embeddings.add(generateMockEmbedding(text));
        }
        return embeddings;
    }

    @Override
    public double[] embedSingle(String text) {
        return generateMockEmbedding(text);
    }

    @Override
    public List<ModelInfo> getAvailableModels() {
        return null;
    }

    @Override
    public void setPreferredModel(String modelName) {
        // Mock implementation
    }

    @Override
    public String getPreferredModel() {
        return "mock-model";
    }

    @Override
    public boolean isModelAvailable(String modelName) {
        return true;
    }

    /**
     * 生成模拟响应
     */
    private String generateMockResponse(ChatRequest request) {
        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "模拟响应";
        }

        Message lastMessage = messages.get(messages.size() - 1);
        String userContent = lastMessage.getContent();

        // 根据内容类型返回不同的模拟响应
        if (userContent.contains("研究主题") || userContent.contains("分解")) {
            return "1. 什么是并发编程?\\n" +
                    "2. Java并发的核心机制有哪些?\\n" +
                    "3. 常见的并发问题及解决方案?\\n" +
                    "4. 最佳实践总结";
        } else if (userContent.contains("检索") || userContent.contains("提取")) {
            return "根据检索结果，关键信息包括：\\n" +
                    "- Java并发编程使用线程实现\\n" +
                    "- 提供synchronized和Lock机制\\n" +
                    "- 线程池是重要的资源管理工具";
        } else if (userContent.contains("分析") || userContent.contains("深度")) {
            return "深度分析结果：\\n" +
                    "1. 并发编程的核心挑战是线程安全\\n" +
                    "2. 合理使用同步机制可以避免数据竞争\\n" +
                    "3. 线程池比直接创建线程更高效";
        } else if (userContent.contains("报告") || userContent.contains("撰写")) {
            return "# Java并发编程最佳实践研究报告\\n\\n" +
                    "## 执行摘要\\n" +
                    "本报告全面研究了Java并发编程的最佳实践。\\n\\n" +
                    "## 主要发现\\n" +
                    "1. 线程安全是核心关注点\\n" +
                    "2. 使用线程池管理资源\\n" +
                    "3. 合理选择同步机制\\n\\n" +
                    "## 结论\\n" +
                    "遵循最佳实践可以编写高效、安全的并发程序。";
        }

        return "这是对\"" + userContent.substring(0, Math.min(30, userContent.length())) + "...\"的模拟响应";
    }

    /**
     * 生成模拟向量
     */
    private double[] generateMockEmbedding(String text) {
        Random random = new Random(text.hashCode());
        double[] embedding = new double[768];
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] = random.nextDouble();
        }
        return embedding;
    }

    /**
     * Usage类
     */
    public static class Usage {
        private int promptTokens;
        private int completionTokens;
        private int totalTokens;

        public int getPromptTokens() {
            return promptTokens;
        }

        public void setPromptTokens(int promptTokens) {
            this.promptTokens = promptTokens;
        }

        public int getCompletionTokens() {
            return completionTokens;
        }

        public void setCompletionTokens(int completionTokens) {
            this.completionTokens = completionTokens;
        }

        public int getTotalTokens() {
            return totalTokens;
        }

        public void setTotalTokens(int totalTokens) {
            this.totalTokens = totalTokens;
        }
    }
}
