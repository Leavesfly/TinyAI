package io.leavesfly.tinyai.agent.multi;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * 消息总线系统
 * 负责Agent间的通信，支持点对点消息和广播消息
 * 
 * @author 山泽
 */
public class MessageBus {
    
    // 订阅者管理：Agent ID -> 消息处理回调列表
    private final Map<String, List<Consumer<AgentMessage>>> subscribers;
    
    // 消息历史记录
    private final List<AgentMessage> messageHistory;
    
    // 最大历史记录数
    private final int maxHistory;
    
    // 异步消息处理线程池
    private final ExecutorService executorService;
    
    // 统计信息
    private volatile long totalMessages = 0;
    private volatile long broadcastMessages = 0;
    private volatile long pointToPointMessages = 0;
    
    public MessageBus() {
        this(1000);
    }
    
    public MessageBus(int maxHistory) {
        this.subscribers = new ConcurrentHashMap<>();
        this.messageHistory = new CopyOnWriteArrayList<>();
        this.maxHistory = maxHistory;
        this.executorService = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "MessageBus-Worker");
            t.setDaemon(true);
            return t;
        });
    }
    
    /**
     * 订阅消息
     * @param agentId Agent ID
     * @param callback 消息处理回调函数
     */
    public void subscribe(String agentId, Consumer<AgentMessage> callback) {
        subscribers.computeIfAbsent(agentId, k -> new CopyOnWriteArrayList<>()).add(callback);
    }
    
    /**
     * 取消订阅
     * @param agentId Agent ID
     */
    public void unsubscribe(String agentId) {
        subscribers.remove(agentId);
    }
    
    /**
     * 发布消息（异步）
     * @param message 要发布的消息
     */
    public void publishAsync(AgentMessage message) {
        // 记录消息历史
        addToHistory(message);
        
        // 异步处理消息分发
        executorService.submit(() -> {
            try {
                if ("broadcast".equals(message.getReceiverId())) {
                    // 广播消息
                    handleBroadcast(message);
                } else {
                    // 点对点消息
                    handlePointToPoint(message);
                }
            } catch (Exception e) {
                System.err.println("消息处理异常: " + e.getMessage());
                e.printStackTrace();
            }
        });
    }
    
    /**
     * 发布消息（同步）
     * @param message 要发布的消息
     */
    public void publish(AgentMessage message) {
        // 记录消息历史
        addToHistory(message);
        
        try {
            if ("broadcast".equals(message.getReceiverId())) {
                // 广播消息
                handleBroadcast(message);
            } else {
                // 点对点消息
                handlePointToPoint(message);
            }
        } catch (Exception e) {
            System.err.println("消息处理异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 处理广播消息
     */
    private void handleBroadcast(AgentMessage message) {
        broadcastMessages++;
        
        for (Map.Entry<String, List<Consumer<AgentMessage>>> entry : subscribers.entrySet()) {
            String agentId = entry.getKey();
            
            // 不发送给自己
            if (!agentId.equals(message.getSenderId())) {
                List<Consumer<AgentMessage>> callbacks = entry.getValue();
                for (Consumer<AgentMessage> callback : callbacks) {
                    try {
                        callback.accept(message);
                    } catch (Exception e) {
                        System.err.println("向Agent " + agentId + " 发送广播消息失败: " + e.getMessage());
                    }
                }
            }
        }
    }
    
    /**
     * 处理点对点消息
     */
    private void handlePointToPoint(AgentMessage message) {
        pointToPointMessages++;
        
        String receiverId = message.getReceiverId();
        List<Consumer<AgentMessage>> callbacks = subscribers.get(receiverId);
        
        if (callbacks != null) {
            for (Consumer<AgentMessage> callback : callbacks) {
                try {
                    callback.accept(message);
                } catch (Exception e) {
                    System.err.println("向Agent " + receiverId + " 发送消息失败: " + e.getMessage());
                }
            }
        } else {
            System.out.println("接收者 " + receiverId + " 未找到，消息未送达");
        }
    }
    
    /**
     * 添加到消息历史
     */
    private void addToHistory(AgentMessage message) {
        totalMessages++;
        messageHistory.add(message);
        
        // 限制历史记录数量
        while (messageHistory.size() > maxHistory) {
            messageHistory.remove(0);
        }
    }
    
    /**
     * 获取两个Agent之间的对话历史
     * @param agent1Id 第一个Agent ID
     * @param agent2Id 第二个Agent ID
     * @param limit 返回的消息数量限制
     * @return 对话历史列表（按时间排序）
     */
    public List<AgentMessage> getConversationHistory(String agent1Id, String agent2Id, int limit) {
        List<AgentMessage> conversation = new ArrayList<>();
        
        // 倒序遍历历史记录
        for (int i = messageHistory.size() - 1; i >= 0 && conversation.size() < limit; i--) {
            AgentMessage msg = messageHistory.get(i);
            
            // 检查是否是两个Agent之间的对话
            if ((msg.getSenderId().equals(agent1Id) && msg.getReceiverId().equals(agent2Id)) ||
                (msg.getSenderId().equals(agent2Id) && msg.getReceiverId().equals(agent1Id))) {
                conversation.add(0, msg); // 添加到列表开头以保持时间顺序
            }
        }
        
        return conversation;
    }
    
    /**
     * 获取Agent的消息历史
     * @param agentId Agent ID
     * @param limit 返回的消息数量限制
     * @return 消息历史列表
     */
    public List<AgentMessage> getAgentMessages(String agentId, int limit) {
        return messageHistory.stream()
                .filter(msg -> msg.getSenderId().equals(agentId) || msg.getReceiverId().equals(agentId))
                .skip(Math.max(0, messageHistory.size() - limit))
                .collect(Collectors.toList());
    }
    
    /**
     * 获取最近的消息
     * @param limit 返回的消息数量限制
     * @return 最近的消息列表
     */
    public List<AgentMessage> getRecentMessages(int limit) {
        int start = Math.max(0, messageHistory.size() - limit);
        return new ArrayList<>(messageHistory.subList(start, messageHistory.size()));
    }
    
    /**
     * 获取所有消息历史
     * @return 完整的消息历史列表
     */
    public List<AgentMessage> getAllMessages() {
        return new ArrayList<>(messageHistory);
    }
    
    /**
     * 清空消息历史
     */
    public void clearHistory() {
        messageHistory.clear();
        totalMessages = 0;
        broadcastMessages = 0;
        pointToPointMessages = 0;
    }
    
    /**
     * 获取当前订阅的Agent列表
     * @return Agent ID列表
     */
    public List<String> getSubscribedAgents() {
        return new ArrayList<>(subscribers.keySet());
    }
    
    /**
     * 检查Agent是否已订阅
     * @param agentId Agent ID
     * @return 是否已订阅
     */
    public boolean isSubscribed(String agentId) {
        return subscribers.containsKey(agentId);
    }
    
    /**
     * 获取统计信息
     * @return 包含统计信息的Map
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new ConcurrentHashMap<>();
        stats.put("totalMessages", totalMessages);
        stats.put("broadcastMessages", broadcastMessages);
        stats.put("pointToPointMessages", pointToPointMessages);
        stats.put("messageHistorySize", messageHistory.size());
        stats.put("subscribedAgents", subscribers.size());
        stats.put("maxHistory", maxHistory);
        return stats;
    }
    
    /**
     * 关闭消息总线
     */
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    @Override
    public String toString() {
        return String.format("MessageBus{totalMessages=%d, subscribers=%d, historySize=%d}", 
                totalMessages, subscribers.size(), messageHistory.size());
    }
}