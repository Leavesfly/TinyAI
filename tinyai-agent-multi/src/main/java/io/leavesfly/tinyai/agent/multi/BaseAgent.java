package io.leavesfly.tinyai.agent.multi;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * Agent基础抽象类
 * 定义所有Agent的核心功能和通用行为
 * 
 * @author 山泽
 */
public abstract class BaseAgent {
    
    // Agent基本信息
    protected final String agentId;
    protected final String name;
    protected final String role;
    
    // Agent状态
    protected volatile AgentState state;
    
    // 核心组件
    protected final MessageBus messageBus;
    protected final LLMSimulator llm;
    
    // Agent能力和配置
    protected final List<String> capabilities;
    protected final Map<String, Object> tools;
    protected final List<Map<String, Object>> memory;
    protected final int maxMemory;
    
    // 任务管理
    protected volatile AgentTask currentTask;
    protected final BlockingQueue<AgentTask> taskQueue;
    protected final List<AgentTask> completedTasks;
    
    // 性能指标
    protected final AgentMetrics metrics;
    
    // 运行控制
    protected final AtomicBoolean running;
    protected final ExecutorService executorService;
    protected CompletableFuture<Void> mainLoop;
    
    /**
     * 构造函数
     */
    public BaseAgent(String agentId, String name, String role, MessageBus messageBus, LLMSimulator llm) {
        this.agentId = agentId;
        this.name = name;
        this.role = role;
        this.state = AgentState.OFFLINE;
        
        this.messageBus = messageBus;
        this.llm = llm;
        
        this.capabilities = new ArrayList<>();
        this.tools = new HashMap<>();
        this.memory = new ArrayList<>();
        this.maxMemory = 100;
        
        this.taskQueue = new LinkedBlockingQueue<>();
        this.completedTasks = new ArrayList<>();
        
        this.metrics = new AgentMetrics();
        this.running = new AtomicBoolean(false);
        this.executorService = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "Agent-" + name);
            t.setDaemon(true);
            return t;
        });
        
        // 初始化能力
        initializeCapabilities();
        
        // 订阅消息
        this.messageBus.subscribe(this.agentId, this::handleMessage);
    }
    
    /**
     * 启动Agent
     */
    public CompletableFuture<Void> start() {
        if (running.compareAndSet(false, true)) {
            state = AgentState.IDLE;
            metrics.setStartTime(System.currentTimeMillis());
            
            // 启动主循环
            mainLoop = CompletableFuture.runAsync(this::runMainLoop, executorService);
            
            System.out.println(String.format("Agent %s (%s) 已启动", name, role));
            return mainLoop;
        }
        return CompletableFuture.completedFuture(null);
    }
    
    /**
     * 停止Agent
     */
    public CompletableFuture<Void> stop() {
        if (running.compareAndSet(true, false)) {
            state = AgentState.OFFLINE;
            
            // 取消订阅
            messageBus.unsubscribe(agentId);
            
            // 停止主循环
            if (mainLoop != null) {
                mainLoop.cancel(true);
            }
            
            System.out.println(String.format("Agent %s (%s) 已停止", name, role));
        }
        
        return CompletableFuture.runAsync(() -> {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        });
    }
    
    /**
     * 主运行循环
     */
    private void runMainLoop() {
        while (running.get()) {
            try {
                if (state == AgentState.IDLE && !taskQueue.isEmpty()) {
                    // 执行队列中的任务
                    AgentTask task = taskQueue.poll();
                    if (task != null) {
                        executeTask(task);
                    }
                } else {
                    // 执行周期性工作
                    performPeriodicWork();
                }
                
                // 短暂休眠以避免过度占用CPU
                Thread.sleep(100);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println(String.format("Agent %s 主循环异常: %s", name, e.getMessage()));
                state = AgentState.ERROR;
                metrics.recordError();
                
                try {
                    Thread.sleep(1000); // 错误后等待1秒
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }
    
    /**
     * 处理接收到的消息
     */
    private void handleMessage(AgentMessage message) {
        try {
            metrics.recordMessageReceived();
            
            // 记录到内存
            Map<String, Object> memoryItem = new HashMap<>();
            memoryItem.put("type", "message_received");
            memoryItem.put("message", message.toMap());
            memoryItem.put("timestamp", LocalDateTime.now().toString());
            remember(memoryItem);
            
            // 根据消息类型处理
            switch (message.getMessageType()) {
                case TASK:
                    handleTaskMessage(message);
                    break;
                case TEXT:
                    handleTextMessage(message);
                    break;
                case SYSTEM:
                    handleSystemMessage(message);
                    break;
                case BROADCAST:
                    handleBroadcastMessage(message);
                    break;
                default:
                    System.out.println(String.format("Agent %s 收到未知类型消息: %s", name, message.getMessageType()));
            }
            
        } catch (Exception e) {
            System.err.println(String.format("Agent %s 处理消息失败: %s", name, e.getMessage()));
            metrics.recordError();
        }
    }
    
    /**
     * 处理任务消息
     */
    private void handleTaskMessage(AgentMessage message) {
        try {
            // 解析任务内容
            if (message.getContent() instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> taskData = (Map<String, Object>) message.getContent();
                
                AgentTask task = new AgentTask();
                task.setTitle((String) taskData.get("title"));
                task.setDescription((String) taskData.get("description"));
                task.setCreatedBy(message.getSenderId());
                task.setAssignedTo(agentId);
                
                // 添加到任务队列
                taskQueue.offer(task);
                metrics.recordTaskAssigned();
                
                System.out.println(String.format("Agent %s 收到新任务: %s", name, task.getTitle()));
            }
        } catch (Exception e) {
            System.err.println(String.format("Agent %s 处理任务消息失败: %s", name, e.getMessage()));
        }
    }
    
    /**
     * 处理文本消息
     */
    private void handleTextMessage(AgentMessage message) {
        CompletableFuture.runAsync(() -> {
            try {
                state = AgentState.COMMUNICATING;
                
                // 构建对话上下文
                List<Map<String, String>> context = buildConversationContext(message.getSenderId());
                
                // 生成回复
                String response = generateResponse(message.getContent().toString(), context);
                
                // 发送回复
                sendMessage(message.getSenderId(), response, MessageType.TEXT);
                
                state = AgentState.IDLE;
                
            } catch (Exception e) {
                System.err.println(String.format("Agent %s 生成回复失败: %s", name, e.getMessage()));
                state = AgentState.ERROR;
            }
        }, executorService);
    }
    
    /**
     * 处理系统消息
     */
    private void handleSystemMessage(AgentMessage message) {
        if (message.getContent() instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> systemData = (Map<String, Object>) message.getContent();
            
            String command = (String) systemData.get("command");
            if ("status".equals(command)) {
                // 返回状态信息
                Map<String, Object> status = getStatus();
                sendMessage(message.getSenderId(), status, MessageType.RESULT);
            }
        }
    }
    
    /**
     * 处理广播消息
     */
    private void handleBroadcastMessage(AgentMessage message) {
        System.out.println(String.format("Agent %s 收到广播: %s", name, message.getContent()));
        
        // 记录到内存
        Map<String, Object> memoryItem = new HashMap<>();
        memoryItem.put("type", "broadcast_received");
        memoryItem.put("content", message.getContent());
        memoryItem.put("sender", message.getSenderId());
        memoryItem.put("timestamp", LocalDateTime.now().toString());
        remember(memoryItem);
    }
    
    /**
     * 执行任务
     */
    private void executeTask(AgentTask task) {
        state = AgentState.BUSY;
        currentTask = task;
        task.setStatus(TaskStatus.IN_PROGRESS);
        
        long startTime = System.currentTimeMillis();
        
        try {
            // 执行具体任务（由子类实现）
            Object result = performTask(task);
            
            // 更新任务状态
            task.setStatus(TaskStatus.COMPLETED);
            task.setResult(result);
            completedTasks.add(task);
            
            long executionTime = System.currentTimeMillis() - startTime;
            metrics.recordTaskCompleted(executionTime);
            
            System.out.println(String.format("Agent %s 完成任务: %s (耗时: %dms)", name, task.getTitle(), executionTime));
            
            // 通知任务创建者
            if (task.getCreatedBy() != null && !task.getCreatedBy().equals(agentId)) {
                Map<String, Object> resultData = new HashMap<>();
                resultData.put("taskId", task.getId());
                resultData.put("status", "completed");
                resultData.put("result", result);
                resultData.put("executionTime", executionTime);
                sendMessage(task.getCreatedBy(), resultData, MessageType.RESULT);
            }
            
        } catch (Exception e) {
            task.setStatus(TaskStatus.FAILED);
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", e.getMessage());
            task.setResult(errorResult);
            metrics.recordTaskFailed();
            
            System.err.println(String.format("Agent %s 任务执行失败: %s - %s", name, task.getTitle(), e.getMessage()));
            
        } finally {
            currentTask = null;
            state = AgentState.IDLE;
        }
    }
    
    /**
     * 发送消息
     */
    public void sendMessage(String receiverId, Object content, MessageType messageType) {
        sendMessage(receiverId, content, messageType, 1);
    }
    
    public void sendMessage(String receiverId, Object content, MessageType messageType, int priority) {
        AgentMessage message = new AgentMessage(agentId, receiverId, messageType, content, priority);
        messageBus.publishAsync(message);
        metrics.recordMessageSent();
        
        // 记录到内存
        Map<String, Object> memoryItem = new HashMap<>();
        memoryItem.put("type", "message_sent");
        memoryItem.put("message", message.toMap());
        memoryItem.put("timestamp", LocalDateTime.now().toString());
        remember(memoryItem);
    }
    
    /**
     * 构建对话上下文
     */
    private List<Map<String, String>> buildConversationContext(String otherAgentId) {
        List<AgentMessage> messages = messageBus.getConversationHistory(agentId, otherAgentId, 5);
        
        List<Map<String, String>> context = new ArrayList<>();
        for (AgentMessage msg : messages) {
            String role = msg.getSenderId().equals(agentId) ? "assistant" : "user";
            Map<String, String> contextItem = new HashMap<>();
            contextItem.put("role", role);
            contextItem.put("content", msg.getContent().toString());
            context.add(contextItem);
        }
        
        return context;
    }
    
    /**
     * 生成回复
     */
    private String generateResponse(String inputText, List<Map<String, String>> context) {
        state = AgentState.THINKING;
        
        // 添加系统提示
        String systemPrompt = llm.generateSystemPrompt(getAgentType(), name, role);
        
        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> systemMessage = new HashMap<>();
        systemMessage.put("role", "system");
        systemMessage.put("content", systemPrompt);
        messages.add(systemMessage);
        
        messages.addAll(context);
        
        Map<String, String> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", inputText);
        messages.add(userMessage);
        
        try {
            return llm.chatCompletion(messages, getAgentType());
        } catch (Exception e) {
            return "抱歉，我遇到了一些问题：" + e.getMessage();
        }
    }
    
    /**
     * 记录到内存
     */
    protected void remember(Map<String, Object> memoryItem) {
        memory.add(memoryItem);
        if (memory.size() > maxMemory) {
            memory.remove(0);
        }
    }
    
    /**
     * 获取Agent状态
     */
    public Map<String, Object> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("agentId", agentId);
        status.put("name", name);
        status.put("role", role);
        status.put("state", state.getValue());
        status.put("currentTask", currentTask != null ? currentTask.getId() : null);
        status.put("taskQueueLength", taskQueue.size());
        status.put("completedTasks", completedTasks.size());
        status.put("capabilities", new ArrayList<>(capabilities));
        status.put("metrics", metrics.toString());
        return status;
    }
    
    // 抽象方法 - 子类必须实现
    
    /**
     * 初始化Agent能力
     */
    protected abstract void initializeCapabilities();
    
    /**
     * 执行具体任务
     */
    protected abstract Object performTask(AgentTask task) throws Exception;
    
    /**
     * 获取Agent类型（用于LLM生成回复）
     */
    protected abstract String getAgentType();
    
    /**
     * 执行周期性工作
     */
    protected void performPeriodicWork() throws InterruptedException {
        // 默认实现：简单等待
        Thread.sleep(1000);
    }
    
    // Getter 方法
    public String getAgentId() { return agentId; }
    public String getName() { return name; }
    public String getRole() { return role; }
    public AgentState getState() { return state; }
    public List<String> getCapabilities() { return new ArrayList<>(capabilities); }
    public AgentMetrics getMetrics() { return metrics; }
    public boolean isRunning() { return running.get(); }
    public int getTaskQueueSize() { return taskQueue.size(); }
    public int getCompletedTasksCount() { return completedTasks.size(); }
}