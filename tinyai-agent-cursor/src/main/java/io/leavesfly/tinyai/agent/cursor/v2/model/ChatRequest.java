package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.*;

/**
 * 聊天请求数据结构
 * 封装向LLM发送的聊天请求
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ChatRequest {
    
    /**
     * 模型名称（如：deepseek-chat, qwen-max等）
     */
    private String model;
    
    /**
     * 消息列表
     */
    private List<Message> messages;
    
    /**
     * 可用工具列表
     */
    private List<ToolDefinition> tools;
    
    /**
     * 工具选择策略（auto, none, required）
     */
    private String toolChoice;
    
    /**
     * 采样温度（0-2之间，默认1.0）
     */
    private Double temperature;
    
    /**
     * 核采样参数（0-1之间）
     */
    private Double topP;
    
    /**
     * 最大生成token数
     */
    private Integer maxTokens;
    
    /**
     * 是否启用流式响应
     */
    private Boolean stream;
    
    /**
     * 停止词列表
     */
    private List<String> stop;
    
    /**
     * 扩展参数
     */
    private Map<String, Object> extraParams;
    
    public ChatRequest() {
        this.messages = new ArrayList<>();
        this.stream = false;
    }
    
    public ChatRequest(String model, List<Message> messages) {
        this.model = model;
        this.messages = messages != null ? messages : new ArrayList<>();
        this.stream = false;
    }
    
    /**
     * 添加消息
     */
    public ChatRequest addMessage(Message message) {
        this.messages.add(message);
        return this;
    }
    
    /**
     * 添加用户消息
     */
    public ChatRequest addUserMessage(String content) {
        this.messages.add(Message.user(content));
        return this;
    }
    
    /**
     * 添加系统消息
     */
    public ChatRequest addSystemMessage(String content) {
        this.messages.add(Message.system(content));
        return this;
    }
    
    /**
     * 添加助手消息
     */
    public ChatRequest addAssistantMessage(String content) {
        this.messages.add(Message.assistant(content));
        return this;
    }
    
    /**
     * 添加工具定义
     */
    public ChatRequest addTool(ToolDefinition tool) {
        if (this.tools == null) {
            this.tools = new ArrayList<>();
        }
        this.tools.add(tool);
        return this;
    }
    
    /**
     * 设置扩展参数
     */
    public ChatRequest setExtraParam(String key, Object value) {
        if (this.extraParams == null) {
            this.extraParams = new HashMap<>();
        }
        this.extraParams.put(key, value);
        return this;
    }
    
    // Builder模式
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private final ChatRequest request;
        
        public Builder() {
            this.request = new ChatRequest();
        }
        
        public Builder model(String model) {
            request.model = model;
            return this;
        }
        
        public Builder messages(List<Message> messages) {
            request.messages = messages;
            return this;
        }
        
        public Builder addMessage(Message message) {
            request.addMessage(message);
            return this;
        }
        
        public Builder addUserMessage(String content) {
            request.addUserMessage(content);
            return this;
        }
        
        public Builder addSystemMessage(String content) {
            request.addSystemMessage(content);
            return this;
        }
        
        public Builder tools(List<ToolDefinition> tools) {
            request.tools = tools;
            return this;
        }
        
        public Builder addTool(ToolDefinition tool) {
            request.addTool(tool);
            return this;
        }
        
        public Builder toolChoice(String toolChoice) {
            request.toolChoice = toolChoice;
            return this;
        }
        
        public Builder temperature(Double temperature) {
            request.temperature = temperature;
            return this;
        }
        
        public Builder topP(Double topP) {
            request.topP = topP;
            return this;
        }
        
        public Builder maxTokens(Integer maxTokens) {
            request.maxTokens = maxTokens;
            return this;
        }
        
        public Builder stream(Boolean stream) {
            request.stream = stream;
            return this;
        }
        
        public Builder stop(List<String> stop) {
            request.stop = stop;
            return this;
        }
        
        public Builder extraParam(String key, Object value) {
            request.setExtraParam(key, value);
            return this;
        }
        
        public ChatRequest build() {
            return request;
        }
    }
    
    // Getters and Setters
    
    public String getModel() {
        return model;
    }
    
    public void setModel(String model) {
        this.model = model;
    }
    
    public List<Message> getMessages() {
        return messages;
    }
    
    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }
    
    public List<ToolDefinition> getTools() {
        return tools;
    }
    
    public void setTools(List<ToolDefinition> tools) {
        this.tools = tools;
    }
    
    public String getToolChoice() {
        return toolChoice;
    }
    
    public void setToolChoice(String toolChoice) {
        this.toolChoice = toolChoice;
    }
    
    public Double getTemperature() {
        return temperature;
    }
    
    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }
    
    public Double getTopP() {
        return topP;
    }
    
    public void setTopP(Double topP) {
        this.topP = topP;
    }
    
    public Integer getMaxTokens() {
        return maxTokens;
    }
    
    public void setMaxTokens(Integer maxTokens) {
        this.maxTokens = maxTokens;
    }
    
    public Boolean getStream() {
        return stream;
    }
    
    public void setStream(Boolean stream) {
        this.stream = stream;
    }
    
    public List<String> getStop() {
        return stop;
    }
    
    public void setStop(List<String> stop) {
        this.stop = stop;
    }
    
    public Map<String, Object> getExtraParams() {
        return extraParams;
    }
    
    public void setExtraParams(Map<String, Object> extraParams) {
        this.extraParams = extraParams;
    }
}
