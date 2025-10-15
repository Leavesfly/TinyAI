package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.List;
import java.util.Map;

/**
 * 聊天响应数据结构
 * 封装LLM返回的聊天响应
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ChatResponse {
    
    /**
     * 响应ID
     */
    private String id;
    
    /**
     * 对象类型（chat.completion）
     */
    private String object;
    
    /**
     * 创建时间戳
     */
    private Long created;
    
    /**
     * 使用的模型名称
     */
    private String model;
    
    /**
     * 生成的选择列表
     */
    private List<Choice> choices;
    
    /**
     * 使用量统计
     */
    private Usage usage;
    
    /**
     * 错误信息（如有）
     */
    private Error error;
    
    public ChatResponse() {
    }
    
    /**
     * 是否成功
     */
    public boolean isSuccess() {
        return error == null && choices != null && !choices.isEmpty();
    }
    
    /**
     * 获取第一个选择的消息
     */
    public Message getFirstMessage() {
        if (choices != null && !choices.isEmpty()) {
            return choices.get(0).getMessage();
        }
        return null;
    }
    
    /**
     * 获取第一个选择的文本内容
     */
    public String getContent() {
        Message message = getFirstMessage();
        return message != null ? message.getContent() : null;
    }
    
    /**
     * 获取第一个选择的工具调用
     */
    public List<ToolCall> getToolCalls() {
        Message message = getFirstMessage();
        return message != null ? message.getToolCalls() : null;
    }
    
    /**
     * 是否需要调用工具
     */
    public boolean hasToolCalls() {
        List<ToolCall> toolCalls = getToolCalls();
        return toolCalls != null && !toolCalls.isEmpty();
    }
    
    /**
     * 获取结束原因
     */
    public String getFinishReason() {
        if (choices != null && !choices.isEmpty()) {
            return choices.get(0).getFinishReason();
        }
        return null;
    }
    
    // Getters and Setters
    
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getObject() {
        return object;
    }
    
    public void setObject(String object) {
        this.object = object;
    }
    
    public Long getCreated() {
        return created;
    }
    
    public void setCreated(Long created) {
        this.created = created;
    }
    
    public String getModel() {
        return model;
    }
    
    public void setModel(String model) {
        this.model = model;
    }
    
    public List<Choice> getChoices() {
        return choices;
    }
    
    public void setChoices(List<Choice> choices) {
        this.choices = choices;
    }
    
    public Usage getUsage() {
        return usage;
    }
    
    public void setUsage(Usage usage) {
        this.usage = usage;
    }
    
    public Error getError() {
        return error;
    }
    
    public void setError(Error error) {
        this.error = error;
    }
    
    /**
     * 选择项
     */
    public static class Choice {
        private Integer index;
        private Message message;
        private String finishReason;
        private Message delta; // 用于流式响应
        
        public Choice() {
        }
        
        public Integer getIndex() {
            return index;
        }
        
        public void setIndex(Integer index) {
            this.index = index;
        }
        
        public Message getMessage() {
            return message;
        }
        
        public void setMessage(Message message) {
            this.message = message;
        }
        
        public String getFinishReason() {
            return finishReason;
        }
        
        public void setFinishReason(String finishReason) {
            this.finishReason = finishReason;
        }
        
        public Message getDelta() {
            return delta;
        }
        
        public void setDelta(Message delta) {
            this.delta = delta;
        }
    }
    
    /**
     * 使用量统计
     */
    public static class Usage {
        private Integer promptTokens;
        private Integer completionTokens;
        private Integer totalTokens;
        
        public Usage() {
        }
        
        public Integer getPromptTokens() {
            return promptTokens;
        }
        
        public void setPromptTokens(Integer promptTokens) {
            this.promptTokens = promptTokens;
        }
        
        public Integer getCompletionTokens() {
            return completionTokens;
        }
        
        public void setCompletionTokens(Integer completionTokens) {
            this.completionTokens = completionTokens;
        }
        
        public Integer getTotalTokens() {
            return totalTokens;
        }
        
        public void setTotalTokens(Integer totalTokens) {
            this.totalTokens = totalTokens;
        }
    }
    
    /**
     * 错误信息
     */
    public static class Error {
        private String code;
        private String message;
        private String type;
        private Map<String, Object> details;
        
        public Error() {
        }
        
        public Error(String code, String message) {
            this.code = code;
            this.message = message;
        }
        
        public String getCode() {
            return code;
        }
        
        public void setCode(String code) {
            this.code = code;
        }
        
        public String getMessage() {
            return message;
        }
        
        public void setMessage(String message) {
            this.message = message;
        }
        
        public String getType() {
            return type;
        }
        
        public void setType(String type) {
            this.type = type;
        }
        
        public Map<String, Object> getDetails() {
            return details;
        }
        
        public void setDetails(Map<String, Object> details) {
            this.details = details;
        }
    }
}
