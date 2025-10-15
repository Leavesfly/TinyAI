package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.ArrayList;
import java.util.List;

/**
 * 聊天消息数据结构
 * 支持文本、工具调用等多种消息类型
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class Message {
    
    /**
     * 消息角色
     */
    private Role role;
    
    /**
     * 消息内容
     */
    private String content;
    
    /**
     * 工具调用列表（仅当role为assistant且需要调用工具时使用）
     */
    private List<ToolCall> toolCalls;
    
    /**
     * 工具调用ID（仅当role为tool时使用，关联到assistant的toolCall）
     */
    private String toolCallId;
    
    /**
     * 消息名称（可选，用于标识消息来源）
     */
    private String name;
    
    public Message() {
    }
    
    public Message(Role role, String content) {
        this.role = role;
        this.content = content;
    }
    
    /**
     * 创建用户消息
     */
    public static Message user(String content) {
        return new Message(Role.USER, content);
    }
    
    /**
     * 创建助手消息
     */
    public static Message assistant(String content) {
        return new Message(Role.ASSISTANT, content);
    }
    
    /**
     * 创建系统消息
     */
    public static Message system(String content) {
        return new Message(Role.SYSTEM, content);
    }
    
    /**
     * 创建工具消息
     */
    public static Message tool(String content, String toolCallId) {
        Message message = new Message(Role.TOOL, content);
        message.setToolCallId(toolCallId);
        return message;
    }
    
    /**
     * 创建带工具调用的助手消息
     */
    public static Message assistantWithTools(String content, List<ToolCall> toolCalls) {
        Message message = new Message(Role.ASSISTANT, content);
        message.setToolCalls(toolCalls);
        return message;
    }
    
    /**
     * 添加工具调用
     */
    public void addToolCall(ToolCall toolCall) {
        if (this.toolCalls == null) {
            this.toolCalls = new ArrayList<>();
        }
        this.toolCalls.add(toolCall);
    }
    
    // Getters and Setters
    
    public Role getRole() {
        return role;
    }
    
    public void setRole(Role role) {
        this.role = role;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }
    
    public void setToolCalls(List<ToolCall> toolCalls) {
        this.toolCalls = toolCalls;
    }
    
    public String getToolCallId() {
        return toolCallId;
    }
    
    public void setToolCallId(String toolCallId) {
        this.toolCallId = toolCallId;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    /**
     * 消息角色枚举
     */
    public enum Role {
        /** 系统消息 */
        SYSTEM("system"),
        /** 用户消息 */
        USER("user"),
        /** 助手消息 */
        ASSISTANT("assistant"),
        /** 工具消息 */
        TOOL("tool");
        
        private final String value;
        
        Role(String value) {
            this.value = value;
        }
        
        public String getValue() {
            return value;
        }
        
        public static Role fromValue(String value) {
            for (Role role : values()) {
                if (role.value.equals(value)) {
                    return role;
                }
            }
            throw new IllegalArgumentException("Unknown role: " + value);
        }
    }
    
    @Override
    public String toString() {
        return "Message{" +
                "role=" + role +
                ", content='" + content + '\'' +
                ", toolCalls=" + toolCalls +
                ", toolCallId='" + toolCallId + '\'' +
                ", name='" + name + '\'' +
                '}';
    }
}
