package io.leavesfly.tinyai.agent.cursor.v2.model;

import java.util.HashMap;
import java.util.Map;

/**
 * 模型信息数据结构
 * 描述可用的大语言模型信息
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ModelInfo {
    
    /**
     * 模型名称
     */
    private String name;
    
    /**
     * 模型显示名称
     */
    private String displayName;
    
    /**
     * 模型提供商（如：deepseek, qwen, openai等）
     */
    private String provider;
    
    /**
     * 模型描述
     */
    private String description;
    
    /**
     * 是否支持工具调用
     */
    private boolean supportsToolCalling;
    
    /**
     * 是否支持流式响应
     */
    private boolean supportsStreaming;
    
    /**
     * 最大上下文长度（tokens）
     */
    private int maxContextLength;
    
    /**
     * 最大输出长度（tokens）
     */
    private int maxOutputLength;
    
    /**
     * 模型类型（chat, completion, embedding等）
     */
    private String type;
    
    /**
     * 是否可用
     */
    private boolean available;
    
    /**
     * 扩展属性
     */
    private Map<String, Object> properties;
    
    public ModelInfo() {
        this.properties = new HashMap<>();
        this.available = true;
    }
    
    public ModelInfo(String name, String provider) {
        this();
        this.name = name;
        this.provider = provider;
    }
    
    /**
     * 设置属性
     */
    public void setProperty(String key, Object value) {
        this.properties.put(key, value);
    }
    
    /**
     * 获取属性
     */
    public Object getProperty(String key) {
        return this.properties.get(key);
    }
    
    // Getters and Setters
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getDisplayName() {
        return displayName;
    }
    
    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }
    
    public String getProvider() {
        return provider;
    }
    
    public void setProvider(String provider) {
        this.provider = provider;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public boolean isSupportsToolCalling() {
        return supportsToolCalling;
    }
    
    public void setSupportsToolCalling(boolean supportsToolCalling) {
        this.supportsToolCalling = supportsToolCalling;
    }
    
    public boolean isSupportsStreaming() {
        return supportsStreaming;
    }
    
    public void setSupportsStreaming(boolean supportsStreaming) {
        this.supportsStreaming = supportsStreaming;
    }
    
    public int getMaxContextLength() {
        return maxContextLength;
    }
    
    public void setMaxContextLength(int maxContextLength) {
        this.maxContextLength = maxContextLength;
    }
    
    public int getMaxOutputLength() {
        return maxOutputLength;
    }
    
    public void setMaxOutputLength(int maxOutputLength) {
        this.maxOutputLength = maxOutputLength;
    }
    
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
    }
    
    public boolean isAvailable() {
        return available;
    }
    
    public void setAvailable(boolean available) {
        this.available = available;
    }
    
    public Map<String, Object> getProperties() {
        return properties;
    }
    
    public void setProperties(Map<String, Object> properties) {
        this.properties = properties;
    }
    
    @Override
    public String toString() {
        return "ModelInfo{" +
                "name='" + name + '\'' +
                ", displayName='" + displayName + '\'' +
                ", provider='" + provider + '\'' +
                ", type='" + type + '\'' +
                ", available=" + available +
                ", supportsToolCalling=" + supportsToolCalling +
                ", supportsStreaming=" + supportsStreaming +
                ", maxContextLength=" + maxContextLength +
                ", maxOutputLength=" + maxOutputLength +
                '}';
    }
}
