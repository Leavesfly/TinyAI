package io.leavesfly.tinyai.agent.pattern;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 步骤记录类
 * 记录Agent执行过程中的每个步骤
 * @author 山泽
 */
public class Step {
    /** 步骤类型：thought, action, observation, reflection, plan等 */
    private final String stepType;
    
    /** 步骤内容 */
    private final String content;
    
    /** 时间戳 */
    private final LocalDateTime timestamp;
    
    /** 元数据 */
    private final Map<String, Object> metadata;
    
    public Step(String stepType, String content) {
        this(stepType, content, new HashMap<>());
    }
    
    public Step(String stepType, String content, Map<String, Object> metadata) {
        this.stepType = stepType;
        this.content = content;
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>(metadata);
    }
    
    public String getStepType() {
        return stepType;
    }
    
    public String getContent() {
        return content;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata);
    }
    
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    @Override
    public String toString() {
        return String.format("%s: %s", stepType.toUpperCase(), content);
    }
}