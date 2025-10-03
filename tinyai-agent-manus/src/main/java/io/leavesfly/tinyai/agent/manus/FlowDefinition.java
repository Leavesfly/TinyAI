package io.leavesfly.tinyai.agent.manus;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Flow定义类
 * 
 * @author 山泽
 */
public class FlowDefinition {
    private String name;
    private String description;
    private Map<String, Object> nodes;
    private LocalDateTime createdAt;
    
    public FlowDefinition() {
        this.nodes = new HashMap<>();
        this.createdAt = LocalDateTime.now();
    }
    
    public FlowDefinition(String name, String description) {
        this();
        this.name = name;
        this.description = description;
    }
    
    // Getter和Setter方法
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public Map<String, Object> getNodes() {
        return nodes;
    }
    
    public void setNodes(Map<String, Object> nodes) {
        this.nodes = nodes != null ? nodes : new HashMap<>();
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    /**
     * 添加节点
     */
    public void addNode(String key, Object value) {
        this.nodes.put(key, value);
    }
    
    /**
     * 移除节点
     */
    public Object removeNode(String key) {
        return this.nodes.remove(key);
    }
    
    /**
     * 获取节点
     */
    public Object getNode(String key) {
        return this.nodes.get(key);
    }
    
    @Override
    public String toString() {
        return String.format("FlowDefinition{name='%s', description='%s', nodes=%d}", 
                           name, description, nodes.size());
    }
}