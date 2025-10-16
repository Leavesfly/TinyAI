package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;

/**
 * MCP 资源定义
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class Resource {
    /**
     * 资源唯一标识符，如 file:///path/to/file
     */
    private String uri;
    
    /**
     * 资源名称
     */
    private String name;
    
    /**
     * 资源类型
     */
    private ResourceType resourceType;
    
    /**
     * 资源描述
     */
    private String description;
    
    /**
     * MIME类型
     */
    private String mimeType;
    
    /**
     * 元数据
     */
    private Map<String, Object> metadata;
    
    public Resource(String uri, String name, ResourceType resourceType) {
        this.uri = uri;
        this.name = name;
        this.resourceType = resourceType;
        this.description = "";
        this.metadata = new HashMap<>();
    }
    
    public Resource(String uri, String name, ResourceType resourceType, String description) {
        this.uri = uri;
        this.name = name;
        this.resourceType = resourceType;
        this.description = description;
        this.metadata = new HashMap<>();
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("uri", uri);
        map.put("name", name);
        map.put("type", resourceType.getValue());
        map.put("description", description);
        map.put("mimeType", mimeType);
        map.put("metadata", metadata);
        return map;
    }
    
    // Getters and Setters
    public String getUri() {
        return uri;
    }
    
    public void setUri(String uri) {
        this.uri = uri;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public ResourceType getResourceType() {
        return resourceType;
    }
    
    public void setResourceType(ResourceType resourceType) {
        this.resourceType = resourceType;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public String getMimeType() {
        return mimeType;
    }
    
    public void setMimeType(String mimeType) {
        this.mimeType = mimeType;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
}
