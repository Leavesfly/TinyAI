package io.leavesfly.tinyai.agent.mcp;

import java.util.HashMap;
import java.util.Map;

/**
 * 资源内容
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class ResourceContent {
    /**
     * 资源URI
     */
    private String uri;
    
    /**
     * 资源内容
     */
    private Object content;
    
    /**
     * MIME类型
     */
    private String mimeType;
    
    public ResourceContent(String uri, Object content) {
        this.uri = uri;
        this.content = content;
    }
    
    public ResourceContent(String uri, Object content, String mimeType) {
        this.uri = uri;
        this.content = content;
        this.mimeType = mimeType;
    }
    
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("uri", uri);
        map.put("content", content);
        map.put("mimeType", mimeType);
        return map;
    }
    
    // Getters and Setters
    public String getUri() {
        return uri;
    }
    
    public void setUri(String uri) {
        this.uri = uri;
    }
    
    public Object getContent() {
        return content;
    }
    
    public void setContent(Object content) {
        this.content = content;
    }
    
    public String getMimeType() {
        return mimeType;
    }
    
    public void setMimeType(String mimeType) {
        this.mimeType = mimeType;
    }
}
