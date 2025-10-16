package io.leavesfly.tinyai.agent.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MCP 工具类
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class MCPUtils {
    
    /**
     * 创建 JSON Schema
     * 
     * @param properties 属性定义
     * @param required 必需字段列表
     * @return JSON Schema Map
     */
    public static Map<String, Object> createJsonSchema(Map<String, Map<String, Object>> properties, 
                                                       List<String> required) {
        Map<String, Object> schema = new HashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        if (required != null && !required.isEmpty()) {
            schema.put("required", required);
        }
        return schema;
    }
    
    /**
     * 创建属性定义
     */
    public static Map<String, Object> createProperty(String type, String description) {
        Map<String, Object> prop = new HashMap<>();
        prop.put("type", type);
        prop.put("description", description);
        return prop;
    }
    
    /**
     * 创建带默认值的属性定义
     */
    public static Map<String, Object> createProperty(String type, String description, Object defaultValue) {
        Map<String, Object> prop = new HashMap<>();
        prop.put("type", type);
        prop.put("description", description);
        prop.put("default", defaultValue);
        return prop;
    }
}
