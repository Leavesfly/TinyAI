package io.leavesfly.tinyai.minimind.api;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;

/**
 * JSON工具类
 * 
 * 基于org.json库的JSON解析和生成工具,替代原有的手写解析器,
 * 提供更完整的JSON支持(嵌套对象/数组、Unicode转义、数值精度等)
 * 
 * @author leavesfly
 * @since 2024
 */
public class SimpleJSON {
    
    /**
     * 将Map/List/基本类型转换为JSON字符串
     */
    public static String toJSON(Object obj) {
        if (obj == null) {
            return "null";
        }
        if (obj instanceof Map) {
            return new JSONObject((Map<?, ?>) obj).toString();
        }
        if (obj instanceof List) {
            return new JSONArray((List<?>) obj).toString();
        }
        if (obj instanceof String) {
            return JSONObject.quote((String) obj);
        }
        if (obj instanceof Number || obj instanceof Boolean) {
            return obj.toString();
        }
        return JSONObject.quote(obj.toString());
    }
    
    /**
     * 解析JSON字符串为Map
     */
    public static Map<String, Object> parseJSON(String json) {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        JSONObject jsonObject = new JSONObject(json);
        return jsonObjectToMap(jsonObject);
    }
    
    /**
     * 将JSONObject递归转换为Map
     */
    private static Map<String, Object> jsonObjectToMap(JSONObject jsonObject) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (String key : jsonObject.keySet()) {
            Object value = jsonObject.get(key);
            result.put(key, convertValue(value));
        }
        return result;
    }
    
    /**
     * 将JSONArray递归转换为List
     */
    private static List<Object> jsonArrayToList(JSONArray jsonArray) {
        List<Object> result = new ArrayList<>(jsonArray.length());
        for (int i = 0; i < jsonArray.length(); i++) {
            result.add(convertValue(jsonArray.get(i)));
        }
        return result;
    }
    
    /**
     * 递归转换JSON值为Java标准类型
     */
    private static Object convertValue(Object value) {
        if (value == JSONObject.NULL) {
            return null;
        }
        if (value instanceof JSONObject) {
            return jsonObjectToMap((JSONObject) value);
        }
        if (value instanceof JSONArray) {
            return jsonArrayToList((JSONArray) value);
        }
        return value;
    }
}
