package io.leavesfly.tinyai.agent.context;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * AdvancedAgent 的辅助类和方法
 * 包含工具调用规范、工具函数等
 * 
 * @author 山泽
 */
public class AgentHelper {
    
    // 全局笔记存储
    private static final Map<String, Map<String, Object>> noteStorage = new HashMap<>();
    private static int nextNoteId = 1;
    
    /**
     * 工具调用规范类
     */
    public static class ToolCallSpec {
        private String name;
        private Map<String, Object> arguments;
        
        public ToolCallSpec(String name, Map<String, Object> arguments) {
            this.name = name;
            this.arguments = arguments;
        }
        
        public String getName() {
            return name;
        }
        
        public Map<String, Object> getArguments() {
            return arguments;
        }
    }
    
    /**
     * 生成会话ID
     */
    public static String generateSessionId() {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            String input = String.valueOf(System.currentTimeMillis());
            byte[] hash = md.digest(input.getBytes());
            
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return hexString.toString().substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            return "session_" + System.currentTimeMillis();
        }
    }
    
    /**
     * 生成文档ID
     */
    public static String generateDocumentId(String content) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(content.getBytes());
            
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return "doc_" + hexString.toString().substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            return "doc_" + System.currentTimeMillis() + "_" + Math.abs(content.hashCode());
        }
    }
    
    /**
     * 解析工具调用
     */
    public static List<ToolCallSpec> parseToolCalls(String text) {
        List<ToolCallSpec> toolCalls = new ArrayList<>();
        
        // 寻找工具调用模式：[tool:toolName(arg1="value1", arg2="value2")]
        Pattern pattern = Pattern.compile("\\[tool:(\\w+)\\(([^\\]]*)\\)\\]");
        Matcher matcher = pattern.matcher(text);
        
        while (matcher.find()) {
            String toolName = matcher.group(1);
            String argsStr = matcher.group(2);
            
            Map<String, Object> args = new HashMap<>();
            if (argsStr != null && !argsStr.trim().isEmpty()) {
                // 解析参数
                args = parseArguments(argsStr);
            }
            
            toolCalls.add(new ToolCallSpec(toolName, args));
        }
        
        return toolCalls;
    }
    
    /**
     * 解析参数字符串
     */
    private static Map<String, Object> parseArguments(String argsStr) {
        Map<String, Object> args = new HashMap<>();
        
        // 简单的参数解析：key="value" 或 key=value
        Pattern argPattern = Pattern.compile("(\\w+)\\s*=\\s*[\"']?([^,\"']*)[\"']?");
        Matcher matcher = argPattern.matcher(argsStr);
        
        while (matcher.find()) {
            String key = matcher.group(1).trim();
            String value = matcher.group(2).trim();
            
            // 尝试转换数值类型
            Object parsedValue = parseValue(value);
            args.put(key, parsedValue);
        }
        
        return args;
    }
    
    /**
     * 解析值类型
     */
    private static Object parseValue(String value) {
        if (value == null || value.isEmpty()) {
            return value;
        }
        
        // 尝试解析为数字
        try {
            if (value.contains(".")) {
                return Double.parseDouble(value);
            } else {
                return Integer.parseInt(value);
            }
        } catch (NumberFormatException e) {
            // 保持字符串类型
            return value;
        }
    }
    
    /**
     * 计算器工具
     */
    public static Object calculatorTool(Map<String, Object> arguments) {
        try {
            String operation = (String) arguments.get("operation");
            Object aObj = arguments.get("a");
            Object bObj = arguments.get("b");
            
            if (operation == null || aObj == null || bObj == null) {
                Map<String, Object> error = new HashMap<>();
                error.put("error", "缺少必要参数");
                return error;
            }
            
            double a = convertToDouble(aObj);
            double b = convertToDouble(bObj);
            double result;
            
            switch (operation.toLowerCase()) {
                case "add":
                    result = a + b;
                    break;
                case "subtract":
                    result = a - b;
                    break;
                case "multiply":
                    result = a * b;
                    break;
                case "divide":
                    if (b == 0) {
                        Map<String, Object> error = new HashMap<>();
                        error.put("error", "除零错误");
                        return error;
                    }
                    result = a / b;
                    break;
                default:
                    Map<String, Object> error = new HashMap<>();
                    error.put("error", "不支持的操作: " + operation);
                    return error;
            }
            
            Map<String, Object> response = new HashMap<>();
            response.put("result", result);
            response.put("expression", String.format("%.2f %s %.2f = %.2f", a, operation, b, result));
            return response;
            
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", e.getMessage());
            return error;
        }
    }
    
    /**
     * 时间工具
     */
    public static Object timeTool(Map<String, Object> arguments) {
        LocalDateTime now = LocalDateTime.now();
        
        Map<String, Object> response = new HashMap<>();
        response.put("current_time", now.toString());
        response.put("formatted_time", now.format(DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss")));
        response.put("timestamp", System.currentTimeMillis());
        
        return response;
    }
    
    /**
     * 笔记工具
     */
    public static Object noteTool(Map<String, Object> arguments) {
        String action = (String) arguments.get("action");
        if (action == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "缺少action参数");
            return error;
        }
        
        switch (action.toLowerCase()) {
            case "create":
                return createNote(arguments);
            case "list":
                return listNotes();
            case "get":
                return getNote(arguments);
            case "delete":
                return deleteNote(arguments);
            default:
                Map<String, Object> error = new HashMap<>();
                error.put("error", "不支持的操作: " + action);
                return error;
        }
    }
    
    /**
     * 创建笔记
     */
    private static Object createNote(Map<String, Object> arguments) {
        String content = (String) arguments.get("content");
        if (content == null || content.trim().isEmpty()) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "笔记内容不能为空");
            return error;
        }
        
        String noteId = String.valueOf(nextNoteId++);
        Map<String, Object> note = new HashMap<>();
        note.put("id", noteId);
        note.put("content", content);
        note.put("created_at", LocalDateTime.now().toString());
        
        noteStorage.put(noteId, note);
        
        Map<String, Object> response = new HashMap<>();
        response.put("message", "已创建笔记 " + noteId);
        response.put("note", note);
        return response;
    }
    
    /**
     * 列出所有笔记
     */
    private static Object listNotes() {
        Map<String, Object> response = new HashMap<>();
        response.put("notes", new ArrayList<>(noteStorage.values()));
        return response;
    }
    
    /**
     * 获取指定笔记
     */
    private static Object getNote(Map<String, Object> arguments) {
        String noteId = (String) arguments.get("note_id");
        if (noteId == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "缺少note_id参数");
            return error;
        }
        
        Map<String, Object> note = noteStorage.get(noteId);
        if (note == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "笔记 " + noteId + " 不存在");
            return error;
        }
        
        Map<String, Object> response = new HashMap<>();
        response.put("note", note);
        return response;
    }
    
    /**
     * 删除笔记
     */
    private static Object deleteNote(Map<String, Object> arguments) {
        String noteId = (String) arguments.get("note_id");
        if (noteId == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "缺少note_id参数");
            return error;
        }
        
        Map<String, Object> deletedNote = noteStorage.remove(noteId);
        if (deletedNote == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "笔记 " + noteId + " 不存在");
            return error;
        }
        
        Map<String, Object> response = new HashMap<>();
        response.put("message", "已删除笔记 " + noteId);
        response.put("deleted_note", deletedNote);
        return response;
    }
    
    /**
     * 转换为double类型
     */
    private static double convertToDouble(Object obj) {
        if (obj instanceof Number) {
            return ((Number) obj).doubleValue();
        } else if (obj instanceof String) {
            return Double.parseDouble((String) obj);
        } else {
            throw new IllegalArgumentException("无法转换为数字: " + obj);
        }
    }
}