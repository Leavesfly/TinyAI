package io.leavesfly.tinyai.agent.context;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.function.Function;

/**
 * 工具注册表
 * 管理Agent可用的工具函数
 *
 * @author 山泽
 */
public class ToolRegistry {

    private final Map<String, Tool> tools;  // 注册的工具

    // 构造函数
    public ToolRegistry() {
        this.tools = new HashMap<>();
    }

    /**
     * 注册工具
     *
     * @param name        工具名称
     * @param function    工具函数
     * @param description 工具描述
     * @param parameters  参数描述
     */
    public void register(String name, Function<Map<String, Object>, Object> function,
                         String description, Map<String, Object> parameters) {
        if (name == null || function == null) {
            throw new IllegalArgumentException("工具名称和函数不能为空");
        }

        Tool tool = new Tool(name, function, description, parameters);
        tools.put(name, tool);
    }

    /**
     * 注册工具（简化版本）
     */
    public void register(String name, Function<Map<String, Object>, Object> function, String description) {
        register(name, function, description, new HashMap<>());
    }

    /**
     * 获取工具
     */
    public Tool getTool(String name) {
        return tools.get(name);
    }

    /**
     * 列出所有工具
     */
    public List<ToolInfo> listTools() {
        List<ToolInfo> toolList = new ArrayList<>();

        for (Tool tool : tools.values()) {
            ToolInfo info = new ToolInfo(
                    tool.getName(),
                    tool.getDescription(),
                    tool.getParameters()
            );
            toolList.add(info);
        }

        return toolList;
    }

    /**
     * 调用工具
     *
     * @param name      工具名称
     * @param arguments 调用参数
     * @return 工具调用结果
     */
    public ToolCall callTool(String name, Map<String, Object> arguments) {
        String toolCallId = generateToolCallId(name);
        ToolCall toolCall = new ToolCall(toolCallId, name, arguments);

        Tool tool = tools.get(name);
        if (tool == null) {
            toolCall.setError("工具 '" + name + "' 不存在");
            return toolCall;
        }

        try {
            Object result = tool.getFunction().apply(arguments != null ? arguments : new HashMap<>());
            toolCall.setResult(result);
        } catch (Exception e) {
            toolCall.setError("工具执行异常: " + e.getMessage());
        }

        return toolCall;
    }

    /**
     * 检查工具是否存在
     */
    public boolean hasTool(String name) {
        return tools.containsKey(name);
    }

    /**
     * 移除工具
     */
    public boolean removeTool(String name) {
        return tools.remove(name) != null;
    }

    /**
     * 获取工具数量
     */
    public int getToolCount() {
        return tools.size();
    }

    /**
     * 清空所有工具
     */
    public void clear() {
        tools.clear();
    }

    /**
     * 生成工具调用ID
     */
    private String generateToolCallId(String toolName) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            String input = toolName + System.currentTimeMillis();
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
            return "tool_" + System.currentTimeMillis() + "_" + Math.abs(toolName.hashCode());
        }
    }

    /**
     * 工具类
     */
    public static class Tool {
        private final String name;
        private final Function<Map<String, Object>, Object> function;
        private final String description;
        private final Map<String, Object> parameters;
        private final LocalDateTime registeredAt;

        public Tool(String name, Function<Map<String, Object>, Object> function,
                    String description, Map<String, Object> parameters) {
            this.name = name;
            this.function = function;
            this.description = description;
            this.parameters = parameters != null ? parameters : new HashMap<>();
            this.registeredAt = LocalDateTime.now();
        }

        // Getter 方法
        public String getName() {
            return name;
        }

        public Function<Map<String, Object>, Object> getFunction() {
            return function;
        }

        public String getDescription() {
            return description;
        }

        public Map<String, Object> getParameters() {
            return parameters;
        }

        public LocalDateTime getRegisteredAt() {
            return registeredAt;
        }

        @Override
        public String toString() {
            return String.format("Tool{name='%s', description='%s'}", name, description);
        }
    }

    /**
     * 工具信息类（用于列出工具）
     */
    public static class ToolInfo {
        private final String name;
        private final String description;
        private final Map<String, Object> parameters;

        public ToolInfo(String name, String description, Map<String, Object> parameters) {
            this.name = name;
            this.description = description;
            this.parameters = parameters;
        }

        public String getName() {
            return name;
        }

        public String getDescription() {
            return description;
        }

        public Map<String, Object> getParameters() {
            return parameters;
        }

        @Override
        public String toString() {
            return String.format("ToolInfo{name='%s', description='%s'}", name, description);
        }
    }
}