package io.leavesfly.tinyai.minimind.training.agent;

import org.json.JSONObject;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Agent 工具系统 - 工具定义、解析与模拟执行
 * <p>
 * 对标 Python minimind3 train_agent.py 的工具系统，包含：
 * 1. 6 种工具的定义和参数 Schema
 * 2. 模拟数据（天气、时间、汇率、翻译等）
 * 3. 工具调用解析（从文本中提取 &lt;tool_call&gt; 标签）
 * 4. 工具参数校验
 * 5. 工具模拟执行
 *
 * @author TinyAI Team
 * @since 2025
 */
public class AgentTool {

    /**
     * 工具定义
     */
    public static class ToolDefinition {
        private final String name;
        private final String description;
        private final List<String> requiredParams;

        public ToolDefinition(String name, String description, List<String> requiredParams) {
            this.name = name;
            this.description = description;
            this.requiredParams = requiredParams;
        }

        public String getName() { return name; }
        public String getDescription() { return description; }
        public List<String> getRequiredParams() { return requiredParams; }

        @Override
        public String toString() {
            return String.format("Tool{%s: %s}", name, description);
        }
    }

    // ========== 工具注册表 ==========

    /** 所有可用工具 */
    private static final List<ToolDefinition> ALL_TOOLS = new ArrayList<>();

    /** 工具名 -> 工具定义 */
    private static final Map<String, ToolDefinition> TOOL_MAP = new HashMap<>();

    static {
        // 对标 Python TOOLS 列表 (L67-L93)
        register("calculate_math", "计算数学表达式", Arrays.asList("expression"));
        register("unit_converter", "单位换算", Arrays.asList("value", "from_unit", "to_unit"));
        register("get_current_weather", "获取天气", Arrays.asList("location"));
        register("get_current_time", "获取时间", Collections.emptyList());
        register("get_exchange_rate", "查询汇率", Arrays.asList("from_currency", "to_currency"));
        register("translate_text", "翻译文本", Arrays.asList("text", "target_language"));
    }

    private static void register(String name, String description, List<String> requiredParams) {
        ToolDefinition tool = new ToolDefinition(name, description, requiredParams);
        ALL_TOOLS.add(tool);
        TOOL_MAP.put(name, tool);
    }

    // ========== 模拟数据（对标 Python L96-L112） ==========

    /** 天气数据 */
    private static final Map<String, String[]> WEATHER_DATA = new HashMap<>();

    /** 时间数据 */
    private static final Map<String, String> TIME_DATA = new HashMap<>();

    /** 汇率数据 */
    private static final Map<String, Double> EXCHANGE_DATA = new HashMap<>();

    /** 翻译数据 */
    private static final Map<String, String> TRANSLATE_DATA = new HashMap<>();

    /** 单位换算倍率 */
    private static final Map<String, Double> UNIT_DATA = new HashMap<>();

    static {
        // 天气: location -> [temperature, condition]
        WEATHER_DATA.put("北京", new String[]{"28°C", "晴"});
        WEATHER_DATA.put("上海", new String[]{"15°C", "多云"});
        WEATHER_DATA.put("广州", new String[]{"32°C", "闷热"});
        WEATHER_DATA.put("深圳", new String[]{"30°C", "晴"});
        WEATHER_DATA.put("杭州", new String[]{"22°C", "阴"});
        WEATHER_DATA.put("成都", new String[]{"18°C", "小雨"});
        WEATHER_DATA.put("武汉", new String[]{"25°C", "多云"});
        WEATHER_DATA.put("南京", new String[]{"20°C", "晴"});
        WEATHER_DATA.put("Tokyo", new String[]{"12°C", "晴"});
        WEATHER_DATA.put("New York", new String[]{"8°C", "多云"});
        WEATHER_DATA.put("London", new String[]{"5°C", "小雨"});

        // 时间
        TIME_DATA.put("Asia/Shanghai", "2025-01-15 14:30:00 CST");
        TIME_DATA.put("America/New_York", "2025-01-15 01:30:00 EST");
        TIME_DATA.put("Europe/London", "2025-01-15 06:30:00 GMT");
        TIME_DATA.put("Asia/Tokyo", "2025-01-15 15:30:00 JST");

        // 汇率: "FROM_TO" -> rate
        EXCHANGE_DATA.put("USD_CNY", 7.235);
        EXCHANGE_DATA.put("EUR_CNY", 7.892);
        EXCHANGE_DATA.put("GBP_CNY", 9.156);
        EXCHANGE_DATA.put("JPY_CNY", 0.0482);
        EXCHANGE_DATA.put("CNY_USD", 0.1382);
        EXCHANGE_DATA.put("USD_EUR", 0.917);
        EXCHANGE_DATA.put("USD_JPY", 150.12);

        // 翻译
        TRANSLATE_DATA.put("hello_中文", "你好");
        TRANSLATE_DATA.put("thank you_中文", "谢谢");
        TRANSLATE_DATA.put("你好_English", "Hello");
        TRANSLATE_DATA.put("谢谢_English", "Thank you");
        TRANSLATE_DATA.put("good morning_中文", "早上好");

        // 单位换算: "from_to" -> 倍率
        UNIT_DATA.put("km_m", 1000.0);
        UNIT_DATA.put("m_km", 0.001);
        UNIT_DATA.put("kg_g", 1000.0);
        UNIT_DATA.put("g_kg", 0.001);
        UNIT_DATA.put("mile_km", 1.60934);
        UNIT_DATA.put("km_mile", 0.621371);
        UNIT_DATA.put("inch_cm", 2.54);
        UNIT_DATA.put("cm_inch", 0.393701);
        UNIT_DATA.put("lb_kg", 0.453592);
        UNIT_DATA.put("kg_lb", 2.20462);
    }

    // ========== 工具调用解析（对标 Python parse_tool_calls L149-L156） ==========

    /** 匹配 <tool_call>...</tool_call> 标签 */
    private static final Pattern TOOL_CALL_PATTERN =
            Pattern.compile("<tool_call>(.*?)</tool_call>", Pattern.DOTALL);

    /**
     * 从文本中解析工具调用
     * <p>
     * 对标 Python parse_tool_calls：使用正则提取 &lt;tool_call&gt;...&lt;/tool_call&gt; 中的 JSON
     *
     * @param text 模型生成的文本
     * @return 解析出的工具调用列表，每个元素包含 name 和 arguments
     */
    public static List<Map<String, Object>> parseToolCalls(String text) {
        List<Map<String, Object>> calls = new ArrayList<>();
        Matcher matcher = TOOL_CALL_PATTERN.matcher(text);

        while (matcher.find()) {
            String jsonStr = matcher.group(1).trim();
            try {
                JSONObject json = new JSONObject(jsonStr);
                Map<String, Object> call = new HashMap<>();
                call.put("name", json.optString("name", ""));

                // arguments 可能是 JSON 对象或字符串
                Object argsObj = json.opt("arguments");
                Map<String, Object> args = new HashMap<>();
                if (argsObj instanceof JSONObject) {
                    JSONObject argsJson = (JSONObject) argsObj;
                    for (String key : argsJson.keySet()) {
                        args.put(key, argsJson.get(key));
                    }
                } else if (argsObj instanceof String) {
                    try {
                        JSONObject argsJson = new JSONObject((String) argsObj);
                        for (String key : argsJson.keySet()) {
                            args.put(key, argsJson.get(key));
                        }
                    } catch (Exception ignored) {
                    }
                }
                call.put("arguments", args);
                calls.add(call);
            } catch (Exception ignored) {
                // JSON 解析失败，跳过此调用
            }
        }
        return calls;
    }

    // ========== 参数校验（对标 Python CHECK_ARGS L138-L145） ==========

    /**
     * 校验工具调用参数是否有效
     *
     * @param name 工具名
     * @param args 参数映射
     * @return 参数是否有效
     */
    @SuppressWarnings("unchecked")
    public static boolean validateArgs(String name, Map<String, Object> args) {
        if (!TOOL_MAP.containsKey(name) || args == null) return false;

        switch (name) {
            case "calculate_math":
                return args.containsKey("expression") && args.get("expression") instanceof String;
            case "unit_converter":
                return args.containsKey("value") && args.containsKey("from_unit") && args.containsKey("to_unit");
            case "get_current_weather":
                return args.containsKey("location") && args.get("location") instanceof String;
            case "get_current_time":
                return true; // timezone 是可选参数
            case "get_exchange_rate":
                return args.containsKey("from_currency") && args.containsKey("to_currency");
            case "translate_text":
                return args.containsKey("text") && args.containsKey("target_language");
            default:
                return false;
        }
    }

    // ========== 工具执行（对标 Python execute_tool L159-L174） ==========

    /**
     * 执行工具（模拟执行，返回预设结果）
     *
     * @param name 工具名
     * @param args 参数
     * @return 执行结果的 JSON 字符串，失败返回 null
     */
    @SuppressWarnings("unchecked")
    public static String executeTool(String name, Map<String, Object> args) {
        try {
            switch (name) {
                case "calculate_math":
                    return executeMath(args);
                case "unit_converter":
                    return executeUnitConverter(args);
                case "get_current_weather":
                    return executeWeather(args);
                case "get_current_time":
                    return executeTime(args);
                case "get_exchange_rate":
                    return executeExchangeRate(args);
                case "translate_text":
                    return executeTranslate(args);
                default:
                    return "{\"error\": \"tool not found\"}";
            }
        } catch (Exception e) {
            return "{\"error\": \"" + e.getMessage() + "\"}";
        }
    }

    private static String executeMath(Map<String, Object> args) {
        String expression = String.valueOf(args.get("expression"));
        // 简单数学表达式计算（仅支持基本运算）
        try {
            double result = evaluateSimpleExpression(expression);
            return new JSONObject().put("result", result).toString();
        } catch (Exception e) {
            return new JSONObject().put("error", "无法计算表达式: " + expression).toString();
        }
    }

    private static String executeUnitConverter(Map<String, Object> args) {
        double value = Double.parseDouble(String.valueOf(args.get("value")));
        String fromUnit = String.valueOf(args.get("from_unit")).toLowerCase();
        String toUnit = String.valueOf(args.get("to_unit")).toLowerCase();
        String key = fromUnit + "_" + toUnit;
        Double rate = UNIT_DATA.get(key);
        if (rate != null) {
            double result = value * rate;
            return new JSONObject()
                    .put("result", Math.round(result * 100.0) / 100.0)
                    .put("from", value + " " + fromUnit)
                    .put("to", result + " " + toUnit).toString();
        }
        return new JSONObject().put("error", "不支持的单位换算: " + fromUnit + " -> " + toUnit).toString();
    }

    private static String executeWeather(Map<String, Object> args) {
        String location = String.valueOf(args.get("location"));
        String[] data = WEATHER_DATA.get(location);
        if (data != null) {
            return new JSONObject()
                    .put("location", location)
                    .put("temperature", data[0])
                    .put("condition", data[1]).toString();
        }
        // 未知城市返回默认值
        return new JSONObject()
                .put("location", location)
                .put("temperature", "20°C")
                .put("condition", "晴").toString();
    }

    private static String executeTime(Map<String, Object> args) {
        String timezone = args.containsKey("timezone")
                ? String.valueOf(args.get("timezone")) : "Asia/Shanghai";
        String time = TIME_DATA.getOrDefault(timezone, "2025-01-15 12:00:00 UTC");
        return new JSONObject()
                .put("timezone", timezone)
                .put("datetime", time).toString();
    }

    private static String executeExchangeRate(Map<String, Object> args) {
        String from = String.valueOf(args.get("from_currency")).toUpperCase();
        String to = String.valueOf(args.get("to_currency")).toUpperCase();
        String key = from + "_" + to;
        Double rate = EXCHANGE_DATA.get(key);
        if (rate != null) {
            return new JSONObject()
                    .put("from", from)
                    .put("to", to)
                    .put("rate", rate).toString();
        }
        return new JSONObject().put("error", "不支持的汇率查询: " + from + " -> " + to).toString();
    }

    private static String executeTranslate(Map<String, Object> args) {
        String text = String.valueOf(args.get("text")).toLowerCase();
        String targetLang = String.valueOf(args.get("target_language"));
        String key = text + "_" + targetLang;
        String translation = TRANSLATE_DATA.get(key);
        if (translation != null) {
            return new JSONObject()
                    .put("original", text)
                    .put("translated", translation)
                    .put("target_language", targetLang).toString();
        }
        // 未知翻译返回原文
        return new JSONObject()
                .put("original", text)
                .put("translated", "[翻译: " + text + "]")
                .put("target_language", targetLang).toString();
    }

    /**
     * 简单数学表达式求值（支持 +, -, *, /）
     */
    private static double evaluateSimpleExpression(String expr) {
        expr = expr.replaceAll("\\s+", "");
        // 简单实现：仅支持两个操作数的基本运算
        for (int i = expr.length() - 1; i >= 1; i--) {
            char c = expr.charAt(i);
            if ((c == '+' || c == '-') && i > 0 && expr.charAt(i - 1) != '*' && expr.charAt(i - 1) != '/') {
                double left = evaluateSimpleExpression(expr.substring(0, i));
                double right = evaluateSimpleExpression(expr.substring(i + 1));
                return c == '+' ? left + right : left - right;
            }
        }
        for (int i = expr.length() - 1; i >= 1; i--) {
            char c = expr.charAt(i);
            if (c == '*' || c == '/') {
                double left = evaluateSimpleExpression(expr.substring(0, i));
                double right = evaluateSimpleExpression(expr.substring(i + 1));
                return c == '*' ? left * right : left / right;
            }
        }
        return Double.parseDouble(expr);
    }

    // ========== 工具列表访问 ==========

    /**
     * 获取所有可用工具
     */
    public static List<ToolDefinition> getAvailableTools() {
        return Collections.unmodifiableList(ALL_TOOLS);
    }

    /**
     * 根据名称获取工具定义
     */
    public static ToolDefinition getTool(String name) {
        return TOOL_MAP.get(name);
    }

    /**
     * 获取有效工具名称集合
     */
    public static Set<String> getValidToolNames() {
        return Collections.unmodifiableSet(TOOL_MAP.keySet());
    }

    /**
     * 根据工具名列表获取对应的工具定义列表
     */
    public static List<ToolDefinition> getToolsByNames(List<String> names) {
        List<ToolDefinition> tools = new ArrayList<>();
        for (String name : names) {
            ToolDefinition tool = TOOL_MAP.get(name);
            if (tool != null) {
                tools.add(tool);
            }
        }
        return tools;
    }
}
