package io.leavesfly.tinyai.agent.pattern;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * 示例工具类
 * 提供一些常用的工具函数供Agent使用
 * @author 山泽
 */
public class SampleTools {
    
    /**
     * 创建天气查询工具
     */
    public static Function<Map<String, Object>, Object> createWeatherTool() {
        return args -> {
            String city = (String) args.get("city");
            if (city == null) {
                return "天气信息: 城市名称为空";
            }
            
            Map<String, String> weatherData = new HashMap<>();
            weatherData.put("北京", "晴天，25°C");
            weatherData.put("上海", "多云，22°C");
            weatherData.put("广州", "雨天，28°C");
            weatherData.put("深圳", "晴天，30°C");
            weatherData.put("杭州", "阴天，24°C");
            
            return weatherData.getOrDefault(city, city + "天气信息暂不可用");
        };
    }
    
    /**
     * 创建新闻查询工具
     */
    public static Function<Map<String, Object>, Object> createNewsTool() {
        return args -> {
            String category = (String) args.get("category");
            if (category == null) {
                category = "综合";
            }
            
            Map<String, String> newsData = new HashMap<>();
            newsData.put("科技", "AI技术取得新突破，GPT-5即将发布");
            newsData.put("财经", "股市今日上涨2%，科技股表现优异");
            newsData.put("体育", "世界杯预选赛结果公布");
            newsData.put("娱乐", "新电影获得票房冠军");
            newsData.put("综合", "今日要闻综合报道");
            
            return newsData.getOrDefault(category, category + "新闻暂无更新");
        };
    }
    
    /**
     * 创建翻译工具
     */
    public static Function<Map<String, Object>, Object> createTranslateTool() {
        return args -> {
            String text = (String) args.get("text");
            String targetLang = (String) args.get("target_lang");
            
            if (text == null) {
                return "翻译错误: 文本为空";
            }
            
            if (targetLang == null) {
                targetLang = "英文";
            }
            
            // 简单的翻译示例
            Map<String, String> translations = new HashMap<>();
            translations.put("你好", "Hello");
            translations.put("谢谢", "Thank you");
            translations.put("再见", "Goodbye");
            
            String result = translations.get(text);
            if (result != null) {
                return "翻译结果(" + targetLang + "): " + result;
            }
            
            return "翻译结果(" + targetLang + "): [模拟翻译] " + text;
        };
    }
    
    /**
     * 创建时间工具
     */
    public static Function<Map<String, Object>, Object> createTimeTool() {
        return args -> {
            String format = (String) args.get("format");
            if (format == null) {
                format = "standard";
            }
            
            java.time.LocalDateTime now = java.time.LocalDateTime.now();
            
            switch (format) {
                case "date":
                    return "当前日期: " + now.toLocalDate();
                case "time":
                    return "当前时间: " + now.toLocalTime().toString().substring(0, 8);
                case "timestamp":
                    return "时间戳: " + System.currentTimeMillis();
                default:
                    return "当前时间: " + now.toString().substring(0, 19);
            }
        };
    }
    
    /**
     * 创建单位转换工具
     */
    public static Function<Map<String, Object>, Object> createUnitConverterTool() {
        return args -> {
            Double value = null;
            Object valueObj = args.get("value");
            if (valueObj instanceof Number) {
                value = ((Number) valueObj).doubleValue();
            } else if (valueObj instanceof String) {
                try {
                    value = Double.parseDouble((String) valueObj);
                } catch (NumberFormatException e) {
                    return "转换错误: 无效的数值";
                }
            }
            
            if (value == null) {
                return "转换错误: 数值为空";
            }
            
            String fromUnit = (String) args.get("from_unit");
            String toUnit = (String) args.get("to_unit");
            
            if (fromUnit == null || toUnit == null) {
                return "转换错误: 单位信息不完整";
            }
            
            // 简单的长度单位转换
            if ("m".equals(fromUnit) && "cm".equals(toUnit)) {
                return value + "米 = " + (value * 100) + "厘米";
            } else if ("cm".equals(fromUnit) && "m".equals(toUnit)) {
                return value + "厘米 = " + (value / 100) + "米";
            } else if ("kg".equals(fromUnit) && "g".equals(toUnit)) {
                return value + "千克 = " + (value * 1000) + "克";
            } else if ("g".equals(fromUnit) && "kg".equals(toUnit)) {
                return value + "克 = " + (value / 1000) + "千克";
            }
            
            return "转换结果: " + value + " " + fromUnit + " 转换为 " + toUnit + " (模拟转换)";
        };
    }
}