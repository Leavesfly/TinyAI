package io.leavesfly.tinyai.agent.mcp;

import java.util.*;

/**
 * 数据分析 MCP Server 示例
 * 
 * @author 山泽
 * @since 2025-10-16
 */
public class DataAnalysisMCPServer extends MCPServer {
    
    public DataAnalysisMCPServer() {
        super("Data Analysis Server", "1.0.0");
        setupResources();
        setupTools();
        setupPrompts();
    }
    
    /**
     * 设置数据资源
     */
    private void setupResources() {
        // 模拟用户数据
        Map<String, Object> userData = new HashMap<>();
        userData.put("database", "users");
        
        List<Map<String, Object>> userRecords = new ArrayList<>();
        userRecords.add(createUserRecord(1, "Alice", 25, "北京"));
        userRecords.add(createUserRecord(2, "Bob", 30, "上海"));
        userRecords.add(createUserRecord(3, "Charlie", 28, "深圳"));
        userData.put("records", userRecords);
        
        Resource userResource = new Resource("db://users", "用户数据", ResourceType.DATABASE);
        userResource.setDescription("用户信息数据库");
        userResource.setMimeType("application/json");
        registerResource(userResource);
        setResourceContent("db://users", userData);
        
        // 模拟销售数据
        Map<String, Object> salesData = new HashMap<>();
        salesData.put("database", "sales");
        
        List<Map<String, Object>> salesRecords = new ArrayList<>();
        salesRecords.add(createSalesRecord("笔记本", 5000, "2024-01-15"));
        salesRecords.add(createSalesRecord("手机", 3000, "2024-01-16"));
        salesRecords.add(createSalesRecord("平板", 2000, "2024-01-17"));
        salesData.put("records", salesRecords);
        
        Resource salesResource = new Resource("db://sales", "销售数据", ResourceType.DATABASE);
        salesResource.setDescription("销售记录数据库");
        salesResource.setMimeType("application/json");
        registerResource(salesResource);
        setResourceContent("db://sales", salesData);
    }
    
    /**
     * 设置数据分析工具
     */
    private void setupTools() {
        // 统计计算工具
        registerTool(new Tool(
            "calculate_statistics",
            "计算数据的统计信息（总和、平均值、最大值、最小值）",
            ToolCategory.COMPUTATION,
            MCPUtils.createJsonSchema(
                new HashMap<String, Map<String, Object>>() {{
                    put("data_uri", MCPUtils.createProperty("string", "数据源URI"));
                    put("field", MCPUtils.createProperty("string", "要统计的字段名"));
                }},
                Arrays.asList("data_uri", "field")
            ),
            args -> calculateStatistics((String) args.get("data_uri"), (String) args.get("field"))
        ));
        
        // 数据查询工具
        registerTool(new Tool(
            "query_data",
            "查询和过滤数据",
            ToolCategory.DATA_ACCESS,
            MCPUtils.createJsonSchema(
                new HashMap<String, Map<String, Object>>() {{
                    put("data_uri", MCPUtils.createProperty("string", "数据源URI"));
                    put("filter_field", MCPUtils.createProperty("string", "过滤字段"));
                    put("filter_value", MCPUtils.createProperty("string", "过滤值"));
                }},
                Arrays.asList("data_uri")
            ),
            args -> queryData(
                (String) args.get("data_uri"),
                (String) args.get("filter_field"),
                args.get("filter_value")
            )
        ));
    }
    
    /**
     * 设置提示词模板
     */
    private void setupPrompts() {
        String template = "# 数据分析报告\n\n" +
                         "## 数据源\n{data_source}\n\n" +
                         "## 统计结果\n{statistics}\n\n" +
                         "## 分析结论\n" +
                         "请基于以上数据提供：\n" +
                         "1. 数据分布特征\n" +
                         "2. 异常值识别\n" +
                         "3. 趋势分析\n" +
                         "4. 业务建议";
        
        List<Map<String, Object>> arguments = new ArrayList<>();
        Map<String, Object> arg1 = new HashMap<>();
        arg1.put("name", "data_source");
        arg1.put("type", "string");
        arg1.put("required", true);
        arguments.add(arg1);
        
        Map<String, Object> arg2 = new HashMap<>();
        arg2.put("name", "statistics");
        arg2.put("type", "string");
        arg2.put("required", true);
        arguments.add(arg2);
        
        registerPrompt(new Prompt(
            "data_analysis_report",
            "数据分析报告模板",
            template,
            arguments
        ));
    }
    
    /**
     * 计算统计信息
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> calculateStatistics(String dataUri, String field) {
        ResourceContent resourceContent = getResource(dataUri);
        if (resourceContent == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "数据源不存在");
            return error;
        }
        
        Map<String, Object> data = (Map<String, Object>) resourceContent.getContent();
        List<Map<String, Object>> records = (List<Map<String, Object>>) data.get("records");
        
        List<Number> values = new ArrayList<>();
        for (Map<String, Object> record : records) {
            Object value = record.get(field);
            if (value instanceof Number) {
                values.add((Number) value);
            }
        }
        
        if (values.isEmpty()) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "字段 " + field + " 不包含数值数据");
            return error;
        }
        
        double sum = 0;
        double min = values.get(0).doubleValue();
        double max = values.get(0).doubleValue();
        
        for (Number value : values) {
            double v = value.doubleValue();
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        
        double average = sum / values.size();
        
        Map<String, Object> result = new HashMap<>();
        result.put("field", field);
        result.put("count", values.size());
        result.put("sum", sum);
        result.put("average", average);
        result.put("min", min);
        result.put("max", max);
        return result;
    }
    
    /**
     * 查询数据
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> queryData(String dataUri, String filterField, Object filterValue) {
        ResourceContent resourceContent = getResource(dataUri);
        if (resourceContent == null) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "数据源不存在");
            return error;
        }
        
        Map<String, Object> data = (Map<String, Object>) resourceContent.getContent();
        List<Map<String, Object>> records = (List<Map<String, Object>>) data.get("records");
        
        List<Map<String, Object>> filtered = new ArrayList<>();
        
        if (filterField != null && filterValue != null) {
            for (Map<String, Object> record : records) {
                Object value = record.get(filterField);
                if (value != null && value.equals(filterValue)) {
                    filtered.add(record);
                }
            }
        } else {
            filtered = records;
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("total", records.size());
        result.put("filtered", filtered.size());
        result.put("results", filtered);
        return result;
    }
    
    // 辅助方法
    private Map<String, Object> createUserRecord(int id, String name, int age, String city) {
        Map<String, Object> record = new HashMap<>();
        record.put("id", id);
        record.put("name", name);
        record.put("age", age);
        record.put("city", city);
        return record;
    }
    
    private Map<String, Object> createSalesRecord(String product, int amount, String date) {
        Map<String, Object> record = new HashMap<>();
        record.put("product", product);
        record.put("amount", amount);
        record.put("date", date);
        return record;
    }
}
