package io.leavesfly.tinyai.agent.cursor;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 代码生成器 - 基于上下文和需求生成代码
 * 支持生成Java函数、类、测试代码等
 * 
 * @author 山泽
 */
public class CodeGenerator {
    
    private final Map<String, String> templates;
    private final Random random;
    
    public CodeGenerator() {
        this.templates = initializeTemplates();
        this.random = new Random();
    }
    
    /**
     * 初始化代码模板
     */
    private Map<String, String> initializeTemplates() {
        Map<String, String> templates = new HashMap<>();
        
        // Java方法模板
        templates.put("java_method", 
            "/**\n" +
            " * {description}\n" +
            " *\n" +
            "{param_docs}" +
            " * @return {return_doc}\n" +
            " */\n" +
            "{modifier} {return_type} {name}({parameters}) {\n" +
            "{body}" +
            "    return {return_value};\n" +
            "}");
        
        // Java类模板
        templates.put("java_class",
            "/**\n" +
            " * {description}\n" +
            " *\n" +
            " * @author 山泽\n" +
            " */\n" +
            "{modifier} class {name}{inheritance} {\n" +
            "\n" +
            "{fields}" +
            "\n" +
            "    /**\n" +
            "     * 构造函数\n" +
            "     */\n" +
            "    public {name}({constructor_params}) {\n" +
            "{constructor_body}" +
            "    }\n" +
            "\n" +
            "{methods}" +
            "}");
        
        // Java测试方法模板
        templates.put("java_test",
            "@Test\n" +
            "public void test{name}() {\n" +
            "    // 准备测试数据\n" +
            "{test_setup}" +
            "\n" +
            "    // 执行测试\n" +
            "    {result_type} result = {method_call};\n" +
            "\n" +
            "    // 验证结果\n" +
            "{assertions}" +
            "}");
        
        // Java接口模板
        templates.put("java_interface",
            "/**\n" +
            " * {description}\n" +
            " *\n" +
            " * @author 山泽\n" +
            " */\n" +
            "public interface {name} {\n" +
            "\n" +
            "{methods}" +
            "}");
        
        return templates;
    }
    
    /**
     * 生成Java方法
     * @param name 方法名
     * @param description 方法描述
     * @param parameters 参数列表
     * @param returnType 返回类型
     * @param modifier 访问修饰符
     * @return 生成的方法代码
     */
    public String generateJavaMethod(String name, String description, List<String> parameters, 
                                   String returnType, String modifier) {
        if (name == null || name.isEmpty()) {
            name = "newMethod";
        }
        if (description == null || description.isEmpty()) {
            description = "生成的方法";
        }
        if (returnType == null || returnType.isEmpty()) {
            returnType = "void";
        }
        if (modifier == null || modifier.isEmpty()) {
            modifier = "public";
        }
        
        // 构建参数文档
        StringBuilder paramDocs = new StringBuilder();
        StringBuilder paramList = new StringBuilder();
        
        if (parameters != null && !parameters.isEmpty()) {
            for (int i = 0; i < parameters.size(); i++) {
                String param = parameters.get(i);
                String[] parts = param.split("\\s+");
                String paramName = parts.length > 1 ? parts[1] : "param" + (i + 1);
                String paramType = parts.length > 0 ? parts[0] : "Object";
                
                paramDocs.append(" * @param ").append(paramName).append(" 参数描述\n");
                if (i > 0) paramList.append(", ");
                paramList.append(paramType).append(" ").append(paramName);
            }
        }
        
        // 生成方法体
        String body = generateMethodBody(returnType, name);
        String returnValue = generateReturnValue(returnType);
        String returnDoc = returnType.equals("void") ? "无返回值" : "返回结果";
        
        return templates.get("java_method")
                .replace("{description}", description)
                .replace("{param_docs}", paramDocs.toString())
                .replace("{return_doc}", returnDoc)
                .replace("{modifier}", modifier)
                .replace("{return_type}", returnType)
                .replace("{name}", name)
                .replace("{parameters}", paramList.toString())
                .replace("{body}", body)
                .replace("{return_value}", returnValue);
    }
    
    /**
     * 生成Java类
     * @param name 类名
     * @param description 类描述
     * @param modifier 访问修饰符
     * @param inheritance 继承或实现关系
     * @param fields 字段列表
     * @param methods 方法列表
     * @return 生成的类代码
     */
    public String generateJavaClass(String name, String description, String modifier, 
                                  String inheritance, List<String> fields, List<String> methods) {
        if (name == null || name.isEmpty()) {
            name = "NewClass";
        }
        if (description == null || description.isEmpty()) {
            description = "生成的类";
        }
        if (modifier == null || modifier.isEmpty()) {
            modifier = "public";
        }
        
        // 构建继承关系
        String inheritanceClause = "";
        if (inheritance != null && !inheritance.isEmpty()) {
            if (inheritance.startsWith("extends") || inheritance.startsWith("implements")) {
                inheritanceClause = " " + inheritance;
            } else {
                inheritanceClause = " extends " + inheritance;
            }
        }
        
        // 构建字段
        StringBuilder fieldsCode = new StringBuilder();
        if (fields != null && !fields.isEmpty()) {
            for (String field : fields) {
                fieldsCode.append("    private ").append(field).append(";\n");
            }
        }
        
        // 构建构造函数参数和体
        String constructorParams = "";
        StringBuilder constructorBody = new StringBuilder();
        if (fields != null && !fields.isEmpty()) {
            List<String> paramList = new ArrayList<>();
            for (String field : fields) {
                String[] parts = field.split("\\s+");
                if (parts.length >= 2) {
                    String type = parts[0];
                    String fieldName = parts[1];
                    paramList.add(type + " " + fieldName);
                    constructorBody.append("        this.").append(fieldName).append(" = ").append(fieldName).append(";\n");
                }
            }
            constructorParams = String.join(", ", paramList);
        }
        
        // 构建方法
        StringBuilder methodsCode = new StringBuilder();
        if (methods != null && !methods.isEmpty()) {
            for (String method : methods) {
                methodsCode.append("\n    ").append(generateSimpleMethod(method)).append("\n");
            }
        }
        
        return templates.get("java_class")
                .replace("{description}", description)
                .replace("{modifier}", modifier)
                .replace("{name}", name)
                .replace("{inheritance}", inheritanceClause)
                .replace("{fields}", fieldsCode.toString())
                .replace("{constructor_params}", constructorParams)
                .replace("{constructor_body}", constructorBody.toString())
                .replace("{methods}", methodsCode.toString());
    }
    
    /**
     * 生成Java测试方法
     * @param methodName 被测试的方法名
     * @param className 被测试的类名
     * @param returnType 返回类型
     * @return 生成的测试代码
     */
    public String generateJavaTest(String methodName, String className, String returnType) {
        if (methodName == null || methodName.isEmpty()) {
            methodName = "method";
        }
        if (className == null || className.isEmpty()) {
            className = "TestClass";
        }
        if (returnType == null || returnType.isEmpty()) {
            returnType = "Object";
        }
        
        String testName = capitalize(methodName);
        String testSetup = generateTestSetup(className);
        String methodCall = generateMethodCall(className, methodName);
        String assertions = generateAssertions(returnType);
        
        return templates.get("java_test")
                .replace("{name}", testName)
                .replace("{test_setup}", testSetup)
                .replace("{result_type}", returnType)
                .replace("{method_call}", methodCall)
                .replace("{assertions}", assertions);
    }
    
    /**
     * 生成Java接口
     * @param name 接口名
     * @param description 接口描述
     * @param methods 方法签名列表
     * @return 生成的接口代码
     */
    public String generateJavaInterface(String name, String description, List<String> methods) {
        if (name == null || name.isEmpty()) {
            name = "NewInterface";
        }
        if (description == null || description.isEmpty()) {
            description = "生成的接口";
        }
        
        StringBuilder methodsCode = new StringBuilder();
        if (methods != null && !methods.isEmpty()) {
            for (String method : methods) {
                methodsCode.append("    /**\n")
                          .append("     * ").append(method).append("方法\n")
                          .append("     */\n")
                          .append("    ").append(method).append(";\n\n");
            }
        }
        
        return templates.get("java_interface")
                .replace("{description}", description)
                .replace("{name}", name)
                .replace("{methods}", methodsCode.toString());
    }
    
    /**
     * 根据请求字符串生成代码
     * @param request 生成请求
     * @return 生成的代码
     */
    public String generateFromRequest(String request) {
        if (request == null || request.trim().isEmpty()) {
            return "// 请提供有效的代码生成请求";
        }
        
        // 保存原始请求
        String originalRequest = request;
        String lowerRequest = request.toLowerCase();
        
        // 解析请求类型（按优先级顺序）
        if (lowerRequest.contains("test")) {
            // 检查是否是测试代码生成请求
            if (lowerRequest.contains("test method") || lowerRequest.contains("unit test") || 
                lowerRequest.contains("test case") || lowerRequest.matches(".*test\\s+\\w.*")) {
                return generateTestFromRequest(originalRequest);
            }
        }
        
        if (lowerRequest.contains("method") || lowerRequest.contains("function")) {
            return generateMethodFromRequest(originalRequest);
        } else if (lowerRequest.contains("class")) {
            return generateClassFromRequest(originalRequest);
        } else if (lowerRequest.contains("interface")) {
            return generateInterfaceFromRequest(originalRequest);
        } else {
            // 默认生成方法
            return generateMethodFromRequest(originalRequest);
        }
    }
    
    /**
     * 从请求生成方法
     */
    private String generateMethodFromRequest(String request) {
        // 提取方法名
        String methodName = extractMethodName(request);
        
        // 提取参数
        List<String> parameters = extractParameters(request);
        
        // 推断返回类型
        String returnType = inferReturnType(request);
        
        // 生成描述
        String description = "根据请求生成的方法: " + request;
        
        return generateJavaMethod(methodName, description, parameters, returnType, "public");
    }
    
    /**
     * 从请求生成类
     */
    private String generateClassFromRequest(String request) {
        // 提取类名
        String className = extractClassName(request);
        
        // 生成字段
        List<String> fields = generateFieldsFromRequest(request);
        
        // 生成方法
        List<String> methods = generateMethodsFromRequest(request);
        
        // 生成描述
        String description = "根据请求生成的类: " + request;
        
        return generateJavaClass(className, description, "public", null, fields, methods);
    }
    
    /**
     * 从请求生成测试
     */
    private String generateTestFromRequest(String request) {
        String methodName = extractMethodName(request);
        if (methodName.equals("newMethod")) {
            methodName = "testMethod"; // 为测试提供更合适的默认名称
        }
        String className = extractClassName(request);
        if (className.equals("NewClass")) {
            className = "TestClass"; // 为测试提供更合适的默认类名
        }
        String returnType = inferReturnType(request);
        
        return generateJavaTest(methodName, className, returnType);
    }
    
    /**
     * 从请求生成接口
     */
    private String generateInterfaceFromRequest(String request) {
        String interfaceName = extractClassName(request);
        List<String> methods = Arrays.asList("void doSomething()", "String getName()", "boolean isValid()");
        String description = "根据请求生成的接口: " + request;
        
        return generateJavaInterface(interfaceName, description, methods);
    }
    
    /**
     * 提取方法名
     */
    private String extractMethodName(String request) {
        // 查找常见模式
        Pattern pattern = Pattern.compile("(method|function)\\s+(\\w+)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(request);
        
        if (matcher.find()) {
            // 保持原始的大小写
            return matcher.group(2);
        }
        
        // 查找动词
        String[] verbs = {"calculate", "get", "set", "process", "handle", "validate", "check", "create", "update", "delete"};
        for (String verb : verbs) {
            if (request.toLowerCase().contains(verb)) {
                return verb + "Data";
            }
        }
        
        return "newMethod";
    }
    
    /**
     * 提取类名
     */
    private String extractClassName(String request) {
        // 查找class关键字后的单词
        Pattern pattern = Pattern.compile("class\\s+(\\w+)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(request);
        
        if (matcher.find()) {
            // 保持原始的大小写
            return matcher.group(1);
        }
        
        // 查找名词
        String[] nouns = {"manager", "service", "controller", "processor", "handler", "validator", "creator", "updater", "deleter"};
        for (String noun : nouns) {
            if (request.toLowerCase().contains(noun)) {
                return capitalize(noun.replace("er", "")) + "Manager";
            }
        }
        
        return "NewClass";
    }
    
    /**
     * 提取参数
     */
    private List<String> extractParameters(String request) {
        List<String> parameters = new ArrayList<>();
        
        String lowerRequest = request.toLowerCase();
        // 简单的参数推断
        if (lowerRequest.contains("string") || lowerRequest.contains("text")) {
            parameters.add("String input");
        }
        if (lowerRequest.contains("number") || lowerRequest.contains("int")) {
            parameters.add("int value");
        }
        if (lowerRequest.contains("list") || lowerRequest.contains("array")) {
            parameters.add("List<String> items");
        }
        
        // 如果没有找到参数，添加默认参数
        if (parameters.isEmpty()) {
            parameters.add("Object data");
        }
        
        return parameters;
    }
    
    /**
     * 推断返回类型
     */
    private String inferReturnType(String request) {
        String lowerRequest = request.toLowerCase();
        if (lowerRequest.contains("void") || lowerRequest.contains("nothing")) {
            return "void";
        }
        if (lowerRequest.contains("string") || lowerRequest.contains("text")) {
            return "String";
        }
        if (lowerRequest.contains("int") || lowerRequest.contains("number")) {
            return "int";
        }
        if (lowerRequest.contains("boolean") || lowerRequest.contains("true") || lowerRequest.contains("false")) {
            return "boolean";
        }
        if (lowerRequest.contains("list") || lowerRequest.contains("array")) {
            return "List<Object>";
        }
        
        return "Object";
    }
    
    /**
     * 生成方法体
     */
    private String generateMethodBody(String returnType, String methodName) {
        StringBuilder body = new StringBuilder();
        body.append("    // TODO: 实现").append(methodName).append("方法\n");
        
        if (!returnType.equals("void")) {
            body.append("    \n");
        }
        
        return body.toString();
    }
    
    /**
     * 生成返回值
     */
    private String generateReturnValue(String returnType) {
        switch (returnType.toLowerCase()) {
            case "void":
                return "";
            case "string":
                return "\"\"";
            case "int":
            case "integer":
                return "0";
            case "boolean":
                return "false";
            case "double":
            case "float":
                return "0.0";
            case "list":
            case "list<string>":
            case "list<object>":
                return "new ArrayList<>()";
            case "map":
            case "map<string, object>":
                return "new HashMap<>()";
            default:
                return "null";
        }
    }
    
    /**
     * 生成简单方法
     */
    private String generateSimpleMethod(String methodSignature) {
        return "/**\n" +
               "     * " + methodSignature + "方法\n" +
               "     */\n" +
               "    public " + methodSignature + " {\n" +
               "        // TODO: 实现方法逻辑\n" +
               "    }";
    }
    
    /**
     * 生成测试设置代码
     */
    private String generateTestSetup(String className) {
        return "    " + className + " instance = new " + className + "();";
    }
    
    /**
     * 生成方法调用代码
     */
    private String generateMethodCall(String className, String methodName) {
        return "instance." + methodName + "()";
    }
    
    /**
     * 生成断言代码
     */
    private String generateAssertions(String returnType) {
        switch (returnType.toLowerCase()) {
            case "void":
                return "    // 验证方法执行成功";
            case "string":
                return "    assertNotNull(result);\n    assertFalse(result.isEmpty());";
            case "boolean":
                return "    assertTrue(result);";
            case "int":
            case "integer":
                return "    assertTrue(result >= 0);";
            default:
                return "    assertNotNull(result);";
        }
    }
    
    /**
     * 从请求生成字段
     */
    private List<String> generateFieldsFromRequest(String request) {
        List<String> fields = new ArrayList<>();
        
        String lowerRequest = request.toLowerCase();
        if (lowerRequest.contains("name")) {
            fields.add("String name");
        }
        if (lowerRequest.contains("id")) {
            fields.add("Long id");
        }
        if (lowerRequest.contains("value")) {
            fields.add("Object value");
        }
        if (lowerRequest.contains("data")) {
            fields.add("Map<String, Object> data");
        }
        
        // 默认字段
        if (fields.isEmpty()) {
            fields.add("String name");
            fields.add("Object value");
        }
        
        return fields;
    }
    
    /**
     * 从请求生成方法
     */
    private List<String> generateMethodsFromRequest(String request) {
        List<String> methods = new ArrayList<>();
        
        methods.add("String getName()");
        methods.add("void setName(String name)");
        methods.add("Object getValue()");
        methods.add("void setValue(Object value)");
        methods.add("String toString()");
        
        return methods;
    }
    
    /**
     * 首字母大写
     */
    private String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return str.substring(0, 1).toUpperCase() + str.substring(1);
    }
    
    /**
     * 获取可用模板列表
     */
    public List<String> getAvailableTemplates() {
        return new ArrayList<>(templates.keySet());
    }
    
    /**
     * 添加自定义模板
     */
    public void addTemplate(String name, String template) {
        templates.put(name, template);
    }
    
    /**
     * 获取模板
     */
    public String getTemplate(String name) {
        return templates.get(name);
    }
}