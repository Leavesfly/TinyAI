package io.leavesfly.tinyai.agent.cursor;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 重构代理 - 智能分析和建议代码重构
 * 基于代码分析结果，提供具体的重构建议和代码示例
 * 
 * @author 山泽
 */
public class RefactorAgent {
    
    private final CodeAnalyzer analyzer;
    private final Map<String, RefactorPattern> refactorPatterns;
    
    /**
     * 重构模式内部类
     */
    public static class RefactorPattern {
        private final String description;
        private final int threshold;
        private final String strategy;
        private final String impactLevel;
        
        public RefactorPattern(String description, int threshold, String strategy, String impactLevel) {
            this.description = description;
            this.threshold = threshold;
            this.strategy = strategy;
            this.impactLevel = impactLevel;
        }
        
        // Getter 方法
        public String getDescription() { return description; }
        public int getThreshold() { return threshold; }
        public String getStrategy() { return strategy; }
        public String getImpactLevel() { return impactLevel; }
    }
    
    public RefactorAgent(CodeAnalyzer analyzer) {
        this.analyzer = analyzer;
        this.refactorPatterns = loadRefactorPatterns();
    }
    
    /**
     * 加载重构模式
     */
    private Map<String, RefactorPattern> loadRefactorPatterns() {
        Map<String, RefactorPattern> patterns = new HashMap<>();
        
        patterns.put("long_method", new RefactorPattern(
            "分解长方法", 50, "extract_method", "中等"));
        
        patterns.put("duplicate_code", new RefactorPattern(
            "消除重复代码", 3, "extract_common", "高"));
        
        patterns.put("large_class", new RefactorPattern(
            "分解大类", 500, "split_class", "高"));
        
        patterns.put("complex_condition", new RefactorPattern(
            "简化复杂条件", 10, "simplify_condition", "中等"));
        
        patterns.put("long_parameter_list", new RefactorPattern(
            "减少参数列表", 5, "parameter_object", "低"));
        
        patterns.put("deep_nesting", new RefactorPattern(
            "减少嵌套层次", 6, "early_return", "中等"));
        
        patterns.put("magic_numbers", new RefactorPattern(
            "消除魔法数字", 1, "extract_constant", "低"));
        
        patterns.put("switch_statements", new RefactorPattern(
            "替换switch语句", 1, "polymorphism", "高"));
        
        return patterns;
    }
    
    /**
     * 分析重构机会
     * @param code 待分析的代码
     * @return 重构建议列表
     */
    public List<RefactorSuggestion> analyzeRefactorOpportunities(String code) {
        if (code == null || code.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        // 获取代码分析结果
        Map<String, Object> analysis = analyzer.analyzeJavaCode(code);
        
        if (!(Boolean) analysis.getOrDefault("syntax_valid", false)) {
            // 如果语法无效，先修复语法问题
            suggestions.add(createSyntaxFixSuggestion());
            return suggestions;
        }
        
        // 检查各种重构模式
        suggestions.addAll(checkLongMethods(code, analysis));
        suggestions.addAll(checkDuplicateCode(code));
        suggestions.addAll(checkComplexConditions(code));
        suggestions.addAll(checkDeepNesting(code));
        suggestions.addAll(checkLongParameterLists(code));
        suggestions.addAll(checkMagicNumbers(code));
        suggestions.addAll(checkSwitchStatements(code));
        suggestions.addAll(checkLargeClass(code, analysis));
        suggestions.addAll(checkNamingConventions(code));
        
        // 按优先级排序
        suggestions.sort((a, b) -> Integer.compare(b.getPriority(), a.getPriority()));
        
        return suggestions;
    }
    
    /**
     * 创建语法修复建议
     */
    private RefactorSuggestion createSyntaxFixSuggestion() {
        return new RefactorSuggestion(
            "syntax_fix",
            "代码存在语法错误，需要先修复语法问题",
            "// 包含语法错误的代码",
            "// 修复后的代码",
            Arrays.asList("修复语法错误", "使代码可编译"),
            "critical"
        );
    }
    
    /**
     * 检查长方法
     */
    private List<RefactorSuggestion> checkLongMethods(String code, Map<String, Object> analysis) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> methods = (List<Map<String, Object>>) analysis.get("methods");
        
        if (methods != null) {
            for (Map<String, Object> method : methods) {
                String methodName = (String) method.get("name");
                Integer line = (Integer) method.get("line");
                
                if (isLongMethod(code, methodName, line)) {
                    suggestions.add(createLongMethodSuggestion(methodName));
                }
            }
        }
        
        return suggestions;
    }
    
    /**
     * 检查是否为长方法
     */
    private boolean isLongMethod(String code, String methodName, Integer startLine) {
        if (startLine == null) return false;
        
        String[] lines = code.split("\n");
        int methodLength = calculateMethodLength(lines, startLine - 1);
        
        return methodLength > refactorPatterns.get("long_method").getThreshold();
    }
    
    /**
     * 计算方法长度
     */
    private int calculateMethodLength(String[] lines, int startLine) {
        if (startLine < 0 || startLine >= lines.length) return 0;
        
        int length = 0;
        int braceCount = 0;
        boolean inMethod = false;
        
        for (int i = startLine; i < lines.length; i++) {
            String line = lines[i];
            
            for (char c : line.toCharArray()) {
                if (c == '{') {
                    braceCount++;
                    inMethod = true;
                } else if (c == '}') {
                    braceCount--;
                    if (braceCount == 0 && inMethod) {
                        return length;
                    }
                }
            }
            
            if (inMethod && !line.trim().isEmpty()) {
                length++;
            }
        }
        
        return length;
    }
    
    /**
     * 创建长方法重构建议
     */
    private RefactorSuggestion createLongMethodSuggestion(String methodName) {
        String originalCode = "public void " + methodName + "() {\n    // 长方法内容\n    // ...\n}";
        String refactoredCode = "public void " + methodName + "() {\n    helperMethod1();\n    helperMethod2();\n}\n\n" +
                               "private void helperMethod1() {\n    // 提取的逻辑1\n}\n\n" +
                               "private void helperMethod2() {\n    // 提取的逻辑2\n}";
        
        return new RefactorSuggestion(
            "extract_method",
            "方法 '" + methodName + "' 过长，建议分解为多个小方法",
            originalCode,
            refactoredCode,
            Arrays.asList("提高代码可读性", "便于单元测试", "降低维护成本", "提高代码复用性"),
            "中等"
        );
    }
    
    /**
     * 检查重复代码
     */
    private List<RefactorSuggestion> checkDuplicateCode(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        Map<String, List<Integer>> duplicateMap = findDuplicateLines(code);
        
        for (Map.Entry<String, List<Integer>> entry : duplicateMap.entrySet()) {
            if (entry.getValue().size() >= refactorPatterns.get("duplicate_code").getThreshold()) {
                suggestions.add(createDuplicateCodeSuggestion(entry.getKey(), entry.getValue()));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 查找重复行
     */
    private Map<String, List<Integer>> findDuplicateLines(String code) {
        Map<String, List<Integer>> lineMap = new HashMap<>();
        String[] lines = code.split("\n");
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (!line.isEmpty() && !line.startsWith("//") && line.length() > 10) {
                lineMap.computeIfAbsent(line, k -> new ArrayList<>()).add(i + 1);
            }
        }
        
        // 只返回重复的行
        Map<String, List<Integer>> duplicates = new HashMap<>();
        for (Map.Entry<String, List<Integer>> entry : lineMap.entrySet()) {
            if (entry.getValue().size() > 1) {
                duplicates.put(entry.getKey(), entry.getValue());
            }
        }
        
        return duplicates;
    }
    
    /**
     * 创建重复代码重构建议
     */
    private RefactorSuggestion createDuplicateCodeSuggestion(String duplicateLine, List<Integer> lineNumbers) {
        String originalCode = "// 重复代码出现在行: " + lineNumbers + "\n" + duplicateLine;
        String refactoredCode = "private void extractedMethod() {\n    " + duplicateLine + "\n}\n\n" +
                               "// 在原位置调用: extractedMethod();";
        
        return new RefactorSuggestion(
            "extract_common",
            "发现重复代码，建议提取为公共方法",
            originalCode,
            refactoredCode,
            Arrays.asList("消除代码重复", "提高维护性", "减少错误风险"),
            "高"
        );
    }
    
    /**
     * 检查复杂条件
     */
    private List<RefactorSuggestion> checkComplexConditions(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        Pattern complexConditionPattern = Pattern.compile("if\\s*\\([^)]{50,}\\)");
        Matcher matcher = complexConditionPattern.matcher(code);
        
        while (matcher.find()) {
            int lineNumber = getLineNumber(code, matcher.start());
            suggestions.add(createComplexConditionSuggestion(matcher.group(), lineNumber));
        }
        
        return suggestions;
    }
    
    /**
     * 创建复杂条件重构建议
     */
    private RefactorSuggestion createComplexConditionSuggestion(String condition, int lineNumber) {
        String originalCode = condition;
        String refactoredCode = "private boolean isValidCondition() {\n    return " + 
                               condition.substring(condition.indexOf('(') + 1, condition.lastIndexOf(')')) + 
                               ";\n}\n\nif (isValidCondition()) {";
        
        return new RefactorSuggestion(
            "simplify_condition",
            "第" + lineNumber + "行的条件表达式过于复杂，建议提取为方法",
            originalCode,
            refactoredCode,
            Arrays.asList("提高代码可读性", "便于调试", "提高可测试性"),
            "中等"
        );
    }
    
    /**
     * 检查深层嵌套
     */
    private List<RefactorSuggestion> checkDeepNesting(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        String[] lines = code.split("\n");
        
        for (int i = 0; i < lines.length; i++) {
            int nestingLevel = calculateNestingLevel(lines[i]);
            if (nestingLevel > refactorPatterns.get("deep_nesting").getThreshold()) {
                suggestions.add(createDeepNestingSuggestion(i + 1, nestingLevel));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 计算嵌套级别
     */
    private int calculateNestingLevel(String line) {
        int level = 0;
        for (char c : line.toCharArray()) {
            if (c == ' ') level++;
            else if (c == '\t') level += 4;
            else break;
        }
        return level / 4;
    }
    
    /**
     * 创建深层嵌套重构建议
     */
    private RefactorSuggestion createDeepNestingSuggestion(int lineNumber, int level) {
        String originalCode = "// 第" + lineNumber + "行存在" + level + "层嵌套";
        String refactoredCode = "// 使用早期返回减少嵌套\nif (!condition1) {\n    return;\n}\n\nif (!condition2) {\n    return;\n}\n\n// 主逻辑";
        
        return new RefactorSuggestion(
            "early_return",
            "第" + lineNumber + "行嵌套过深(" + level + "层)，建议使用早期返回模式",
            originalCode,
            refactoredCode,
            Arrays.asList("降低复杂度", "提高可读性", "减少认知负担"),
            "中等"
        );
    }
    
    /**
     * 检查长参数列表
     */
    private List<RefactorSuggestion> checkLongParameterLists(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        Pattern methodPattern = Pattern.compile("\\w+\\s+\\w+\\s*\\(([^)]+)\\)");
        Matcher matcher = methodPattern.matcher(code);
        
        while (matcher.find()) {
            String parameters = matcher.group(1);
            int paramCount = parameters.split(",").length;
            
            if (paramCount > refactorPatterns.get("long_parameter_list").getThreshold()) {
                int lineNumber = getLineNumber(code, matcher.start());
                suggestions.add(createLongParameterListSuggestion(paramCount, lineNumber));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 创建长参数列表重构建议
     */
    private RefactorSuggestion createLongParameterListSuggestion(int paramCount, int lineNumber) {
        String originalCode = "public void method(param1, param2, param3, param4, param5, param6)";
        String refactoredCode = "public class MethodParams {\n    // 参数字段\n}\n\npublic void method(MethodParams params)";
        
        return new RefactorSuggestion(
            "parameter_object",
            "第" + lineNumber + "行的方法参数过多(" + paramCount + "个)，建议使用参数对象",
            originalCode,
            refactoredCode,
            Arrays.asList("简化方法签名", "提高参数管理", "便于扩展"),
            "低"
        );
    }
    
    /**
     * 检查魔法数字
     */
    private List<RefactorSuggestion> checkMagicNumbers(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        Pattern magicNumberPattern = Pattern.compile("\\b\\d{2,}\\b");
        Matcher matcher = magicNumberPattern.matcher(code);
        
        Set<String> foundNumbers = new HashSet<>();
        while (matcher.find()) {
            String number = matcher.group();
            if (!foundNumbers.contains(number) && !isCommonNumber(number)) {
                foundNumbers.add(number);
                int lineNumber = getLineNumber(code, matcher.start());
                suggestions.add(createMagicNumberSuggestion(number, lineNumber));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 检查是否为常见数字
     */
    private boolean isCommonNumber(String number) {
        int num = Integer.parseInt(number);
        return num <= 10 || num == 100 || num == 1000;
    }
    
    /**
     * 创建魔法数字重构建议
     */
    private RefactorSuggestion createMagicNumberSuggestion(String number, int lineNumber) {
        String originalCode = "// 使用魔法数字: " + number;
        String refactoredCode = "private static final int MEANINGFUL_NAME = " + number + ";\n// 使用: MEANINGFUL_NAME";
        
        return new RefactorSuggestion(
            "extract_constant",
            "第" + lineNumber + "行发现魔法数字" + number + "，建议提取为常量",
            originalCode,
            refactoredCode,
            Arrays.asList("提高代码可读性", "便于维护", "避免硬编码"),
            "低"
        );
    }
    
    /**
     * 检查Switch语句
     */
    private List<RefactorSuggestion> checkSwitchStatements(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        Pattern switchPattern = Pattern.compile("switch\\s*\\([^)]+\\)\\s*\\{");
        Matcher matcher = switchPattern.matcher(code);
        
        while (matcher.find()) {
            int lineNumber = getLineNumber(code, matcher.start());
            int caseCount = countSwitchCases(code, matcher.end());
            
            if (caseCount > 5) {
                suggestions.add(createSwitchStatementSuggestion(lineNumber, caseCount));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 计算Switch语句中的case数量
     */
    private int countSwitchCases(String code, int switchStart) {
        int caseCount = 0;
        int braceCount = 1;
        int pos = switchStart;
        
        while (pos < code.length() && braceCount > 0) {
            char c = code.charAt(pos);
            if (c == '{') {
                braceCount++;
            } else if (c == '}') {
                braceCount--;
            } else if (code.substring(pos).startsWith("case ")) {
                caseCount++;
            }
            pos++;
        }
        
        return caseCount;
    }
    
    /**
     * 创建Switch语句重构建议
     */
    private RefactorSuggestion createSwitchStatementSuggestion(int lineNumber, int caseCount) {
        String originalCode = "switch (type) {\n    case TYPE1: ...\n    case TYPE2: ...\n    // " + caseCount + " cases\n}";
        String refactoredCode = "// 使用多态替换Switch\ninterface TypeHandler {\n    void handle();\n}\n\nMap<Type, TypeHandler> handlers = ...";
        
        return new RefactorSuggestion(
            "polymorphism",
            "第" + lineNumber + "行的Switch语句过于复杂(" + caseCount + "个case)，建议使用多态",
            originalCode,
            refactoredCode,
            Arrays.asList("消除Switch语句", "提高扩展性", "符合开闭原则"),
            "高"
        );
    }
    
    /**
     * 检查大类
     */
    private List<RefactorSuggestion> checkLargeClass(String code, Map<String, Object> analysis) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        @SuppressWarnings("unchecked")
        Map<String, Object> metrics = (Map<String, Object>) analysis.get("metrics");
        
        if (metrics != null) {
            Integer codeLines = (Integer) metrics.get("code_lines");
            if (codeLines != null && codeLines > refactorPatterns.get("large_class").getThreshold()) {
                suggestions.add(createLargeClassSuggestion(codeLines));
            }
        }
        
        return suggestions;
    }
    
    /**
     * 创建大类重构建议
     */
    private RefactorSuggestion createLargeClassSuggestion(int codeLines) {
        String originalCode = "public class LargeClass {\n    // " + codeLines + " 行代码\n    // 太多职责...\n}";
        String refactoredCode = "// 分解为多个类\npublic class ClassA {\n    // 职责A\n}\n\npublic class ClassB {\n    // 职责B\n}";
        
        return new RefactorSuggestion(
            "split_class",
            "类过大(" + codeLines + "行)，建议按职责分解为多个类",
            originalCode,
            refactoredCode,
            Arrays.asList("符合单一职责原则", "提高可维护性", "便于测试"),
            "高"
        );
    }
    
    /**
     * 检查命名规范
     */
    private List<RefactorSuggestion> checkNamingConventions(String code) {
        List<RefactorSuggestion> suggestions = new ArrayList<>();
        
        // 检查类名
        Pattern classPattern = Pattern.compile("class\\s+([a-z][a-zA-Z0-9_]*)");
        Matcher classMatcher = classPattern.matcher(code);
        while (classMatcher.find()) {
            int lineNumber = getLineNumber(code, classMatcher.start());
            suggestions.add(createNamingConventionSuggestion("类名", classMatcher.group(1), lineNumber, "PascalCase"));
        }
        
        // 检查变量名
        Pattern variablePattern = Pattern.compile("(int|String|boolean|double)\\s+([A-Z][a-zA-Z0-9_]*)");
        Matcher variableMatcher = variablePattern.matcher(code);
        while (variableMatcher.find()) {
            int lineNumber = getLineNumber(code, variableMatcher.start());
            suggestions.add(createNamingConventionSuggestion("变量名", variableMatcher.group(2), lineNumber, "camelCase"));
        }
        
        return suggestions;
    }
    
    /**
     * 创建命名规范重构建议
     */
    private RefactorSuggestion createNamingConventionSuggestion(String type, String name, int lineNumber, String convention) {
        String originalCode = type + ": " + name;
        String refactoredCode = type + ": " + (convention.equals("PascalCase") ? 
                                               Character.toUpperCase(name.charAt(0)) + name.substring(1) :
                                               Character.toLowerCase(name.charAt(0)) + name.substring(1));
        
        return new RefactorSuggestion(
            "naming_convention",
            "第" + lineNumber + "行的" + type + "不符合" + convention + "命名规范",
            originalCode,
            refactoredCode,
            Arrays.asList("符合Java命名规范", "提高代码一致性", "便于团队协作"),
            "低"
        );
    }
    
    /**
     * 获取行号
     */
    private int getLineNumber(String code, int position) {
        int lineNumber = 1;
        for (int i = 0; i < position && i < code.length(); i++) {
            if (code.charAt(i) == '\n') {
                lineNumber++;
            }
        }
        return lineNumber;
    }
    
    /**
     * 获取重构模式信息
     */
    public Map<String, RefactorPattern> getRefactorPatterns() {
        return new HashMap<>(refactorPatterns);
    }
    
    /**
     * 添加自定义重构模式
     */
    public void addRefactorPattern(String name, RefactorPattern pattern) {
        refactorPatterns.put(name, pattern);
    }
}