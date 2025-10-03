package io.leavesfly.tinyai.agent.cursor;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 代码分析器 - 分析代码结构、质量和潜在问题
 * 基于静态分析技术，支持Java代码的深度解析
 * 
 * @author 山泽
 */
public class CodeAnalyzer {
    
    private final Map<String, Object> analysisCache;
    private final List<Pattern> syntaxPatterns;
    
    public CodeAnalyzer() {
        this.analysisCache = new HashMap<>();
        this.syntaxPatterns = initializeSyntaxPatterns();
    }
    
    /**
     * 分析Java代码
     * @param code 待分析的代码
     * @return 分析结果
     */
    public Map<String, Object> analyzeJavaCode(String code) {
        if (code == null || code.trim().isEmpty()) {
            return createErrorResult("代码不能为空");
        }
        
        // 检查缓存
        String cacheKey = String.valueOf(code.hashCode());
        if (analysisCache.containsKey(cacheKey)) {
            return (Map<String, Object>) analysisCache.get(cacheKey);
        }
        
        Map<String, Object> analysis = new HashMap<>();
        
        try {
            // 基本语法检查
            List<CodeIssue> syntaxIssues = checkSyntax(code);
            analysis.put("syntax_valid", syntaxIssues.isEmpty());
            analysis.put("syntax_issues", syntaxIssues);
            
            // 提取代码结构
            analysis.put("imports", extractImports(code));
            analysis.put("classes", extractClasses(code));
            analysis.put("methods", extractMethods(code));
            analysis.put("variables", extractVariables(code));
            
            // 计算代码度量
            analysis.put("metrics", calculateMetrics(code));
            analysis.put("complexity", calculateComplexity(code));
            
            // 发现代码问题
            List<CodeIssue> codeIssues = findCodeIssues(code);
            analysis.put("issues", codeIssues);
            
            // 缓存结果
            analysisCache.put(cacheKey, analysis);
            
        } catch (Exception e) {
            return createErrorResult("分析过程中发生错误: " + e.getMessage());
        }
        
        return analysis;
    }
    
    /**
     * 创建错误结果
     */
    private Map<String, Object> createErrorResult(String errorMessage) {
        Map<String, Object> result = new HashMap<>();
        result.put("syntax_valid", false);
        result.put("error", errorMessage);
        result.put("analysis_time", System.currentTimeMillis());
        return result;
    }
    
    /**
     * 初始化语法模式
     */
    private List<Pattern> initializeSyntaxPatterns() {
        List<Pattern> patterns = new ArrayList<>();
        
        // 基本语法检查模式
        patterns.add(Pattern.compile("\\bclass\\s+\\w+", Pattern.MULTILINE));
        patterns.add(Pattern.compile("\\binterface\\s+\\w+", Pattern.MULTILINE));
        patterns.add(Pattern.compile("\\bpublic\\s+.*\\s+\\w+\\s*\\(.*\\)", Pattern.MULTILINE));
        patterns.add(Pattern.compile("\\bprivate\\s+.*\\s+\\w+\\s*\\(.*\\)", Pattern.MULTILINE));
        
        return patterns;
    }
    
    /**
     * 检查基本语法
     */
    private List<CodeIssue> checkSyntax(String code) {
        List<CodeIssue> issues = new ArrayList<>();
        String[] lines = code.split("\n");
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            int lineNumber = i + 1;
            
            // 检查括号匹配
            if (!checkBracketMatching(line)) {
                issues.add(new CodeIssue("bracket_mismatch", "high", 
                    "括号不匹配", lineNumber, "检查括号配对"));
            }
            
            // 检查分号
            if (needsSemicolon(line) && !line.endsWith(";") && !line.endsWith("{") && !line.endsWith("}")) {
                issues.add(new CodeIssue("missing_semicolon", "medium", 
                    "可能缺少分号", lineNumber, "在语句末尾添加分号"));
            }
            
            // 检查命名规范
            checkNamingConventions(line, lineNumber, issues);
        }
        
        return issues;
    }
    
    /**
     * 检查括号匹配
     */
    private boolean checkBracketMatching(String line) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : line.toCharArray()) {
            switch (c) {
                case '(':
                case '[':
                case '{':
                    stack.push(c);
                    break;
                case ')':
                    if (stack.isEmpty() || stack.pop() != '(') return false;
                    break;
                case ']':
                    if (stack.isEmpty() || stack.pop() != '[') return false;
                    break;
                case '}':
                    if (stack.isEmpty() || stack.pop() != '{') return false;
                    break;
            }
        }
        
        return stack.isEmpty();
    }
    
    /**
     * 检查是否需要分号
     */
    private boolean needsSemicolon(String line) {
        line = line.trim();
        if (line.isEmpty() || line.startsWith("//") || line.startsWith("/*")) {
            return false;
        }
        
        // 这些语句不需要分号
        String[] noSemicolonKeywords = {"if", "else", "for", "while", "do", "switch", 
                                       "try", "catch", "finally", "class", "interface", 
                                       "public class", "private class", "protected class"};
        
        for (String keyword : noSemicolonKeywords) {
            if (line.startsWith(keyword + " ") || line.startsWith(keyword + "(")) {
                return false;
            }
        }
        
        return !line.endsWith("{") && !line.endsWith("}") && line.length() > 2;
    }
    
    /**
     * 检查命名规范
     */
    private void checkNamingConventions(String line, int lineNumber, List<CodeIssue> issues) {
        // 检查类名（应该使用PascalCase）
        Pattern classPattern = Pattern.compile("\\bclass\\s+([a-z][a-zA-Z0-9_]*)");
        Matcher classMatcher = classPattern.matcher(line);
        if (classMatcher.find()) {
            issues.add(new CodeIssue("naming_convention", "low", 
                "类名应该使用PascalCase命名法", lineNumber, 
                "将类名改为大写字母开头"));
        }
        
        // 检查变量名（应该使用camelCase）
        Pattern variablePattern = Pattern.compile("\\b(int|String|boolean|double|float)\\s+([A-Z][a-zA-Z0-9_]*)");
        Matcher variableMatcher = variablePattern.matcher(line);
        if (variableMatcher.find()) {
            issues.add(new CodeIssue("naming_convention", "low", 
                "变量名应该使用camelCase命名法", lineNumber, 
                "将变量名改为小写字母开头"));
        }
    }
    
    /**
     * 提取导入语句
     */
    private List<String> extractImports(String code) {
        List<String> imports = new ArrayList<>();
        Pattern importPattern = Pattern.compile("import\\s+([\\w\\.\\*]+);");
        Matcher matcher = importPattern.matcher(code);
        
        while (matcher.find()) {
            imports.add(matcher.group(1));
        }
        
        return imports;
    }
    
    /**
     * 提取类定义
     */
    private List<Map<String, Object>> extractClasses(String code) {
        List<Map<String, Object>> classes = new ArrayList<>();
        Pattern classPattern = Pattern.compile("(public\\s+|private\\s+|protected\\s+)?class\\s+(\\w+)\\s*(extends\\s+(\\w+))?\\s*(implements\\s+([\\w,\\s]+))?");
        Matcher matcher = classPattern.matcher(code);
        
        while (matcher.find()) {
            Map<String, Object> classInfo = new HashMap<>();
            classInfo.put("name", matcher.group(2));
            classInfo.put("modifier", matcher.group(1) != null ? matcher.group(1).trim() : "default");
            classInfo.put("extends", matcher.group(4));
            classInfo.put("implements", matcher.group(6));
            classInfo.put("line", getLineNumber(code, matcher.start()));
            
            classes.add(classInfo);
        }
        
        return classes;
    }
    
    /**
     * 提取方法定义
     */
    private List<Map<String, Object>> extractMethods(String code) {
        List<Map<String, Object>> methods = new ArrayList<>();
        Pattern methodPattern = Pattern.compile("(public|private|protected)?\\s*(static)?\\s*(\\w+)\\s+(\\w+)\\s*\\([^)]*\\)");
        Matcher matcher = methodPattern.matcher(code);
        
        while (matcher.find()) {
            Map<String, Object> methodInfo = new HashMap<>();
            methodInfo.put("modifier", matcher.group(1));
            methodInfo.put("static", matcher.group(2) != null);
            methodInfo.put("returnType", matcher.group(3));
            methodInfo.put("name", matcher.group(4));
            methodInfo.put("line", getLineNumber(code, matcher.start()));
            
            methods.add(methodInfo);
        }
        
        return methods;
    }
    
    /**
     * 提取变量定义
     */
    private List<Map<String, Object>> extractVariables(String code) {
        List<Map<String, Object>> variables = new ArrayList<>();
        Pattern variablePattern = Pattern.compile("(private|public|protected)?\\s*(static)?\\s*(final)?\\s*(\\w+)\\s+(\\w+)");
        Matcher matcher = variablePattern.matcher(code);
        
        while (matcher.find()) {
            String type = matcher.group(4);
            // 过滤掉方法和类关键字
            if (!isKeyword(type)) {
                Map<String, Object> variableInfo = new HashMap<>();
                variableInfo.put("modifier", matcher.group(1));
                variableInfo.put("static", matcher.group(2) != null);
                variableInfo.put("final", matcher.group(3) != null);
                variableInfo.put("type", type);
                variableInfo.put("name", matcher.group(5));
                variableInfo.put("line", getLineNumber(code, matcher.start()));
                
                variables.add(variableInfo);
            }
        }
        
        return variables;
    }
    
    /**
     * 计算代码度量
     */
    private Map<String, Object> calculateMetrics(String code) {
        Map<String, Object> metrics = new HashMap<>();
        String[] lines = code.split("\n");
        
        int totalLines = lines.length;
        int codeLines = 0;
        int commentLines = 0;
        int blankLines = 0;
        
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                blankLines++;
            } else if (trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith("*")) {
                commentLines++;
            } else {
                codeLines++;
            }
        }
        
        metrics.put("total_lines", totalLines);
        metrics.put("code_lines", codeLines);
        metrics.put("comment_lines", commentLines);
        metrics.put("blank_lines", blankLines);
        metrics.put("comment_ratio", totalLines > 0 ? (double) commentLines / totalLines : 0.0);
        
        return metrics;
    }
    
    /**
     * 计算圈复杂度
     */
    private int calculateComplexity(String code) {
        int complexity = 1; // 基础复杂度
        
        // 查找控制流语句
        String[] complexityKeywords = {"if", "else if", "while", "for", "do", "switch", "case", "catch", "?", "&&", "||"};
        
        for (String keyword : complexityKeywords) {
            Pattern pattern = Pattern.compile("\\b" + Pattern.quote(keyword) + "\\b");
            Matcher matcher = pattern.matcher(code);
            while (matcher.find()) {
                complexity++;
            }
        }
        
        return complexity;
    }
    
    /**
     * 发现代码问题
     */
    private List<CodeIssue> findCodeIssues(String code) {
        List<CodeIssue> issues = new ArrayList<>();
        String[] lines = code.split("\n");
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            int lineNumber = i + 1;
            
            // 检查长行
            if (line.length() > 120) {
                issues.add(new CodeIssue("long_line", "low", 
                    "代码行过长 (" + line.length() + " 字符)", lineNumber, 
                    "将长行分解为多行"));
            }
            
            // 检查过多的嵌套
            int indentLevel = getIndentLevel(line);
            if (indentLevel > 6) {
                issues.add(new CodeIssue("deep_nesting", "medium", 
                    "嵌套层次过深", lineNumber, 
                    "考虑提取方法以减少嵌套"));
            }
            
            // 检查硬编码字符串
            if (line.contains("\"") && !line.trim().startsWith("//")) {
                long stringCount = line.chars().filter(ch -> ch == '"').count();
                if (stringCount >= 4) { // 至少两个字符串
                    issues.add(new CodeIssue("hardcoded_string", "low", 
                        "发现硬编码字符串", lineNumber, 
                        "考虑使用常量或配置文件"));
                }
            }
        }
        
        // 检查方法长度
        checkMethodLength(code, issues);
        
        // 检查重复代码
        checkDuplicateCode(code, issues);
        
        return issues;
    }
    
    /**
     * 检查方法长度
     */
    private void checkMethodLength(String code, List<CodeIssue> issues) {
        Pattern methodPattern = Pattern.compile("(public|private|protected)?\\s*\\w+\\s+\\w+\\s*\\([^)]*\\)\\s*\\{");
        Matcher matcher = methodPattern.matcher(code);
        
        while (matcher.find()) {
            int methodStart = matcher.start();
            int methodStartLine = getLineNumber(code, methodStart);
            
            // 简单计算方法长度（直到找到匹配的}）
            int braceCount = 1;
            int pos = matcher.end();
            int currentLine = methodStartLine;
            
            while (pos < code.length() && braceCount > 0) {
                char c = code.charAt(pos);
                if (c == '{') braceCount++;
                else if (c == '}') braceCount--;
                else if (c == '\n') currentLine++;
                pos++;
            }
            
            int methodLength = currentLine - methodStartLine;
            if (methodLength > 50) {
                issues.add(new CodeIssue("long_method", "medium", 
                    "方法过长 (" + methodLength + " 行)", methodStartLine, 
                    "考虑将大方法分解为多个小方法"));
            }
        }
    }
    
    /**
     * 检查重复代码
     */
    private void checkDuplicateCode(String code, List<CodeIssue> issues) {
        String[] lines = code.split("\n");
        Map<String, List<Integer>> lineMap = new HashMap<>();
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (!line.isEmpty() && !line.startsWith("//") && line.length() > 10) {
                lineMap.computeIfAbsent(line, k -> new ArrayList<>()).add(i + 1);
            }
        }
        
        for (Map.Entry<String, List<Integer>> entry : lineMap.entrySet()) {
            if (entry.getValue().size() > 1) {
                int firstLine = entry.getValue().get(0);
                issues.add(new CodeIssue("duplicate_code", "low", 
                    "发现重复代码行", firstLine, 
                    "考虑提取公共方法或常量"));
            }
        }
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
     * 获取缩进级别
     */
    private int getIndentLevel(String line) {
        int level = 0;
        for (char c : line.toCharArray()) {
            if (c == ' ') level++;
            else if (c == '\t') level += 4;
            else break;
        }
        return level / 4; // 假设4个空格为一个缩进级别
    }
    
    /**
     * 检查是否为关键字
     */
    private boolean isKeyword(String word) {
        String[] keywords = {"class", "interface", "extends", "implements", "public", "private", 
                            "protected", "static", "final", "abstract", "synchronized", "native", 
                            "strictfp", "transient", "volatile", "if", "else", "while", "for", 
                            "do", "switch", "case", "default", "break", "continue", "return", 
                            "try", "catch", "finally", "throw", "throws", "import", "package"};
        
        for (String keyword : keywords) {
            if (keyword.equals(word)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 清空缓存
     */
    public void clearCache() {
        analysisCache.clear();
    }
    
    /**
     * 获取缓存大小
     */
    public int getCacheSize() {
        return analysisCache.size();
    }
}