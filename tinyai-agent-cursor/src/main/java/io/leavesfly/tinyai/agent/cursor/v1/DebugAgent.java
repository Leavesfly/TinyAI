package io.leavesfly.tinyai.agent.cursor.v1;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 调试代理 - 基于LLM的智能错误诊断和修复建议
 * 结合传统静态分析和LLM智能推理，提供更准确的错误诊断和解决方案
 * 
 * @author 山泽
 */
public class DebugAgent {
    
    private final CodeAnalyzer analyzer;
    private final Map<String, ErrorPattern> errorPatterns;
    private final CursorLLMSimulator llmSimulator;  // 新增LLM模拟器
    
    /**
     * 错误模式内部类
     */
    public static class ErrorPattern {
        private final String description;
        private final String commonCauses;
        private final String fixStrategy;
        private final String severity;
        
        public ErrorPattern(String description, String commonCauses, String fixStrategy, String severity) {
            this.description = description;
            this.commonCauses = commonCauses;
            this.fixStrategy = fixStrategy;
            this.severity = severity;
        }
        
        // Getter 方法
        public String getDescription() { return description; }
        public String getCommonCauses() { return commonCauses; }
        public String getFixStrategy() { return fixStrategy; }
        public String getSeverity() { return severity; }
    }
    
    public DebugAgent(CodeAnalyzer analyzer) {
        this.analyzer = analyzer;
        this.errorPatterns = loadErrorPatterns();
        this.llmSimulator = new CursorLLMSimulator();  // 初始化LLM模拟器
    }
    
    /**
     * 加载错误模式
     */
    private Map<String, ErrorPattern> loadErrorPatterns() {
        Map<String, ErrorPattern> patterns = new HashMap<>();
        
        patterns.put("SyntaxError", new ErrorPattern(
            "语法错误", 
            "缺少分号、括号不匹配、缩进错误", 
            "检查语法结构和符号匹配", 
            "critical"));
        
        patterns.put("NullPointerException", new ErrorPattern(
            "空指针异常", 
            "未初始化对象、空引用调用", 
            "添加空值检查，确保对象初始化", 
            "high"));
        
        patterns.put("ArrayIndexOutOfBounds", new ErrorPattern(
            "数组越界", 
            "索引超出数组长度", 
            "检查数组边界，添加边界验证", 
            "high"));
        
        patterns.put("ClassNotFound", new ErrorPattern(
            "类未找到", 
            "缺少依赖、包路径错误", 
            "检查类路径和import语句", 
            "medium"));
        
        patterns.put("IllegalArgument", new ErrorPattern(
            "非法参数", 
            "参数类型不匹配、参数值无效", 
            "验证参数有效性，添加参数检查", 
            "medium"));
        
        patterns.put("ConcurrentModification", new ErrorPattern(
            "并发修改异常", 
            "迭代时修改集合", 
            "使用Iterator安全删除，或使用并发安全集合", 
            "medium"));
        
        patterns.put("StackOverflow", new ErrorPattern(
            "栈溢出", 
            "无限递归、递归层次过深", 
            "添加递归终止条件，检查递归逻辑", 
            "high"));
        
        patterns.put("OutOfMemory", new ErrorPattern(
            "内存溢出", 
            "内存泄漏、对象创建过多", 
            "优化内存使用，检查对象生命周期", 
            "critical"));
        
        return patterns;
    }
    
    /**
     * 诊断代码错误 - 增强LLM智能诊断
     * @param code 待诊断的代码
     * @param errorMessage 可选的错误消息
     * @return 诊断结果
     */
    public Map<String, Object> diagnoseError(String code, String errorMessage) {
        Map<String, Object> diagnosis = new HashMap<>();
        
        System.out.println("🔍 启动LLM增强错误诊断...");
        
        // 初始化诊断结果
        diagnosis.put("error_found", false);
        diagnosis.put("error_type", "");
        diagnosis.put("error_line", 0);
        diagnosis.put("diagnosis", "");
        diagnosis.put("suggestions", new ArrayList<>());
        diagnosis.put("fixed_code", "");
        diagnosis.put("confidence", 0.0);
        
        if (code == null || code.trim().isEmpty()) {
            diagnosis.put("error_found", true);
            diagnosis.put("error_type", "EmptyCode");
            diagnosis.put("diagnosis", "代码为空");
            diagnosis.put("suggestions", Arrays.asList("请提供有效的代码"));
            return diagnosis;
        }
        
        try {
            // 1. 传统静态分析
            Map<String, Object> staticAnalysis = performStaticAnalysis(code, errorMessage);
            
            // 2. LLM智能诊断
            String llmDiagnosis = llmSimulator.generateDebugAdvice(code, errorMessage);
            diagnosis.put("llm_diagnosis", llmDiagnosis);
            
            // 3. LLM智能修复建议
            String fixSuggestion = llmSimulator.generateCodingResponse(
                "请为以下错误提供详细的修复方案: " + errorMessage,
                code, "debug");
            diagnosis.put("llm_fix_suggestion", fixSuggestion);
            
            // 4. 结合传统和LLM结果
            diagnosis = mergeAnalysisResults(diagnosis, staticAnalysis, llmDiagnosis);
            
            // 5. 生成综合建议
            List<String> comprehensiveSuggestions = generateComprehensiveSuggestions(
                staticAnalysis, llmDiagnosis, errorMessage);
            diagnosis.put("comprehensive_suggestions", comprehensiveSuggestions);
            
            // 6. 智能修复代码
            if ((Boolean) diagnosis.get("error_found")) {
                String smartFixedCode = generateSmartFixedCode(code, diagnosis);
                diagnosis.put("smart_fixed_code", smartFixedCode);
            }
            
            System.out.println("✅ LLM增强诊断完成，置信度: " + diagnosis.get("confidence"));
            
        } catch (Exception e) {
            System.err.println("❌ LLM诊断失败: " + e.getMessage());
            // 回退到传统诊断
            return performTraditionalDiagnosis(code, errorMessage);
        }
        
        return diagnosis;
    }
    
    /**
     * 诊断语法错误
     */
    private Map<String, Object> diagnoseSyntaxError(Map<String, Object> analysis, String code) {
        Map<String, Object> diagnosis = new HashMap<>();
        
        diagnosis.put("error_found", true);
        diagnosis.put("error_type", "SyntaxError");
        diagnosis.put("error_line", analysis.getOrDefault("line", 0));
        diagnosis.put("diagnosis", analysis.getOrDefault("error", "语法错误"));
        diagnosis.put("confidence", 0.9);
        
        // 获取语法修复建议
        List<String> suggestions = getSyntaxFixSuggestions(analysis, code);
        diagnosis.put("suggestions", suggestions);
        
        // 尝试自动修复语法错误
        String fixedCode = attemptSyntaxFix(code, analysis);
        diagnosis.put("fixed_code", fixedCode);
        
        return diagnosis;
    }
    
    /**
     * 获取语法修复建议
     */
    private List<String> getSyntaxFixSuggestions(Map<String, Object> analysis, String code) {
        List<String> suggestions = new ArrayList<>();
        String errorMsg = (String) analysis.getOrDefault("error", "");
        
        if (errorMsg.toLowerCase().contains("';' expected")) {
            suggestions.add("在语句末尾添加分号 (;)");
            suggestions.add("检查是否缺少语句结束符");
        } else if (errorMsg.toLowerCase().contains("')' expected")) {
            suggestions.add("检查括号是否匹配");
            suggestions.add("确保所有左括号都有对应的右括号");
        } else if (errorMsg.toLowerCase().contains("'}' expected")) {
            suggestions.add("检查花括号是否匹配");
            suggestions.add("确保所有代码块都正确闭合");
        } else if (errorMsg.toLowerCase().contains("identifier expected")) {
            suggestions.add("检查变量名或方法名是否正确");
            suggestions.add("确保使用有效的标识符");
        } else if (errorMsg.toLowerCase().contains("illegal start of expression")) {
            suggestions.add("检查表达式语法");
            suggestions.add("确保操作符和操作数正确");
        } else {
            suggestions.add("检查语法结构是否正确");
            suggestions.add("参考Java语法规范");
            suggestions.add("使用IDE的语法检查功能");
        }
        
        return suggestions;
    }
    
    /**
     * 尝试自动修复语法错误
     */
    private String attemptSyntaxFix(String code, Map<String, Object> analysis) {
        String[] lines = code.split("\n");
        Integer errorLine = (Integer) analysis.get("line");
        
        if (errorLine == null || errorLine <= 0 || errorLine > lines.length) {
            return code;
        }
        
        String problematicLine = lines[errorLine - 1];
        String fixedLine = attemptLineFix(problematicLine);
        
        if (!fixedLine.equals(problematicLine)) {
            lines[errorLine - 1] = fixedLine;
            return String.join("\n", lines);
        }
        
        return code;
    }
    
    /**
     * 尝试修复单行代码
     */
    private String attemptLineFix(String line) {
        String trimmed = line.trim();
        
        // 修复缺少分号
        if (!trimmed.isEmpty() && 
            !trimmed.endsWith(";") && 
            !trimmed.endsWith("{") && 
            !trimmed.endsWith("}") &&
            !trimmed.startsWith("//") &&
            !trimmed.startsWith("/*") &&
            !trimmed.contains("if ") &&
            !trimmed.contains("else") &&
            !trimmed.contains("for ") &&
            !trimmed.contains("while ") &&
            !trimmed.contains("class ") &&
            !trimmed.contains("interface ")) {
            return line + ";";
        }
        
        // 修复常见的打字错误
        String fixed = line;
        fixed = fixed.replace("System.out.println(", "System.out.println(");
        fixed = fixed.replace("pubic ", "public ");
        fixed = fixed.replace("privte ", "private ");
        fixed = fixed.replace("protcted ", "protected ");
        fixed = fixed.replace("retur ", "return ");
        
        return fixed;
    }
    
    /**
     * 查找逻辑错误
     */
    private List<String> findLogicErrors(String code) {
        List<String> errors = new ArrayList<>();
        
        // 检查可能的空指针访问
        if (hasNullPointerRisk(code)) {
            errors.add("可能存在空指针访问风险，建议添加空值检查");
        }
        
        // 检查无限循环风险
        if (hasInfiniteLoopRisk(code)) {
            errors.add("可能存在无限循环风险，检查循环条件");
        }
        
        // 检查数组越界风险
        if (hasArrayBoundsRisk(code)) {
            errors.add("可能存在数组越界风险，检查数组索引");
        }
        
        // 检查资源泄漏风险
        if (hasResourceLeakRisk(code)) {
            errors.add("可能存在资源泄漏风险，确保及时关闭资源");
        }
        
        // 检查类型转换风险
        if (hasClassCastRisk(code)) {
            errors.add("可能存在类型转换异常风险，建议添加类型检查");
        }
        
        return errors;
    }
    
    /**
     * 检查空指针风险
     */
    private boolean hasNullPointerRisk(String code) {
        // 查找可能的空指针访问模式
        Pattern[] patterns = {
            Pattern.compile("\\w+\\.\\w+\\(.*\\).*\\."), // 链式调用
            Pattern.compile("\\w+\\[.*\\]\\.\\w+"), // 数组元素直接调用
            Pattern.compile("\\(\\w+\\)\\s*\\w+\\.") // 强制转换后直接调用
        };
        
        for (Pattern pattern : patterns) {
            if (pattern.matcher(code).find()) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 检查无限循环风险
     */
    private boolean hasInfiniteLoopRisk(String code) {
        // 简单检查：while(true) 或 for(;;) 但没有break
        if ((code.contains("while(true)") || code.contains("for(;;)")) && 
            !code.contains("break")) {
            return true;
        }
        
        // 检查循环变量没有修改的情况
        Pattern whilePattern = Pattern.compile("while\\s*\\(\\s*(\\w+)\\s*[<>=!]+.*\\)\\s*\\{([^}]+)\\}");
        Matcher matcher = whilePattern.matcher(code);
        
        while (matcher.find()) {
            String variable = matcher.group(1);
            String loopBody = matcher.group(2);
            
            if (!loopBody.contains(variable + "++") && 
                !loopBody.contains(variable + "--") && 
                !loopBody.contains(variable + " =") &&
                !loopBody.contains(variable + "+=") &&
                !loopBody.contains(variable + "-=")) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 检查数组越界风险
     */
    private boolean hasArrayBoundsRisk(String code) {
        // 查找数组访问模式
        Pattern arrayAccessPattern = Pattern.compile("\\w+\\[\\d+\\]");
        Matcher matcher = arrayAccessPattern.matcher(code);
        
        // 简单检查：使用固定索引访问数组
        return matcher.find();
    }
    
    /**
     * 检查资源泄漏风险
     */
    private boolean hasResourceLeakRisk(String code) {
        // 检查是否使用了需要关闭的资源但没有try-with-resources或finally
        String[] resources = {"FileInputStream", "FileOutputStream", "BufferedReader", 
                             "BufferedWriter", "Socket", "Connection"};
        
        for (String resource : resources) {
            if (code.contains(resource) && 
                !code.contains("try-with-resources") && 
                !code.contains("finally") &&
                !code.contains(".close()")) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 检查类型转换风险
     */
    private boolean hasClassCastRisk(String code) {
        // 查找强制类型转换
        Pattern castPattern = Pattern.compile("\\(\\w+\\)\\s*\\w+");
        Matcher matcher = castPattern.matcher(code);
        
        return matcher.find() && !code.contains("instanceof");
    }
    
    /**
     * 查找运行时错误风险
     */
    private List<String> findRuntimeErrorRisks(String code) {
        List<String> risks = new ArrayList<>();
        
        // 检查除零风险
        if (hasDivisionByZeroRisk(code)) {
            risks.add("可能存在除零异常风险，添加除数检查");
        }
        
        // 检查字符串空值操作
        if (hasStringNullRisk(code)) {
            risks.add("字符串操作可能遇到空值，添加空值检查");
        }
        
        // 检查集合操作风险
        if (hasCollectionRisk(code)) {
            risks.add("集合操作可能抛出异常，检查集合状态");
        }
        
        // 检查线程安全风险
        if (hasThreadSafetyRisk(code)) {
            risks.add("可能存在线程安全问题，考虑同步机制");
        }
        
        return risks;
    }
    
    /**
     * 检查除零风险
     */
    private boolean hasDivisionByZeroRisk(String code) {
        return code.contains("/") && !code.contains("if") && !code.contains("!=");
    }
    
    /**
     * 检查字符串空值风险
     */
    private boolean hasStringNullRisk(String code) {
        Pattern stringMethodPattern = Pattern.compile("\\w+\\.(length|charAt|substring|indexOf)\\(");
        return stringMethodPattern.matcher(code).find() && !code.contains("!= null");
    }
    
    /**
     * 检查集合操作风险
     */
    private boolean hasCollectionRisk(String code) {
        String[] riskyMethods = {".get(", ".remove(", ".set("};
        for (String method : riskyMethods) {
            if (code.contains(method) && !code.contains("size()") && !code.contains("isEmpty()")) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 检查线程安全风险
     */
    private boolean hasThreadSafetyRisk(String code) {
        return (code.contains("static") && code.contains("=")) || 
               (code.contains("ArrayList") || code.contains("HashMap")) && 
               (code.contains("thread") || code.contains("Thread"));
    }
    
    /**
     * 诊断特定错误
     */
    private Map<String, Object> diagnoseSpecificError(String code, String errorMessage) {
        Map<String, Object> diagnosis = new HashMap<>();
        diagnosis.put("error_found", false);
        
        // 根据错误消息匹配错误类型
        String errorType = matchErrorType(errorMessage);
        if (errorType != null) {
            ErrorPattern pattern = errorPatterns.get(errorType);
            if (pattern != null) {
                diagnosis.put("error_found", true);
                diagnosis.put("error_type", errorType);
                diagnosis.put("diagnosis", pattern.getDescription() + ": " + errorMessage);
                diagnosis.put("suggestions", generateSpecificSuggestions(errorType, pattern, errorMessage));
                diagnosis.put("confidence", 0.8);
                
                // 尝试特定修复
                String fixedCode = attemptSpecificFix(code, errorType, errorMessage);
                diagnosis.put("fixed_code", fixedCode);
            }
        }
        
        return diagnosis;
    }
    
    /**
     * 匹配错误类型
     */
    private String matchErrorType(String errorMessage) {
        String lowerMessage = errorMessage.toLowerCase();
        
        if (lowerMessage.contains("nullpointerexception")) {
            return "NullPointerException";
        } else if (lowerMessage.contains("arrayindexoutofbounds")) {
            return "ArrayIndexOutOfBounds";
        } else if (lowerMessage.contains("classnotfound")) {
            return "ClassNotFound";
        } else if (lowerMessage.contains("illegalargument")) {
            return "IllegalArgument";
        } else if (lowerMessage.contains("concurrentmodification")) {
            return "ConcurrentModification";
        } else if (lowerMessage.contains("stackoverfloweerror")) {
            return "StackOverflow";
        } else if (lowerMessage.contains("outofmemoryerror")) {
            return "OutOfMemory";
        }
        
        return null;
    }
    
    /**
     * 生成特定建议
     */
    private List<String> generateSpecificSuggestions(String errorType, ErrorPattern pattern, String errorMessage) {
        List<String> suggestions = new ArrayList<>();
        
        suggestions.add("错误类型：" + pattern.getDescription());
        suggestions.add("常见原因：" + pattern.getCommonCauses());
        suggestions.add("修复策略：" + pattern.getFixStrategy());
        
        // 根据错误类型添加具体建议
        switch (errorType) {
            case "NullPointerException":
                suggestions.add("在调用方法前检查对象是否为null");
                suggestions.add("使用Optional类来处理可能为null的值");
                break;
                
            case "ArrayIndexOutOfBounds":
                suggestions.add("检查数组长度：if (index < array.length)");
                suggestions.add("使用增强for循环避免索引错误");
                break;
                
            case "ClassNotFound":
                suggestions.add("检查类路径配置");
                suggestions.add("确认依赖jar包已正确添加");
                break;
                
            case "ConcurrentModification":
                suggestions.add("使用Iterator的remove()方法");
                suggestions.add("考虑使用ConcurrentHashMap等线程安全集合");
                break;
        }
        
        return suggestions;
    }
    
    /**
     * 尝试特定修复
     */
    private String attemptSpecificFix(String code, String errorType, String errorMessage) {
        switch (errorType) {
            case "NullPointerException":
                return attemptNullPointerFix(code);
                
            case "ArrayIndexOutOfBounds":
                return attemptArrayBoundsFix(code);
                
            default:
                return code;
        }
    }
    
    /**
     * 尝试修复空指针问题
     */
    private String attemptNullPointerFix(String code) {
        // 简单的空值检查插入
        Pattern methodCallPattern = Pattern.compile("(\\w+)\\.(\\w+)\\(");
        Matcher matcher = methodCallPattern.matcher(code);
        
        StringBuffer fixed = new StringBuffer();
        while (matcher.find()) {
            String object = matcher.group(1);
            String replacement = "if (" + object + " != null) {\n    " + matcher.group(0);
            matcher.appendReplacement(fixed, replacement);
        }
        matcher.appendTail(fixed);
        
        return fixed.toString();
    }
    
    /**
     * 尝试修复数组越界问题
     */
    private String attemptArrayBoundsFix(String code) {
        // 为数组访问添加边界检查
        Pattern arrayAccessPattern = Pattern.compile("(\\w+)\\[(\\w+)\\]");
        Matcher matcher = arrayAccessPattern.matcher(code);
        
        StringBuffer fixed = new StringBuffer();
        while (matcher.find()) {
            String array = matcher.group(1);
            String index = matcher.group(2);
            String replacement = "(" + index + " >= 0 && " + index + " < " + array + ".length) ? " + 
                               matcher.group(0) + " : null";
            matcher.appendReplacement(fixed, replacement);
        }
        matcher.appendTail(fixed);
        
        return fixed.toString();
    }
    
    /**
     * 尝试自动修复
     */
    private String attemptAutoFix(String code, String errorType) {
        switch (errorType) {
            case "SyntaxError":
                return attemptSyntaxFix(code, null);
            case "LogicError":
                return attemptLogicFix(code);
            case "RuntimeRisk":
                return attemptRuntimeRiskFix(code);
            default:
                return code;
        }
    }
    
    /**
     * 尝试修复逻辑错误
     */
    private String attemptLogicFix(String code) {
        String fixed = code;
        
        // 添加基本的空值检查
        if (hasNullPointerRisk(code)) {
            fixed = addNullChecks(fixed);
        }
        
        // 添加数组边界检查
        if (hasArrayBoundsRisk(code)) {
            fixed = addBoundaryChecks(fixed);
        }
        
        return fixed;
    }
    
    /**
     * 添加空值检查
     */
    private String addNullChecks(String code) {
        // 简化实现：在方法调用前添加注释提醒
        return "// 建议添加空值检查\n" + code;
    }
    
    /**
     * 添加边界检查
     */
    private String addBoundaryChecks(String code) {
        // 简化实现：添加注释提醒
        return "// 建议添加数组边界检查\n" + code;
    }
    
    /**
     * 尝试修复运行时风险
     */
    private String attemptRuntimeRiskFix(String code) {
        return "// 建议添加异常处理\ntry {\n" + code + "\n} catch (Exception e) {\n    // 处理异常\n}";
    }
    
    /**
     * 获取错误模式信息
     */
    public Map<String, ErrorPattern> getErrorPatterns() {
        return new HashMap<>(errorPatterns);
    }
    
    /**
     * 添加自定义错误模式
     */
    public void addErrorPattern(String name, ErrorPattern pattern) {
        errorPatterns.put(name, pattern);
    }
    
    // ========== LLM增强方法 ==========
    
    /**
     * 执行传统静态分析
     */
    private Map<String, Object> performStaticAnalysis(String code, String errorMessage) {
        Map<String, Object> analysis = new HashMap<>();
        
        // 语法分析
        Map<String, Object> codeAnalysis = analyzer.analyzeJavaCode(code);
        analysis.put("syntax_valid", codeAnalysis.getOrDefault("syntax_valid", true));
        analysis.put("syntax_issues", codeAnalysis.getOrDefault("syntax_issues", new ArrayList<>()));
        
        // 逻辑错误检查
        List<String> logicErrors = findLogicErrors(code);
        analysis.put("logic_errors", logicErrors);
        
        // 运行时风险检查
        List<String> runtimeRisks = findRuntimeErrorRisks(code);
        analysis.put("runtime_risks", runtimeRisks);
        
        // 特定错误分析
        if (errorMessage != null && !errorMessage.trim().isEmpty()) {
            Map<String, Object> specificError = diagnoseSpecificError(code, errorMessage);
            analysis.put("specific_error", specificError);
        }
        
        return analysis;
    }
    
    /**
     * 合并分析结果
     */
    private Map<String, Object> mergeAnalysisResults(Map<String, Object> diagnosis, 
                                                     Map<String, Object> staticAnalysis, 
                                                     String llmDiagnosis) {
        // 检查是否发现错误
        boolean errorFound = false;
        String errorType = "Unknown";
        double confidence = 0.5;
        
        // 从静态分析中提取信息
        if (!(Boolean) staticAnalysis.getOrDefault("syntax_valid", true)) {
            errorFound = true;
            errorType = "SyntaxError";
            confidence = 0.9;
        } else if (!((List<?>) staticAnalysis.getOrDefault("logic_errors", new ArrayList<>())).isEmpty()) {
            errorFound = true;
            errorType = "LogicError";
            confidence = 0.7;
        } else if (!((List<?>) staticAnalysis.getOrDefault("runtime_risks", new ArrayList<>())).isEmpty()) {
            errorFound = true;
            errorType = "RuntimeRisk";
            confidence = 0.6;
        }
        
        // 结合LLM分析提高置信度
        if (llmDiagnosis.contains("错误") || llmDiagnosis.contains("问题")) {
            errorFound = true;
            confidence = Math.min(1.0, confidence + 0.2);
        }
        
        diagnosis.put("error_found", errorFound);
        diagnosis.put("error_type", errorType);
        diagnosis.put("confidence", confidence);
        diagnosis.put("static_analysis", staticAnalysis);
        
        return diagnosis;
    }
    
    /**
     * 生成综合建议
     */
    private List<String> generateComprehensiveSuggestions(Map<String, Object> staticAnalysis, 
                                                         String llmDiagnosis, 
                                                         String errorMessage) {
        List<String> suggestions = new ArrayList<>();
        
        // 添加静态分析建议
        @SuppressWarnings("unchecked")
        List<String> logicErrors = (List<String>) staticAnalysis.getOrDefault("logic_errors", new ArrayList<>());
        suggestions.addAll(logicErrors);
        
        @SuppressWarnings("unchecked")
        List<String> runtimeRisks = (List<String>) staticAnalysis.getOrDefault("runtime_risks", new ArrayList<>());
        suggestions.addAll(runtimeRisks);
        
        // 添加LLM建议
        if (llmDiagnosis != null && !llmDiagnosis.isEmpty()) {
            suggestions.add("LLM分析结果: " + llmDiagnosis);
        }
        
        // 添加基于错误类型的建议
        if (errorMessage != null) {
            suggestions.addAll(generateErrorSpecificSuggestions(errorMessage));
        }
        
        // 如果没有建议，添加通用建议
        if (suggestions.isEmpty()) {
            suggestions.add("代码看起来基本正常，建议检查运行时环境和输入数据");
            suggestions.add("建议添加更多的单元测试来验证代码正确性");
        }
        
        return suggestions;
    }
    
    /**
     * 生成智能修复代码
     */
    private String generateSmartFixedCode(String originalCode, Map<String, Object> diagnosis) {
        StringBuilder fixedCode = new StringBuilder();
        
        fixedCode.append("// === 智能修复建议 ===\n");
        fixedCode.append("// 错误类型: ").append(diagnosis.get("error_type")).append("\n");
        fixedCode.append("// 置信度: ").append(diagnosis.get("confidence")).append("\n\n");
        
        String errorType = (String) diagnosis.get("error_type");
        
        switch (errorType) {
            case "SyntaxError":
                fixedCode.append("传统修复: 检查语法结构\n");
                fixedCode.append(attemptSyntaxFix(originalCode, null));
                break;
                
            case "LogicError":
                fixedCode.append("逻辑修复: 添加检查和验证\n");
                fixedCode.append(attemptLogicFix(originalCode));
                break;
                
            case "RuntimeRisk":
                fixedCode.append("运行时修复: 添加异常处理\n");
                fixedCode.append(attemptRuntimeRiskFix(originalCode));
                break;
                
            default:
                fixedCode.append("通用修复: 添加注释和验证\n");
                fixedCode.append("// 建议添加适当的错误处理和日志\n");
                fixedCode.append(originalCode);
        }
        
        return fixedCode.toString();
    }
    
    /**
     * 执行传统诊断（回退方案）
     */
    private Map<String, Object> performTraditionalDiagnosis(String code, String errorMessage) {
        Map<String, Object> diagnosis = new HashMap<>();
        
        // 初始化
        diagnosis.put("error_found", false);
        diagnosis.put("error_type", "");
        diagnosis.put("error_line", 0);
        diagnosis.put("diagnosis", "");
        diagnosis.put("suggestions", new ArrayList<>());
        diagnosis.put("fixed_code", "");
        diagnosis.put("confidence", 0.0);
        
        // 基础分析
        Map<String, Object> analysis = analyzer.analyzeJavaCode(code);
        
        // 检查语法错误
        if (!(Boolean) analysis.getOrDefault("syntax_valid", true)) {
            return diagnoseSyntaxError(analysis, code);
        }
        
        // 检查逻辑错误
        List<String> logicErrors = findLogicErrors(code);
        if (!logicErrors.isEmpty()) {
            diagnosis.put("error_found", true);
            diagnosis.put("error_type", "LogicError");
            diagnosis.put("diagnosis", "发现潜在逻辑错误");
            diagnosis.put("suggestions", logicErrors);
            diagnosis.put("confidence", 0.7);
        }
        
        return diagnosis;
    }
    
    /**
     * 生成错误特定建议
     */
    private List<String> generateErrorSpecificSuggestions(String errorMessage) {
        List<String> suggestions = new ArrayList<>();
        
        if (errorMessage.toLowerCase().contains("null")) {
            suggestions.add("在使用对象前检查是否为null");
            suggestions.add("使用Optional类处理可能为null的值");
        }
        
        if (errorMessage.toLowerCase().contains("array") || errorMessage.toLowerCase().contains("index")) {
            suggestions.add("检查数组边界条件");
            suggestions.add("使用增强for循环避免索引错误");
        }
        
        return suggestions;
    }
}