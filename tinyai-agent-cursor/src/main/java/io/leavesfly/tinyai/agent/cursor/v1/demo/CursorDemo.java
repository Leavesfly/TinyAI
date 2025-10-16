package io.leavesfly.tinyai.agent.cursor.v1.demo;

import io.leavesfly.tinyai.agent.cursor.v1.AICodingCursor;
import io.leavesfly.tinyai.agent.cursor.v1.CodeIssue;
import io.leavesfly.tinyai.agent.cursor.v1.RefactorSuggestion;

import java.util.*;
import java.util.Scanner;



/**
 * AI Coding Cursor 演示程序
 * 展示智能编程助手的所有核心功能
 * 
 * @author 山泽
 */
public class CursorDemo {
    
    private static final AICodingCursor cursor = new AICodingCursor("演示助手");
    private static final Scanner scanner = new Scanner(System.in);
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("🚀 AI Coding Cursor 智能编程助手演示");
        System.out.println("=".repeat(60));
        
        // 运行基础功能演示
        demonstrateBasicFeatures();
        
        // 运行高级功能演示
        demonstrateAdvancedFeatures();
        
        // 交互式演示
        runInteractiveDemo();
        
        System.out.println("\n🎉 演示完成！感谢使用 AI Coding Cursor!");
    }
    
    /**
     * 基础功能演示
     */
    private static void demonstrateBasicFeatures() {
        System.out.println("\n📋 1. 基础功能演示");
        System.out.println("-".repeat(40));
        
        // 演示代码分析
        demonstrateCodeAnalysis();
        
        // 演示代码生成
        demonstrateCodeGeneration();
        
        // 演示重构建议
        demonstrateRefactorSuggestions();
        
        // 演示错误调试
        demonstrateErrorDebugging();
    }
    
    /**
     * 演示代码分析
     */
    private static void demonstrateCodeAnalysis() {
        System.out.println("\n🔍 代码分析演示:");
        
        String testCode = """
            public class Calculator {
                private int value;
                
                public Calculator() {
                    this.value = 0;
                }
                
                public int add(int a, int b) {
                    if (a < 0 || b < 0) {
                        throw new IllegalArgumentException("负数not allowed");
                    }
                    return a + b;
                }
                
                public int divide(int a, int b) {
                    return a / b; // 潜在的除零错误
                }
                
                public void processArray(int[] arr) {
                    for (int i = 0; i <= arr.length; i++) { // 数组越界错误
                        System.out.println(arr[i]);
                    }
                }
            }
            """;
        
        Map<String, Object> analysis = cursor.analyzeCode(testCode);
        
        System.out.println("分析结果:");
        System.out.println("• 语法有效: " + analysis.get("syntax_valid"));
        
        @SuppressWarnings("unchecked")
        Map<String, Object> metrics = (Map<String, Object>) analysis.get("metrics");
        if (metrics != null) {
            System.out.println("• 总行数: " + metrics.get("total_lines"));
            System.out.println("• 代码行数: " + metrics.get("code_lines"));
            System.out.println("• 注释行数: " + metrics.get("comment_lines"));
        }
        
        System.out.println("• 圈复杂度: " + analysis.get("complexity"));
        
        @SuppressWarnings("unchecked")
        List<CodeIssue> issues = (List<CodeIssue>) analysis.get("issues");
        if (issues != null && !issues.isEmpty()) {
            System.out.println("• 发现问题:");
            for (int i = 0; i < Math.min(3, issues.size()); i++) {
                CodeIssue issue = issues.get(i);
                System.out.println("  - [" + issue.getSeverity() + "] " + issue.getMessage());
            }
        }
    }
    
    /**
     * 演示代码生成
     */
    private static void demonstrateCodeGeneration() {
        System.out.println("\n🤖 代码生成演示:");
        
        // 生成方法
        System.out.println("生成验证邮箱的方法:");
        String methodCode = cursor.generateCode("method validateEmail");
        System.out.println(methodCode.substring(0, Math.min(200, methodCode.length())) + "...");
        
        // 生成类
        System.out.println("\n生成用户管理类:");
        String classCode = cursor.generateCode("class UserManager");
        System.out.println(classCode.substring(0, Math.min(300, classCode.length())) + "...");
        
        // 生成测试
        System.out.println("\n生成测试代码:");
        String testCode = cursor.generateCode("test method for Calculator");
        System.out.println(testCode.substring(0, Math.min(200, testCode.length())) + "...");
    }
    
    /**
     * 演示重构建议
     */
    private static void demonstrateRefactorSuggestions() {
        System.out.println("\n🔧 重构建议演示:");
        
        String complexCode = """
            public class DataProcessor {
                public void processData(String data1, String data2, String data3, String data4, String data5, String data6) {
                    if (data1 != null && data2 != null && data3 != null && data4 != null && data5 != null && data6 != null) {
                        if (data1.length() > 0 && data2.length() > 0) {
                            if (data3.contains("valid") && data4.contains("valid")) {
                                if (data5.startsWith("prefix") && data6.endsWith("suffix")) {
                                    // 处理逻辑
                                    System.out.println("Processing...");
                                    System.out.println("Processing...");
                                    System.out.println("Processing...");
                                    // 重复代码
                                    int result = calculateSomething(100);
                                    System.out.println(result);
                                }
                            }
                        }
                    }
                }
                
                private int calculateSomething(int value) {
                    return value * 100 / 50 + 25;
                }
            }
            """;
        
        List<RefactorSuggestion> suggestions = cursor.suggestRefactor(complexCode);
        
        if (!suggestions.isEmpty()) {
            System.out.println("发现 " + suggestions.size() + " 个重构建议:");
            for (int i = 0; i < Math.min(3, suggestions.size()); i++) {
                RefactorSuggestion suggestion = suggestions.get(i);
                System.out.println("• [" + suggestion.getEstimatedImpact() + "] " + suggestion.getDescription());
                System.out.println("  收益: " + suggestion.getBenefitsSummary());
            }
        } else {
            System.out.println("✅ 代码结构良好，无需重构");
        }
    }
    
    /**
     * 演示错误调试
     */
    private static void demonstrateErrorDebugging() {
        System.out.println("\n🐛 错误调试演示:");
        
        // 语法错误代码
        String buggyCode = """
            public class BuggyClass {
                public void method1() {
                    int x = 10
                    String str = null;
                    int length = str.length(); // 空指针风险
                    
                    int[] arr = new int[5];
                    System.out.println(arr[10]); // 数组越界
                }
                
                public int divide(int a, int b) {
                    return a / b; // 除零风险
                }
            }
            """;
        
        Map<String, Object> debugResult = cursor.debugCode(buggyCode);
        
        System.out.println("调试结果:");
        System.out.println("• 发现错误: " + debugResult.get("error_found"));
        
        if ((Boolean) debugResult.get("error_found")) {
            System.out.println("• 错误类型: " + debugResult.get("error_type"));
            System.out.println("• 诊断信息: " + debugResult.get("diagnosis"));
            
            @SuppressWarnings("unchecked")
            List<String> suggestions = (List<String>) debugResult.get("suggestions");
            if (suggestions != null && !suggestions.isEmpty()) {
                System.out.println("• 修复建议:");
                for (int i = 0; i < Math.min(3, suggestions.size()); i++) {
                    System.out.println("  - " + suggestions.get(i));
                }
            }
        }
    }
    
    /**
     * 高级功能演示
     */
    private static void demonstrateAdvancedFeatures() {
        System.out.println("\n📋 2. 高级功能演示");
        System.out.println("-".repeat(40));
        
        // 综合代码审查
        demonstrateCodeReview();
        
        // AI对话功能
        demonstrateAIChat();
        
        // 系统状态监控
        demonstrateSystemStatus();
    }
    
    /**
     * 演示代码审查
     */
    private static void demonstrateCodeReview() {
        System.out.println("\n📋 综合代码审查演示:");
        
        String reviewCode = """
            public class UserService {
                private List<User> users = new ArrayList<>();
                
                public User createUser(String name, String email) {
                    // 简单的用户创建逻辑
                    User user = new User();
                    user.setName(name);
                    user.setEmail(email);
                    users.add(user);
                    return user;
                }
                
                public List<User> getAllUsers() {
                    return users;
                }
                
                public User findUserById(Long id) {
                    for (User user : users) {
                        if (user.getId().equals(id)) {
                            return user;
                        }
                    }
                    return null;
                }
            }
            """;
        
        Map<String, Object> reviewResult = cursor.reviewCode(reviewCode);
        
        System.out.println("审查结果:");
        System.out.println("• 质量评分: " + String.format("%.1f", (Double) reviewResult.get("overall_score")));
        
        @SuppressWarnings("unchecked")
        List<String> recommendations = (List<String>) reviewResult.get("recommendations");
        if (recommendations != null && !recommendations.isEmpty()) {
            System.out.println("• 改进建议:");
            for (int i = 0; i < Math.min(3, recommendations.size()); i++) {
                System.out.println("  - " + recommendations.get(i));
            }
        }
    }
    
    /**
     * 演示AI对话
     */
    private static void demonstrateAIChat() {
        System.out.println("\n💬 AI对话功能演示:");
        
        String[] testQuestions = {
            "如何避免空指针异常？",
            "Java中什么是单例模式？",
            "如何优化数据库查询性能？"
        };
        
        for (String question : testQuestions) {
            System.out.println("\n👤 问题: " + question);
            String response = cursor.chat(question);
            System.out.println("🤖 回答: " + response.substring(0, Math.min(150, response.length())) + "...");
        }
    }
    
    /**
     * 演示系统状态
     */
    private static void demonstrateSystemStatus() {
        System.out.println("\n📊 系统状态演示:");
        
        Map<String, Object> status = cursor.getSystemStatus();
        System.out.println("• 系统名称: " + status.get("name"));
        System.out.println("• 运行时长: " + status.get("uptime_minutes") + " 分钟");
        System.out.println("• 会话操作数: " + status.get("session_operations"));
        
        @SuppressWarnings("unchecked")
        Map<String, Integer> stats = (Map<String, Integer>) status.get("operation_stats");
        if (stats != null && !stats.isEmpty()) {
            System.out.println("• 操作统计:");
            stats.forEach((operation, count) -> 
                System.out.println("  - " + operation + ": " + count + " 次"));
        }
    }
    
    /**
     * 交互式演示
     */
    private static void runInteractiveDemo() {
        System.out.println("\n📋 3. 交互式演示");
        System.out.println("-".repeat(40));
        System.out.println("输入 'help' 查看可用命令，输入 'quit' 退出");
        
        while (true) {
            System.out.print("\n> ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                break;
            }
            
            if (input.equalsIgnoreCase("help")) {
                System.out.println(cursor.getHelp());
                continue;
            }
            
            if (input.startsWith("analyze:")) {
                String code = input.substring(8).trim();
                if (!code.isEmpty()) {
                    Map<String, Object> result = cursor.analyzeCode(code);
                    System.out.println("分析完成，语法有效: " + result.get("syntax_valid"));
                } else {
                    System.out.println("请提供要分析的代码");
                }
                continue;
            }
            
            if (input.startsWith("generate:")) {
                String request = input.substring(9).trim();
                if (!request.isEmpty()) {
                    String code = cursor.generateCode(request);
                    System.out.println("生成的代码:\n" + code.substring(0, Math.min(300, code.length())) + "...");
                } else {
                    System.out.println("请提供生成请求");
                }
                continue;
            }
            
            if (input.startsWith("chat:")) {
                String message = input.substring(5).trim();
                if (!message.isEmpty()) {
                    String response = cursor.chat(message);
                    System.out.println("AI回复: " + response);
                } else {
                    System.out.println("请提供对话内容");
                }
                continue;
            }
            
            if (input.equalsIgnoreCase("status")) {
                Map<String, Object> status = cursor.getSystemStatus();
                System.out.println("系统状态: " + status.get("name") + " - 运行 " + status.get("uptime_minutes") + " 分钟");
                continue;
            }
            
            if (input.equalsIgnoreCase("clear")) {
                cursor.clearSessionHistory();
                continue;
            }
            
            // 默认当作对话处理
            if (!input.isEmpty()) {
                String response = cursor.chat(input);
                System.out.println("AI回复: " + response);
            }
        }
    }
    
    /**
     * 创建示例用户类（用于演示）
     */
    public static class User {
        private Long id;
        private String name;
        private String email;
        
        // 构造函数
        public User() {}
        
        // Getter和Setter方法
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
    }
}