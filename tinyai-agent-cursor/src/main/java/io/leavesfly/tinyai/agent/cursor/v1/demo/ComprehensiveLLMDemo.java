package io.leavesfly.tinyai.agent.cursor.v1.demo;

import io.leavesfly.tinyai.agent.cursor.v1.AICodingCursor;
import io.leavesfly.tinyai.agent.cursor.v1.RefactorSuggestion;

import java.util.List;
import java.util.Map;



/**
 * 综合LLM增强演示程序
 * 全面展示所有组件的LLM增强功能，包括分析、生成、重构、调试等
 * 
 * @author 山泽
 */
public class ComprehensiveLLMDemo {
    
    public static void main(String[] args) {
        System.out.println("🚀 TinyAI Cursor 综合LLM增强功能演示");
        System.out.println("========================================");
        
        // 创建完全LLM增强的AI编程助手
        AICodingCursor cursor = new AICodingCursor("Comprehensive LLM Enhanced Cursor");
        
        // 演示用的复杂代码样例
        String complexCode = """
            import java.util.*;
            
            public class UserManager {
                private Map<String, User> users = new HashMap<>();
                private List<String> logs = new ArrayList<>();
                
                public void addUser(String name, String email, int age) {
                    if (name == null || email == null) {
                        System.out.println("Invalid input");
                        return;
                    }
                    if (name.length() < 2) {
                        System.out.println("Name too short");
                        return;
                    }
                    if (!email.contains("@")) {
                        System.out.println("Invalid email");
                        return;
                    }
                    if (age < 0 || age > 150) {
                        System.out.println("Invalid age");
                        return;
                    }
                    
                    User user = new User();
                    user.name = name;
                    user.email = email;
                    user.age = age;
                    user.id = UUID.randomUUID().toString();
                    user.createdAt = new Date();
                    user.isActive = true;
                    
                    users.put(user.id, user);
                    logs.add("User added: " + name + " at " + new Date());
                    System.out.println("User successfully added");
                }
                
                public User findUser(String id) {
                    if (id == null) return null;
                    return users.get(id);
                }
                
                public List<User> getAllUsers() {
                    List<User> result = new ArrayList<>();
                    for (String key : users.keySet()) {
                        User user = users.get(key);
                        if (user != null) {
                            result.add(user);
                        }
                    }
                    return result;
                }
                
                public boolean updateUser(String id, String name, String email, int age) {
                    User user = users.get(id);
                    if (user == null) {
                        System.out.println("User not found");
                        return false;
                    }
                    
                    // 重复的验证逻辑
                    if (name == null || email == null) {
                        System.out.println("Invalid input");
                        return false;
                    }
                    if (name.length() < 2) {
                        System.out.println("Name too short");
                        return false;
                    }
                    if (!email.contains("@")) {
                        System.out.println("Invalid email");
                        return false;
                    }
                    if (age < 0 || age > 150) {
                        System.out.println("Invalid age");
                        return false;
                    }
                    
                    user.name = name;
                    user.email = email;
                    user.age = age;
                    user.updatedAt = new Date();
                    
                    logs.add("User updated: " + name + " at " + new Date());
                    System.out.println("User successfully updated");
                    return true;
                }
                
                static class User {
                    String id;
                    String name;
                    String email;
                    int age;
                    Date createdAt;
                    Date updatedAt;
                    boolean isActive;
                }
            }
            """;
        
        // 1. 综合代码分析演示
        demonstrateComprehensiveAnalysis(cursor, complexCode);
        
        // 2. 智能代码生成演示
        demonstrateIntelligentGeneration(cursor);
        
        // 3. 智能重构建议演示
        demonstrateSmartRefactoring(cursor, complexCode);
        
        // 4. 智能错误诊断演示
        demonstrateSmartDebugging(cursor);
        
        // 5. 综合代码审查演示
        demonstrateComprehensiveReview(cursor, complexCode);
        
        // 6. LLM增强对话演示
        demonstrateLLMEnhancedChat(cursor);
        
        // 7. 系统状态和性能展示
        displaySystemPerformance(cursor);
        
        System.out.println("\n🎉 综合LLM增强功能演示完成！");
        System.out.println("✨ 所有组件都已成功集成LLM增强功能，大幅提升了智能化水平！");
    }
    
    /**
     * 演示综合代码分析功能
     */
    private static void demonstrateComprehensiveAnalysis(AICodingCursor cursor, String code) {
        System.out.println("\n📊 === 综合代码分析演示 ===");
        
        // LLM增强的代码分析
        System.out.println("\n🔍 执行LLM增强代码分析...");
        Map<String, Object> analysis = cursor.analyzeCode(code);
        
        // 显示传统分析结果
        System.out.println("\n📋 传统分析结果:");
        System.out.println("- 语法有效性: " + analysis.get("syntax_valid"));
        System.out.println("- 复杂度: " + analysis.get("complexity"));
        System.out.println("- 类数量: " + ((List<?>) analysis.get("classes")).size());
        System.out.println("- 方法数量: " + ((List<?>) analysis.get("methods")).size());
        
        // 显示LLM增强分析结果
        if (analysis.containsKey("llm_analysis")) {
            System.out.println("\n🤖 LLM智能分析:");
            System.out.println(analysis.get("llm_analysis"));
        }
        
        if (analysis.containsKey("llm_suggestions")) {
            System.out.println("\n💡 LLM智能建议:");
            System.out.println(analysis.get("llm_suggestions"));
        }
        
        // 使用CodeAnalyzer的新功能
        System.out.println("\n📈 生成智能分析报告...");
        String report = cursor.getAnalyzer().generateSmartAnalysisReport(code);
        System.out.println(report);
    }
    
    /**
     * 演示智能代码生成功能
     */
    private static void demonstrateIntelligentGeneration(AICodingCursor cursor) {
        System.out.println("\n🤖 === 智能代码生成演示 ===");
        
        String[] requests = {
            "创建一个线程安全的缓存类",
            "生成一个用户认证的方法",
            "实现一个简单的观察者模式",
            "创建一个数据校验工具类"
        };
        
        for (String request : requests) {
            System.out.println("\n📝 生成请求: " + request);
            String generatedCode = cursor.generateCode(request);
            System.out.println("生成的代码:");
            System.out.println(generatedCode.substring(0, Math.min(200, generatedCode.length())) + "...");
        }
        
        // 演示CodeGenerator的新增功能
        System.out.println("\n🎯 演示代码建议功能...");
        String suggestion = cursor.getGenerator().generateCodeSuggestion(
            "需要处理大量数据的场景", 
            "实现一个高性能的数据处理器");
        System.out.println("智能建议:");
        System.out.println(suggestion.substring(0, Math.min(300, suggestion.length())) + "...");
    }
    
    /**
     * 演示智能重构建议功能
     */
    private static void demonstrateSmartRefactoring(AICodingCursor cursor, String code) {
        System.out.println("\n🔧 === 智能重构建议演示 ===");
        
        System.out.println("\n🔍 分析重构机会...");
        List<RefactorSuggestion> suggestions = cursor.suggestRefactor(code);
        
        System.out.println("发现 " + suggestions.size() + " 个重构建议:");
        
        int count = 0;
        for (RefactorSuggestion suggestion : suggestions) {
            if (count >= 5) break; // 只显示前5个建议
            
            System.out.println("\n" + (count + 1) + ". " + suggestion.getDescription());
            System.out.println("   类型: " + suggestion.getSuggestionType());
            System.out.println("   优先级: " + suggestion.getPriority());
            if (suggestion.getBenefits() != null && !suggestion.getBenefits().isEmpty()) {
                System.out.println("   收益: " + String.join(", ", suggestion.getBenefits()));
            }
            count++;
        }
    }
    
    /**
     * 演示智能错误诊断功能
     */
    private static void demonstrateSmartDebugging(AICodingCursor cursor) {
        System.out.println("\n🐛 === 智能错误诊断演示 ===");
        
        // 故意有问题的代码
        String buggyCode = """
            public class Calculator {
                public int divide(int a, int b) {
                    return a / b;  // 可能除零错误
                }
                
                public String processData(String[] data) {
                    String result = null;
                    for (int i = 0; i <= data.length; i++) {  // 越界错误
                        result += data[i];
                    }
                    return result;
                }
            }
            """;
        
        // 诊断不同类型的错误
        String[] errorMessages = {
            "ArithmeticException: / by zero",
            "ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 5",
            "NullPointerException"
        };
        
        for (String errorMsg : errorMessages) {
            System.out.println("\n🔍 诊断错误: " + errorMsg);
            Map<String, Object> debugResult = cursor.debugCode(buggyCode, errorMsg);
            
            if (debugResult.containsKey("llm_debug_advice")) {
                System.out.println("🤖 LLM智能诊断:");
                System.out.println(debugResult.get("llm_debug_advice"));
            }
            
            if (debugResult.containsKey("smart_solution")) {
                System.out.println("💡 智能解决方案:");
                String solution = (String) debugResult.get("smart_solution");
                System.out.println(solution.substring(0, Math.min(200, solution.length())) + "...");
            }
        }
    }
    
    /**
     * 演示综合代码审查功能
     */
    private static void demonstrateComprehensiveReview(AICodingCursor cursor, String code) {
        System.out.println("\n📋 === 综合代码审查演示 ===");
        
        System.out.println("\n🔍 执行综合代码审查...");
        Map<String, Object> review = cursor.reviewCode(code);
        
        System.out.println("📊 审查结果:");
        System.out.println("- 整体质量评分: " + review.get("overall_score"));
        
        if (review.containsKey("recommendations")) {
            @SuppressWarnings("unchecked")
            List<String> recommendations = (List<String>) review.get("recommendations");
            System.out.println("- 改进建议数量: " + recommendations.size());
            
            System.out.println("\n💡 主要建议:");
            for (int i = 0; i < Math.min(3, recommendations.size()); i++) {
                System.out.println("  " + (i + 1) + ". " + recommendations.get(i));
            }
        }
    }
    
    /**
     * 演示LLM增强对话功能
     */
    private static void demonstrateLLMEnhancedChat(AICodingCursor cursor) {
        System.out.println("\n💬 === LLM增强对话演示 ===");
        
        String[] questions = {
            "如何提高这段代码的性能？",
            "这个设计模式有什么优缺点？",
            "如何进行单元测试？",
            "有什么安全性考虑？"
        };
        
        for (String question : questions) {
            System.out.println("\n❓ 用户问题: " + question);
            String response = cursor.chat(question);
            System.out.println("🤖 AI回答: " + response.substring(0, Math.min(150, response.length())) + "...");
        }
    }
    
    /**
     * 显示系统性能和状态
     */
    private static void displaySystemPerformance(AICodingCursor cursor) {
        System.out.println("\n📈 === 系统性能和状态 ===");
        
        Map<String, Object> status = cursor.getSystemStatus();
        
        System.out.println("🔧 系统状态:");
        System.out.println("- 系统名称: " + status.get("name"));
        System.out.println("- 运行时间: " + status.get("uptime"));
        System.out.println("- LLM模型: " + cursor.getLLMSimulator().getModelName());
        
        System.out.println("\n📊 操作统计:");
        @SuppressWarnings("unchecked")
        Map<String, Integer> stats = (Map<String, Integer>) status.get("operation_stats");
        if (stats != null) {
            stats.forEach((operation, count) -> 
                System.out.println("- " + operation + ": " + count + " 次"));
        }
        
        // 显示各组件的LLM状态
        System.out.println("\n🤖 LLM集成状态:");
        System.out.println("- CodeAnalyzer: " + cursor.getAnalyzer().getLLMStatus().get("llm_enabled"));
        System.out.println("- CodeGenerator: " + cursor.getGenerator().getLLMStatus().get("llm_enabled"));
        System.out.println("- 缓存大小: " + cursor.getAnalyzer().getCacheSize());
    }
}