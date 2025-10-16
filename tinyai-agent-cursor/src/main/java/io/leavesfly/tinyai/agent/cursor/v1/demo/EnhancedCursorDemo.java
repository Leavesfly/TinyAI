package io.leavesfly.tinyai.agent.cursor.v1.demo;

import io.leavesfly.tinyai.agent.cursor.v1.AICodingCursor;
import io.leavesfly.tinyai.agent.cursor.v1.RefactorSuggestion;

import java.util.List;
import java.util.Map;



/**
 * LLM增强版Cursor演示程序
 * 展示基于LLM模拟器的智能编程助手功能
 * 
 * @author 山泽
 */
public class EnhancedCursorDemo {
    
    public static void main(String[] args) {
        System.out.println("🚀 欢迎使用 TinyAI Enhanced Cursor - LLM智能编程助手演示");
        System.out.println("========================================================");
        
        // 创建LLM增强版AI编程助手
        AICodingCursor cursor = new AICodingCursor("LLM Enhanced Cursor");
        
        // 演示代码样例
        String sampleCode = """
            public class Calculator {
                public int calculate(int a, int b, String operation) {
                    if (operation.equals("add")) {
                        return a + b;
                    } else if (operation.equals("subtract")) {
                        return a - b;
                    } else if (operation.equals("multiply")) {
                        return a * b;
                    } else if (operation.equals("divide")) {
                        if (b == 0) {
                            System.out.println("Error: Division by zero!");
                            return 0;
                        }
                        return a / b;
                    } else {
                        System.out.println("Unknown operation: " + operation);
                        return 0;
                    }
                }
                
                public void processLargeDataSet() {
                    // 这是一个很长的方法示例
                    System.out.println("Processing data...");
                    for (int i = 0; i < 1000; i++) {
                        if (i % 2 == 0) {
                            System.out.println("Even: " + i);
                        } else {
                            System.out.println("Odd: " + i);
                        }
                    }
                    System.out.println("Data processing complete");
                    // 更多重复的代码...
                }
            }
            """;
        
        // 演示1：LLM增强代码分析
        demonstrateCodeAnalysis(cursor, sampleCode);
        
        // 演示2：LLM智能代码生成
        demonstrateCodeGeneration(cursor);
        
        // 演示3：LLM智能重构建议
        demonstrateRefactorSuggestions(cursor, sampleCode);
        
        // 演示4：LLM智能错误诊断
        demonstrateErrorDiagnosis(cursor, sampleCode);
        
        // 演示5：LLM智能对话
        demonstrateIntelligentChat(cursor);
        
        // 显示系统状态
        displaySystemStatus(cursor);
        
        System.out.println("\n🎉 LLM增强版Cursor演示完成！");
        System.out.println("通过集成LLM模拟器，编程助手的智能化水平显著提升！");
    }
    
    /**
     * 演示LLM增强代码分析功能
     */
    private static void demonstrateCodeAnalysis(AICodingCursor cursor, String code) {
        System.out.println("\n📊 === LLM增强代码分析演示 ===");
        
        Map<String, Object> analysis = cursor.analyzeCode(code);
        
        System.out.println("\n🔍 基础分析结果：");
        System.out.println("- 语法有效性: " + analysis.get("syntax_valid"));
        System.out.println("- 复杂度: " + analysis.get("complexity"));
        
        System.out.println("\n🤖 LLM智能分析：");
        System.out.println(analysis.get("llm_analysis"));
        
        System.out.println("\n💡 智能改进建议：");
        System.out.println(analysis.get("smart_suggestions"));
    }
    
    /**
     * 演示LLM智能代码生成功能
     */
    private static void demonstrateCodeGeneration(AICodingCursor cursor) {
        System.out.println("\n🤖 === LLM智能代码生成演示 ===");
        
        String[] requests = {
            "创建一个用户管理类，包含添加、删除、查询用户的方法",
            "生成一个处理文件上传的方法",
            "实现一个简单的缓存机制"
        };
        
        for (String request : requests) {
            System.out.println("\n📝 生成请求: " + request);
            System.out.println("生成结果:");
            String generatedCode = cursor.generateCode(request);
            
            // 显示LLM生成的部分（前几行）
            String[] lines = generatedCode.split("\n");
            for (int i = 0; i < Math.min(10, lines.length); i++) {
                System.out.println(lines[i]);
            }
            if (lines.length > 10) {
                System.out.println("... (省略更多内容)");
            }
            System.out.println();
        }
    }
    
    /**
     * 演示LLM智能重构建议功能
     */
    private static void demonstrateRefactorSuggestions(AICodingCursor cursor, String code) {
        System.out.println("\n🔧 === LLM智能重构建议演示 ===");
        
        List<RefactorSuggestion> suggestions = cursor.suggestRefactor(code);
        
        System.out.println("发现 " + suggestions.size() + " 个重构建议：\n");
        
        for (int i = 0; i < Math.min(3, suggestions.size()); i++) {
            RefactorSuggestion suggestion = suggestions.get(i);
            System.out.println((i + 1) + ". " + suggestion.getDescription());
            System.out.println("   优先级: " + suggestion.getPriority());
            System.out.println("   收益: " + suggestion.getBenefitsSummary());
            System.out.println();
        }
    }
    
    /**
     * 演示LLM智能错误诊断功能
     */
    private static void demonstrateErrorDiagnosis(AICodingCursor cursor, String code) {
        System.out.println("\n🐛 === LLM智能错误诊断演示 ===");
        
        // 模拟一个包含错误的代码
        String buggyCode = """
            public class BuggyExample {
                public void processArray(int[] arr) {
                    for (int i = 0; i <= arr.length; i++) {  // 数组越界风险
                        System.out.println(arr[i]);
                    }
                }
                
                public String processText(String text) {
                    return text.toUpperCase();  // 空指针风险
                }
            }
            """;
        
        Map<String, Object> debugResult = cursor.debugCode(buggyCode, "ArrayIndexOutOfBoundsException");
        
        System.out.println("🔍 错误诊断结果：");
        System.out.println("- 发现错误: " + debugResult.get("error_found"));
        System.out.println("- 错误类型: " + debugResult.get("error_type"));
        System.out.println("- 置信度: " + debugResult.get("confidence"));
        
        System.out.println("\n🤖 LLM诊断分析：");
        System.out.println(debugResult.get("llm_diagnosis"));
        
        System.out.println("\n💡 智能修复建议：");
        System.out.println(debugResult.get("smart_solution"));
    }
    
    /**
     * 演示LLM智能对话功能
     */
    private static void demonstrateIntelligentChat(AICodingCursor cursor) {
        System.out.println("\n💬 === LLM智能对话演示 ===");
        
        String[] questions = {
            "如何优化Java代码的性能？",
            "什么是设计模式，能举个例子吗？",
            "如何处理并发编程中的线程安全问题？"
        };
        
        for (String question : questions) {
            System.out.println("\n❓ 用户问题: " + question);
            String response = cursor.chat(question);
            System.out.println("🤖 AI回答: " + response);
        }
    }
    
    /**
     * 显示系统状态
     */
    private static void displaySystemStatus(AICodingCursor cursor) {
        System.out.println("\n📊 === 系统状态报告 ===");
        
        Map<String, Object> status = cursor.getSystemStatus();
        System.out.println("系统名称: " + status.get("name"));
        System.out.println("运行时间: " + status.get("uptime_minutes") + " 分钟");
        System.out.println("执行操作数: " + status.get("session_operations"));
        System.out.println("LLM模型: " + cursor.getLLMSimulator().getModelName());
        System.out.println("缓存大小: " + status.get("cache_size"));
        
        System.out.println("\n📈 操作统计:");
        Map<String, Integer> stats = cursor.getOperationStats();
        stats.forEach((operation, count) -> 
            System.out.println("- " + operation + ": " + count + " 次"));
        
        System.out.println("\n📝 最近操作历史:");
        List<String> history = cursor.getSessionHistory();
        int recentCount = Math.min(3, history.size());
        for (int i = history.size() - recentCount; i < history.size(); i++) {
            System.out.println("- " + history.get(i));
        }
    }
}