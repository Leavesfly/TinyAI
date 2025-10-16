package io.leavesfly.tinyai.agent.cursor.v1.demo;


import io.leavesfly.tinyai.agent.cursor.v1.AICodingCursor;
import io.leavesfly.tinyai.agent.cursor.v1.CursorLLMSimulator;

/**
 * LLM集成效果对比演示
 * 展示传统方式与LLM增强方式的差异
 * 
 * @author 山泽
 */
public class LLMComparisonDemo {
    
    public static void main(String[] args) {
        System.out.println("🔬 LLM集成效果对比演示");
        System.out.println("========================");
        
        // 创建增强版助手
        AICodingCursor enhancedCursor = new AICodingCursor("LLM Enhanced");
        
        // 创建LLM模拟器进行直接对比
        CursorLLMSimulator llmSimulator = new CursorLLMSimulator();
        
        String testCode = """
            public class Example {
                public void longMethod() {
                    System.out.println("Step 1");
                    System.out.println("Step 2");
                    System.out.println("Step 3");
                    // ... 很多代码行
                }
            }
            """;
        
        // 对比1：代码分析能力
        demonstrateAnalysisComparison(enhancedCursor, llmSimulator, testCode);
        
        // 对比2：错误诊断能力  
        demonstrateDiagnosisComparison(enhancedCursor, llmSimulator, testCode);
        
        // 对比3：代码生成能力
        demonstrateGenerationComparison(enhancedCursor, llmSimulator);
        
        System.out.println("\n✨ 对比总结：");
        System.out.println("- LLM增强版提供了更智能、更人性化的分析和建议");
        System.out.println("- 传统方式与LLM分析相结合，提高了准确性和实用性");
        System.out.println("- LLM模拟器为编程助手带来了类似真实AI的智能体验");
    }
    
    private static void demonstrateAnalysisComparison(AICodingCursor cursor, 
                                                     CursorLLMSimulator llm, String code) {
        System.out.println("\n📊 代码分析能力对比：");
        System.out.println("-------------------");
        
        // LLM增强分析
        System.out.println("🤖 LLM增强版分析：");
        var analysis = cursor.analyzeCode(code);
        System.out.println(analysis.get("llm_analysis"));
        
        System.out.println("\n🔍 纯LLM分析：");
        String llmAnalysis = llm.generateCodeAnalysis(code, "structure");
        System.out.println(llmAnalysis);
    }
    
    private static void demonstrateDiagnosisComparison(AICodingCursor cursor, 
                                                      CursorLLMSimulator llm, String code) {
        System.out.println("\n🐛 错误诊断能力对比：");
        System.out.println("-------------------");
        
        String errorMsg = "Method too long";
        
        // LLM增强诊断
        System.out.println("🤖 LLM增强版诊断：");
        var debugResult = cursor.debugCode(code, errorMsg);
        System.out.println(debugResult.get("llm_debug_advice"));
        
        System.out.println("\n🔍 纯LLM诊断：");
        String llmDebug = llm.generateDebugAdvice(code, errorMsg);
        System.out.println(llmDebug);
    }
    
    private static void demonstrateGenerationComparison(AICodingCursor cursor, 
                                                       CursorLLMSimulator llm) {
        System.out.println("\n🤖 代码生成能力对比：");
        System.out.println("-------------------");
        
        String request = "创建一个用户验证方法";
        
        // LLM增强生成
        System.out.println("🤖 LLM增强版生成：");
        String enhanced = cursor.generateCode(request);
        System.out.println(enhanced.substring(0, Math.min(200, enhanced.length())) + "...");
        
        System.out.println("\n🔍 纯LLM生成：");
        String llmGenerated = llm.generateCodeImplementation(request, "");
        System.out.println(llmGenerated);
    }
}