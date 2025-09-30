package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nlp.SimpleTokenizer;

/**
 * DeepSeek-R1 模型演示类
 * 
 * 这个演示类展示了如何使用DeepSeek-R1模型进行各种推理任务，
 * 包括数学问题求解、逻辑推理、代码生成等。
 * 
 * 演示功能：
 * 1. 基础推理能力展示
 * 2. 思维链推理过程展示
 * 3. 多步推理演示
 * 4. 不同模型规模对比
 * 5. 推理质量评估
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Demo {
    
    private SimpleTokenizer tokenizer;
    
    public DeepSeekR1Demo() {
        this.tokenizer = new SimpleTokenizer();
    }
    
    /**
     * 运行所有演示
     */
    public static void main(String[] args) {
        DeepSeekR1Demo demo = new DeepSeekR1Demo();
        
        System.out.println("=== DeepSeek-R1 模型演示开始 ===\n");
        
        try {
            // 1. 基础功能演示
            demo.demonstrateBasicFunctionality();
            
            // 2. 数学推理演示
            demo.demonstrateMathReasoning();
            
            // 3. 逻辑推理演示
            demo.demonstrateLogicalReasoning();
            
            // 4. 思维链演示
            demo.demonstrateChainOfThought();
            
            // 5. 模型对比演示
            demo.demonstrateModelComparison();
            
            // 6. 推理质量评估
            demo.demonstrateReasoningQuality();
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== DeepSeek-R1 模型演示结束 ===");
    }
    
    /**
     * 演示基础功能
     */
    public void demonstrateBasicFunctionality() {
        System.out.println("=== 1. 基础功能演示 ===");
        
        // 创建调试模型进行快速演示
        DeepSeekR1Model model = DeepSeekR1Factory.createDebugModel("demo_basic");
        
        // 打印模型信息
        model.printModelInfo();
        
        // 创建简单输入
        String inputText = "什么是人工智能？";
        System.out.println("输入问题: " + inputText);
        
        // 转换为token ID
        int[] tokenIds = tokenizer.encode(inputText);
        NdArray input = createInputArray(tokenIds);
        
        // 执行推理
        System.out.println("执行推理...");
        ReasoningResult result = model.performReasoning(input);
        
        // 显示结果
        System.out.println("推理结果:");
        System.out.println(result.getDetailedReport());
        
        System.out.println("✓ 基础功能演示完成\n");
    }
    
    /**
     * 演示数学推理
     */
    public void demonstrateMathReasoning() {
        System.out.println("=== 2. 数学推理演示 ===");
        
        // 创建数学推理优化模型
        DeepSeekR1Model model = DeepSeekR1Factory.createMathReasoningModel("demo_math");
        
        // 数学问题
        String mathProblem = "如果一个数的3倍加5等于20，这个数是多少？";
        System.out.println("数学问题: " + mathProblem);
        
        int[] tokenIds = tokenizer.encode(mathProblem);
        NdArray input = createInputArray(tokenIds);
        
        // 启用推理模式
        model.setReasoningMode(true);
        
        System.out.println("执行数学推理...");
        ReasoningResult result = model.performReasoning(input);
        
        System.out.println("数学推理结果:");
        System.out.println(result.getFormattedReasoningSteps());
        System.out.printf("置信度: %.4f\n", result.getConfidenceScore());
        System.out.printf("推理步骤数: %d\n", result.getNumReasoningSteps());
        
        System.out.println("✓ 数学推理演示完成\n");
    }
    
    /**
     * 演示逻辑推理
     */
    public void demonstrateLogicalReasoning() {
        System.out.println("=== 3. 逻辑推理演示 ===");
        
        DeepSeekR1Model model = DeepSeekR1Factory.createReasoningModel("demo_logic");
        
        // 逻辑推理问题
        String logicProblem = "所有鸟都有羽毛，企鹅是鸟，那么企鹅有羽毛吗？";
        System.out.println("逻辑问题: " + logicProblem);
        
        int[] tokenIds = tokenizer.encode(logicProblem);
        NdArray input = createInputArray(tokenIds);
        
        System.out.println("执行逻辑推理...");
        ReasoningResult result = model.performReasoning(input);
        
        if (result.isSuccess()) {
            System.out.println("逻辑推理成功!");
            System.out.println("推理过程:");
            for (int i = 0; i < result.getReasoningSteps().size(); i++) {
                System.out.printf("  步骤 %d: %s\n", i + 1, result.getReasoningSteps().get(i));
            }
        } else {
            System.out.println("逻辑推理失败: " + result.getErrorMessage());
        }
        
        System.out.println("✓ 逻辑推理演示完成\n");
    }
    
    /**
     * 演示思维链推理过程
     */
    public void demonstrateChainOfThought() {
        System.out.println("=== 4. 思维链推理演示 ===");
        
        DeepSeekR1Model model = DeepSeekR1Factory.createMediumModel("demo_cot");
        
        // 复杂问题
        String complexProblem = "一个正方形的边长是5cm，如果我们在这个正方形内画一个最大的圆，这个圆的面积是多少？";
        System.out.println("复杂问题: " + complexProblem);
        
        int[] tokenIds = tokenizer.encode(complexProblem);
        NdArray input = createInputArray(tokenIds);
        
        System.out.println("启动思维链推理...");
        long startTime = System.currentTimeMillis();
        
        ReasoningResult result = model.performReasoning(input);
        
        long endTime = System.currentTimeMillis();
        result.setReasoningTimeMs(endTime - startTime);
        
        System.out.println("思维链推理完成!");
        System.out.println(result.getDetailedReport());
        
        // 分析推理质量
        analyzeReasoningQuality(result);
        
        System.out.println("✓ 思维链推理演示完成\n");
    }
    
    /**
     * 演示不同模型规模对比
     */
    public void demonstrateModelComparison() {
        System.out.println("=== 5. 模型规模对比演示 ===");
        
        // 创建不同规模的模型
        DeepSeekR1Model tinyModel = DeepSeekR1Factory.createTinyModel("demo_tiny");
        DeepSeekR1Model mediumModel = DeepSeekR1Factory.createMediumModel("demo_medium");
        
        String testProblem = "解释机器学习的基本概念";
        System.out.println("测试问题: " + testProblem);
        
        int[] tokenIds = tokenizer.encode(testProblem);
        NdArray input = createInputArray(tokenIds);
        
        // 测试小型模型
        System.out.println("\n--- 小型模型测试 ---");
        testModel(tinyModel, input, "Tiny");
        
        // 测试中型模型
        System.out.println("\n--- 中型模型测试 ---");
        testModel(mediumModel, input, "Medium");
        
        System.out.println("✓ 模型对比演示完成\n");
    }
    
    /**
     * 演示推理质量评估
     */
    public void demonstrateReasoningQuality() {
        System.out.println("=== 6. 推理质量评估演示 ===");
        
        DeepSeekR1Model model = DeepSeekR1Factory.createReasoningModel("demo_quality");
        
        String[] testQuestions = {
            "2 + 2 = ?",
            "解释量子物理的基本原理",
            "编写一个计算斐波那契数列的函数",
            "分析全球变暖的主要原因"
        };
        
        System.out.println("对多个问题进行推理质量评估...\n");
        
        for (int i = 0; i < testQuestions.length; i++) {
            String question = testQuestions[i];
            System.out.printf("问题 %d: %s\n", i + 1, question);
            
            int[] tokenIds = tokenizer.encode(question);
            NdArray input = createInputArray(tokenIds);
            
            ReasoningResult result = model.performReasoning(input);
            
            System.out.printf("  推理质量评分: %.2f/10\n", result.getConfidenceScore() * 10);
            System.out.printf("  推理步骤数: %d\n", result.getNumReasoningSteps());
            System.out.printf("  推理状态: %s\n", result.isSuccess() ? "成功" : "失败");
            System.out.println();
        }
        
        System.out.println("✓ 推理质量评估演示完成\n");
    }
    
    /**
     * 测试单个模型
     */
    private void testModel(DeepSeekR1Model model, NdArray input, String modelType) {
        long startTime = System.currentTimeMillis();
        
        try {
            ReasoningResult result = model.performReasoning(input);
            long endTime = System.currentTimeMillis();
            
            System.out.printf("%s模型性能:\n", modelType);
            System.out.printf("  推理时间: %d ms\n", endTime - startTime);
            System.out.printf("  推理步骤: %d\n", result.getNumReasoningSteps());
            System.out.printf("  置信度: %.4f\n", result.getConfidenceScore());
            System.out.printf("  状态: %s\n", result.isSuccess() ? "成功" : "失败");
            
        } catch (Exception e) {
            System.out.printf("%s模型测试失败: %s\n", modelType, e.getMessage());
        }
    }
    
    /**
     * 分析推理质量
     */
    private void analyzeReasoningQuality(ReasoningResult result) {
        System.out.println("\n--- 推理质量分析 ---");
        
        double confidence = result.getConfidenceScore();
        int steps = result.getNumReasoningSteps();
        
        // 质量评级
        String qualityRating;
        if (confidence >= 0.9) {
            qualityRating = "优秀";
        } else if (confidence >= 0.7) {
            qualityRating = "良好";
        } else if (confidence >= 0.5) {
            qualityRating = "一般";
        } else {
            qualityRating = "较差";
        }
        
        System.out.printf("推理质量评级: %s (%.1f%%)\n", qualityRating, confidence * 100);
        System.out.printf("推理效率: %s\n", steps <= 5 ? "高效" : steps <= 10 ? "中等" : "复杂");
        System.out.printf("推理深度: %d 步\n", steps);
        
        // 推理建议
        if (confidence < 0.7) {
            System.out.println("建议: 考虑增加推理步骤或调整推理阈值");
        }
        if (steps > 15) {
            System.out.println("建议: 推理步骤较多，可能需要优化问题分解策略");
        }
    }
    
    /**
     * 创建输入数组
     */
    private NdArray createInputArray(int[] tokenIds) {
        // 创建批大小为1的输入
        NdArray input = NdArray.of(Shape.of(1, tokenIds.length));
        
        for (int i = 0; i < tokenIds.length; i++) {
            input.set(tokenIds[i], 0, i);
        }
        
        return input;
    }
    
    /**
     * 交互式演示
     * 允许用户输入问题进行实时推理
     */
    public void runInteractiveDemo() {
        System.out.println("=== DeepSeek-R1 交互式演示 ===");
        System.out.println("输入问题进行推理（输入 'quit' 退出）:");
        
        DeepSeekR1Model model = DeepSeekR1Factory.createMediumModel("interactive");
        
        try (java.util.Scanner scanner = new java.util.Scanner(System.in)) {
            while (true) {
                System.out.print("\n问题: ");
                String question = scanner.nextLine();
                
                if ("quit".equalsIgnoreCase(question.trim())) {
                    break;
                }
                
                if (question.trim().isEmpty()) {
                    continue;
                }
                
                try {
                    int[] tokenIds = tokenizer.encode(question);
                    NdArray input = createInputArray(tokenIds);
                    
                    System.out.println("推理中...");
                    ReasoningResult result = model.performReasoning(input);
                    
                    System.out.println("\n推理结果:");
                    System.out.println(result.getFormattedReasoningSteps());
                    System.out.printf("置信度: %.2f%%\n", result.getConfidenceScore() * 100);
                    
                } catch (Exception e) {
                    System.out.println("推理失败: " + e.getMessage());
                }
            }
        }
        
        System.out.println("交互式演示结束");
    }
    
    /**
     * 基准测试
     * 对模型进行性能基准测试
     */
    public void runBenchmarkTest() {
        System.out.println("=== DeepSeek-R1 基准测试 ===");
        
        String[] testCases = {
            "1 + 1 = ?",
            "什么是深度学习？",
            "解释相对论的基本概念",
            "编写冒泡排序算法",
            "分析人工智能的发展趋势"
        };
        
        DeepSeekR1Model[] models = {
            DeepSeekR1Factory.createTinyModel("bench_tiny"),
            DeepSeekR1Factory.createMediumModel("bench_medium"),
            DeepSeekR1Factory.createReasoningModel("bench_reasoning")
        };
        
        String[] modelNames = {"Tiny", "Medium", "Reasoning"};
        
        System.out.printf("%-15s %-10s %-10s %-10s %-10s\n", 
                         "Model", "Avg Time", "Avg Steps", "Avg Conf", "Success Rate");
        // 打印分隔线
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 65; i++) {
            sb.append("-");
        }
        System.out.println(sb.toString());
        
        for (int m = 0; m < models.length; m++) {
            DeepSeekR1Model model = models[m];
            String modelName = modelNames[m];
            
            long totalTime = 0;
            int totalSteps = 0;
            double totalConfidence = 0;
            int successCount = 0;
            
            for (String testCase : testCases) {
                try {
                    int[] tokenIds = tokenizer.encode(testCase);
                    NdArray input = createInputArray(tokenIds);
                    
                    long startTime = System.currentTimeMillis();
                    ReasoningResult result = model.performReasoning(input);
                    long endTime = System.currentTimeMillis();
                    
                    totalTime += (endTime - startTime);
                    totalSteps += result.getNumReasoningSteps();
                    totalConfidence += result.getConfidenceScore();
                    if (result.isSuccess()) successCount++;
                    
                } catch (Exception e) {
                    // 测试失败，继续下一个
                }
            }
            
            double avgTime = totalTime / (double) testCases.length;
            double avgSteps = totalSteps / (double) testCases.length;
            double avgConfidence = totalConfidence / testCases.length;
            double successRate = successCount / (double) testCases.length;
            
            System.out.printf("%-15s %-10.1f %-10.1f %-10.3f %-10.1f%%\n",
                             modelName, avgTime, avgSteps, avgConfidence, successRate * 100);
        }
        
        System.out.println("\n✓ 基准测试完成");
    }
}