package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * DeepSeek R1 演示和测试类
 * 
 * 展示DeepSeek R1模型的各种功能，包括：
 * 1. 基础推理能力
 * 2. 思维链推理
 * 3. 自我反思
 * 4. 强化学习训练
 * 5. 多种推理策略
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TestR1 {
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek R1 演示开始 ===");
        
        try {
            // 运行所有演示
            demonstrateBasicReasoning();
            demonstrateChainOfThought();
            demonstrateReflection();
            demonstrateTraining();
            demonstrateBatchProcessing();
            
            System.out.println("\n=== 所有演示完成 ===");
            
        } catch (Exception e) {
            System.err.println("演示过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示基础推理能力
     */
    public static void demonstrateBasicReasoning() {
        System.out.println("\n=== 1. 基础推理演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model(
            "demo_r1", 
            1000,  // 词汇表大小
            256,   // 模型维度
            6      // Transformer层数
        );
        
        // 打印模型信息
        model.printModelInfo();
        
        // 创建示例输入
        NdArray inputIds = createSampleInput();
        
        // 执行推理
        System.out.println("执行推理...");
        DeepSeekR1Result result = model.performReasoning(inputIds, "演示问题：2 + 3 等于多少？");
        
        // 显示结果
        System.out.println("推理结果:");
        System.out.println(result.getSummary());
        
        if (result.getReasoningChain() != null) {
            System.out.println("\n推理链详情:");
            System.out.println(result.getReasoningChain().getSummary());
        }
        
        if (result.getReflectionResult() != null) {
            System.out.println("\n反思结果:");
            System.out.printf("质量分数: %.3f\n", result.getReflectionResult().getQualityScore());
            System.out.printf("置信度: %.3f\n", result.getFinalConfidence());
        }
    }
    
    /**
     * 演示思维链推理
     */
    public static void demonstrateChainOfThought() {
        System.out.println("\n=== 2. 思维链推理演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("cot_r1", 1000, 256, 4);
        
        // 创建思维链处理器
        ChainOfThoughtPrompting cotProcessor = new ChainOfThoughtPrompting(model);
        
        // 测试不同类型的问题
        String[] testQuestions = {
            "比较深度学习和机器学习的区别",
            "分析为什么深度学习在图像识别中效果很好",
            "如何解决过拟合问题？",
            "从几个例子中总结神经网络的特点"
        };
        
        for (String question : testQuestions) {
            System.out.printf("\n处理问题: %s\n", question);
            
            // 自动选择策略
            ChainOfThoughtPrompting.ReasoningStrategy strategy = cotProcessor.selectStrategy(question);
            System.out.printf("选择策略: %s\n", strategy);
            
            // 生成提示
            String prompt = cotProcessor.generateCoTPrompt(question, strategy);
            System.out.println("生成的提示:");
            System.out.println(prompt.substring(0, Math.min(200, prompt.length())) + "...");
            
            // 处理问题
            CoTProcessingResult result = cotProcessor.processWithCoT(question, strategy);
            System.out.println("处理结果: " + result.getSummary());
            System.out.println("效果评级: " + result.getEffectivenessRating());
        }
    }
    
    /**
     * 演示自我反思功能
     */
    public static void demonstrateReflection() {
        System.out.println("\n=== 3. 自我反思演示 ===");
        
        // 创建反思模块
        ReflectionModule reflectionModule = new ReflectionModule("demo_reflection", 256);
        
        // 创建模拟的推理输出和原始输入
        NdArray reasoningOutput = NdArray.randn(Shape.of(1, 256));
        NdArray originalInput = NdArray.randn(Shape.of(1, 256));
        
        // 执行反思
        System.out.println("执行自我反思...");
        ReflectionModule.ReflectionResult reflection = reflectionModule.performReflection(
            new io.leavesfly.tinyai.func.Variable(reasoningOutput),
            new io.leavesfly.tinyai.func.Variable(originalInput)
        );
        
        // 显示反思结果
        System.out.println("反思报告:");
        System.out.println(reflection.getReflectionReport());
    }
    
    /**
     * 演示强化学习训练
     */
    public static void demonstrateTraining() {
        System.out.println("\n=== 4. 强化学习训练演示 ===");
        
        // 创建模型和训练器
        DeepSeekR1Model model = new DeepSeekR1Model("train_r1", 1000, 128, 3);
        RLTrainer trainer = new RLTrainer(model, 0.001);
        
        // 创建训练数据
        TrainingData trainData = createTrainingData();
        
        System.out.printf("训练数据: %d 个样本\n", trainData.size());
        
        // 训练几个步骤进行演示
        System.out.println("\n开始训练演示（3个样本）...");
        for (int i = 0; i < Math.min(3, trainData.size()); i++) {
            TrainingData.Sample sample = trainData.getSample(i);
            
            TrainingMetrics metrics = trainer.trainStep(
                sample.getInputIds(),
                sample.getTargetIds(),
                sample.getQuestion()
            );
            
            System.out.printf("步骤 %d: %s\n", i + 1, metrics.getSummary());
        }
        
        System.out.println("训练演示完成");
        System.out.printf("训练器状态: %s\n", trainer.toString());
    }
    
    /**
     * 演示批量处理
     */
    public static void demonstrateBatchProcessing() {
        System.out.println("\n=== 5. 批量处理演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("batch_r1", 1000, 128, 3);
        ChainOfThoughtPrompting cotProcessor = new ChainOfThoughtPrompting(model);
        
        // 准备问题列表
        List<String> questions = Arrays.asList(
            "什么是人工智能？",
            "深度学习有什么优势？",
            "如何评估模型性能？"
        );
        
        // 批量处理
        List<CoTProcessingResult> results = cotProcessor.batchProcess(questions);
        
        // 分析结果
        System.out.println("\n批量处理分析:");
        int successCount = 0;
        double totalConfidence = 0.0;
        
        for (CoTProcessingResult result : results) {
            if (!result.isError()) {
                successCount++;
                totalConfidence += result.getFinalConfidence();
            }
            
            System.out.printf("  问题: %s\n", result.getOriginalQuestion());
            System.out.printf("  结果: %s\n", result.getEffectivenessRating());
            System.out.printf("  置信度: %.3f\n\n", result.getFinalConfidence());
        }
        
        System.out.printf("总体统计:\n");
        System.out.printf("  成功率: %.1f%% (%d/%d)\n", 
                        100.0 * successCount / results.size(), successCount, results.size());
        
        if (successCount > 0) {
            System.out.printf("  平均置信度: %.3f\n", totalConfidence / successCount);
        }
    }
    
    /**
     * 创建示例输入
     */
    private static NdArray createSampleInput() {
        // 创建一个简单的token序列
        NdArray input = NdArray.of(Shape.of(1, 10));
        
        // 填充一些示例token ID
        for (int i = 0; i < 10; i++) {
            input.set(i % 100, 0, i);  // 简单的重复模式
        }
        
        return input;
    }
    
    /**
     * 创建训练数据
     */
    private static TrainingData createTrainingData() {
        TrainingData data = new TrainingData();
        
        // 添加一些示例训练样本
        for (int i = 0; i < 5; i++) {
            NdArray input = NdArray.of(Shape.of(1, 8));
            NdArray target = NdArray.of(Shape.of(1, 8));
            
            // 填充示例数据
            for (int j = 0; j < 8; j++) {
                input.set((i * 10 + j) % 100, 0, j);
                target.set((i * 10 + j + 1) % 100, 0, j);
            }
            
            data.addSample(input, target, String.format("示例问题 %d", i + 1));
        }
        
        return data;
    }
    
    /**
     * 性能基准测试
     */
    public static void runBenchmark() {
        System.out.println("\n=== 性能基准测试 ===");
        
        // 测试不同规模的模型
        int[] modelSizes = {128, 256, 512};
        int[] layerCounts = {3, 6, 12};
        
        for (int dModel : modelSizes) {
            for (int numLayers : layerCounts) {
                System.out.printf("\n测试配置: dModel=%d, layers=%d\n", dModel, numLayers);
                
                long startTime = System.currentTimeMillis();
                
                // 创建模型
                DeepSeekR1Model model = new DeepSeekR1Model(
                    String.format("benchmark_%d_%d", dModel, numLayers),
                    1000, dModel, numLayers
                );
                
                // 执行推理
                NdArray input = createSampleInput();
                DeepSeekR1Result result = model.performReasoning(input, "基准测试问题");
                
                long endTime = System.currentTimeMillis();
                
                System.out.printf("  用时: %d ms\n", endTime - startTime);
                System.out.printf("  成功: %s\n", result.isSuccessful() ? "是" : "否");
                System.out.printf("  置信度: %.3f\n", result.getFinalConfidence());
                
                if (result.getReasoningChain() != null) {
                    System.out.printf("  推理步骤: %d\n", result.getReasoningChain().getStepCount());
                }
            }
        }
    }
    
    /**
     * 错误处理演示
     */
    public static void demonstrateErrorHandling() {
        System.out.println("\n=== 错误处理演示 ===");
        
        try {
            // 测试无效参数
            System.out.println("测试无效模型参数...");
            DeepSeekR1Model invalidModel = new DeepSeekR1Model("invalid", -1, 256, 6);
        } catch (IllegalArgumentException e) {
            System.out.println("成功捕获参数错误: " + e.getMessage());
        }
        
        try {
            // 测试过长序列
            System.out.println("测试过长输入序列...");
            DeepSeekR1Model model = new DeepSeekR1Model("test", 1000, 128, 3);
            NdArray longInput = NdArray.of(Shape.of(1, 1000));  // 超过maxSeqLength
            model.performReasoning(longInput, "测试问题");
        } catch (IllegalArgumentException e) {
            System.out.println("成功捕获序列长度错误: " + e.getMessage());
        }
        
        System.out.println("错误处理测试完成");
    }
}
