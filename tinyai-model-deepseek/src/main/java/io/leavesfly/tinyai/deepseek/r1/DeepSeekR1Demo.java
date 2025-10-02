package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek R1演示程序
 * 
 * 该演示程序展示了DeepSeek R1模型的主要功能，包括：
 * 1. 模型创建和配置
 * 2. 基础推理功能
 * 3. 推理细节展示
 * 4. 思维链推理演示
 * 5. 文本生成功能
 * 6. 模型统计信息
 * 
 * 这是一个完整的使用示例，展示了如何在TinyAI框架中使用DeepSeek R1
 */
public class DeepSeekR1Demo {
    
    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("    DeepSeek R1 TinyAI实现演示");
        System.out.println("========================================\n");
        
        try {
            // 1. 创建和配置模型
            demonstrateModelCreation();
            
            // 2. 基础推理演示
            demonstrateBasicInference();
            
            // 3. 详细推理演示
            demonstrateDetailedInference();
            
            // 4. 思维链推理演示
            demonstrateChainOfThought();
            
            // 5. 文本生成演示
            demonstrateTextGeneration();
            
            // 6. 模型统计演示
            demonstrateModelStatistics();
            
            System.out.println("\n========================================");
            System.out.println("         演示程序执行完成");
            System.out.println("========================================");
            
        } catch (Exception e) {
            System.err.println("演示程序执行出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示模型创建和配置
     */
    private static void demonstrateModelCreation() {
        System.out.println("=== 1. 模型创建和配置演示 ===");
        
        // 创建DeepSeek R1模型
        DeepSeekR1Model model = new DeepSeekR1Model(
            "DeepSeekR1-Demo",  // 模型名称
            1000,               // 词汇表大小
            256,                // 模型维度
            4,                  // Transformer层数
            8,                  // 注意力头数
            1024,               // 前馈网络维度
            128,                // 最大序列长度
            0.1                 // Dropout比率
        );
        
        // 验证配置
        boolean isValid = model.validateConfiguration();
        System.out.println("模型配置验证: " + (isValid ? "通过" : "失败"));
        
        // 打印模型详细信息
        model.printModelDetails();
        
        System.out.println("模型创建完成!\n");
    }
    
    /**
     * 演示基础推理功能
     */
    private static void demonstrateBasicInference() {
        System.out.println("=== 2. 基础推理功能演示 ===");
        
        // 创建简化模型进行快速演示
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeekR1-Basic", 100, 64);
        
        // 准备测试输入
        NdArray inputIds = createSampleInput(5); // 创建长度为5的示例输入
        
        System.out.println("输入序列: " + arrayToString(inputIds));
        
        // 执行推理
        io.leavesfly.tinyai.func.Variable output = model.inference(inputIds);
        
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println("推理输出前3个logits: " + getFirstLogits(output.getValue(), 3));
        
        System.out.println("基础推理演示完成!\n");
    }
    
    /**
     * 演示详细推理功能
     */
    private static void demonstrateDetailedInference() {
        System.out.println("=== 3. 详细推理功能演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeekR1-Detailed", 100, 64);
        
        // 准备输入
        NdArray inputIds = createSampleInput(8);
        System.out.println("输入序列: " + arrayToString(inputIds));
        
        // 执行详细推理
        DeepSeekR1Block.DeepSeekR1Result result = model.inferenceWithDetails(inputIds, null);
        
        // 显示推理结果
        System.out.println("推理结果:");
        System.out.println("- Logits形状: " + result.getLogits().getValue().getShape());
        System.out.println("- 推理输出形状: " + result.getReasoningOutput().getValue().getShape());
        System.out.println("- Transformer输出形状: " + result.getTransformerOutput().getValue().getShape());
        
        // 显示反思结果
        ReflectionBlock.ReflectionResult reflection = result.getReflectionResult();
        System.out.println("\n反思结果:");
        System.out.println("- 质量分数: " + String.format("%.3f", reflection.getQualityScore()));
        System.out.println("- 需要改进: " + (reflection.needsRefinement() ? "是" : "否"));
        System.out.println("- 质量描述: " + reflection.getQualityDescription());
        
        System.out.println("详细推理演示完成!\n");
    }
    
    /**
     * 演示思维链推理
     */
    private static void demonstrateChainOfThought() {
        System.out.println("=== 4. 思维链推理演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeekR1-CoT", 100, 64);
        
        // 准备输入token序列
        List<Integer> inputTokens = Arrays.asList(1, 15, 23, 8, 42, 7, 19);
        System.out.println("输入token序列: " + inputTokens);
        
        // 执行思维链推理
        DeepSeekR1Model.ChainOfThoughtResult cotResult = 
            model.chainOfThoughtReasoning(inputTokens, 5);
        
        // 显示思维链过程
        cotResult.printChainOfThought();
        
        System.out.println("思维链推理演示完成!\n");
    }
    
    /**
     * 演示文本生成功能
     */
    private static void demonstrateTextGeneration() {
        System.out.println("=== 5. 文本生成功能演示 ===");
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeekR1-Generate", 100, 64);
        
        // 准备种子序列
        List<Integer> seedTokens = Arrays.asList(1, 5, 10, 15);
        System.out.println("种子token序列: " + seedTokens);
        
        // 生成文本
        List<Integer> generatedTokens = model.generateText(seedTokens, 10);
        System.out.println("生成的token序列: " + generatedTokens);
        
        // 显示生成统计
        System.out.println("原始长度: " + seedTokens.size());
        System.out.println("生成后长度: " + generatedTokens.size());
        System.out.println("新增token数: " + (generatedTokens.size() - seedTokens.size()));
        
        System.out.println("文本生成演示完成!\n");
    }
    
    /**
     * 演示模型统计信息
     */
    private static void demonstrateModelStatistics() {
        System.out.println("=== 6. 模型统计信息演示 ===");
        
        // 创建不同规模的模型进行比较
        DeepSeekR1Model smallModel = new DeepSeekR1Model("Small-Model", 500, 128);
        DeepSeekR1Model largeModel = new DeepSeekR1Model("Large-Model", 2000, 512);
        
        // 显示小模型统计
        System.out.println("小模型统计:");
        Map<String, Object> smallStats = smallModel.getModelStatistics();
        printModelStats(smallStats);
        
        System.out.println("\n大模型统计:");
        Map<String, Object> largeStats = largeModel.getModelStatistics();
        printModelStats(largeStats);
        
        // 参数数量比较
        long smallParams = (Long) smallStats.get("total_parameters");
        long largeParams = (Long) largeStats.get("total_parameters");
        double ratio = (double) largeParams / smallParams;
        
        System.out.println("\n模型比较:");
        System.out.println("参数数量比例: " + String.format("%.2fx", ratio));
        
        System.out.println("模型统计演示完成!\n");
    }
    
    /**
     * 创建示例输入
     */
    private static NdArray createSampleInput(int seqLen) {
        NdArray input = NdArray.zeros(Shape.of(1, seqLen));
        
        // 填充随机token ID
        for (int i = 0; i < seqLen; i++) {
            input.set((int) (Math.random() * 50) + 1, 0, i); // 1-50范围的随机token
        }
        
        return input;
    }
    
    /**
     * 将数组转换为字符串显示
     */
    private static String arrayToString(NdArray array) {
        if (array.getShape().getDimNum() != 2) {
            return array.getShape().toString();
        }
        
        StringBuilder sb = new StringBuilder("[");
        int seqLen = array.getShape().getDimension(1);
        for (int i = 0; i < seqLen; i++) {
            if (i > 0) sb.append(", ");
            sb.append((int) array.get(0, i));
        }
        sb.append("]");
        
        return sb.toString();
    }
    
    /**
     * 获取前几个logits值
     */
    private static String getFirstLogits(NdArray logits, int count) {
        StringBuilder sb = new StringBuilder("[");
        Shape shape = logits.getShape();
        
        int actualCount = Math.min(count, shape.getDimension(shape.getDimNum() - 1));
        for (int i = 0; i < actualCount; i++) {
            if (i > 0) sb.append(", ");
            // 获取第一个batch、最后一个时间步的前几个logits
            if (shape.getDimNum() == 3) {
                sb.append(String.format("%.3f", logits.get(0, shape.getDimension(1) - 1, i)));
            } else if (shape.getDimNum() == 2) {
                sb.append(String.format("%.3f", logits.get(0, i)));
            }
        }
        sb.append("]");
        
        return sb.toString();
    }
    
    /**
     * 打印模型统计信息
     */
    private static void printModelStats(Map<String, Object> stats) {
        System.out.println("  词汇表大小: " + stats.get("vocab_size"));
        System.out.println("  模型维度: " + stats.get("d_model"));
        System.out.println("  层数: " + stats.get("num_layers"));
        System.out.println("  注意力头数: " + stats.get("num_heads"));
        System.out.println("  前馈维度: " + stats.get("d_ff"));
        System.out.println("  最大序列长度: " + stats.get("max_seq_len"));
        System.out.println("  总参数数: " + formatNumber((Long) stats.get("total_parameters")));
        System.out.println("  推理步骤数: " + stats.get("reasoning_steps"));
    }
    
    /**
     * 格式化大数字
     */
    private static String formatNumber(long number) {
        if (number >= 1_000_000) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        } else {
            return String.valueOf(number);
        }
    }
    
    /**
     * 简单的基准测试
     */
    public static void runBenchmark() {
        System.out.println("=== DeepSeek R1 性能基准测试 ===");
        
        DeepSeekR1Model model = new DeepSeekR1Model("Benchmark-Model", 1000, 256);
        NdArray input = createSampleInput(64);
        
        // 预热
        for (int i = 0; i < 5; i++) {
            model.inference(input);
        }
        
        // 基准测试
        int iterations = 10;
        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < iterations; i++) {
            model.inference(input);
        }
        
        long endTime = System.currentTimeMillis();
        double avgTime = (endTime - startTime) / (double) iterations;
        
        System.out.println("平均推理时间: " + String.format("%.2f ms", avgTime));
        System.out.println("吞吐量: " + String.format("%.2f inferences/sec", 1000.0 / avgTime));
        
        // 详细推理基准
        startTime = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            model.inferenceWithDetails(input, null);
        }
        endTime = System.currentTimeMillis();
        double avgDetailedTime = (endTime - startTime) / (double) iterations;
        
        System.out.println("详细推理平均时间: " + String.format("%.2f ms", avgDetailedTime));
        System.out.println("详细推理开销: " + String.format("%.1fx", avgDetailedTime / avgTime));
        
        System.out.println("基准测试完成!\n");
    }
}