package io.leavesfly.tinyai.gpt1.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * GPT-1 模型使用示例
 * 
 * 展示如何使用GPT-1模型进行：
 * 1. 模型初始化和配置
 * 2. 前向传播和预测
 * 3. 文本生成
 * 4. 模型信息展示
 * 5. 简单的训练演示
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1Example {
    
    public static void main(String[] args) {
        System.out.println("=== GPT-1 模型使用示例 ===\n");
        
        // 1. 基础模型演示
        demonstrateBasicUsage();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // 2. 不同规模模型对比
        demonstrateModelSizes();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // 3. 文本生成演示
        demonstrateTextGeneration();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // 4. 批量处理演示
        demonstrateBatchProcessing();
        
        System.out.println("\n=== GPT-1 示例演示完成 ===");
    }
    
    /**
     * 基础使用演示
     */
    public static void demonstrateBasicUsage() {
        System.out.println("1. === 基础使用演示 ===");
        
        // 创建小型GPT-1模型
        GPT1Model model = GPT1Model.createTinyModel("tiny-gpt1");
        
        // 显示模型信息
        model.printModelInfo();
        
        // 创建示例输入
        int[] inputTokens = {1, 2, 3, 4, 5};  // 示例token序列
        
        System.out.println("输入token序列: " + Arrays.toString(inputTokens));
        
        try {
            // 前向传播
            Variable logits = model.predictNextToken(inputTokens);
            System.out.println("预测输出形状: " + logits.getValue().getShape());
            
            // 验证输入
            Variable input = createTestInput(inputTokens);
            model.validateInput(input);
            System.out.println("输入验证通过 ✓");
            
        } catch (Exception e) {
            System.out.println("执行过程中出现错误: " + e.getMessage());
        }
    }
    
    /**
     * 不同规模模型对比演示
     */
    public static void demonstrateModelSizes() {
        System.out.println("2. === 不同规模模型对比 ===");
        
        // 小型模型
        GPT1Model tinyModel = GPT1Model.createTinyModel("tiny");
        System.out.println("小型模型: " + tinyModel.getModelCapacity());
        
        // 中型模型
        GPT1Model mediumModel = GPT1Model.createMediumModel("medium");
        System.out.println("中型模型: " + mediumModel.getModelCapacity());
        
        // 完整模型（原论文配置）
        GPT1Model fullModel = GPT1Model.createFullModel("full", 40000);
        System.out.println("完整模型: " + fullModel.getModelCapacity());
        
        // 配置对比
        System.out.println("\n配置对比:");
        System.out.printf("%-10s %-8s %-8s %-8s %-8s%n", 
                         "模型", "词汇表", "序列长度", "隐藏维度", "层数");
        System.out.println("-".repeat(50));
        
        printModelComparison("小型", tinyModel);
        printModelComparison("中型", mediumModel);
        printModelComparison("完整", fullModel);
    }
    
    /**
     * 文本生成演示
     */
    public static void demonstrateTextGeneration() {
        System.out.println("3. === 文本生成演示 ===");
        
        GPT1Model model = GPT1Model.createTinyModel("generator");
        
        // 示例提示词
        List<Integer> promptTokens = Arrays.asList(1, 2, 3);
        System.out.println("提示词tokens: " + promptTokens);
        
        try {
            // 生成文本（不同温度）
            System.out.println("\n不同温度参数的生成结果:");
            
            double[] temperatures = {0.5, 1.0, 1.5};
            for (double temp : temperatures) {
                List<Integer> generated = model.generateText(promptTokens, 10, temp);
                System.out.printf("温度 %.1f: %s%n", temp, generated);
            }
            
        } catch (Exception e) {
            System.out.println("文本生成过程中出现错误: " + e.getMessage());
        }
    }
    
    /**
     * 批量处理演示
     */
    public static void demonstrateBatchProcessing() {
        System.out.println("4. === 批量处理演示 ===");
        
        GPT1Model model = GPT1Model.createTinyModel("batch-processor");
        
        // 创建批量输入
        int batchSize = 3;
        int sequenceLength = 8;
        
        try {
            Variable batchInput = createBatchInput(batchSize, sequenceLength, model.getVocabSize());
            System.out.printf("批量输入形状: %s%n", batchInput.getValue().getShape());
            
            // 批量预测
            Variable batchOutput = model.batchPredict(batchInput);
            System.out.printf("批量输出形状: %s%n", batchOutput.getValue().getShape());
            
            System.out.println("批量处理完成 ✓");
            
        } catch (Exception e) {
            System.out.println("批量处理过程中出现错误: " + e.getMessage());
        }
    }
    
    /**
     * 演示模型配置自定义
     */
    public static void demonstrateCustomConfiguration() {
        System.out.println("5. === 自定义配置演示 ===");
        
        // 创建自定义配置
        GPT1Config customConfig = new GPT1Config(
            5000,    // 词汇表大小
            256,     // 最大序列长度
            512,     // 隐藏层维度
            8,       // Transformer层数
            8,       // 注意力头数
            2048,    // 前馈网络维度
            0.1,     // 残差dropout
            0.1,     // 嵌入dropout
            0.1,     // 注意力dropout
            1e-5,    // 层归一化epsilon
            0.02,    // 初始化范围
            "gelu"   // 激活函数
        );
        
        // 验证配置
        try {
            customConfig.validate();
            System.out.println("自定义配置验证通过 ✓");
            System.out.println(customConfig.toString());
            
            // 使用自定义配置创建模型
            GPT1Model customModel = new GPT1Model("custom-gpt1", customConfig);
            System.out.println("自定义模型创建成功: " + customModel.getModelCapacity());
            
        } catch (Exception e) {
            System.out.println("自定义配置验证失败: " + e.getMessage());
        }
    }
    
    /**
     * 演示训练准备和简单训练循环
     */
    public static void demonstrateTrainingSetup() {
        System.out.println("6. === 训练准备演示 ===");
        
        GPT1Model model = GPT1Model.createTinyModel("training-demo");
        
        // 创建虚拟训练数据
        System.out.println("准备训练数据...");
        List<int[]> trainingSequences = createDummyTrainingData(100, 32, model.getVocabSize());
        System.out.printf("训练数据: %d 个序列, 序列长度: %d%n", 
                         trainingSequences.size(), trainingSequences.get(0).length);
        
        // 简单的训练循环演示
        System.out.println("\n开始简单训练演示...");
        int epochs = 3;
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            System.out.printf("轮次 %d/%d:%n", epoch, epochs);
            
            double totalLoss = 0.0;
            int batchCount = 0;
            
            // 简化的批量处理
            for (int i = 0; i < Math.min(5, trainingSequences.size()); i++) {
                int[] sequence = trainingSequences.get(i);
                
                try {
                    // 创建输入和目标
                    Variable input = createTestInput(Arrays.copyOf(sequence, sequence.length - 1));
                    Variable target = createTestInput(Arrays.copyOfRange(sequence, 1, sequence.length));
                    
                    // 计算损失（占位符）
                    double loss = model.computeLanguageModelingLoss(input, target);
                    totalLoss += loss;
                    batchCount++;
                    
                } catch (Exception e) {
                    System.out.printf("  批次 %d 处理失败: %s%n", i + 1, e.getMessage());
                }
            }
            
            double avgLoss = batchCount > 0 ? totalLoss / batchCount : 0.0;
            System.out.printf("  平均损失: %.4f%n", avgLoss);
        }
        
        System.out.println("训练演示完成!");
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 创建测试输入
     */
    private static Variable createTestInput(int[] tokens) {
        float[][] input = new float[1][tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            input[0][i] = tokens[i];
        }
        return new Variable(NdArray.of(input));
    }
    
    /**
     * 创建批量输入
     */
    private static Variable createBatchInput(int batchSize, int sequenceLength, int vocabSize) {
        Random random = new Random(42);
        float[][] batchInput = new float[batchSize][sequenceLength];
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                batchInput[b][s] = random.nextInt(vocabSize);
            }
        }
        
        return new Variable(NdArray.of(batchInput));
    }
    
    /**
     * 创建虚拟训练数据
     */
    private static List<int[]> createDummyTrainingData(int numSequences, int sequenceLength, int vocabSize) {
        Random random = new Random(42);
        List<int[]> data = new ArrayList<>();
        
        for (int i = 0; i < numSequences; i++) {
            int[] sequence = new int[sequenceLength];
            for (int j = 0; j < sequenceLength; j++) {
                sequence[j] = random.nextInt(vocabSize);
            }
            data.add(sequence);
        }
        
        return data;
    }
    
    /**
     * 打印模型对比信息
     */
    private static void printModelComparison(String name, GPT1Model model) {
        System.out.printf("%-10s %-8d %-8d %-8d %-8d%n",
                         name,
                         model.getVocabSize(),
                         model.getMaxSequenceLength(),
                         model.getHiddenSize(),
                         model.getNumLayers());
    }
    
    /**
     * 运行完整示例
     */
    public static void runFullExample() {
        demonstrateBasicUsage();
        System.out.println();
        
        demonstrateModelSizes();
        System.out.println();
        
        demonstrateTextGeneration();
        System.out.println();
        
        demonstrateBatchProcessing();
        System.out.println();
        
        demonstrateCustomConfiguration();
        System.out.println();
        
        demonstrateTrainingSetup();
    }
}