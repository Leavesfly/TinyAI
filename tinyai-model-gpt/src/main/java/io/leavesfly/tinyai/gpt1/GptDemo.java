package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * GPT-1 综合演示程序
 * 
 * 提供GPT-1模型的完整演示，包括：
 * 1. 模型创建和配置
 * 2. 基础功能演示
 * 3. 文本生成展示
 * 4. 性能测试
 * 5. 不同配置对比
 * 
 * @author 山泽
 * @version 1.0
 */
public class GptDemo {
    
    public static void main(String[] args) {
        System.out.println("🤖 欢迎使用 TinyAI GPT-1 演示程序!");
        System.out.println("基于TinyAI框架实现的GPT-1 Transformer解码器模型\n");
        
        try {
            // 1. 快速开始演示
            quickStartDemo();
            
            System.out.println("\n" + "=".repeat(60) + "\n");
            
            // 2. 详细功能演示
            detailedFunctionalityDemo();
            
            System.out.println("\n" + "=".repeat(60) + "\n");
            
            // 3. 架构展示
            architectureDemo();
            
            System.out.println("\n" + "=".repeat(60) + "\n");
            
            // 4. 性能基准测试
            performanceBenchmark();
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n🎉 GPT-1 演示完成! 感谢使用 TinyAI 框架!");
    }
    
    /**
     * 快速开始演示
     */
    public static void quickStartDemo() {
        System.out.println("🚀 === 快速开始演示 ===");
        
        // 创建小型GPT-1模型
        System.out.println("正在创建小型GPT-1模型...");
        GPT1Model model = GPT1Model.createTinyModel("demo-gpt1");
        
        System.out.println("✅ 模型创建成功!");
        System.out.println("📊 " + model.getModelCapacity());
        
        // 简单的前向传播测试
        System.out.println("\n正在测试前向传播...");
        int[] testInput = {1, 2, 3, 4, 5};
        
        try {
            Variable result = model.predictNextToken(testInput);
            System.out.println("✅ 前向传播成功!");
            System.out.printf("输入: %s\n", Arrays.toString(testInput));
            System.out.printf("输出形状: %s\n", result.getValue().getShape());
        } catch (Exception e) {
            System.out.println("❌ 前向传播失败: " + e.getMessage());
        }
    }
    
    /**
     * 详细功能演示
     */
    public static void detailedFunctionalityDemo() {
        System.out.println("🔧 === 详细功能演示 ===");
        
        // 1. 不同规模模型对比
        System.out.println("\n1. 模型规模对比:");
        compareModelSizes();
        
        // 2. 配置验证演示
        System.out.println("\n2. 配置验证演示:");
        demonstrateConfigValidation();
        
        // 3. 文本生成演示
        System.out.println("\n3. 文本生成演示:");
        demonstrateTextGeneration();
        
        // 4. 批量处理演示
        System.out.println("\n4. 批量处理演示:");
        demonstrateBatchProcessing();
    }
    
    /**
     * 架构展示
     */
    public static void architectureDemo() {
        System.out.println("🏗️ === GPT-1 架构展示 ===");
        
        GPT1Model model = GPT1Model.createMediumModel("architecture-demo");
        
        // 显示详细的模型信息
        model.printModelInfo();
        
        // 显示组件信息
        System.out.println("\n📦 模型组件结构:");
        GPT1Block block = model.getGPT1Block();
        
        System.out.printf("├── Token嵌入层: %s\n", block.getTokenEmbedding().getClass().getSimpleName());
        System.out.printf("├── Transformer块: %d 层\n", block.getTransformerBlocks().size());
        
        for (int i = 0; i < block.getTransformerBlocks().size(); i++) {
            String prefix = (i == block.getTransformerBlocks().size() - 1) ? "│   └──" : "│   ├──";
            System.out.printf("%s 第%d层: MultiHeadAttention + FeedForward\n", prefix, i + 1);
        }
        
        System.out.printf("├── 最终LayerNorm: %s\n", block.getFinalLayerNorm().getClass().getSimpleName());
        System.out.printf("└── 输出头: %s\n", block.getOutputHead().getClass().getSimpleName());
    }
    
    /**
     * 性能基准测试
     */
    public static void performanceBenchmark() {
        System.out.println("⚡ === 性能基准测试 ===");
        
        GPT1Model model = GPT1Model.createTinyModel("benchmark");
        
        // 测试不同序列长度的推理速度
        int[] sequenceLengths = {8, 16, 32, 64};
        System.out.println("\n测试不同序列长度的推理性能:");
        System.out.println("序列长度 | 推理时间(ms) | 状态");
        System.out.println("-".repeat(35));
        
        for (int seqLen : sequenceLengths) {
            if (seqLen <= model.getMaxSequenceLength()) {
                long startTime = System.currentTimeMillis();
                
                try {
                    int[] testInput = createTestSequence(seqLen, model.getVocabSize());
                    model.predictNextToken(testInput);
                    
                    long endTime = System.currentTimeMillis();
                    System.out.printf("%-8d | %-11d | ✅ 成功\n", seqLen, endTime - startTime);
                    
                } catch (Exception e) {
                    System.out.printf("%-8d | %-11s | ❌ 失败\n", seqLen, "N/A");
                }
            } else {
                System.out.printf("%-8d | %-11s | ⚠️ 超长\n", seqLen, "N/A");
            }
        }
        
        // 内存使用测试
        System.out.println("\n💾 内存使用情况:");
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("当前内存使用: %.2f MB\n", usedMemory / 1024.0 / 1024.0);
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 比较不同规模的模型
     */
    private static void compareModelSizes() {
        GPT1Model tiny = GPT1Model.createTinyModel("tiny");
        GPT1Model medium = GPT1Model.createMediumModel("medium");
        GPT1Model full = GPT1Model.createFullModel("full", 10000);
        
        System.out.printf("%-8s | %-8s | %-8s | %-8s | %s\n", 
                         "规模", "词汇表", "序列长度", "隐藏维度", "参数量");
        System.out.println("-".repeat(60));
        
        System.out.printf("%-8s | %-8d | %-8d | %-8d | %s\n", 
                         "Tiny", tiny.getVocabSize(), tiny.getMaxSequenceLength(), 
                         tiny.getHiddenSize(), tiny.getModelCapacity());
        
        System.out.printf("%-8s | %-8d | %-8d | %-8d | %s\n", 
                         "Medium", medium.getVocabSize(), medium.getMaxSequenceLength(), 
                         medium.getHiddenSize(), medium.getModelCapacity());
        
        System.out.printf("%-8s | %-8d | %-8d | %-8d | %s\n", 
                         "Full", full.getVocabSize(), full.getMaxSequenceLength(), 
                         full.getHiddenSize(), full.getModelCapacity());
    }
    
    /**
     * 演示配置验证
     */
    private static void demonstrateConfigValidation() {
        System.out.println("测试有效配置:");
        try {
            GPT1Config validConfig = new GPT1Config(1000, 128, 256, 6, 8);
            validConfig.validate();
            System.out.println("✅ 有效配置验证通过");
        } catch (Exception e) {
            System.out.println("❌ 有效配置验证失败: " + e.getMessage());
        }
        
        System.out.println("\n测试无效配置:");
        try {
            GPT1Config invalidConfig = new GPT1Config(1000, 128, 257, 6, 8); // 257不能被8整除
            invalidConfig.validate();
            System.out.println("❌ 无效配置验证应该失败但通过了");
        } catch (Exception e) {
            System.out.println("✅ 无效配置验证正确失败: " + e.getMessage());
        }
    }
    
    /**
     * 演示文本生成
     */
    private static void demonstrateTextGeneration() {
        GPT1Model model = GPT1Model.createTinyModel("text-gen");
        
        List<Integer> prompt = Arrays.asList(1, 2, 3);
        System.out.printf("提示词: %s\n", prompt);
        
        try {
            List<Integer> generated = model.generateText(prompt, 10, 1.0);
            System.out.printf("生成结果: %s\n", generated);
            System.out.println("✅ 文本生成成功");
        } catch (Exception e) {
            System.out.println("❌ 文本生成失败: " + e.getMessage());
        }
    }
    
    /**
     * 演示批量处理
     */
    private static void demonstrateBatchProcessing() {
        GPT1Model model = GPT1Model.createTinyModel("batch");
        
        try {
            // 创建批量输入
            float[][] batchData = {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}
            };
            
            Variable batchInput = new Variable(NdArray.of(batchData));
            Variable result = model.batchPredict(batchInput);
            
            System.out.printf("批量输入形状: %s\n", batchInput.getValue().getShape());
            System.out.printf("批量输出形状: %s\n", result.getValue().getShape());
            System.out.println("✅ 批量处理成功");
            
        } catch (Exception e) {
            System.out.println("❌ 批量处理失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建测试序列
     */
    private static int[] createTestSequence(int length, int vocabSize) {
        int[] sequence = new int[length];
        for (int i = 0; i < length; i++) {
            sequence[i] = i % vocabSize;
        }
        return sequence;
    }
    
    /**
     * 运行特定演示模块
     * 
     * @param demoType 演示类型: "quick", "detailed", "architecture", "performance", "all"
     */
    public static void runDemo(String demoType) {
        switch (demoType.toLowerCase()) {
            case "quick":
                quickStartDemo();
                break;
            case "detailed":
                detailedFunctionalityDemo();
                break;
            case "architecture":
                architectureDemo();
                break;
            case "performance":
                performanceBenchmark();
                break;
            case "all":
            default:
                main(new String[0]);
                break;
        }
    }
}
