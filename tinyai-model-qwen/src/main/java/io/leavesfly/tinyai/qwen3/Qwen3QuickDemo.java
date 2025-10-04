package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Qwen3模型快速演示
 * 
 * 提供最简单的使用示例，展示Qwen3模型的基本功能
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3QuickDemo {
    
    /**
     * 快速演示Qwen3模型的基本使用
     */
    public static void quickStart() {
        System.out.println("🚀 Qwen3 快速开始演示");
        System.out.println("=" + "=".repeat(30));
        
        try {
            // 1. 创建模型配置
            System.out.println("\n1. 创建模型配置...");
            Qwen3Config config = Qwen3Config.createDemoConfig();
            System.out.println("✓ 配置创建完成");
            System.out.println("  - 词汇表大小: " + config.getVocabSize());
            System.out.println("  - 隐藏维度: " + config.getHiddenSize());
            System.out.println("  - 层数: " + config.getNumHiddenLayers());
            
            // 2. 创建模型
            System.out.println("\n2. 初始化Qwen3模型...");
            Qwen3Model model = new Qwen3Model("qwen3-quick-demo", config);
            System.out.println("✓ 模型初始化完成");
            System.out.println("  - 参数数量: " + String.format("%,d", model.countParameters()));
            System.out.println("  - 模型大小: " + String.format("%.2f MB", model.getModelSizeMB()));
            
            // 3. 准备输入数据
            System.out.println("\n3. 准备输入数据...");
            int batchSize = 2;
            int seqLen = 10;
            int vocabSize = config.getVocabSize();
            
            // 创建随机token ID作为输入
            NdArray inputIds = NdArray.of(Shape.of(batchSize, seqLen));
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < seqLen; j++) {
                    int tokenId = (int) (Math.random() * vocabSize);
                    inputIds.set(tokenId, i, j);
                }
            }
            
            System.out.println("✓ 输入数据创建完成");
            System.out.println("  - 输入形状: [" + batchSize + ", " + seqLen + "]");
            System.out.println("  - 示例token IDs: [" + 
                (int)inputIds.get(0, 0) + ", " + (int)inputIds.get(0, 1) + ", " + 
                (int)inputIds.get(0, 2) + ", ...]");
            
            // 4. 模型前向传播
            System.out.println("\n4. 执行模型前向传播...");
            long startTime = System.currentTimeMillis();
            
            Variable output = model.forward(new Variable(inputIds));
            
            long endTime = System.currentTimeMillis();
            System.out.println("✓ 前向传播完成");
            System.out.println("  - 输出形状: " + output.getValue().getShape());
            System.out.println("  - 执行时间: " + (endTime - startTime) + " ms");
            
            // 5. 分析输出
            System.out.println("\n5. 分析模型输出...");
            NdArray logits = output.getValue();
            Shape outputShape = logits.getShape();
            
            System.out.println("✓ 输出分析完成");
            System.out.println("  - 批次大小: " + outputShape.getDimension(0));
            System.out.println("  - 序列长度: " + outputShape.getDimension(1));
            System.out.println("  - 词汇表大小: " + outputShape.getDimension(2));
            
            // 验证输出形状正确性
            boolean shapeCorrect = outputShape.getDimension(0) == batchSize &&
                                 outputShape.getDimension(1) == seqLen &&
                                 outputShape.getDimension(2) == vocabSize;
            
            System.out.println("  - 形状验证: " + (shapeCorrect ? "✓ 正确" : "✗ 错误"));
            
            // 6. 显示部分输出值
            System.out.println("\n6. 输出值样例:");
            System.out.print("  - 第一个样本第一个位置的前5个logits: [");
            for (int i = 0; i < 5; i++) {
                System.out.printf("%.3f", logits.get(0, 0, i));
                if (i < 4) System.out.print(", ");
            }
            System.out.println("]");
            
        } catch (Exception e) {
            System.err.println("❌ 演示过程中发生错误:");
            System.err.println("  错误信息: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示各种配置的模型
     */
    public static void configDemo() {
        System.out.println("\n📊 不同配置模型对比");
        System.out.println("=" + "=".repeat(30));
        
        try {
            // 测试不同配置
            Qwen3Config[] configs = {
                Qwen3Config.createSmallConfig(),
                Qwen3Config.createDemoConfig()
            };
            
            String[] configNames = {"小型配置", "演示配置"};
            
            for (int i = 0; i < configs.length; i++) {
                System.out.println("\n" + (i + 1) + ". " + configNames[i] + ":");
                
                Qwen3Config config = configs[i];
                Qwen3Model model = new Qwen3Model("test-" + i, config);
                
                System.out.println("  - 词汇表大小: " + String.format("%,d", config.getVocabSize()));
                System.out.println("  - 隐藏维度: " + config.getHiddenSize());
                System.out.println("  - 层数: " + config.getNumHiddenLayers());
                System.out.println("  - 注意力头: " + config.getNumAttentionHeads());
                System.out.println("  - 参数数量: " + String.format("%,d", model.countParameters()));
                System.out.println("  - 模型大小: " + String.format("%.2f MB", model.getModelSizeMB()));
                
                // 快速测试
                NdArray testInput = NdArray.of(Shape.of(1, 5));
                for (int j = 0; j < 5; j++) {
                    testInput.set((int)(Math.random() * config.getVocabSize()), 0, j);
                }
                
                long startTime = System.currentTimeMillis();
                Variable output = model.forward(new Variable(testInput));
                long endTime = System.currentTimeMillis();
                
                System.out.println("  - 推理时间: " + (endTime - startTime) + " ms");
                System.out.println("  - 输出形状: " + output.getValue().getShape());
            }
            
        } catch (Exception e) {
            System.err.println("❌ 配置演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 性能测试演示
     */
    public static void performanceDemo() {
        System.out.println("\n⚡ 性能测试演示");
        System.out.println("=" + "=".repeat(30));
        
        try {
            Qwen3Model model = new Qwen3Model("perf-test", Qwen3Config.createDemoConfig());
            
            // 不同批次大小测试
            int[] batchSizes = {1, 2, 4};
            int seqLen = 10;
            
            System.out.println("\n批次大小性能对比:");
            System.out.println("序列长度: " + seqLen);
            System.out.println("格式: [批次大小] -> 推理时间");
            
            for (int batchSize : batchSizes) {
                // 创建测试输入
                NdArray input = NdArray.of(Shape.of(batchSize, seqLen));
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        input.set((int)(Math.random() * 1000), i, j);
                    }
                }
                
                // 预热
                model.forward(new Variable(input));
                
                // 性能测试
                long startTime = System.currentTimeMillis();
                Variable output = model.forward(new Variable(input));
                long endTime = System.currentTimeMillis();
                
                long executionTime = endTime - startTime;
                System.out.println("  [" + batchSize + "] -> " + executionTime + " ms");
            }
            
            // 不同序列长度测试
            int[] seqLens = {5, 10, 20};
            int batchSize = 1;
            
            System.out.println("\n序列长度性能对比:");
            System.out.println("批次大小: " + batchSize);
            System.out.println("格式: [序列长度] -> 推理时间");
            
            for (int testSeqLen : seqLens) {
                // 创建测试输入
                NdArray input = NdArray.of(Shape.of(batchSize, testSeqLen));
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < testSeqLen; j++) {
                        input.set((int)(Math.random() * 1000), i, j);
                    }
                }
                
                // 预热和测试
                model.forward(new Variable(input));
                
                long startTime = System.currentTimeMillis();
                Variable output = model.forward(new Variable(input));
                long endTime = System.currentTimeMillis();
                
                long executionTime = endTime - startTime;
                System.out.println("  [" + testSeqLen + "] -> " + executionTime + " ms");
            }
            
        } catch (Exception e) {
            System.err.println("❌ 性能测试过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 主方法 - 运行所有演示
     */
    public static void main(String[] args) {
        System.out.println("🎯 Qwen3模型快速演示程序");
        System.out.println("基于TinyAI框架的现代大语言模型实现");
        System.out.println();
        
        try {
            // 运行各种演示
            quickStart();           // 基础使用演示
            configDemo();           // 配置对比演示
            performanceDemo();      // 性能测试演示
            
            System.out.println("\n🎉 所有演示完成!");
            System.out.println("\n💡 使用提示:");
            System.out.println("- 可以通过修改配置来创建不同规模的模型");
            System.out.println("- 模型支持批量推理以提高效率");
            System.out.println("- 实际使用时需要加载预训练的权重");
            System.out.println("- 完整功能请参考 Qwen3Demo 类");
            
        } catch (Exception e) {
            System.err.println("❌ 程序执行过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}