package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DeepSeek-R1模型示例代码
 * 
 * 演示DeepSeek-R1模型的基本使用方法，包括：
 * 1. 模型创建与配置
 * 2. 基础推理
 * 3. 带详细信息的推理
 * 4. 序列生成
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Demo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-R1 模型示例");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 示例1: 创建模型并打印信息
        example1_CreateModel();
        
        // 示例2: 基础推理
        example2_BasicInference();
        
        // 示例3: 带详细信息的推理
        example3_DetailedInference();
        
        // 示例4: 序列生成
        example4_SequenceGeneration();
        
        System.out.println("=".repeat(80));
        System.out.println("所有示例执行完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1: 创建模型并打印信息
     */
    private static void example1_CreateModel() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例1: 创建模型并打印信息");
        System.out.println("=".repeat(80));
        
        // 创建微型模型（用于快速测试）
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
        
        // 打印模型信息
        model.printModelInfo();
        
        // 打印配置摘要
        System.out.println("\n" + model.getConfigSummary());
    }
    
    /**
     * 示例2: 基础推理
     */
    private static void example2_BasicInference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例2: 基础推理");
        System.out.println("=".repeat(80));
        
        // 创建微型模型
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
        DeepSeekR1Config config = model.getConfig();
        
        // 准备输入 [batch_size=2, seq_len=8]
        int batchSize = 2;
        int seqLen = 8;
        float[][] inputData = new float[batchSize][seqLen];
        
        // 填充随机token ID（范围: 0 ~ vocabSize-1）
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                inputData[b][s] = (float) (Math.random() * config.getVocabSize());
            }
        }
        
        NdArray tokenIds = NdArray.of(inputData);
        System.out.printf("输入形状: %s\n", tokenIds.getShape());
        
        // 执行推理
        Variable logits = model.predict(new Variable(tokenIds));
        
        System.out.printf("输出logits形状: %s\n", logits.getValue().getShape());
        System.out.printf("预期形状: [%d, %d, %d]\n", batchSize, seqLen, config.getVocabSize());
        System.out.println("✓ 基础推理成功!");
    }
    
    /**
     * 示例3: 带详细信息的推理
     */
    private static void example3_DetailedInference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例3: 带详细信息的推理");
        System.out.println("=".repeat(80));
        
        // 创建微型模型
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
        DeepSeekR1Config config = model.getConfig();
        
        // 准备输入 [batch_size=1, seq_len=10]
        int batchSize = 1;
        int seqLen = 10;
        float[][] inputData = new float[batchSize][seqLen];
        
        // 填充随机token ID
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                inputData[b][s] = (float) (Math.random() * config.getVocabSize());
            }
        }
        
        NdArray tokenIds = NdArray.of(inputData);
        
        // 执行带详细信息的推理
        DeepSeekR1Model.ReasoningResult result = model.performReasoning(new Variable(tokenIds));
        
        // 打印推理结果
        System.out.println("\n推理结果详情:");
        System.out.println(result);
        
        System.out.println("\n✓ 带详细信息的推理成功!");
    }
    
    /**
     * 示例4: 序列生成
     */
    private static void example4_SequenceGeneration() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例4: 序列生成");
        System.out.println("=".repeat(80));
        
        // 创建微型模型
        DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("DeepSeek-R1-Tiny");
        DeepSeekR1Config config = model.getConfig();
        
        // 准备提示词 [batch_size=1, prompt_len=5]
        int batchSize = 1;
        int promptLen = 5;
        int maxNewTokens = 10;
        
        float[][] promptData = new float[batchSize][promptLen];
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < promptLen; s++) {
                promptData[b][s] = (float) (Math.random() * config.getVocabSize());
            }
        }
        
        NdArray promptIds = NdArray.of(promptData);
        System.out.printf("提示词形状: %s\n", promptIds.getShape());
        System.out.printf("提示词token IDs (batch 0): [");
        for (int i = 0; i < promptLen; i++) {
            System.out.printf("%.0f", promptIds.get(0, i));
            if (i < promptLen - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        // 生成序列
        System.out.printf("\n开始生成 %d 个新token...\n", maxNewTokens);
        NdArray generated = model.generateSequence(promptIds, maxNewTokens);
        
        System.out.printf("生成序列形状: %s\n", generated.getShape());
        System.out.printf("预期形状: [%d, %d]\n", batchSize, promptLen + maxNewTokens);
        
        System.out.print("生成的完整序列 (batch 0): [");
        for (int i = 0; i < promptLen + maxNewTokens; i++) {
            System.out.printf("%.0f", generated.get(0, i));
            if (i < promptLen + maxNewTokens - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        System.out.println("\n✓ 序列生成成功!");
    }
    
    /**
     * 示例5: 自定义配置模型
     */
    public static void example5_CustomConfig() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例5: 自定义配置模型");
        System.out.println("=".repeat(80));
        
        // 创建自定义配置（使用 MoE 架构）
        DeepSeekR1Config config = new DeepSeekR1Config();
        config.setVocabSize(5000);
        config.setNEmbd(128);
        config.setNLayer(4);
        config.setNHead(4);
        config.setNInner(512);
        config.setNPositions(256);
        
        // MoE 配置（R1 使用与 V3 相同的 MoE 架构）
        config.setNumExperts(8);
        config.setTopK(2);
        config.setExpertHiddenDim(512);
        
        // RL 训练配置（R1 特有）
        config.setRlExplorationRate(0.1);
        config.setRlDiscountFactor(0.99);
        
        // 配置已自动验证（在 BaseConfig 中）
        
        // 创建模型
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Custom", config);
        
        // 打印模型信息
        System.out.println(model);
        System.out.println("\n" + model.getConfigSummary());
        
        System.out.println("\n✓ 自定义配置模型创建成功!");
    }
    
    /**
     * 示例6: 对比不同规模的模型
     */
    public static void example6_CompareModels() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例6: 对比不同规模的模型");
        System.out.println("=".repeat(80));
        
        // 创建不同规模的模型
        DeepSeekR1Model tinyModel = DeepSeekR1Model.createTinyModel("R1-Tiny");
        DeepSeekR1Model smallModel = DeepSeekR1Model.createSmallModel("R1-Small");
        DeepSeekR1Model standardModel = DeepSeekR1Model.createStandardModel("R1-Standard");
        
        // 打印对比信息
        System.out.printf("%-15s | %-10s | %-6s | %-6s | %-8s | %-10s\n",
            "模型", "参数量", "层数", "维度", "注意力头", "推理步骤");
        System.out.println("-".repeat(80));
        
        printModelComparison(tinyModel);
        printModelComparison(smallModel);
        printModelComparison(standardModel);
        
        System.out.println("\n✓ 模型对比完成!");
    }
    
    private static void printModelComparison(DeepSeekR1Model model) {
        DeepSeekR1Config config = model.getConfig();
        System.out.printf("%-15s | %-10s | %-6d | %-6d | %-8d | %-10d\n",
            model.getName(),
            formatParamCount(config.estimateParameterCount()),
            config.getNLayer(),
            config.getNEmbd(),
            config.getNHead(),
            config.getNumExperts()  // 显示专家数量而不是推理步骤
        );
    }
    
    private static String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%.2fK", count / 1_000.0);
        }
    }
}
