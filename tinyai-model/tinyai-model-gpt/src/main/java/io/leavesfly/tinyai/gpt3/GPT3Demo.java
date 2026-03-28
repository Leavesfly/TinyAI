package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.util.PlantUML;
import io.leavesfly.tinyai.util.UmlPrinter;

/**
 * GPT-3模型演示程序
 * <p>
 * 展示GPT-3模型的核心功能:
 * 1. 多规模模型创建
 * 2. 模型架构分析
 * 3. 前向传播演示
 * 4. 文本生成演示
 * 5. 配置对比分析
 *
 * @author leavesfly
 * @version 1.0
 */
public class GPT3Demo {

    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("GPT-3 模型演示程序");
        System.out.println("基于TinyAI框架实现");
        System.out.println("=".repeat(80) + "\n");

        // 1. 模型创建演示
//        demonstrateModelCreation();

        // 2. 架构分析
//        demonstrateArchitectureAnalysis();

        // 3. 前向传播演示
        demonstrateForwardPass();

        // 4. 文本生成演示
//        demonstrateTextGeneration();

        // 5. 配置对比
//        demonstrateConfigComparison();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("演示完成！");
        System.out.println("=".repeat(80) + "\n");

    }

    /**
     * 1. 模型创建演示
     */
    private static void demonstrateModelCreation() {
        printSectionHeader("1. 模型创建演示");

        System.out.println("创建多种规模的GPT-3模型...\n");

        // 创建小型模型
        System.out.println("▶ 小型模型 (125M参数):");
        GPT3Model smallModel = GPT3Model.createSmallModel("gpt3-small");
        System.out.println("  " + smallModel);
        System.out.println("  描述: " + smallModel.getConfig());
        System.out.println();

        // 创建中型模型
        System.out.println("▶ 中型模型 (350M参数):");
        GPT3Model mediumModel = GPT3Model.createMediumModel("gpt3-medium");
        System.out.println("  " + mediumModel);
        System.out.println();

//        // 创建大型模型
//        System.out.println("▶ 大型模型 (1.3B参数):");
//        GPT3Model largeModel = GPT3Model.createLargeModel("gpt3-large");
//        System.out.println("  " + largeModel);
//        System.out.println("  特性: RoPE + 稀疏注意力 + 梯度检查点");
//        System.out.println();
//
//        // 创建超大型模型
//        System.out.println("▶ 超大型模型 (175B参数):");
//        GPT3Model xlModel = GPT3Model.createXLModel("gpt3-xl");
//        System.out.println("  " + xlModel);
//        System.out.println("  特性: 全部优化特性启用");
//        System.out.println();
//
//        System.out.println("✓ 成功创建4种规模的GPT-3模型");
    }

    /**
     * 2. 架构分析演示
     */
    private static void demonstrateArchitectureAnalysis() {
        printSectionHeader("2. 架构分析演示");

        // 使用中型模型进行详细分析
        GPT3Model model = GPT3Model.createMediumModel("gpt3-architecture-demo");

        System.out.println("分析GPT-3中型模型架构...\n");

        // 打印详细模型信息
        model.printModelInfo();

        // 打印配置摘要
        System.out.println("\n配置摘要:");
        System.out.println(model.getConfigSummary());

        // 分析并行架构特点
        System.out.println("\n架构特点:");
        System.out.println("  • 并行注意力: GPT-3同时计算注意力和MLP，提升效率");
        System.out.println("  • Pre-LayerNorm: 在子层之前应用归一化，改善训练稳定性");
        System.out.println("  • 深层堆叠: " + model.getConfig().getNLayer() + "个Transformer块串联");
        System.out.println("  • 多头注意力: " + model.getConfig().getNHead() + "个注意力头并行计算");
    }

    /**
     * 3. 前向传播演示
     */
    private static Variable demonstrateForwardPass() {
        printSectionHeader("3. 前向传播演示");

        // 使用小型模型进行演示（节省计算）
        GPT3Model model = GPT3Model.createNanoModel("gpt3-forward-demo");

        System.out.println("执行前向传播...\n");

        // 准备输入数据
        int batchSize = 2;
        int seqLen = 32;
        NdArray tokenIds = createRandomTokenIds(batchSize, seqLen, 100);

        System.out.println("输入:");
        System.out.printf("  - Batch大小: %d\n", batchSize);
        System.out.printf("  - 序列长度: %d\n", seqLen);
        System.out.printf("  - 形状: %s\n", tokenIds.getShape());
        System.out.println();

        // 执行前向传播
        long startTime = System.currentTimeMillis();
        Variable output = model.predict(new Variable(tokenIds));
        long endTime = System.currentTimeMillis();

        System.out.println("输出:");
        System.out.printf("  - 形状: %s\n", output.getValue().getShape());
        System.out.printf("  - 词汇表大小: %d\n", model.getConfig().getVocabSize());
        System.out.printf("  - 推理时间: %d ms\n", endTime - startTime);
        System.out.println();

        // 验证输出形状
        Shape expectedShape = Shape.of(batchSize, seqLen, model.getConfig().getVocabSize());
        if (output.getValue().getShape().equals(expectedShape)) {
            System.out.println("✓ 输出形状验证通过");
        } else {
            System.out.println("✗ 输出形状验证失败");
        }

//        System.out.printf(PlantUML.generateGraph(output));
        System.out.printf(PlantUML.generateModuleGraph(model.getModule()));

        return output;
    }

    /**
     * 4. 文本生成演示
     */
    private static void demonstrateTextGeneration() {
        printSectionHeader("4. 文本生成演示");

        // 使用小型模型
        GPT3Model model = GPT3Model.createSmallModel("gpt3-generation-demo");

        System.out.println("执行文本生成（贪婪解码）...\n");

        // 准备提示序列
        int batchSize = 1;
        int promptLen = 5;
        NdArray promptIds = createRandomTokenIds(batchSize, promptLen, 5000);

        System.out.println("提示序列:");
        System.out.printf("  - 长度: %d\n", promptLen);
        System.out.printf("  - Token IDs: ");
        for (int i = 0; i < promptLen; i++) {
            System.out.printf("%d ", (int) promptIds.get(0, i));
        }
        System.out.println("\n");

        // 生成新token
        int maxNewTokens = 10;
        System.out.printf("生成 %d 个新token...\n", maxNewTokens);

        long startTime = System.currentTimeMillis();
        NdArray generatedSeq = model.generateSequence(promptIds, maxNewTokens);
        long endTime = System.currentTimeMillis();

        System.out.println("\n生成序列:");
        System.out.printf("  - 总长度: %d (提示: %d + 新生成: %d)\n",
                promptLen + maxNewTokens, promptLen, maxNewTokens);
        System.out.print("  - Token IDs: ");
        for (int i = 0; i < promptLen + maxNewTokens; i++) {
            if (i == promptLen) {
                System.out.print("| ");  // 标记提示和生成的分界
            }
            System.out.printf("%d ", (int) generatedSeq.get(0, i));
        }
        System.out.println();
        System.out.printf("  - 生成时间: %d ms\n", endTime - startTime);
        System.out.printf("  - 平均速度: %.2f tokens/s\n",
                maxNewTokens * 1000.0 / (endTime - startTime));
        System.out.println();

        System.out.println("✓ 文本生成完成");
        System.out.println("注: 实际应用中可使用Top-k、Top-p、Beam Search等高级采样策略");
    }

    /**
     * 5. 配置对比演示
     */
    private static void demonstrateConfigComparison() {
        printSectionHeader("5. 配置对比分析");

        System.out.println("GPT-3各规模配置对比:\n");

        // 创建表头
        System.out.printf("%-15s %-12s %-8s %-8s %-8s %-15s %-15s\n",
                "规模", "参数量", "层数", "维度", "头数", "并行架构", "特殊优化");
        System.out.println("-".repeat(95));

        // 小型
        GPT3Config smallConfig = GPT3Config.createSmallConfig();
        System.out.printf("%-15s %-12s %-8d %-8d %-8d %-15s %-15s\n",
                "Small (125M)", formatParam(smallConfig.estimateParameterCount()),
                smallConfig.getNLayer(), smallConfig.getNEmbd(), smallConfig.getNHead(),
                smallConfig.isParallelAttention() ? "是" : "否",
                "无");

        // 中型
        GPT3Config mediumConfig = GPT3Config.createMediumConfig();
        System.out.printf("%-15s %-12s %-8d %-8d %-8d %-15s %-15s\n",
                "Medium (350M)", formatParam(mediumConfig.estimateParameterCount()),
                mediumConfig.getNLayer(), mediumConfig.getNEmbd(), mediumConfig.getNHead(),
                mediumConfig.isParallelAttention() ? "是" : "否",
                "无");

        // 大型
        GPT3Config largeConfig = GPT3Config.createLargeConfig();
        System.out.printf("%-15s %-12s %-8d %-8d %-8d %-15s %-15s\n",
                "Large (1.3B)", formatParam(largeConfig.estimateParameterCount()),
                largeConfig.getNLayer(), largeConfig.getNEmbd(), largeConfig.getNHead(),
                largeConfig.isParallelAttention() ? "是" : "否",
                "RoPE+稀疏");

        // 超大型
        GPT3Config xlConfig = GPT3Config.createXLConfig();
        System.out.printf("%-15s %-12s %-8d %-8d %-8d %-15s %-15s\n",
                "XL (175B)", formatParam(xlConfig.estimateParameterCount()),
                xlConfig.getNLayer(), xlConfig.getNEmbd(), xlConfig.getNHead(),
                xlConfig.isParallelAttention() ? "是" : "否",
                "全部");

        System.out.println();

        // 关键差异说明
        System.out.println("关键差异:");
        System.out.println("  1. 参数规模: 从125M到175B，跨越3个数量级");
        System.out.println("  2. 架构优化: 大型模型启用RoPE、稀疏注意力、梯度检查点");
        System.out.println("  3. 计算效率: 所有规模均采用并行注意力架构");
        System.out.println("  4. 序列长度: 统一支持2048 tokens（GPT-2为1024）");
    }

    // ==================== 辅助方法 ====================

    /**
     * 打印章节标题
     */
    private static void printSectionHeader(String title) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println(title);
        System.out.println("=".repeat(80) + "\n");
    }

    /**
     * 创建随机token IDs（用于演示）
     */
    private static NdArray createRandomTokenIds(int batchSize, int seqLen, int vocabSize) {
        float[][] data = new float[batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                data[b][s] = (float) (Math.random() * vocabSize);
            }
        }
        return NdArray.of(data);
    }

    /**
     * 格式化参数数量
     */
    private static String formatParam(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
}
