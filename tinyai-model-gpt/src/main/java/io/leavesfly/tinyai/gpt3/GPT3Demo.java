package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-3模型演示程序
 * 
 * 展示GPT-3模型的核心功能：
 * 1. 模型创建和配置
 * 2. 前向传播
 * 3. 文本生成
 * 4. Few-shot学习模拟
 * 5. 模型信息展示
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT3Demo {
    
    /**
     * 主演示方法
     */
    public static void main(String[] args) {
        System.out.println("🚀 GPT-3 模型演示程序");
        System.out.println("=======================\n");
        
        try {
            // 1. 模型创建演示
            demonstrateModelCreation();
            
            // 2. 模型架构分析
            demonstrateArchitectureAnalysis();
            
            // 3. 前向传播演示
            demonstrateForwardPass();
            
            // 4. 文本生成演示
            demonstrateTextGeneration();
            
            // 5. Few-shot学习演示
            demonstrateFewShotLearning();
            
            // 6. 旋转位置编码演示
            demonstrateRotaryEmbedding();
            
            System.out.println("✅ GPT-3演示程序完成！");
            
        } catch (Exception e) {
            System.err.println("❌ 演示过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示模型创建
     */
    private static void demonstrateModelCreation() {
        System.out.println("📝 1. GPT-3模型创建演示");
        System.out.println("------------------------\n");
        
        // 创建不同规模的GPT-3模型
        System.out.println("创建小型GPT-3模型（125M参数）...");
        GPT3Model smallModel = GPT3Model.createSmallModel("gpt3-small");
        System.out.println("✓ 小型模型创建成功");
        
        System.out.println("\n创建中型GPT-3模型（350M参数）...");
        GPT3Model mediumModel = GPT3Model.createMediumModel("gpt3-medium");
        System.out.println("✓ 中型模型创建成功");
        
        System.out.println("\n创建大型GPT-3模型（1.3B参数）...");
        GPT3Model largeModel = GPT3Model.createLargeModel("gpt3-large");
        System.out.println("✓ 大型模型创建成功");
        
        // 展示模型信息
        System.out.println("\n=== 小型GPT-3模型信息 ===");
        smallModel.printModelInfo();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 演示架构分析
     */
    private static void demonstrateArchitectureAnalysis() {
        System.out.println("🏗️ 2. GPT-3架构分析");
        System.out.println("-------------------\n");
        
        // 分析不同规模模型的架构
        GPT3Config[] configs = {
            GPT3Config.createSmallConfig(),
            GPT3Config.createMediumConfig(),
            GPT3Config.createLargeConfig(),
            GPT3Config.createXLConfig()
        };
        
        String[] names = {"小型(125M)", "中型(350M)", "大型(1.3B)", "超大型(175B)"};
        
        System.out.printf("%-12s %-8s %-8s %-8s %-10s %-12s %-10s\n", 
                         "模型规模", "层数", "维度", "头数", "内部维度", "稀疏注意力", "参数估算");
        System.out.println("-".repeat(80));
        
        for (int i = 0; i < configs.length; i++) {
            GPT3Config config = configs[i];
            System.out.printf("%-12s %-8d %-8d %-8d %-10d %-12s %,10d\n",
                             names[i],
                             config.getNLayer(),
                             config.getNEmbd(),
                             config.getNHead(),
                             config.getNInner(),
                             config.isSparseAttention() ? "是" : "否",
                             config.estimateParameterCount());
        }
        
        System.out.println("\n📊 GPT-3关键特性:");
        System.out.println("• 解码器-only Transformer架构");
        System.out.println("• Pre-LayerNorm结构");
        System.out.println("• 并行注意力和MLP计算");
        System.out.println("• 旋转位置编码(RoPE)");
        System.out.println("• 稀疏注意力机制（大型模型）");
        System.out.println("• 强大的Few-shot学习能力");
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 演示前向传播
     */
    private static void demonstrateForwardPass() {
        System.out.println("⚡ 3. 前向传播演示");
        System.out.println("------------------\n");
        
        // 创建小型模型用于演示
        GPT3Model model = GPT3Model.createSmallModel("gpt3-demo");
        GPT3Config config = model.getConfig();
        
        // 创建示例输入
        int batchSize = 2;
        int seqLen = 10;
        NdArray inputTokens = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 填充随机token ID（模拟真实输入）
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int)(Math.random() * Math.min(1000, config.getVocabSize()));
                inputTokens.set(tokenId, b, s);
            }
        }
        
        System.out.println("输入形状: " + inputTokens.getShape());
        System.out.println("输入示例: [" + 
                          inputTokens.get(0, 0) + ", " + 
                          inputTokens.get(0, 1) + ", " + 
                          inputTokens.get(0, 2) + ", ...]");
        
        // 执行前向传播
        System.out.println("\n执行前向传播...");
        long startTime = System.currentTimeMillis();
        Variable output = model.forward(new Variable(inputTokens));
        long endTime = System.currentTimeMillis();
        
        NdArray outputData = output.getValue();
        System.out.println("✓ 前向传播完成");
        System.out.println("输出形状: " + outputData.getShape());
        System.out.println("处理时间: " + (endTime - startTime) + "ms");
        
        // 验证输出
        Shape expectedShape = Shape.of(batchSize, seqLen, config.getVocabSize());
        if (outputData.getShape().equals(expectedShape)) {
            System.out.println("✓ 输出形状验证通过");
        } else {
            System.out.println("❌ 输出形状不匹配，期望: " + expectedShape);
        }
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 演示文本生成
     */
    private static void demonstrateTextGeneration() {
        System.out.println("📖 4. 文本生成演示");
        System.out.println("------------------\n");
        
        GPT3Model model = GPT3Model.createSmallModel("gpt3-generator");
        
        // 创建起始序列
        NdArray startTokens = NdArray.of(Shape.of(1, 5));
        for (int i = 0; i < 5; i++) {
            startTokens.set(i + 1, 0, i);  // 简单的递增序列
        }
        
        System.out.println("起始序列: " + arrayToString(startTokens));
        
        // 生成文本
        System.out.println("开始生成...");
        int maxLength = 10;
        NdArray generated = model.generateSequence(startTokens, maxLength);
        
        System.out.println("生成结果: " + arrayToString(generated));
        System.out.println("生成长度: " + generated.getShape().getDimension(1));
        
        // 测试单步预测
        System.out.println("\n单步预测演示:");
        for (int i = 1; i <= 3; i++) {
            NdArray testSeq = NdArray.of(Shape.of(1, i));
            for (int j = 0; j < i; j++) {
                testSeq.set(j + 1, 0, j);
            }
            int nextToken = model.predictNextToken(testSeq);
            System.out.println("输入: " + arrayToString(testSeq) + " -> 预测: " + nextToken);
        }
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 演示Few-shot学习
     */
    private static void demonstrateFewShotLearning() {
        System.out.println("🎯 5. Few-shot学习演示");
        System.out.println("---------------------\n");
        
        GPT3Model model = GPT3Model.createMediumModel("gpt3-fewshot");
        
        // 创建Few-shot上下文（模拟任务示例）
        System.out.println("创建Few-shot上下文...");
        
        // 模拟分类任务的上下文：
        // 输入: "正面情感" -> 标签: 1
        // 输入: "负面情感" -> 标签: 0
        NdArray context = NdArray.of(Shape.of(1, 20));
        for (int i = 0; i < 20; i++) {
            // 创建模式化的上下文序列
            if (i < 10) {
                context.set(i + 100, 0, i);  // 第一个示例
            } else {
                context.set(i + 200, 0, i);  // 第二个示例
            }
        }
        
        System.out.println("Few-shot上下文: " + arrayToString(context));
        
        // 基于上下文生成
        System.out.println("\n基于上下文生成新输出...");
        int maxNewTokens = 15;
        NdArray fewShotResult = model.fewShotGenerate(context, maxNewTokens);
        
        System.out.println("Few-shot生成结果: " + arrayToString(fewShotResult));
        System.out.println("上下文长度: " + context.getShape().getDimension(1));
        System.out.println("总生成长度: " + fewShotResult.getShape().getDimension(1));
        System.out.println("新生成token数: " + (fewShotResult.getShape().getDimension(1) - context.getShape().getDimension(1)));
        
        System.out.println("\n💡 Few-shot学习优势:");
        System.out.println("• 无需微调即可适应新任务");
        System.out.println("• 通过示例快速理解任务模式");
        System.out.println("• 支持多种任务类型");
        System.out.println("• 展现强大的泛化能力");
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 演示旋转位置编码
     */
    private static void demonstrateRotaryEmbedding() {
        System.out.println("🔄 6. 旋转位置编码演示");
        System.out.println("---------------------\n");
        
        // 创建旋转位置编码
        int rotaryDim = 64;
        int maxSeqLen = 128;
        GPT3RotaryEmbedding rope = new GPT3RotaryEmbedding("demo_rope", rotaryDim, maxSeqLen);
        
        System.out.println("旋转位置编码配置:");
        System.out.println("• 旋转维度: " + rotaryDim);
        System.out.println("• 最大序列长度: " + maxSeqLen);
        System.out.println("• 基础频率: " + rope.getBase());
        
        // 生成位置编码
        int testSeqLen = 10;
        System.out.println("\n为序列长度 " + testSeqLen + " 生成旋转编码...");
        
        NdArray[] cosAndSin = rope.generateRotaryEmbedding(testSeqLen);
        NdArray cos = cosAndSin[0];
        NdArray sin = cosAndSin[1];
        
        System.out.println("Cos编码形状: " + cos.getShape());
        System.out.println("Sin编码形状: " + sin.getShape());
        
        // 显示前几个位置的编码值
        System.out.println("\n前3个位置的编码值示例:");
        for (int pos = 0; pos < Math.min(3, testSeqLen); pos++) {
            System.out.printf("位置%d: cos[0:4]=", pos);
            for (int d = 0; d < Math.min(4, rotaryDim); d++) {
                System.out.printf("%.3f ", cos.get(pos, d));
            }
            System.out.println("...");
        }
        
        // 测试旋转变换
        System.out.println("\n测试旋转变换应用...");
        int batchSize = 1;
        int numHeads = 8;
        int headDim = 64;
        
        NdArray testQuery = NdArray.likeRandomN(Shape.of(batchSize, testSeqLen, numHeads, headDim));
        NdArray testKey = NdArray.likeRandomN(Shape.of(batchSize, testSeqLen, numHeads, headDim));
        
        Variable[] rotated = rope.applyRotaryPositionEmbedding(
            new Variable(testQuery), new Variable(testKey), testSeqLen
        );
        
        System.out.println("✓ 旋转变换应用成功");
        System.out.println("输入Query形状: " + testQuery.getShape());
        System.out.println("输出Query形状: " + rotated[0].getValue().getShape());
        
        System.out.println("\n🔍 RoPE优势:");
        System.out.println("• 相对位置编码特性");
        System.out.println("• 支持任意长度序列");
        System.out.println("• 不增加参数量");
        System.out.println("• 保持向量模长不变");
        
        System.out.println("\n" + "=".repeat(50) + "\n");
    }
    
    /**
     * 辅助方法：将NdArray转换为字符串表示
     */
    private static String arrayToString(NdArray array) {
        if (array.getShape().getDimNum() != 2) {
            return array.getShape().toString();
        }
        
        int seqLen = array.getShape().getDimension(1);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(10, seqLen); i++) {
            if (i > 0) sb.append(", ");
            sb.append((int)array.get(0, i));
        }
        if (seqLen > 10) {
            sb.append(", ...");
        }
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * 性能基准测试
     */
    public static void benchmarkGPT3() {
        System.out.println("⏱️ GPT-3性能基准测试");
        System.out.println("---------------------\n");
        
        GPT3Model[] models = {
            GPT3Model.createSmallModel("small"),
            GPT3Model.createMediumModel("medium")
        };
        
        String[] modelNames = {"小型(125M)", "中型(350M)"};
        int[] seqLengths = {32, 64, 128};
        
        for (int m = 0; m < models.length; m++) {
            System.out.println(modelNames[m] + " 模型性能:");
            
            for (int seqLen : seqLengths) {
                NdArray input = NdArray.of(Shape.of(1, seqLen));
                for (int i = 0; i < seqLen; i++) {
                    input.set(i % 1000, 0, i);
                }
                
                // 预热
                models[m].forward(new Variable(input));
                
                // 测试
                long startTime = System.nanoTime();
                models[m].forward(new Variable(input));
                long endTime = System.nanoTime();
                
                double timeMs = (endTime - startTime) / 1_000_000.0;
                System.out.printf("  序列长度%-4d: %.2f ms\n", seqLen, timeMs);
            }
            System.out.println();
        }
    }
}