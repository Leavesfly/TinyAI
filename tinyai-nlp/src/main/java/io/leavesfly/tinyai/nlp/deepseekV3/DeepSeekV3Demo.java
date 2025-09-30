package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;

/**
 * DeepSeek-V3 演示程序
 * 
 * 展示DeepSeek-V3模型的主要功能，包括：
 * 1. 模型创建和配置
 * 2. 前向传播演示
 * 3. MLA注意力机制演示
 * 4. MoE专家负载均衡分析
 * 5. 内存优化效果展示
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Demo {
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek-V3 模型演示程序 ===");
        System.out.println();
        
        try {
            // 1. 展示不同规模的模型创建
            demonstrateModelCreation();
            
            // 2. 演示基本前向传播
            demonstrateForwardPass();
            
            // 3. 演示MLA注意力机制
            demonstrateMLAAttention();
            
            // 4. 演示MoE专家系统
            demonstrateMoEExperts();
            
            // 5. 演示内存优化效果
            demonstrateMemoryOptimization();
            
            // 6. 演示配置对比
            demonstrateConfigComparison();
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 演示程序结束 ===");
    }
    
    /**
     * 演示不同规模的模型创建
     */
    private static void demonstrateModelCreation() {
        System.out.println("1. === 模型创建演示 ===");
        
        // 创建不同规模的模型
        System.out.println("创建不同规模的DeepSeek-V3模型:");
        
        DeepSeekV3Model tinyModel = DeepSeekV3Factory.createTinyModel("deepseek_v3_tiny");
        DeepSeekV3Model smallModel = DeepSeekV3Factory.createSmallModel("deepseek_v3_small");
        DeepSeekV3Model standardModel = DeepSeekV3Factory.createStandardModel("deepseek_v3_standard");
        
        // 打印模型信息
        System.out.println("\n模型参数对比:");
        System.out.printf("%-12s | %-15s | %-15s | %-10s%n", 
                         "Model", "Total Params", "Active Params", "Efficiency");
        System.out.println("-------------|-----------------|-----------------|----------");
        
        printModelStats("Tiny", tinyModel);
        printModelStats("Small", smallModel);
        printModelStats("Standard", standardModel);
        
        System.out.println();
    }
    
    /**
     * 演示基本前向传播
     */
    private static void demonstrateForwardPass() {
        System.out.println("2. === 前向传播演示 ===");
        
        // 使用小型模型进行演示
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("demo_model");
        
        // 创建模拟输入数据
        int batchSize = 2;
        int seqLen = 16;
        NdArray inputTokens = createRandomTokens(batchSize, seqLen, model.getConfig().getVocabSize());
        
        System.out.printf("输入形状: [%d, %d] (batch_size, seq_len)%n", batchSize, seqLen);
        System.out.printf("词汇表大小: %d%n", model.getConfig().getVocabSize());
        
        // 执行前向传播
        long startTime = System.currentTimeMillis();
        Variable input = new Variable(inputTokens);
        Variable output = model.layerForward(input);
        long endTime = System.currentTimeMillis();
        
        Shape outputShape = output.getValue().getShape();
        System.out.printf("输出形状: [%d, %d, %d] (batch_size, seq_len, vocab_size)%n", 
                         outputShape.getDimension(0), outputShape.getDimension(1), outputShape.getDimension(2));
        System.out.printf("前向传播耗时: %d ms%n", endTime - startTime);
        
        // 显示负载均衡统计
        double loadBalanceLoss = model.computeTotalLoadBalancingLoss();
        System.out.printf("负载均衡损失: %.6f%n", loadBalanceLoss);
        
        System.out.println();
    }
    
    /**
     * 演示MLA注意力机制
     */
    private static void demonstrateMLAAttention() {
        System.out.println("3. === MLA注意力机制演示 ===");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createTinyModel("mla_demo");
        DeepSeekV3Config config = model.getConfig();
        
        System.out.println("MLA配置信息:");
        System.out.printf("  - 模型维度: %d%n", config.getDModel());
        System.out.printf("  - 注意力头数: %d%n", config.getNumHeads());
        System.out.printf("  - MLA潜在维度: %d%n", config.getDMLA());
        System.out.printf("  - 缓存压缩比: %.2fx%n", (double) config.getDModel() / config.getDMLA());
        
        // 计算内存节省
        int seqLen = 1024;
        long memorySavings = model.getTotalMemorySavings(seqLen);
        System.out.printf("  - 内存节省 (seq_len=%d): %.1f KB%n", seqLen, memorySavings / 1024.0);
        
        // 演示KV缓存启用/禁用
        System.out.println("\nKV缓存管理:");
        model.enableKVCache();
        System.out.println("  - KV缓存已启用");
        
        model.disableKVCache();
        System.out.println("  - KV缓存已禁用");
        
        System.out.println();
    }
    
    /**
     * 演示MoE专家系统
     */
    private static void demonstrateMoEExperts() {
        System.out.println("4. === MoE专家系统演示 ===");
        
        DeepSeekV3Model model = DeepSeekV3Factory.createSmallModel("moe_demo");
        DeepSeekV3Config config = model.getConfig();
        
        System.out.println("MoE配置信息:");
        System.out.printf("  - 专家数量: %d%n", config.getNumExperts());
        System.out.printf("  - Top-K选择: %d%n", config.getTopK());
        System.out.printf("  - 共享专家数: %d%n", config.getNumSharedExperts());
        System.out.printf("  - 专家隐藏维度: %d%n", config.getDExpert());
        
        // 执行前向传播以生成统计数据
        NdArray inputTokens = createRandomTokens(4, 32, config.getVocabSize());
        Variable input = new Variable(inputTokens);
        model.layerForward(input);
        
        // 显示专家使用统计
        System.out.println("\n专家使用统计:");
        List<long[]> allLayerUsage = model.getAllLayersExpertUsage();
        for (int i = 0; i < Math.min(3, allLayerUsage.size()); i++) {
            long[] usage = allLayerUsage.get(i);
            System.out.printf("  层 %d: ", i);
            for (int j = 0; j < Math.min(8, usage.length); j++) {
                if (j > 0) System.out.print(", ");
                System.out.printf("%d", usage[j]);
            }
            if (usage.length > 8) System.out.print("...");
            System.out.println();
        }
        
        // 重置统计并再次运行
        model.resetAllMoEStats();
        System.out.println("\n统计信息已重置");
        
        System.out.println();
    }
    
    /**
     * 演示内存优化效果
     */
    private static void demonstrateMemoryOptimization() {
        System.out.println("5. === 内存优化效果演示 ===");
        
        DeepSeekV3Config baseConfig = DeepSeekV3Config.createSmallConfig();
        
        // 创建推理优化模型
        DeepSeekV3Model optimizedModel = DeepSeekV3Factory.createInferenceOptimizedModel(
            "optimized_model", baseConfig);
        
        System.out.println("推理优化配置:");
        System.out.printf("  - FlashAttention: %s%n", baseConfig.isUseFlashAttention() ? "启用" : "禁用");
        System.out.printf("  - KV缓存: %s%n", baseConfig.isUseKVCache() ? "启用" : "禁用");
        System.out.printf("  - MLA缓存: %s%n", baseConfig.isUseMLACache() ? "启用" : "禁用");
        
        // 计算不同序列长度的内存节省
        int[] seqLengths = {512, 1024, 2048, 4096};
        System.out.println("\n内存节省对比:");
        System.out.printf("%-10s | %-15s | %-15s%n", "Seq Length", "Memory Saved", "Savings");
        System.out.println("-----------|-----------------|---------------");
        
        for (int seqLen : seqLengths) {
            long savings = optimizedModel.getTotalMemorySavings(seqLen);
            double savingsPercentage = calculateMemorySavingsPercentage(baseConfig, seqLen);
            System.out.printf("%-10d | %-15s | %-14.1f%%%n", 
                             seqLen, formatBytes(savings), savingsPercentage);
        }
        
        System.out.println();
    }
    
    /**
     * 演示配置对比
     */
    private static void demonstrateConfigComparison() {
        System.out.println("6. === 配置对比演示 ===");
        
        // 打印所有预设配置信息
        DeepSeekV3Factory.printPresetConfigsInfo();
        
        // 创建自定义配置
        System.out.println("\n自定义配置示例:");
        DeepSeekV3Config customConfig = new DeepSeekV3Config(
            32000,  // vocabSize
            512,    // dModel
            8,      // numLayers
            8,      // numHeads
            1024,   // maxSeqLength
            16,     // numExperts
            4       // topK
        );
        
        System.out.println(customConfig.getSummary());
        
        // 验证配置
        try {
            customConfig.validate();
            System.out.println("自定义配置验证通过");
        } catch (Exception e) {
            System.out.println("自定义配置验证失败: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 创建随机token输入
     */
    private static NdArray createRandomTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.zeros(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int randomToken = (int) (Math.random() * vocabSize);
                tokens.set(randomToken, b, s);
            }
        }
        
        return tokens;
    }
    
    /**
     * 打印模型统计信息
     */
    private static void printModelStats(String modelName, DeepSeekV3Model model) {
        long totalParams = model.getTotalParameterCount();
        long activeParams = model.getActiveParameterCount();
        double efficiency = (double) activeParams / totalParams * 100;
        
        System.out.printf("%-12s | %-15s | %-15s | %-9.1f%%%n",
                         modelName,
                         formatNumber(totalParams),
                         formatNumber(activeParams),
                         efficiency);
    }
    
    /**
     * 格式化数字显示
     */
    private static String formatNumber(long number) {
        if (number >= 1_000_000_000L) {
            return String.format("%.1fB", number / 1_000_000_000.0);
        } else if (number >= 1_000_000L) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000L) {
            return String.format("%.1fK", number / 1_000.0);
        } else {
            return String.valueOf(number);
        }
    }
    
    /**
     * 格式化字节显示
     */
    private static String formatBytes(long bytes) {
        if (bytes >= 1_024_000_000L) {
            return String.format("%.1f GB", bytes / 1_024_000_000.0);
        } else if (bytes >= 1_024_000L) {
            return String.format("%.1f MB", bytes / 1_024_000.0);
        } else if (bytes >= 1_024L) {
            return String.format("%.1f KB", bytes / 1_024.0);
        } else {
            return bytes + " B";
        }
    }
    
    /**
     * 计算内存节省百分比
     */
    private static double calculateMemorySavingsPercentage(DeepSeekV3Config config, int seqLen) {
        // 传统注意力缓存大小
        long traditionalCache = 2L * seqLen * config.getDModel() * 4; // 2 * seq_len * d_model * sizeof(float)
        
        // MLA缓存大小
        long mlaCache = 2L * seqLen * config.getDMLA() * 4; // 2 * seq_len * d_mla * sizeof(float)
        
        return ((double) (traditionalCache - mlaCache) / traditionalCache) * 100;
    }
}