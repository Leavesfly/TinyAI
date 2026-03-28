package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DeepSeek-V3模型演示程序
 * 
 * 演示DeepSeek-V3的核心功能：
 * 1. 模型创建和配置
 * 2. 混合专家(MoE)推理
 * 3. 任务感知路由
 * 4. 代码生成优化
 * 5. 多种推理策略
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Demo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 模型演示程序");
        System.out.println("=".repeat(80));
        
        // 运行示例
        example1_CreateModel();
        example5_MoEAnalysis();
        
        System.out.println("=".repeat(80));
        System.out.println("所有示例完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1：创建模型并查看配置
     */
    public static void example1_CreateModel() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例1: 创建DeepSeek-V3模型");
        System.out.println("=".repeat(80));
        
        // 创建小型模型（用于演示）
        DeepSeekV3Model model = DeepSeekV3Model.createSmallModel("DeepSeek-V3-Small");
        
        // 打印模型信息
        model.printModelInfo();
        
        // 打印配置摘要
        System.out.println("\n" + model.getConfigSummary());
        
        System.out.println("\n✅ 模型创建成功");
    }
    
    /**
     * 示例2：MoE分析（专家选择和负载均衡）
     */
    public static void example5_MoEAnalysis() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例5: MoE混合专家分析");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Config config = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-MoE", config);
        
        // 打印MoE配置
        System.out.println("\nMoE配置:");
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getTopK());
        System.out.println("  - 专家隐藏层维度: " + config.getExpertHiddenDim());
        System.out.println("  - 负载均衡损失权重: " + config.getLoadBalanceLossWeight());
        System.out.println("  - 参数激活率: " + String.format("%.2f%%", config.getActivationRatio()));
        
        // 模拟多任务输入
        float[][] inputs = {
            {30, 31, 32, 33}  // 通用任务
        };
        Variable inputVar = new Variable(NdArray.of(inputs));
        
        DeepSeekV3Block.DetailedForwardResult result = model.predictWithDetails(inputVar);
        
        System.out.println("\n执行结果:");
        System.out.println("  - 平均MoE损失: " + String.format("%.6f", result.avgMoELoss));
        
        // 参数效率分析
        System.out.println("\n参数效率分析:");
        long totalParams = config.estimateParameterCount();
        long activeParams = config.estimateActiveParameterCount();
        System.out.println("  - 总参数量: " + formatParamCount(totalParams));
        System.out.println("  - 激活参数量: " + formatParamCount(activeParams));
        System.out.println("  - 节省参数: " + formatParamCount(totalParams - activeParams) +
                          " (" + String.format("%.1f%%", 100.0 - config.getActivationRatio()) + ")");
        
        System.out.println("\n✅ MoE分析完成");
    }
    
    /**
     * 格式化参数数量
     */
    private static String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
