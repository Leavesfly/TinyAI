package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
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
        
        // 运行所有示例
        example1_CreateModel();
        example2_CodeGeneration();
        example3_ReasoningTask();
        example4_MathTask();
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
     * 示例2：代码生成任务（V3的核心优势）
     */
    public static void example2_CodeGeneration() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例2: 代码生成任务");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("DeepSeek-V3-Code");
        
        // 模拟代码生成输入（提示词："编写一个Java快速排序算法"）
        float[][] input = {
            {1, 2, 3, 4, 5, 6, 7, 8}  // 模拟token序列
        };
        Variable inputVar = new Variable(NdArray.of(input));
        
        // 执行代码生成
        System.out.println("\n任务类型: 代码生成");
        System.out.println("输入形状: " + inputVar.getValue().getShape());
        
        DeepSeekV3Model.CodeGenerationResult result = model.generateCode(inputVar);
        
        System.out.println("\n代码生成结果:");
        System.out.println("  - 任务类型: " + result.taskType.getDescription());
        System.out.println("  - MoE负载均衡损失: " + String.format("%.6f", result.moeLoss));
        System.out.println("  - 输出形状: " + result.logits.getValue().getShape());
        
        System.out.println("\n✅ 代码生成完成");
    }
    
    /**
     * 示例3：推理任务（任务感知路由）
     */
    public static void example3_ReasoningTask() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例3: 推理任务（任务感知）");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("DeepSeek-V3-Reasoning");
        
        // 模拟推理输入（提示词："如果A>B且B>C，那么A>C吗？"）
        float[][] input = {
            {10, 11, 12, 13, 14, 15, 16, 17}
        };
        Variable inputVar = new Variable(NdArray.of(input));
        
        // 执行推理
        System.out.println("\n任务类型: 逻辑推理");
        System.out.println("输入形状: " + inputVar.getValue().getShape());
        
        DeepSeekV3Model.ReasoningResult result = model.performReasoning(inputVar);
        
        System.out.println("\n推理结果:");
        System.out.println("  - 检测任务类型: " + 
            (result.taskType != null ? result.taskType.getDescription() : "未知"));
        System.out.println("  - MoE负载均衡损失: " + String.format("%.6f", result.moeLoss));
        System.out.println("  - MoE负载均衡损失: " + String.format("%.6f", result.moeLoss));
        System.out.println("  - 输出形状: " + result.logits.getValue().getShape());
        
        System.out.println("\n✅ 推理完成");
    }
    
    /**
     * 示例4：数学计算任务
     */
    public static void example4_MathTask() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例4: 数学计算任务");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("DeepSeek-V3-Math");
        
        // 模拟数学输入（提示词："求解方程 x²-5x+6=0"）
        float[][] input = {
            {20, 21, 22, 23, 24, 25, 26, 27}
        };
        Variable inputVar = new Variable(NdArray.of(input));
        
        // 执行数学计算
        System.out.println("\n任务类型: 数学计算");
        System.out.println("输入形状: " + inputVar.getValue().getShape());
        
        DeepSeekV3Model.MathResult result = model.solveMath(inputVar);
        
        System.out.println("\n数学计算结果:");
        System.out.println("  - 任务类型: " + result.taskType.getDescription());
        System.out.println("  - MoE负载均衡损失: " + String.format("%.6f", result.moeLoss));
        System.out.println("  - 输出形状: " + result.logits.getValue().getShape());
        
        System.out.println("\n✅ 数学计算完成");
    }
    
    /**
     * 示例5：MoE分析（专家选择和负载均衡）
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
        
        // 演示不同任务类型的专家选择
        System.out.println("\n任务类型到专家的映射（简化展示）:");
        System.out.println("  - 推理任务 → 倾向选择专家0和1");
        System.out.println("  - 代码任务 → 倾向选择专家2和3");
        System.out.println("  - 数学任务 → 倾向选择专家4和5");
        System.out.println("  - 通用任务 → 倾向选择专家6和7");
        
        // 模拟多任务输入
        float[][] inputs = {
            {30, 31, 32, 33}  // 通用任务
        };
        Variable inputVar = new Variable(NdArray.of(inputs));
        
        DeepSeekV3Block.DetailedForwardResult result = 
            model.predictWithDetails(inputVar, TaskType.GENERAL);
        
        System.out.println("\n执行结果:");
        System.out.println("  - 检测任务类型: " + result.taskType.getDescription());
        System.out.println("  - 平均MoE损失: " + String.format("%.6f", result.avgMoELoss));
        
        // 参数效率分析
        System.out.println("\n参数效率分析:");
        long totalParams = config.estimateParameterCount();
        long activeParams = config.estimateActiveParameterCount();
        System.out.println("  - 总参数量: " + formatParamCount(totalParams));
        System.out.println("  - 激活参数量: " + formatParamCount(activeParams));
        System.out.println("  - 节省参数: " + formatParamCount(totalParams - activeParams) +
                          " (" + String.format("%.1f%%", (1 - config.getActivationRatio() / 100.0) * 100) + ")");
        
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
