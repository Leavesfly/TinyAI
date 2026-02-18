package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;

/**
 * DeepSeek-V3训练完整演示
 * 
 * 展示V3模型的完整训练流程：
 * 1. 预训练 (Pretrain) - 大规模因果语言建模，包含MoE负载均衡
 * 2. 后训练 (Posttrain) - 任务感知微调，支持代码生成优化
 * 3. 推理 (Inference) - 多种生成策略，任务感知推理
 * 
 * V3核心特性：
 * - 混合专家模型 (MoE): 8专家+Top-2路由，激活率~25%
 * - 任务感知架构: 5种任务类型自适应
 * - 代码生成增强: 支持10种编程语言
 * - 多模态支持: 图像、视频特征融合
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3TrainDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 完整训练流程演示");
        System.out.println("=".repeat(80));
        System.out.println();
        
        printV3Features();
        
        // 示例1: 预训练
        example1_Pretrain();
        
        // 示例2: 后训练/微调
        example2_Posttrain();
        
        // 示例3: 代码生成后训练
        example3_CodePosttrain();
        
        // 示例4: 推理
        example4_Inference();
        
        // 示例5: 任务感知推理
        example5_TaskAwareInference();
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("所有V3训练示例完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 打印V3核心特性
     */
    private static void printV3Features() {
        System.out.println("""
            DeepSeek-V3 核心特性:
            
            1. 混合专家模型 (MoE)
               - 8个专家网络，Top-2门控路由
               - 参数激活率约25%（只激活2/8专家）
               - MoE负载均衡损失确保专家均匀使用
               
            2. 任务感知架构
               - REASONING: 增强推理任务
               - CODING: 代码生成任务（10种语言）
               - MATH: 数学推理任务
               - GENERAL: 通用对话任务
               - MULTIMODAL: 多模态理解任务
               
            3. 代码生成增强
               - 支持Python, Java, C++, JavaScript等10种语言
               - 4维质量评估: 正确性、可读性、效率、风格
               - 代码质量监控和优化
               
            4. 多模态支持
               - 图像特征融合
               - 视频帧序列理解
               - 跨模态推理能力
            """);
        System.out.println("=".repeat(80));
    }
    
    /**
     * 示例1: 预训练
     */
    private static void example1_Pretrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例1: DeepSeek-V3预训练 (含MoE负载均衡)");
        System.out.println("=".repeat(80));
        
        // 创建极小型模型用于演示（适合默认JVM堆内存）
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        
        System.out.println("\n模型配置:");
        System.out.println("  - 隐藏维度: " + config.getNEmbd());
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - 激活专家数: " + config.getTopK());
        System.out.println("  - 参数激活率: " + 
            String.format("%.1f%%", (float)config.getTopK() / config.getNumExperts() * 100));
        
        // 创建示例数据集
        int numSamples = 100;
        int seqLength = 32;
        int batchSize = 4;
        DeepSeekV3Dataset trainDataset = DeepSeekV3Dataset.createDummyDataset(
            numSamples, seqLength, config.getVocabSize(), batchSize
        );
        
        // 创建预训练器
        DeepSeekV3Pretrain pretrain = new DeepSeekV3Pretrain(model, trainDataset);
        
        // 配置训练参数
        pretrain.configure(
            2,           // maxEpochs (演示用小epoch)
            2.5e-4f,     // learningRate (V3推荐值)
            100,         // warmupSteps
            1.0f         // maxGradNorm
        );
        
        pretrain.setCheckpoint("./checkpoints/v3_pretrain_demo", 500);
        
        // 开始预训练
        System.out.println("\n开始预训练...");
        System.out.println("  - 学习率: 2.5e-4");
        System.out.println("  - MoE负载均衡权重: " + config.getLoadBalanceLossWeight());
        pretrain.train();
        
        // 打印统计
        DeepSeekV3Pretrain.TrainingStats stats = pretrain.getStats();
        System.out.println("\n预训练统计:");
        System.out.println("  - 总步数: " + stats.totalSteps);
        System.out.println("  - 平均语言模型损失: " + String.format("%.4f", stats.avgLoss));
        System.out.println("  - 平均MoE负载损失: " + String.format("%.6f", stats.avgMoeLoss));
        
        System.out.println("\n✓ 预训练示例完成!");
    }
    
    /**
     * 示例2: 后训练/微调
     */
    private static void example2_Posttrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例2: DeepSeek-V3后训练/微调 (任务感知)");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        
        // 创建训练和验证数据集
        int numTrainSamples = 40;
        int numValSamples = 10;
        int seqLength = 16;
        int batchSize = 2;
        
        DeepSeekV3Dataset trainDataset = DeepSeekV3Dataset.createDummyDataset(
            numTrainSamples, seqLength, config.getVocabSize(), batchSize
        );
        DeepSeekV3Dataset valDataset = DeepSeekV3Dataset.createDummyDataset(
            numValSamples, seqLength, config.getVocabSize(), batchSize
        );
        
        // 创建后训练器
        DeepSeekV3Posttrain posttrain = new DeepSeekV3Posttrain(
            model, trainDataset, valDataset
        );
        
        // 配置
        posttrain.configure(
            3,           // maxEpochs
            2.5e-5f,     // learningRate (比预训练小10倍)
            3            // patience (早停耐心值)
        );
        
        // 开始后训练
        System.out.println("\n开始后训练...");
        System.out.println("  - 学习率: 2.5e-5 (降低10倍)");
        System.out.println("  - 早停耐心值: 3");
        System.out.println("  - 任务感知优化: 启用");
        posttrain.train();
        
        System.out.println("\n✓ 后训练示例完成!");
    }
    
    /**
     * 示例3: 代码生成后训练
     */
    private static void example3_CodePosttrain() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例3: DeepSeek-V3代码生成后训练");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        
        // 创建代码生成数据集
        int numTrainSamples = 30;
        int numValSamples = 10;
        int seqLength = 32;  // 代码任务需要更长序列
        int batchSize = 2;
        
        DeepSeekV3Dataset trainDataset = DeepSeekV3Dataset.createCodeDataset(
            numTrainSamples, seqLength, config.getVocabSize(), batchSize,
            new String[]{"python", "java", "cpp", "javascript"}
        );
        DeepSeekV3Dataset valDataset = DeepSeekV3Dataset.createCodeDataset(
            numValSamples, seqLength, config.getVocabSize(), batchSize,
            new String[]{"python", "java"}
        );
        
        System.out.println("\n代码数据集信息:");
        System.out.println("  - 训练样本: " + numTrainSamples);
        System.out.println("  - 验证样本: " + numValSamples);
        System.out.println("  - 序列长度: " + seqLength);
        System.out.println("  - 支持语言: Python, Java, C++, JavaScript");
        
        // 创建后训练器
        DeepSeekV3Posttrain posttrain = new DeepSeekV3Posttrain(
            model, trainDataset, valDataset
        );
        
        // 配置（代码任务使用更小的学习率）
        posttrain.configure(
            4,           // maxEpochs (代码任务需要更多轮次)
            1e-5f,       // learningRate (更小)
            3            // patience
        );
        
        // 开始训练
        System.out.println("\n开始代码生成后训练...");
        posttrain.train();
        
        System.out.println("\n✓ 代码生成后训练完成!");
    }
    
    /**
     * 示例4: 推理
     */
    private static void example4_Inference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例4: DeepSeek-V3推理 (多种生成策略)");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        
        // 创建推理引擎
        DeepSeekV3Inference inference = new DeepSeekV3Inference(model);
        inference.setSeed(42);  // 固定随机种子
        
        // 准备提示词
        int[] promptIds = {1, 2, 3, 4, 5};
        int maxNewTokens = 10;
        
        System.out.println("\n提示词: " + java.util.Arrays.toString(promptIds));
        System.out.println("最大生成token数: " + maxNewTokens);
        
        // 策略1: 贪婪解码
        System.out.println("\n--- 策略1: 贪婪解码 ---");
        DeepSeekV3Inference.GenerationResult result1 = inference.generateGreedy(
            promptIds, maxNewTokens, TaskType.GENERAL
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result1.tokens));
        result1.printReasoningTrace();
        
        // 策略2: Temperature采样
        System.out.println("\n--- 策略2: Temperature采样 (temperature=0.8) ---");
        DeepSeekV3Inference.GenerationResult result2 = inference.generateWithTemperature(
            promptIds, maxNewTokens, 0.8f, TaskType.GENERAL
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result2.tokens));
        result2.printReasoningTrace();
        
        // 策略3: Top-K采样
        System.out.println("\n--- 策略3: Top-K采样 (k=50) ---");
        DeepSeekV3Inference.GenerationResult result3 = inference.generateTopK(
            promptIds, maxNewTokens, 50, TaskType.GENERAL
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result3.tokens));
        
        // 策略4: Top-P (Nucleus)采样
        System.out.println("\n--- 策略4: Top-P采样 (p=0.9) ---");
        DeepSeekV3Inference.GenerationResult result4 = inference.generateTopP(
            promptIds, maxNewTokens, 0.9f, TaskType.GENERAL
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(result4.tokens));
        
        System.out.println("\n✓ 推理示例完成!");
    }
    
    /**
     * 示例5: 任务感知推理
     */
    private static void example5_TaskAwareInference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例5: DeepSeek-V3任务感知推理");
        System.out.println("=".repeat(80));
        
        // 创建模型
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        
        // 创建推理引擎
        DeepSeekV3Inference inference = new DeepSeekV3Inference(model);
        
        // 准备提示词
        int[] promptIds = {1, 2, 3, 4, 5};
        int maxNewTokens = 8;
        
        System.out.println("\n展示不同任务类型的推理结果:");
        
        // 推理任务
        System.out.println("\n--- 任务类型: REASONING (推理) ---");
        var reasoningResult = inference.generateGreedy(
            promptIds, maxNewTokens, TaskType.REASONING
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(reasoningResult.tokens));
        System.out.println("平均MoE损失: " + String.format("%.6f", 
            reasoningResult.reasoningSteps.stream()
                .mapToDouble(s -> s.moeLoss)
                .average().orElse(0.0)));
        
        // 代码生成任务
        System.out.println("\n--- 任务类型: CODING (代码生成) ---");
        var codingResult = inference.generateGreedy(
            promptIds, maxNewTokens, TaskType.CODING
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(codingResult.tokens));
        System.out.println("平均置信度: " + String.format("%.4f", 
            codingResult.reasoningSteps.stream()
                .mapToDouble(s -> s.confidence)
                .average().orElse(0.0)));
        
        // 数学推理任务
        System.out.println("\n--- 任务类型: MATH (数学推理) ---");
        var mathResult = inference.generateGreedy(
            promptIds, maxNewTokens, TaskType.MATH
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(mathResult.tokens));
        
        // 多模态任务
        System.out.println("\n--- 任务类型: MULTIMODAL (多模态) ---");
        var multimodalResult = inference.generateGreedy(
            promptIds, maxNewTokens, TaskType.MULTIMODAL
        );
        System.out.println("生成序列: " + java.util.Arrays.toString(multimodalResult.tokens));
        
        System.out.println("\n✓ 任务感知推理示例完成!");
    }
    
    /**
     * 示例6: 完整训练流水线说明
     */
    public static void example6_FullPipeline() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("示例6: DeepSeek-V3完整训练流水线");
        System.out.println("=".repeat(80));
        
        System.out.println("""
            V3完整训练流程:
            
            ┌─────────────────────────────────────────────────────────────┐
            │ 阶段1: 预训练 (Pretrain)                                     │
            ├─────────────────────────────────────────────────────────────┤
            │ 数据: 大规模无标注文本语料                                    │
            │ 目标: 因果语言建模 + MoE负载均衡                              │
            │ 学习率: 2.5e-4                                               │
            │ 特性:                                                        │
            │   - MoE负载均衡损失确保8个专家均匀使用                        │
            │   - Warmup + Cosine学习率调度                                │
            │   - 梯度裁剪防止训练不稳定                                    │
            └─────────────────────────────────────────────────────────────┘
            
            ┌─────────────────────────────────────────────────────────────┐
            │ 阶段2: 后训练 (Posttrain/Finetune)                          │
            ├─────────────────────────────────────────────────────────────┤
            │ 数据: 任务特定标注数据                                        │
            │ 目标: 任务感知优化                                           │
            │ 学习率: 2.5e-5 (降低10倍)                                    │
            │ 特性:                                                        │
            │   - 5种任务类型: REASONING/CODING/MATH/GENERAL/MULTIMODAL    │
            │   - 代码质量4维评估: 正确性/可读性/效率/风格                  │
            │   - 早停机制防止过拟合 (patience=3)                          │
            │   - 验证集监控训练质量                                        │
            └─────────────────────────────────────────────────────────────┘
            
            ┌─────────────────────────────────────────────────────────────┐
            │ 阶段3: 推理部署 (Inference)                                  │
            ├─────────────────────────────────────────────────────────────┤
            │ 生成策略:                                                    │
            │   1. Greedy贪婪解码 - 确定性生成                             │
            │   2. Temperature采样 - 控制随机性(0.1-2.0)                  │
            │   3. Top-K采样 - 从前K个候选中采样                           │
            │   4. Top-P (Nucleus)采样 - 累积概率采样                     │
            │                                                              │
            │ 任务感知:                                                    │
            │   - 根据任务类型自动调整专家路由                              │
            │   - 代码生成时启用语言特定优化                                │
            │   - 推理追踪展示MoE负载和置信度                              │
            └─────────────────────────────────────────────────────────────┘
            
            关键优势:
            ✓ MoE架构: 25%激活率，显著降低计算成本
            ✓ 任务感知: 自适应5种任务类型，提升专业能力
            ✓ 代码优化: 10种语言+4维质量评估
            ✓ 多模态: 支持文本+图像+视频
            ✓ 可扩展: 轻松增加新专家和任务类型
            
            性能指标:
            - 参数总量: ~100B (假设)
            - 激活参数: ~25B (25%激活率)
            - 推理速度: 比Dense模型快4倍
            - 代码生成质量: SOTA级别
            """);
        
        System.out.println("✓ 流水线说明完成!");
    }
}
