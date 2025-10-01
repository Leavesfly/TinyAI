package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.ml.Monitor;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import java.util.List;

/**
 * DeepSeek V3模型演示类
 * 
 * 展示了DeepSeek V3模型的各种功能和用法，包括：
 * 1. 模型初始化和配置
 * 2. 不同任务类型的推理演示
 * 3. 代码生成功能演示
 * 4. 推理过程分析
 * 5. MoE专家使用统计
 * 6. 强化学习训练演示
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Demo {
    
    /**
     * 重复字符生成辅助方法（Java 8兼容）
     */
    private static String repeatChar(char c, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(c);
        }
        return sb.toString();
    }
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek V3 模型演示 ===\n");
        
        try {
            // 1. 模型初始化演示
            demonstrateModelInitialization();
            
            // 2. 基础推理演示
            demonstrateBasicInference();
            
            // 3. 任务类型感知推理演示
            demonstrateTaskTypeAwareInference();
            
            // 4. 代码生成演示
            demonstrateCodeGeneration();
            
            // 5. 推理过程分析演示
            demonstrateReasoningAnalysis();
            
            // 6. MoE专家使用演示
            demonstrateExpertUsage();
            
            // 7. 模型统计信息演示
            demonstrateModelStatistics();
            
            // 8. 强化学习训练演示（简化版）
            demonstrateRLTraining();
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== DeepSeek V3 演示完成 ===");
    }
    
    /**
     * 1. 模型初始化演示
     */
    private static void demonstrateModelInitialization() {
        System.out.println("1. 模型初始化演示");
        System.out.println(repeatChar('-', 50));
        
        // 创建不同规模的模型配置
        DeepSeekV3Model.V3ModelConfig smallConfig = DeepSeekV3Model.V3ModelConfig.getSmallConfig();
        DeepSeekV3Model.V3ModelConfig defaultConfig = DeepSeekV3Model.V3ModelConfig.getDefaultConfig();
        DeepSeekV3Model.V3ModelConfig largeConfig = DeepSeekV3Model.V3ModelConfig.getLargeConfig();
        
        // 创建模型实例
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Demo", defaultConfig);
        
        // 打印模型架构信息
        model.printArchitecture();
        
        // 显示模型配置对比
        System.out.println("\n模型配置对比:");
        System.out.printf("小型配置: 词汇%d, 维度%d, 层数%d, 专家%d%n", 
                         smallConfig.vocabSize, smallConfig.dModel, smallConfig.numLayers, smallConfig.numExperts);
        System.out.printf("标准配置: 词汇%d, 维度%d, 层数%d, 专家%d%n", 
                         defaultConfig.vocabSize, defaultConfig.dModel, defaultConfig.numLayers, defaultConfig.numExperts);
        System.out.printf("大型配置: 词汇%d, 维度%d, 层数%d, 专家%d%n", 
                         largeConfig.vocabSize, largeConfig.dModel, largeConfig.numLayers, largeConfig.numExperts);
        
        System.out.println();
    }
    
    /**
     * 2. 基础推理演示
     */
    private static void demonstrateBasicInference() {
        System.out.println("2. 基础推理演示");
        System.out.println(repeatChar('-', 50));
        
        // 创建模型
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Basic");
        
        // 创建模拟输入数据
        NdArray inputIds = createSampleInput(2, 10); // batch_size=2, seq_len=10
        
        System.out.println("输入数据形状: " + inputIds.getShape());
        
        // 执行基础推理
        DeepSeekV3Block.DeepSeekV3Output output = model.generate(inputIds);
        
        System.out.println("输出logits形状: " + output.logits.getValue().getShape());
        System.out.println("推理步骤数量: " + output.reasoningSteps.size());
        System.out.println("推理质量评分: " + String.format("%.3f", output.getReasoningQuality()));
        System.out.println("MoE损失: " + String.format("%.4f", output.moeLoss));
        System.out.println("识别的任务类型: " + output.identifiedTaskType);
        
        System.out.println();
    }
    
    /**
     * 3. 任务类型感知推理演示
     */
    private static void demonstrateTaskTypeAwareInference() {
        System.out.println("3. 任务类型感知推理演示");
        System.out.println(repeatChar('-', 50));
        
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-TaskAware");
        NdArray inputIds = createSampleInput(1, 8);
        
        // 测试不同任务类型
        TaskType[] taskTypes = {TaskType.GENERAL, TaskType.REASONING, TaskType.CODING, TaskType.MATH};
        
        for (TaskType taskType : taskTypes) {
            System.out.println("任务类型: " + taskType);
            
            DeepSeekV3Block.DeepSeekV3Output output = model.generateWithTaskType(inputIds, taskType);
            
            System.out.printf("  推理质量: %.3f%n", output.getReasoningQuality());
            System.out.printf("  MoE损失: %.4f%n", output.moeLoss);
            System.out.printf("  推理步骤: %d%n", output.reasoningSteps.size());
            
            if (taskType == TaskType.CODING && output.codeInfo != null) {
                System.out.printf("  代码置信度: %.3f%n", output.getCodeConfidence());
            }
            
            System.out.println();
        }
    }
    
    /**
     * 4. 代码生成演示
     */
    private static void demonstrateCodeGeneration() {
        System.out.println("4. 代码生成演示");
        System.out.println(repeatChar('-', 50));
        
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-CodeGen");
        NdArray inputIds = createSampleInput(1, 12);
        
        // 执行代码生成
        DeepSeekV3Model.CodeGenerationResult codeResult = model.generateCode(inputIds);
        
        System.out.println("代码生成结果:");
        System.out.println("  检测语言: " + codeResult.detectedLanguage);
        System.out.printf("  语法得分: %.3f%n", codeResult.syntaxScore);
        System.out.printf("  质量得分: %.3f%n", codeResult.qualityScore);
        System.out.printf("  代码置信度: %.3f%n", codeResult.codeConfidence);
        System.out.println("  推理步骤数: " + codeResult.reasoningSteps.size());
        
        // 显示推理步骤
        System.out.println("\n代码生成推理步骤:");
        for (int i = 0; i < Math.min(3, codeResult.reasoningSteps.size()); i++) {
            V3ReasoningStep step = codeResult.reasoningSteps.get(i);
            System.out.printf("  步骤%d: %s (置信度: %.3f)%n", 
                             i+1, step.getThought(), step.getConfidence());
        }
        
        System.out.println();
    }
    
    /**
     * 5. 推理过程分析演示
     */
    private static void demonstrateReasoningAnalysis() {
        System.out.println("5. 推理过程分析演示");
        System.out.println(repeatChar('-', 50));
        
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Reasoning");
        NdArray inputIds = createSampleInput(1, 15);
        
        // 执行推理任务
        DeepSeekV3Model.ReasoningResult reasoningResult = model.performReasoning(inputIds);
        
        System.out.println("推理分析结果:");
        System.out.printf("  平均置信度: %.3f%n", reasoningResult.averageConfidence);
        System.out.println("  识别任务类型: " + reasoningResult.identifiedTaskType);
        System.out.println("  推理步骤详情:");
        
        for (int i = 0; i < reasoningResult.reasoningSteps.size(); i++) {
            V3ReasoningStep step = reasoningResult.reasoningSteps.get(i);
            System.out.printf("    第%d步: %s%n", i+1, step.getThought());
            System.out.printf("           行动: %s%n", step.getAction());
            System.out.printf("           置信度: %.3f%n", step.getConfidence());
            System.out.printf("           验证: %s%n", step.getVerification());
            if (step.getSelfCorrection() != null) {
                System.out.printf("           自我纠错: %s%n", step.getSelfCorrection());
            }
            System.out.println();
        }
    }
    
    /**
     * 6. MoE专家使用演示
     */
    private static void demonstrateExpertUsage() {
        System.out.println("6. MoE专家使用演示");
        System.out.println(repeatChar('-', 50));
        
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-MoE");
        
        // 为不同任务类型执行推理，观察专家使用模式
        TaskType[] tasks = {TaskType.CODING, TaskType.MATH, TaskType.REASONING};
        
        for (TaskType taskType : tasks) {
            NdArray inputIds = createSampleInput(1, 10);
            DeepSeekV3Block.DeepSeekV3Output output = model.generateWithTaskType(inputIds, taskType);
            
            System.out.println("任务类型: " + taskType);
            System.out.println("专家使用统计:");
            
            output.getExpertUsageStats().forEach((expertId, count) -> {
                System.out.printf("  %s: %d次%n", expertId, count);
            });
            
            System.out.printf("MoE损失: %.4f%n", output.moeLoss);
            System.out.println();
        }
    }
    
    /**
     * 7. 模型统计信息演示
     */
    private static void demonstrateModelStatistics() {
        System.out.println("7. 模型统计信息演示");
        System.out.println(repeatChar('-', 50));
        
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Stats");
        
        // 执行几次推理以收集统计信息
        for (int i = 0; i < 3; i++) {
            NdArray inputIds = createSampleInput(1, 8 + i * 2);
            TaskType taskType = TaskType.values()[i % TaskType.values().length];
            model.generateWithTaskType(inputIds, taskType);
        }
        
        // 获取和显示统计信息
        DeepSeekV3Model.V3ModelStats stats = model.getModelStats();
        
        System.out.println("模型统计信息:");
        System.out.println("  总参数量: " + stats.totalParameters);
        System.out.println("  词汇表大小: " + stats.vocabSize);
        System.out.println("  模型维度: " + stats.dModel);
        System.out.println("  Transformer层数: " + stats.numLayers);
        System.out.println("  专家数量: " + stats.numExperts);
        System.out.println("  最大序列长度: " + stats.maxSeqLen);
        System.out.printf("  最近MoE损失: %.4f%n", stats.lastMoeLoss);
        System.out.printf("  最近推理质量: %.3f%n", stats.lastReasoningQuality);
        System.out.printf("  最近代码置信度: %.3f%n", stats.lastCodeConfidence);
        
        if (stats.expertUsageStats != null) {
            System.out.println("  专家使用统计: " + stats.expertUsageStats);
        }
        
        // 获取详细推理信息
        DeepSeekV3Model.DetailedInferenceInfo detailInfo = model.getLastInferenceDetails();
        if (detailInfo != null) {
            System.out.println("\n最近推理详情:");
            detailInfo.printSummary();
        }
        
        System.out.println();
    }
    
    /**
     * 8. 强化学习训练演示（简化版）
     */
    private static void demonstrateRLTraining() {
        System.out.println("8. 强化学习训练演示（简化版）");
        System.out.println(repeatChar('-', 50));
        
        // 创建模型和训练器
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-RL", 
                                                   DeepSeekV3Model.V3ModelConfig.getSmallConfig());
        
        Monitor monitor = new Monitor();
        // 创建一个简化的评估器实现
        Evaluator evaluator = new Evaluator() {
            @Override
            public void evaluate() {
                System.out.println("简化评估器: 评估完成");
            }
        };
        
        V3RLTrainer trainer = new V3RLTrainer(2, monitor, evaluator, // 只训练2个epoch用于演示
                                             V3RLTrainer.V3TrainingConfig.getDefaultConfig());
        
        System.out.println("V3强化学习训练器已创建");
        System.out.println("训练配置: 默认配置");
        System.out.println("注意: 这是一个简化的演示，实际训练需要真实数据集");
        
        // 在实际应用中，这里会初始化真实的数据集和损失函数
        System.out.println("训练演示完成（简化版）");
        
        System.out.println();
    }
    
    /**
     * 创建示例输入数据（token IDs）
     */
    private static NdArray createSampleInput(int batchSize, int seqLen) {
        // 创建 token IDs 输入 [batch_size, seq_len]
        NdArray input = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 填充随机token IDs（在实际词汇表范围内）
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int) (Math.random() * 1000); // 随机token ID
                input.set(tokenId, b, s);
            }
        }
        
        return input;
    }
    
    /**
     * 显示推理步骤详情
     */
    private static void printReasoningSteps(List<V3ReasoningStep> steps, int maxSteps) {
        System.out.println("推理步骤详情:");
        
        for (int i = 0; i < Math.min(maxSteps, steps.size()); i++) {
            V3ReasoningStep step = steps.get(i);
            System.out.printf("  步骤%d [%s]: %s%n", 
                             i+1, step.getTaskType(), step.getThought());
            System.out.printf("         置信度: %.3f, 验证: %s%n", 
                             step.getConfidence(), step.getVerification());
        }
        
        if (steps.size() > maxSteps) {
            System.out.printf("  ... 还有 %d 个步骤%n", steps.size() - maxSteps);
        }
    }
    
    /**
     * 运行完整的使用示例
     */
    public static void runComprehensiveExample() {
        System.out.println("\n=== 综合使用示例 ===");
        
        // 1. 创建模型
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Comprehensive");
        
        // 2. 准备不同类型的输入
        NdArray generalInput = createSampleInput(1, 10);
        NdArray codingInput = createSampleInput(1, 12);
        NdArray mathInput = createSampleInput(1, 8);
        
        // 3. 执行不同任务
        System.out.println("执行通用任务...");
        DeepSeekV3Block.DeepSeekV3Output generalOutput = model.generate(generalInput);
        
        System.out.println("执行代码生成任务...");
        DeepSeekV3Model.CodeGenerationResult codeOutput = model.generateCode(codingInput);
        
        System.out.println("执行数学推理任务...");
        DeepSeekV3Model.MathResult mathOutput = model.solveMath(mathInput);
        
        // 4. 比较结果
        System.out.println("\n任务结果比较:");
        System.out.printf("通用任务推理质量: %.3f%n", generalOutput.getReasoningQuality());
        System.out.printf("代码生成置信度: %.3f%n", codeOutput.codeConfidence);
        System.out.printf("数学推理置信度: %.3f%n", mathOutput.mathConfidence);
        
        // 5. 显示模型整体表现
        DeepSeekV3Model.V3ModelStats finalStats = model.getModelStats();
        System.out.println("\n模型整体表现: " + finalStats);
        
        System.out.println("综合示例完成！");
    }
}