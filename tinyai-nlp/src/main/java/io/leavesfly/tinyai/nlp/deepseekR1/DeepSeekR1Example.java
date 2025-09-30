package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nlp.SimpleTokenizer;

/**
 * DeepSeek-R1 模型使用示例
 * 
 * 这个示例展示了DeepSeek-R1模型的基本使用方法，
 * 包括模型创建、配置、推理等核心功能。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Example {
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek-R1 使用示例 ===\n");
        
        try {
            // 1. 基本使用示例
            basicUsageExample();
            
            // 2. 配置自定义示例
            customConfigExample();
            
            // 3. 推理能力示例
            reasoningCapabilityExample();
            
            // 4. 模型对比示例
            modelComparisonExample();
            
        } catch (Exception e) {
            System.err.println("示例运行出错: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 示例运行完成 ===");
    }
    
    /**
     * 基本使用示例
     */
    public static void basicUsageExample() {
        System.out.println("1. 基本使用示例");
        System.out.println(repeatChar("-", 30));
        
        // 创建一个中等规模的模型
        DeepSeekR1Model model = DeepSeekR1Factory.createMediumModel("example_basic");
        
        // 打印模型配置信息
        model.printModelInfo();
        
        // 创建简单的输入
        String question = "解释人工智能的发展历程";
        System.out.println("问题: " + question);
        
        // 使用tokenizer处理输入
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        int[] tokenIds = tokenizer.encode(question);
        
        // 创建输入数组
        NdArray input = NdArray.of(Shape.of(1, tokenIds.length));
        for (int i = 0; i < tokenIds.length; i++) {
            input.set(tokenIds[i], 0, i);
        }
        
        // 执行推理
        System.out.println("执行推理...");
        ReasoningResult result = model.performReasoning(input);
        
        // 显示结果
        if (result.isSuccess()) {
            System.out.println("推理成功!");
            System.out.printf("置信度: %.2f%%\n", result.getConfidenceScore() * 100);
            System.out.printf("推理步骤数: %d\n", result.getNumReasoningSteps());
        } else {
            System.out.println("推理失败: " + result.getErrorMessage());
        }
        
        System.out.println("✓ 基本使用示例完成\n");
    }
    
    /**
     * 自定义配置示例
     */
    public static void customConfigExample() {
        System.out.println("2. 自定义配置示例");
        System.out.println(repeatChar("-", 30));
        
        // 创建自定义配置
        DeepSeekR1Config config = new DeepSeekR1Config();
        
        // 调整推理参数
        config.setMaxReasoningSteps(15);           // 增加推理步骤
        config.setReasoningThreshold(0.8);         // 提高置信度要求
        config.setEnableReasoning(true);           // 启用推理模式
        config.setEnableCoT(true);                 // 启用思维链
        
        // 调整模型参数
        config.setDModel(512);                     // 调整模型维度
        config.setNumLayers(8);                    // 调整层数
        config.setMaxSeqLength(512);               // 调整最大序列长度
        
        // 打印配置信息
        config.printConfig();
        
        // 使用自定义配置创建模型
        DeepSeekR1Model customModel = DeepSeekR1Factory.createModel("custom_model", config);
        
        System.out.println("自定义模型创建成功!");
        System.out.printf("模型参数: %d维度, %d层, %d最大推理步骤\n", 
                         customModel.getDModel(), 
                         customModel.getNumLayers(),
                         customModel.getMaxReasoningSteps());
        
        System.out.println("✓ 自定义配置示例完成\n");
    }
    
    /**
     * 推理能力示例
     */
    public static void reasoningCapabilityExample() {
        System.out.println("3. 推理能力示例");
        System.out.println(repeatChar("-", 30));
        
        // 创建推理优化模型
        DeepSeekR1Model reasoningModel = DeepSeekR1Factory.createReasoningModel("reasoning_example");
        
        // 测试不同类型的推理问题
        String[] testQuestions = {
            "如果所有A都是B，所有B都是C，那么所有A都是C吗？",
            "一个长方形的长是8cm，宽是6cm，它的面积和周长分别是多少？",
            "解释为什么天空是蓝色的？"
        };
        
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        
        for (int i = 0; i < testQuestions.length; i++) {
            String question = testQuestions[i];
            System.out.printf("\n问题 %d: %s\n", i + 1, question);
            
            try {
                // 准备输入
                int[] tokenIds = tokenizer.encode(question);
                NdArray input = NdArray.of(Shape.of(1, tokenIds.length));
                for (int j = 0; j < tokenIds.length; j++) {
                    input.set(tokenIds[j], 0, j);
                }
                
                // 执行推理
                long startTime = System.currentTimeMillis();
                ReasoningResult result = reasoningModel.performReasoning(input);
                long endTime = System.currentTimeMillis();
                
                // 显示结果
                if (result.isSuccess()) {
                    System.out.printf("推理成功 (耗时: %d ms)\n", endTime - startTime);
                    System.out.printf("置信度: %.2f%%, 步骤数: %d\n", 
                                     result.getConfidenceScore() * 100, 
                                     result.getNumReasoningSteps());
                    
                    // 显示推理步骤
                    if (!result.getReasoningSteps().isEmpty()) {
                        System.out.println("推理过程:");
                        for (int s = 0; s < Math.min(3, result.getReasoningSteps().size()); s++) {
                            System.out.printf("  步骤 %d: %s\n", s + 1, result.getReasoningSteps().get(s));
                        }
                    }
                } else {
                    System.out.println("推理失败: " + result.getErrorMessage());
                }
                
            } catch (Exception e) {
                System.out.println("处理失败: " + e.getMessage());
            }
        }
        
        System.out.println("\n✓ 推理能力示例完成\n");
    }
    
    /**
     * 模型对比示例
     */
    public static void modelComparisonExample() {
        System.out.println("4. 模型对比示例");
        System.out.println(repeatChar("-", 30));
        
        // 创建不同规模的模型
        DeepSeekR1Model[] models = {
            DeepSeekR1Factory.createTinyModel("tiny_comparison"),
            DeepSeekR1Factory.createMediumModel("medium_comparison"),
            DeepSeekR1Factory.createReasoningModel("reasoning_comparison")
        };
        
        String[] modelNames = {"Tiny", "Medium", "Reasoning"};
        
        // 测试问题
        String testQuestion = "解释机器学习和深度学习的区别";
        System.out.println("测试问题: " + testQuestion);
        
        // 准备输入
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        int[] tokenIds = tokenizer.encode(testQuestion);
        NdArray input = NdArray.of(Shape.of(1, tokenIds.length));
        for (int i = 0; i < tokenIds.length; i++) {
            input.set(tokenIds[i], 0, i);
        }
        
        System.out.println("\n模型性能对比:");
        System.out.printf("%-12s %-10s %-10s %-10s %-8s\n", 
                         "模型", "推理时间", "推理步骤", "置信度", "状态");
        System.out.println(repeatChar("-", 55));
        
        for (int i = 0; i < models.length; i++) {
            DeepSeekR1Model model = models[i];
            String modelName = modelNames[i];
            
            try {
                long startTime = System.currentTimeMillis();
                ReasoningResult result = model.performReasoning(input);
                long endTime = System.currentTimeMillis();
                
                System.out.printf("%-12s %-10d %-10d %-10.3f %-8s\n",
                                 modelName,
                                 endTime - startTime,
                                 result.getNumReasoningSteps(),
                                 result.getConfidenceScore(),
                                 result.isSuccess() ? "成功" : "失败");
                
            } catch (Exception e) {
                System.out.printf("%-12s %-10s %-10s %-10s %-8s\n",
                                 modelName, "ERROR", "ERROR", "ERROR", "失败");
            }
        }
        
        System.out.println("\n分析:");
        System.out.println("- Tiny模型: 速度快，资源消耗低，适合快速原型");
        System.out.println("- Medium模型: 平衡性能和资源，适合大多数应用");
        System.out.println("- Reasoning模型: 推理能力强，适合复杂推理任务");
        
        System.out.println("✓ 模型对比示例完成\n");
    }
    
    /**
     * 工具方法：打印分隔线
     */
    private static String repeatChar(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}