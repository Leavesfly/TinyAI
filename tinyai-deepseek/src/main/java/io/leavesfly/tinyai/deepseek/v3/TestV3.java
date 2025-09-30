package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;

/**
 * DeepSeek V3 模型测试类
 * 
 * 提供完整的V3模型测试用例，包括：
 * 1. 基础模型创建和前向传播测试
 * 2. 任务类型识别测试
 * 3. MoE专家路由测试
 * 4. 推理模块测试
 * 5. 代码生成模块测试
 * 6. 性能基准测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TestV3 {
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek V3 模型测试开始 ===");
        
        try {
            // 运行所有测试
            testBasicModelCreation();
            testModelForward();
            testTaskTypeRecognition();
            testMoERouting();
            testReasoningModule();
            testCodeGenerationModule();
            testPerformanceBenchmark();
            
            System.out.println("\n=== 所有测试通过! ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试基础模型创建
     */
    public static void testBasicModelCreation() {
        System.out.println("\n1. 测试基础模型创建...");
        
        // 创建迷你模型
        DeepSeekV3Model tinyModel = DeepSeekV3Model.createTinyV3("tiny_v3");
        System.out.println("✓ 迷你V3模型创建成功");
        System.out.println(tinyModel.getV3ModelInfo());
        
        // 创建小型模型
        DeepSeekV3Model smallModel = DeepSeekV3Model.createSmallV3("small_v3");
        System.out.println("\n✓ 小型V3模型创建成功");
        
        // 验证模型参数
        assert tinyModel.getTotalParameterCount() > 0 : "模型参数数量应大于0";
        assert tinyModel.getActiveParameterCount() > 0 : "激活参数数量应大于0";
        assert tinyModel.getActiveParameterCount() <= tinyModel.getTotalParameterCount() : "激活参数不应超过总参数";
        
        System.out.println("✓ 模型参数验证通过");
    }
    
    /**
     * 测试模型前向传播
     */
    public static void testModelForward() {
        System.out.println("\n2. 测试模型前向传播...");
        
        DeepSeekV3Model model = DeepSeekV3Model.createTinyV3("test_forward");
        
        // 创建测试输入
        int batchSize = 2;
        int seqLen = 10;
        NdArray inputIds = createRandomInputIds(batchSize, seqLen, model.getVocabSize());
        Variable input = new Variable(inputIds);
        
        // 前向传播
        Variable output = model.layerForward(input);
        
        // 验证输出形状
        Shape expectedShape = Shape.of(batchSize, seqLen, model.getVocabSize());
        assert output.getValue().getShape().equals(expectedShape) : 
            "输出形状不匹配，期望: " + expectedShape + ", 实际: " + output.getValue().getShape();
        
        System.out.println("✓ 前向传播测试通过");
        System.out.println("  输入形状: " + input.getValue().getShape());
        System.out.println("  输出形状: " + output.getValue().getShape());
        
        // 验证处理统计
        assert model.getTotalTokensProcessed() == batchSize * seqLen : "Token处理统计不正确";
        System.out.println("✓ Token处理统计正确: " + model.getTotalTokensProcessed());
    }
    
    /**
     * 测试任务类型识别
     */
    public static void testTaskTypeRecognition() {
        System.out.println("\n3. 测试任务类型识别...");
        
        // 测试所有任务类型
        for (TaskType taskType : TaskType.values()) {
            System.out.println("  测试任务类型: " + taskType.getValue());
            
            // 验证任务类型属性
            assert taskType.getValue() != null && !taskType.getValue().isEmpty() : 
                "任务类型值不应为空";
            
            // 测试推荐专家数量
            int recommendedExperts = taskType.getRecommendedExpertCount();
            assert recommendedExperts > 0 && recommendedExperts <= 10 : 
                "推荐专家数量应在合理范围内";
            
            System.out.println("    推荐专家数: " + recommendedExperts);
            System.out.println("    是否复杂推理: " + taskType.isComplexReasoning());
            System.out.println("    是否创造性: " + taskType.isCreative());
        }
        
        // 测试任务类型解析
        TaskType parsed = TaskType.fromValue("coding");
        assert parsed == TaskType.CODING : "任务类型解析失败";
        
        System.out.println("✓ 任务类型识别测试通过");
    }
    
    /**
     * 测试MoE专家路由
     */
    public static void testMoERouting() {
        System.out.println("\n4. 测试MoE专家路由...");
        
        DeepSeekV3Model model = DeepSeekV3Model.createTinyV3("test_moe");
        
        // 创建测试输入
        NdArray inputIds = createRandomInputIds(1, 5, model.getVocabSize());
        Variable input = new Variable(inputIds);
        
        // 前向传播以生成路由信息
        model.layerForward(input);
        
        // 检查路由信息
        List<ExpertRoutingInfo> routingInfos = model.getAllRoutingInfo();
        System.out.println("  路由信息层数: " + routingInfos.size());
        
        // 验证专家使用统计
        List<long[]> expertUsage = model.getAllLayersExpertUsage();
        assert expertUsage.size() == model.getNumLayers() : "专家使用统计层数不匹配";
        
        for (int i = 0; i < expertUsage.size(); i++) {
            long[] usage = expertUsage.get(i);
            assert usage.length == model.getNumExperts() : "专家数量不匹配";
            
            long totalUsage = 0;
            for (long count : usage) {
                totalUsage += count;
            }
            System.out.println("  第" + (i+1) + "层专家总使用次数: " + totalUsage);
        }
        
        // 测试负载均衡
        double loadBalanceLoss = model.computeTotalLoadBalancingLoss();
        System.out.println("  总负载均衡损失: " + String.format("%.6f", loadBalanceLoss));
        
        // 重置统计
        model.resetAllMoEStats();
        double resetLoss = model.computeTotalLoadBalancingLoss();
        assert resetLoss == 0.0 : "重置后负载均衡损失应为0";
        
        System.out.println("✓ MoE专家路由测试通过");
    }
    
    /**
     * 测试推理模块
     */
    public static void testReasoningModule() {
        System.out.println("\n5. 测试推理模块...");
        
        DeepSeekV3Model model = DeepSeekV3Model.createTinyV3("test_reasoning");
        
        // 创建测试输入
        NdArray inputIds = createRandomInputIds(1, 8, model.getVocabSize());
        Variable input = new Variable(inputIds);
        
        // 前向传播生成推理链
        model.layerForward(input);
        
        // 检查推理链
        List<V3ReasoningStep> reasoningChain = model.getCurrentReasoningChain();
        System.out.println("  推理步骤数: " + reasoningChain.size());
        
        assert reasoningChain.size() > 0 : "推理链不应为空";
        
        // 验证推理步骤质量
        for (int i = 0; i < reasoningChain.size(); i++) {
            V3ReasoningStep step = reasoningChain.get(i);
            
            assert step.getConfidence() >= 0.0 && step.getConfidence() <= 1.0 : 
                "置信度应在[0,1]范围内";
            assert step.getThought() != null && !step.getThought().isEmpty() : 
                "思考内容不应为空";
            assert step.getTaskType() != null : "任务类型不应为空";
            
            System.out.println("  步骤" + (i+1) + ": " + step.getSummary());
        }
        
        // 测试推理链报告
        String reasoningReport = model.getReasoningChainReport();
        assert reasoningReport != null && !reasoningReport.isEmpty() : "推理报告不应为空";
        
        System.out.println("✓ 推理模块测试通过");
    }
    
    /**
     * 测试代码生成模块
     */
    public static void testCodeGenerationModule() {
        System.out.println("\n6. 测试代码生成模块...");
        
        DeepSeekV3Model model = DeepSeekV3Model.createTinyV3("test_codegen");
        
        // 创建代码生成任务的输入
        NdArray inputIds = createRandomInputIds(1, 6, model.getVocabSize());
        Variable input = new Variable(inputIds);
        
        // 模拟代码任务（需要修改模型以支持任务类型传递）
        model.layerForward(input);
        
        // 获取代码生成报告
        String codeReport = model.getCodeGenerationReport();
        assert codeReport != null && !codeReport.isEmpty() : "代码生成报告不应为空";
        
        System.out.println("  代码生成报告:");
        System.out.println(codeReport);
        
        // 测试代码生成模块的语言支持
        CodeGenerationModule codeGen = model.getCodeGeneration();
        List<String> supportedLangs = codeGen.getSupportedLanguages();
        
        assert supportedLangs.size() > 0 : "应支持至少一种编程语言";
        System.out.println("  支持的编程语言: " + supportedLangs);
        
        System.out.println("✓ 代码生成模块测试通过");
    }
    
    /**
     * 性能基准测试
     */
    public static void testPerformanceBenchmark() {
        System.out.println("\n7. 性能基准测试...");
        
        DeepSeekV3Model model = DeepSeekV3Model.createTinyV3("benchmark");
        
        int numIterations = 10;
        int batchSize = 2;
        int seqLen = 20;
        
        long totalTime = 0;
        
        for (int i = 0; i < numIterations; i++) {
            // 创建随机输入
            NdArray inputIds = createRandomInputIds(batchSize, seqLen, model.getVocabSize());
            Variable input = new Variable(inputIds);
            
            // 计时前向传播
            long startTime = System.currentTimeMillis();
            Variable output = model.layerForward(input);
            long endTime = System.currentTimeMillis();
            
            long iterationTime = endTime - startTime;
            totalTime += iterationTime;
            
            if (i == 0) {
                // 验证第一次的输出
                assert output != null : "输出不应为null";
                assert output.getValue().getShape().size() > 0 : "输出应有正确的大小";
            }
        }
        
        double avgTime = (double) totalTime / numIterations;
        double tokensPerSecond = (batchSize * seqLen * 1000.0) / avgTime;
        
        System.out.println("  性能统计:");
        System.out.println("    迭代次数: " + numIterations);
        System.out.println("    平均推理时间: " + String.format("%.2f", avgTime) + "ms");
        System.out.println("    处理速度: " + String.format("%.1f", tokensPerSecond) + " tokens/秒");
        System.out.println("    总处理Token数: " + model.getTotalTokensProcessed());
        
        // 显示模型完整信息
        System.out.println("\n  模型详细信息:");
        System.out.println(model.getV3ModelInfo());
        
        System.out.println("✓ 性能基准测试完成");
    }
    
    /**
     * 创建随机输入ID
     * 
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @param vocabSize 词汇表大小
     * @return 随机输入ID数组
     */
    private static NdArray createRandomInputIds(int batchSize, int seqLen, int vocabSize) {
        double[] data = new double[batchSize * seqLen];
        
        for (int i = 0; i < data.length; i++) {
            // 生成0到vocabSize-1的随机整数
            data[i] = Math.floor(Math.random() * vocabSize);
        }
        
        return NdArray.of(data).reshape(Shape.of(batchSize, seqLen));
    }
    
    /**
     * 演示不同任务类型的处理
     */
    public static void demonstrateTaskTypes() {
        System.out.println("\n=== 任务类型演示 ===");
        
        DeepSeekV3Model model = DeepSeekV3Model.createSmallV3("task_demo");
        
        // 演示不同任务类型
        TaskType[] taskTypes = {TaskType.REASONING, TaskType.CODING, TaskType.MATH, TaskType.GENERAL};
        
        for (TaskType taskType : taskTypes) {
            System.out.println("\n处理任务类型: " + taskType.getValue());
            
            // 创建该任务类型的测试输入
            NdArray inputIds = createRandomInputIds(1, 12, model.getVocabSize());
            Variable input = new Variable(inputIds);
            
            // 重置模型状态
            model.resetState();
            
            // 前向传播
            Variable output = model.layerForward(input);
            
            // 显示结果
            System.out.println("  推理步骤数: " + model.getCurrentReasoningChain().size());
            System.out.println("  MoE负载均衡损失: " + 
                String.format("%.6f", model.computeTotalLoadBalancingLoss()));
            
            if (taskType == TaskType.CODING && model.getCodeInfo() != null) {
                System.out.println("  代码置信度: " + 
                    String.format("%.3f", model.getCodeInfo().getCodeConfidence()));
            }
        }
    }
}
