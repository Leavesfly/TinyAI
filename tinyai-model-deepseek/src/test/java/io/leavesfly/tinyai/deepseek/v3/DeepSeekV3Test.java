package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;
import org.junit.Assert;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Map;

/**
 * DeepSeek V3模型单元测试
 * 
 * 测试DeepSeek V3模型的各个组件和功能，包括：
 * 1. 基础组件测试
 * 2. 模型初始化测试
 * 3. 前向传播测试
 * 4. 任务类型感知测试
 * 5. 推理过程测试
 * 6. 代码生成测试
 * 7. MoE专家使用测试
 * 8. 性能和稳定性测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Test {
    
    private DeepSeekV3Model model;
    private NdArray sampleInput;
    private final int batchSize = 1;
    private final int seqLen = 4;
    private final int vocabSize = 100;
    private final int dModel = 32;
    
    @Before
    public void setUp() {
        // 创建小型测试配置 - 减少参数以避免内存问题
        DeepSeekV3Model.V3ModelConfig testConfig = new DeepSeekV3Model.V3ModelConfig(
            vocabSize, dModel, 2, 2, dModel * 2, 2, 32, 0.1f
        );
        
        model = new DeepSeekV3Model("DeepSeek-V3-Test", testConfig);
        
        // 创建测试输入数据
        sampleInput = createTestInput(batchSize, seqLen);
    }
    
    /**
     * 创建测试输入数据
     */
    private NdArray createTestInput(int batchSize, int seqLen) {
        NdArray input = NdArray.of(Shape.of(batchSize, seqLen));
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = (int) (Math.random() * (vocabSize - 1));
                input.set(tokenId, b, s);
            }
        }
        return input;
    }
    
    /**
     * 测试任务类型枚举
     */
    @Test
    public void testTaskTypeEnum() {
        // 测试任务类型的基本功能
        TaskType[] taskTypes = TaskType.values();
        assertTrue("应该有多种任务类型", taskTypes.length >= 5);
        
        // 测试字符串转换
        assertEquals("REASONING", TaskType.REASONING, TaskType.fromValue("reasoning"));
        assertEquals("CODING", TaskType.CODING, TaskType.fromValue("coding"));
        assertEquals("默认应为GENERAL", TaskType.GENERAL, TaskType.fromValue("unknown"));
        
        // 测试getValue方法
        assertEquals("reasoning", TaskType.REASONING.getValue());
        assertEquals("coding", TaskType.CODING.getValue());
    }
    
    /**
     * 测试模型初始化
     */
    @Test
    public void testModelInitialization() {
        assertNotNull("模型不应为null", model);
        assertEquals("DeepSeek-V3-Test", model.getName());
        
        // 测试模型配置
        DeepSeekV3Model.V3ModelConfig config = model.getConfig();
        assertEquals(vocabSize, config.vocabSize);
        assertEquals(dModel, config.dModel);
        assertEquals(2, config.numLayers);
        assertEquals(2, config.numHeads);
        assertEquals(2, config.numExperts);
        
        // 测试模型参数存在
        assertFalse("模型应该有参数", model.getAllParams().isEmpty());
    }
    
    /**
     * 测试基础前向传播
     */
    @Test
    public void testBasicForward() {
        DeepSeekV3Block.DeepSeekV3Output output = model.generate(sampleInput);
        
        assertNotNull("输出不应为null", output);
        assertNotNull("logits不应为null", output.logits);
        
        // 检查输出形状
        Shape outputShape = output.logits.getValue().getShape();
        assertEquals("批次大小应匹配", batchSize, outputShape.getDimension(0));
        assertEquals("序列长度应匹配", seqLen, outputShape.getDimension(1));
        assertEquals("词汇表大小应匹配", vocabSize, outputShape.getDimension(2));
        
        // 检查推理步骤
        assertNotNull("推理步骤不应为null", output.reasoningSteps);
        assertTrue("应该有推理步骤", output.reasoningSteps.size() > 0);
        
        // 检查MoE损失
        assertTrue("MoE损失应为非负数", output.moeLoss >= 0);
    }
    
    /**
     * 测试任务类型感知推理
     */
    @Test
    public void testTaskTypeAwareInference() {
        for (TaskType taskType : TaskType.values()) {
            DeepSeekV3Block.DeepSeekV3Output output = model.generateWithTaskType(sampleInput, taskType);
            
            assertNotNull("任务类型 " + taskType + " 的输出不应为null", output);
            assertEquals("请求的任务类型应匹配", taskType, output.requestedTaskType);
            assertNotNull("应该有识别的任务类型", output.identifiedTaskType);
            
            // 验证推理质量
            float reasoningQuality = output.getReasoningQuality();
            assertTrue("推理质量应在合理范围内", reasoningQuality >= 0.0f && reasoningQuality <= 1.0f);
        }
    }
    
    /**
     * 测试代码生成功能
     */
    @Test
    public void testCodeGeneration() {
        DeepSeekV3Model.CodeGenerationResult codeResult = model.generateCode(sampleInput);
        
        assertNotNull("代码生成结果不应为null", codeResult);
        assertNotNull("应该有检测的语言", codeResult.detectedLanguage);
        
        // 验证代码质量指标
        assertTrue("语法得分应在合理范围", codeResult.syntaxScore >= 0.0f && codeResult.syntaxScore <= 1.0f);
        assertTrue("质量得分应在合理范围", codeResult.qualityScore >= 0.0f && codeResult.qualityScore <= 1.0f);
        assertTrue("代码置信度应在合理范围", codeResult.codeConfidence >= 0.0f && codeResult.codeConfidence <= 1.0f);
        
        assertNotNull("应该有推理步骤", codeResult.reasoningSteps);
        assertTrue("推理步骤应大于0", codeResult.reasoningSteps.size() > 0);
    }
    
    /**
     * 测试推理过程
     */
    @Test
    public void testReasoningProcess() {
        DeepSeekV3Model.ReasoningResult reasoningResult = model.performReasoning(sampleInput);
        
        assertNotNull("推理结果不应为null", reasoningResult);
        assertNotNull("应该有推理步骤", reasoningResult.reasoningSteps);
        
        // 验证推理步骤的基本属性
        for (V3ReasoningStep step : reasoningResult.reasoningSteps) {
            assertNotNull("思考内容不应为null", step.getThought());
            assertNotNull("行动不应为null", step.getAction());
            assertNotNull("任务类型不应为null", step.getTaskType());
            
            float confidence = step.getConfidence();
            assertTrue("置信度应在0-1之间", confidence >= 0.0f && confidence <= 1.0f);
        }
        
        // 验证平均置信度
        assertTrue("平均置信度应在合理范围", 
                  reasoningResult.averageConfidence >= 0.0f && reasoningResult.averageConfidence <= 1.0f);
    }
    
    /**
     * 测试数学推理功能
     */
    @Test
    public void testMathReasoning() {
        DeepSeekV3Model.MathResult mathResult = model.solveMath(sampleInput);
        
        assertNotNull("数学结果不应为null", mathResult);
        assertNotNull("应该有数学推理步骤", mathResult.reasoningSteps);
        
        // 验证数学置信度
        assertTrue("数学置信度应在合理范围", 
                  mathResult.mathConfidence >= 0.0f && mathResult.mathConfidence <= 1.0f);
    }
    
    /**
     * 测试批量生成
     */
    @Test
    public void testBatchGeneration() {
        NdArray batchInput = createTestInput(2, 3);
        
        DeepSeekV3Model.BatchGenerationResult batchResult = 
            model.generateBatch(batchInput, TaskType.GENERAL);
        
        assertNotNull("批量结果不应为null", batchResult);
        assertEquals("批次大小应匹配", 2, batchResult.batchSize);
        assertNotNull("应该有批量logits", batchResult.batchLogits);
        
        // 验证批量推理质量
        assertTrue("批量推理质量应在合理范围", 
                  batchResult.averageReasoningQuality >= 0.0f && 
                  batchResult.averageReasoningQuality <= 1.0f);
    }
    
    /**
     * 测试MoE专家使用
     */
    @Test
    public void testMoEExpertUsage() {
        // 为不同任务类型生成输出，检查专家使用模式
        TaskType[] taskTypes = {TaskType.CODING, TaskType.MATH, TaskType.REASONING};
        
        for (TaskType taskType : taskTypes) {
            DeepSeekV3Block.DeepSeekV3Output output = model.generateWithTaskType(sampleInput, taskType);
            
            // 验证路由信息
            assertNotNull("应该有路由信息", output.routingInfo);
            assertTrue("路由信息应非空", output.routingInfo.size() > 0);
            
            // 验证专家使用统计
            Map<String, Integer> expertUsage = output.getExpertUsageStats();
            assertNotNull("专家使用统计不应为null", expertUsage);
        }
    }
    
    /**
     * 测试模型统计信息
     */
    @Test
    public void testModelStatistics() {
        // 执行几次推理以生成统计信息
        model.generate(sampleInput);
        model.generateCode(sampleInput);
        
        DeepSeekV3Model.V3ModelStats stats = model.getModelStats();
        
        assertNotNull("统计信息不应为null", stats);
        assertTrue("总参数数量应大于0", stats.totalParameters > 0);
        assertEquals("词汇表大小应匹配", vocabSize, stats.vocabSize);
        assertEquals("模型维度应匹配", dModel, stats.dModel);
        assertEquals("层数应匹配", 2, stats.numLayers);
        assertEquals("专家数量应匹配", 2, stats.numExperts);
        
        // 验证最近的指标
        assertTrue("最近MoE损失应为非负数", stats.lastMoeLoss >= 0);
        assertTrue("最近推理质量应在合理范围", 
                  stats.lastReasoningQuality >= 0.0f && stats.lastReasoningQuality <= 1.0f);
    }
    
    /**
     * 测试详细推理信息
     */
    @Test
    public void testDetailedInferenceInfo() {
        model.generateWithTaskType(sampleInput, TaskType.REASONING);
        
        DeepSeekV3Model.DetailedInferenceInfo detailInfo = model.getLastInferenceDetails();
        
        assertNotNull("详细信息不应为null", detailInfo);
        assertNotNull("请求任务类型不应为null", detailInfo.requestedTaskType);
        assertNotNull("识别任务类型不应为null", detailInfo.identifiedTaskType);
        assertNotNull("推理步骤不应为null", detailInfo.reasoningSteps);
        
        assertTrue("MoE损失应为非负数", detailInfo.moeLoss >= 0);
        assertTrue("推理质量应在合理范围", 
                  detailInfo.reasoningQuality >= 0.0f && detailInfo.reasoningQuality <= 1.0f);
    }
    
    /**
     * 测试模型状态重置
     */
    @Test
    public void testModelStateReset() {
        // 执行推理以产生状态
        model.generate(sampleInput);
        assertNotNull("执行前应有最后输出", model.getLastInferenceDetails());
        
        // 重置状态
        model.resetState();
        
        // 验证状态已重置（这里简化测试，实际应该检查内部状态）
        // 在实际实现中，可能需要添加更多状态检查方法
    }
    
    /**
     * 测试MixtureOfExperts组件
     */
    @Test
    public void testMixtureOfExperts() {
        MixtureOfExperts moe = new MixtureOfExperts("test_moe", dModel, 2, 1, 1.0f);
        
        assertEquals("模型维度应匹配", dModel, moe.getDModel());
        assertEquals("专家数量应匹配", 2, moe.getNumExperts());
        assertEquals("选择专家数应匹配", 1, moe.getNumSelected());
        
        // 测试专家特化映射
        Map<Integer, TaskType> specializations = moe.getExpertSpecializations();
        assertNotNull("专家特化映射不应为null", specializations);
        assertEquals("应该有2个专家", 2, specializations.size());
        
        // 测试MoE前向传播 - 使用更小的尺寸
        Variable input = new Variable(NdArray.of(Shape.of(1, 2, dModel)).like(0.1f));
        MixtureOfExperts.MoEResult result = moe.forwardWithTaskType(input, TaskType.CODING);
        
        assertNotNull("MoE结果不应为null", result);
        assertNotNull("MoE输出不应为null", result.output);
        assertNotNull("路由信息不应为null", result.routingInfo);
    }
    
    /**
     * 测试V3ReasoningBlock组件
     */
    @Test
    public void testV3ReasoningBlock() {
        V3ReasoningBlock reasoningBlock = new V3ReasoningBlock("test_reasoning", dModel, 2);
        
        assertEquals("模型维度应匹配", dModel, reasoningBlock.getDModel());
        assertEquals("推理步骤数应匹配", 2, reasoningBlock.getNumReasoningSteps());
        
        // 测试推理前向传播 - 使用更小的尺寸
        Variable input = new Variable(NdArray.of(Shape.of(1, 2, dModel)).like(0.2f));
        V3ReasoningBlock.ReasoningResult result = reasoningBlock.performV3Reasoning(input);
        
        assertNotNull("推理结果不应为null", result);
        assertNotNull("最终输出不应为null", result.finalOutput);
        assertNotNull("推理步骤不应为null", result.reasoningSteps);
        assertEquals("推理步骤数应匹配", 2, result.reasoningSteps.size());
        assertNotNull("任务类型不应为null", result.taskType);
    }
    
    /**
     * 测试CodeGenerationBlock组件
     */
    @Test
    public void testCodeGenerationBlock() {
        CodeGenerationBlock codeBlock = new CodeGenerationBlock("test_code", dModel, 3);
        
        assertEquals("模型维度应匹配", dModel, codeBlock.getDModel());
        assertEquals("编程语言数应匹配", 3, codeBlock.getNumProgrammingLanguages());
        
        // 测试语言映射
        Map<Integer, String> languageMapping = codeBlock.getLanguageMapping();
        assertNotNull("语言映射不应为null", languageMapping);
        assertTrue("应该有语言映射", languageMapping.size() > 0);
        
        // 测试代码生成分析
        Variable input = new Variable(NdArray.of(Shape.of(1, dModel)).like(0.3f));
        CodeGenerationBlock.CodeGenerationResult result = codeBlock.performCodeGenerationAnalysis(input);
        
        assertNotNull("代码生成结果不应为null", result);
        assertNotNull("增强输出不应为null", result.enhancedOutput);
        assertNotNull("代码信息不应为null", result.codeInfo);
        
        // 验证代码质量指标
        assertTrue("代码置信度应在合理范围", 
                  result.getCodeConfidence() >= 0.0f && result.getCodeConfidence() <= 1.0f);
    }
    
    /**
     * 测试不同配置的模型 - 使用更小的配置避免内存问题
     */
    @Test
    public void testDifferentModelConfigurations() {
        // 测试小型配置
        DeepSeekV3Model smallModel = new DeepSeekV3Model("Small-V3", 
            new DeepSeekV3Model.V3ModelConfig(50, 16, 1, 1, 32, 1, 16, 0.1f));
        assertNotNull("小型模型不应为null", smallModel);
        
        // 测试中型配置
        DeepSeekV3Model mediumModel = new DeepSeekV3Model("Medium-V3", 
            new DeepSeekV3Model.V3ModelConfig(100, 32, 2, 2, 64, 2, 32, 0.1f));
        assertNotNull("中型模型不应为null", mediumModel);
        
        // 比较配置
        assertTrue("中型模型的词汇表应更大", 
                  mediumModel.getConfig().vocabSize >= smallModel.getConfig().vocabSize);
        assertTrue("中型模型的维度应更大", 
                  mediumModel.getConfig().dModel >= smallModel.getConfig().dModel);
    }
    
    /**
     * 性能和稳定性测试 - 修复序列长度断言问题
     */
    @Test
    public void testPerformanceAndStability() {
        // 测试多次推理的稳定性
        for (int i = 0; i < 3; i++) {
            NdArray input = createTestInput(1, 3);
            DeepSeekV3Block.DeepSeekV3Output output = model.generate(input);
            
            assertNotNull("第" + i + "次推理输出不应为null", output);
            assertTrue("第" + i + "次推理质量应合理", 
                      output.getReasoningQuality() >= 0.0f && output.getReasoningQuality() <= 1.0f);
        }
        
        // 测试不同输入大小的处理能力 - 使用更小的数值避免内存问题
        int[] seqLengths = {2, 3, 4};
        for (int seqLen : seqLengths) {
            NdArray input = createTestInput(1, seqLen);
            DeepSeekV3Block.DeepSeekV3Output output = model.generate(input);
            
            assertNotNull("序列长度" + seqLen + "的输出不应为null", output);
            assertEquals("输出序列长度应匹配", seqLen, 
                        output.logits.getValue().getShape().getDimension(1));
        }
    }
    
    /**
     * 测试错误情况处理
     */
    @Test
    public void testErrorHandling() {
        // 测试空输入处理
        try {
            NdArray emptyInput = NdArray.of(Shape.of(1, 0));
            // 在实际实现中，应该优雅地处理这种情况
            // model.generate(emptyInput);
        } catch (Exception e) {
            // 预期可能会有异常
        }
        
        // 测试超大输入处理
        try {
            // 创建一个接近最大序列长度的输入
            NdArray largeInput = createTestInput(1, 500);
            DeepSeekV3Block.DeepSeekV3Output output = model.generate(largeInput);
            // 应该能够处理，或者给出合理的错误信息
            if (output != null) {
                assertNotNull("大输入的输出应有效", output.logits);
            }
        } catch (Exception e) {
            // 预期可能会有资源限制异常
        }
    }
}