package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeek R1模型单元测试
 * 
 * 测试涵盖：
 * 1. 各个组件的基本功能
 * 2. 模型集成测试
 * 3. 推理和反思功能
 * 4. 边界条件和异常处理
 */
public class DeepSeekR1Test {
    
    private DeepSeekR1Model testModel;
    private static final int VOCAB_SIZE = 100;
    private static final int D_MODEL = 64;
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 4;
    private static final int MAX_SEQ_LEN = 32;
    
    @BeforeEach
    void setUp() {
        // 创建测试用的小型模型
        testModel = new DeepSeekR1Model(
            "TestModel", VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, 
            D_MODEL * 2, MAX_SEQ_LEN, 0.1
        );
    }
    
    @Test
    @DisplayName("测试模型配置验证")
    void testModelConfigurationValidation() {
        // 测试有效配置
        assertTrue(testModel.validateConfiguration(), "有效配置应该通过验证");
        
        // 测试无效的词汇表大小
        DeepSeekR1Model invalidModel1 = new DeepSeekR1Model("Invalid", 0, D_MODEL);
        assertFalse(invalidModel1.validateConfiguration(), "词汇表大小为0应该验证失败");
        
        // 测试无效的模型维度（不能被头数整除）
        DeepSeekR1Model invalidModel2 = new DeepSeekR1Model("Invalid", VOCAB_SIZE, 63, 2, 4);
        assertFalse(invalidModel2.validateConfiguration(), "模型维度不能被头数整除应该验证失败");
    }
    
    @Test
    @DisplayName("测试ReasoningBlock基本功能")
    void testReasoningBlock() {
        ReasoningBlock reasoningBlock = new ReasoningBlock("test_reasoning", D_MODEL, 3);
        
        // 测试初始化
        assertEquals(D_MODEL, reasoningBlock.getDModel());
        assertEquals(3, reasoningBlock.getNumReasoningSteps());
        
        // 测试前向传播
        NdArray input = NdArray.ones(Shape.of(1, 8, D_MODEL));
        Variable result = reasoningBlock.layerForward(new Variable(input));
        
        // 验证输出形状
        assertEquals(Shape.of(1, D_MODEL), result.getValue().getShape());
        
        // 验证输出不全为零
        boolean hasNonZero = false;
        NdArray resultValue = result.getValue();
        // 获取一个典型元素进行验证
        if (resultValue.getShape().size() > 0) {
            float value = resultValue.get(0, 0);
            if (Math.abs(value) > 1e-6) {
                hasNonZero = true;
            }
        }
        assertTrue(hasNonZero, "推理块输出不应该全为零");
    }
    
    @Test
    @DisplayName("测试ReflectionBlock基本功能")
    void testReflectionBlock() {
        ReflectionBlock reflectionBlock = new ReflectionBlock("test_reflection", D_MODEL, 0.7);
        
        // 测试初始化
        assertEquals(D_MODEL, reflectionBlock.getDModel());
        assertEquals(0.7, reflectionBlock.getQualityThreshold(), 1e-6);
        
        // 测试前向传播
        NdArray reasoningOutput = NdArray.ones(Shape.of(1, D_MODEL));
        NdArray originalInput = NdArray.ones(Shape.of(1, D_MODEL)).mulNum(0.5f);
        
        Variable result = reflectionBlock.layerForward(
            new Variable(reasoningOutput), 
            new Variable(originalInput)
        );
        
        // 验证输出形状
        assertEquals(Shape.of(1, D_MODEL), result.getValue().getShape());
        
        // 测试反思功能
        ReflectionBlock.ReflectionResult reflectionResult = 
            reflectionBlock.performReflection(
                new Variable(reasoningOutput), 
                new Variable(originalInput)
            );
        
        assertNotNull(reflectionResult);
        assertTrue(reflectionResult.getQualityScore() >= 0.0f && 
                  reflectionResult.getQualityScore() <= 1.0f, 
                  "质量分数应该在0-1之间");
        assertNotNull(reflectionResult.getQualityDescription());
    }
    
    @Test
    @DisplayName("测试TransformerBlock基本功能")
    void testTransformerBlock() {
        TransformerBlock transformerBlock = new TransformerBlock(
            "test_transformer", D_MODEL, NUM_HEADS, D_MODEL * 2, 0.1
        );
        
        // 测试配置
        assertEquals(D_MODEL, transformerBlock.getDModel());
        assertEquals(NUM_HEADS, transformerBlock.getNumHeads());
        
        // 测试前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(1, 8, D_MODEL));
        Variable result = transformerBlock.layerForward(new Variable(input));
        
        // 验证输出形状
        assertEquals(input.getShape(), result.getValue().getShape());
        
        // 验证残差连接（输出应该与输入有相关性但不完全相同）
        NdArray inputData = input;
        NdArray outputData = result.getValue();
        
        // 简单验证：输出不应该与输入完全相同
        boolean isDifferent = false;
        
        // 只比较几个典型元素
        for (int i = 0; i < Math.min(3, inputData.getShape().getDimension(1)); i++) {
            for (int j = 0; j < Math.min(3, inputData.getShape().getDimension(2)); j++) {
                float inputValue = inputData.get(0, i, j);
                float outputValue = outputData.get(0, i, j);
                if (Math.abs(inputValue - outputValue) > 1e-6) {
                    isDifferent = true;
                    break;
                }
            }
            if (isDifferent) break;
        }
        assertTrue(isDifferent, "Transformer输出应该与输入不同（经过了变换）");
    }
    
    @Test
    @DisplayName("测试DeepSeekR1Model基础推理")
    void testBasicInference() {
        // 准备输入
        NdArray inputIds = createTestInput(8);
        
        // 执行推理
        Variable output = testModel.inference(inputIds);
        
        // 验证输出形状: [batch_size, vocab_size]
        Shape expectedShape = Shape.of(1, VOCAB_SIZE);
        assertEquals(expectedShape, output.getValue().getShape());
        
        // 验证输出值的合理性（不应该全为NaN或无穷大）
        NdArray outputData = output.getValue();
        // 检查一些典型元素
        for (int i = 0; i < Math.min(10, outputData.getShape().getDimension(1)); i++) {
            float value = outputData.get(0, i);
            assertTrue(Float.isFinite(value), "输出值应该是有限的数值");
        }
    }
    
    @Test
    @DisplayName("测试DeepSeekR1Model详细推理")
    void testDetailedInference() {
        NdArray inputIds = createTestInput(8);
        
        // 执行详细推理
        DeepSeekR1Block.DeepSeekR1Result result = 
            testModel.inferenceWithDetails(inputIds, null);
        
        // 验证结果组件
        assertNotNull(result.getLogits());
        assertNotNull(result.getReasoningOutput());
        assertNotNull(result.getReflectionResult());
        assertNotNull(result.getTransformerOutput());
        
        // 验证反思结果
        ReflectionBlock.ReflectionResult reflection = result.getReflectionResult();
        assertTrue(reflection.getQualityScore() >= 0.0f && 
                  reflection.getQualityScore() <= 1.0f);
        assertNotNull(reflection.getQualityDescription());
        
        // 验证各组件输出形状
        assertEquals(Shape.of(1, VOCAB_SIZE), result.getLogits().getValue().getShape());
        assertEquals(Shape.of(1, D_MODEL), result.getReasoningOutput().getValue().getShape());
    }
    
    @Test
    @DisplayName("测试文本生成功能")
    void testTextGeneration() {
        List<Integer> seedTokens = Arrays.asList(1, 5, 10, 15);
        int maxNewTokens = 6;
        
        List<Integer> generatedTokens = testModel.generateText(seedTokens, maxNewTokens);
        
        // 验证生成结果
        assertNotNull(generatedTokens);
        assertTrue(generatedTokens.size() >= seedTokens.size(), 
                  "生成序列长度应该不小于种子序列");
        assertTrue(generatedTokens.size() <= seedTokens.size() + maxNewTokens,
                  "生成序列长度不应超过限制");
        
        // 验证种子部分保持不变
        for (int i = 0; i < seedTokens.size(); i++) {
            assertEquals(seedTokens.get(i), generatedTokens.get(i),
                        "种子token应该保持不变");
        }
        
        // 验证生成的token在有效范围内
        for (int token : generatedTokens) {
            assertTrue(token >= 0 && token < VOCAB_SIZE,
                      "生成的token应该在词汇表范围内");
        }
    }
    
    @Test
    @DisplayName("测试思维链推理")
    void testChainOfThoughtReasoning() {
        List<Integer> inputTokens = Arrays.asList(1, 10, 20, 30);
        int maxSteps = 3;
        
        DeepSeekR1Model.ChainOfThoughtResult cotResult = 
            testModel.chainOfThoughtReasoning(inputTokens, maxSteps);
        
        // 验证思维链结果
        assertNotNull(cotResult);
        assertNotNull(cotResult.getReasoningSteps());
        assertNotNull(cotResult.getReflectionResult());
        assertNotNull(cotResult.getFinalAnswer());
        
        // 验证推理步骤
        List<DeepSeekR1Model.ReasoningStep> steps = cotResult.getReasoningSteps();
        assertTrue(steps.size() <= maxSteps, "推理步骤数不应超过限制");
        
        for (DeepSeekR1Model.ReasoningStep step : steps) {
            assertTrue(step.getStepNumber() > 0, "步骤号应该大于0");
            assertNotNull(step.getThought(), "思考内容不应为空");
            assertNotNull(step.getAction(), "行动内容不应为空");
            assertTrue(step.getConfidence() >= 0.0f && step.getConfidence() <= 1.0f,
                      "置信度应该在0-1之间");
        }
    }
    
    @Test
    @DisplayName("测试模型统计信息")
    void testModelStatistics() {
        Map<String, Object> stats = testModel.getModelStatistics();
        
        // 验证基本配置统计
        assertEquals(VOCAB_SIZE, stats.get("vocab_size"));
        assertEquals(D_MODEL, stats.get("d_model"));
        assertEquals(NUM_LAYERS, stats.get("num_layers"));
        assertEquals(NUM_HEADS, stats.get("num_heads"));
        assertEquals(MAX_SEQ_LEN, stats.get("max_seq_len"));
        
        // 验证参数统计
        Long totalParams = (Long) stats.get("total_parameters");
        assertNotNull(totalParams);
        assertTrue(totalParams > 0, "总参数数应该大于0");
        
        // 验证推理相关统计
        assertNotNull(stats.get("reasoning_steps"));
        assertNotNull(stats.get("reflection_threshold"));
    }
    
    @Test
    @DisplayName("测试边界条件 - 空输入")
    void testEmptyInput() {
        // 测试长度为1的最小输入
        NdArray minInput = NdArray.zeros(Shape.of(1, 1));
        minInput.set(1, 0, 0); // 设置一个有效token
        
        assertDoesNotThrow(() -> {
            Variable result = testModel.inference(minInput);
            assertNotNull(result);
        }, "最小输入应该能正常处理");
    }
    
    @Test
    @DisplayName("测试边界条件 - 最大序列长度")
    void testMaxSequenceLength() {
        // 测试最大序列长度输入
        NdArray maxInput = createTestInput(MAX_SEQ_LEN);
        
        assertDoesNotThrow(() -> {
            Variable result = testModel.inference(maxInput);
            assertNotNull(result);
        }, "最大序列长度输入应该能正常处理");
    }
    
    @Test
    @DisplayName("测试异常处理 - 无效token ID")
    void testInvalidTokenId() {
        NdArray invalidInput = NdArray.zeros(Shape.of(1, 4));
        invalidInput.set(VOCAB_SIZE + 10, 0, 0); // 超出词汇表范围的token
        invalidInput.set(5, 0, 1);
        invalidInput.set(10, 0, 2);
        invalidInput.set(15, 0, 3);
        
        // 注意：实际行为取决于具体实现，这里测试是否能graceful处理
        assertDoesNotThrow(() -> {
            testModel.inference(invalidInput);
        }, "应该能处理无效token ID（可能通过截断或其他方式）");
    }
    
    // 辅助方法
    
    /**
     * 创建测试输入数据
     */
    private NdArray createTestInput(int seqLen) {
        NdArray input = NdArray.zeros(Shape.of(1, seqLen));
        
        // 填充有效的随机token ID
        for (int i = 0; i < seqLen; i++) {
            int tokenId = (int) (Math.random() * (VOCAB_SIZE - 1)) + 1; // 1 to VOCAB_SIZE-1
            input.set(tokenId, 0, i);
        }
        
        return input;
    }
    
    /**
     * 运行简单的性能测试
     */
    @Test
    @DisplayName("简单性能测试")
    void testPerformance() {
        NdArray input = createTestInput(16);
        
        // 预热
        for (int i = 0; i < 3; i++) {
            testModel.inference(input);
        }
        
        // 测试推理时间
        long startTime = System.currentTimeMillis();
        int iterations = 5;
        
        for (int i = 0; i < iterations; i++) {
            testModel.inference(input);
        }
        
        long endTime = System.currentTimeMillis();
        double avgTime = (endTime - startTime) / (double) iterations;
        
        System.out.println("平均推理时间: " + avgTime + " ms");
        
        // 验证推理时间在合理范围内（这个阈值可能需要根据实际情况调整）
        assertTrue(avgTime < 10000, "推理时间应该在合理范围内（< 10秒）");
    }
}