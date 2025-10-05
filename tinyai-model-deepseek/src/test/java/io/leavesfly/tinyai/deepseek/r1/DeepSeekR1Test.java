package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

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
    private static final int VOCAB_SIZE = 50;
    private static final int D_MODEL = 32;
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 2;
    private static final int MAX_SEQ_LEN = 16;
    
    @Before
    public void setUp() {
        // 创建测试用的小型模型
        testModel = new DeepSeekR1Model(
            "TestModel", VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, 
            D_MODEL * 2, MAX_SEQ_LEN, 0.1
        );
    }
    
    @Test
    public void testModelConfigurationValidation() {
        // 测试有效配置
        assertTrue("有效配置应该通过验证", testModel.validateConfiguration());
        
        // 测试无效的词汇表大小
        DeepSeekR1Model invalidModel1 = new DeepSeekR1Model("Invalid", 0, D_MODEL);
        assertFalse("词汇表大小为0应该验证失败", invalidModel1.validateConfiguration());
        
        // 测试无效的模型维度（不能被头数整除）
        DeepSeekR1Model invalidModel2 = new DeepSeekR1Model("Invalid", VOCAB_SIZE, 30, 2, 4);
        assertFalse("模型维度不能被头数整除应该验证失败", invalidModel2.validateConfiguration());
    }
    
    @Test
    public void testReasoningBlock() {
        ReasoningBlock reasoningBlock = new ReasoningBlock("test_reasoning", D_MODEL, 2);
        
        // 测试初始化
        assertEquals(D_MODEL, reasoningBlock.getDModel());
        assertEquals(2, reasoningBlock.getNumReasoningSteps());
        
        // 测试前向传播
        NdArray input = NdArray.ones(Shape.of(1, 4, D_MODEL));
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
        assertTrue("推理块输出不应该全为零", hasNonZero);
    }
    
    @Test
    public void testReflectionBlock() {
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
        assertTrue("质量分数应该在0-1之间",
                  reflectionResult.getQualityScore() >= 0.0f && 
                  reflectionResult.getQualityScore() <= 1.0f);
        assertNotNull(reflectionResult.getQualityDescription());
    }
    
    @Test
    public void testBasicInference() {
        // 准备输入
        NdArray inputIds = createTestInput(4);
        
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
            assertTrue("输出值应该是有限的数值", Float.isFinite(value));
        }
    }
    
    @Test
    public void testDetailedInference() {
        NdArray inputIds = createTestInput(4);
        
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
        assertTrue("质量分数应该在0-1之间",
                  reflection.getQualityScore() >= 0.0f && 
                  reflection.getQualityScore() <= 1.0f);
        assertNotNull(reflection.getQualityDescription());
        
        // 验证各组件输出形状
        assertEquals(Shape.of(1, VOCAB_SIZE), result.getLogits().getValue().getShape());
        assertEquals(Shape.of(1, D_MODEL), result.getReasoningOutput().getValue().getShape());
    }
    
    @Test
    public void testTextGeneration() {
        List<Integer> seedTokens = Arrays.asList(1, 5, 10, 15);
        int maxNewTokens = 3;
        
        List<Integer> generatedTokens = testModel.generateText(seedTokens, maxNewTokens);
        
        // 验证生成结果
        assertNotNull(generatedTokens);
        assertTrue("生成序列长度应该不小于种子序列", 
                  generatedTokens.size() >= seedTokens.size());
        assertTrue("生成序列长度不应超过限制",
                  generatedTokens.size() <= seedTokens.size() + maxNewTokens);
        
        // 验证种子部分保持不变
        for (int i = 0; i < seedTokens.size(); i++) {
            assertEquals("种子token应该保持不变",
                        seedTokens.get(i), generatedTokens.get(i));
        }
        
        // 验证生成的token在有效范围内
        for (int token : generatedTokens) {
            assertTrue("生成的token应该在词汇表范围内",
                      token >= 0 && token < VOCAB_SIZE);
        }
    }
    
    @Test
    public void testModelStatistics() {
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
        assertTrue("总参数数应该大于0", totalParams > 0);
        
        // 验证推理相关统计
        assertNotNull(stats.get("reasoning_steps"));
        assertNotNull(stats.get("reflection_threshold"));
    }
    
    @Test
    public void testPerformance() {
        NdArray input = createTestInput(8);
        
        // 预热
        for (int i = 0; i < 2; i++) {
            testModel.inference(input);
        }
        
        // 测试推理时间
        long startTime = System.currentTimeMillis();
        int iterations = 3;
        
        for (int i = 0; i < iterations; i++) {
            testModel.inference(input);
        }
        
        long endTime = System.currentTimeMillis();
        double avgTime = (endTime - startTime) / (double) iterations;
        
        System.out.println("平均推理时间: " + avgTime + " ms");
        
        // 验证推理时间在合理范围内（这个阈值可能需要根据实际情况调整）
        assertTrue("推理时间应该在合理范围内（< 5秒）", avgTime < 5000);
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
}