// 替换所有的断言调用为正确的JUnit 4语法
package io.leavesfly.tinyai.nlp.deepseekR1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * DeepSeek-R1 模型单元测试
 * 
 * 测试DeepSeek-R1模型的核心功能：
 * 1. 模型创建和初始化
 * 2. 前向传播
 * 3. 推理功能
 * 4. 思维链推理
 * 5. 配置管理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1ModelTest {
    
    private DeepSeekR1Model model;
    private DeepSeekR1Config config;
    private NdArray testInput;
    
    @Before
    public void setUp() {
        // 创建测试配置
        config = DeepSeekR1Config.createDebugConfig();
        
        // 创建测试模型
        model = DeepSeekR1Factory.createModel("test_model", config);
        
        // 创建测试输入
        int batchSize = 2;
        int seqLen = 10;
        testInput = NdArray.of(Shape.of(batchSize, seqLen));
        
        // 填充测试数据
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                testInput.set(b * seqLen + t + 1, b, t); // 简单的递增序列
            }
        }
    }
    
    @Test
    public void testModelCreationAndBasicProperties() {
        assertNotNull("模型应该成功创建", model);
        assertEquals("模型名称应该正确", "test_model", model.getName());
        
        // 测试模型配置
        assertEquals(config.getVocabSize(), model.getVocabSize());
        assertEquals(config.getDModel(), model.getDModel());
        assertEquals(config.getNumLayers(), model.getNumLayers());
        assertEquals(config.getNumHeads(), model.getNumHeads());
        assertEquals(config.getMaxSeqLength(), model.getMaxSeqLength());
        assertEquals(config.getMaxReasoningSteps(), model.getMaxReasoningSteps());
    }
    
    @Test
    public void testModelComponentsInitialization() {
        // 检查关键组件是否正确初始化
        assertNotNull("模型组件初始化 - Token嵌入层", model.getTokenEmbedding());
        assertNotNull("模型组件初始化 - Transformer块", model.getTransformerBlocks());
        assertNotNull("模型组件初始化 - 思维链推理层", model.getCotReasoningLayer());
        assertNotNull("模型组件初始化 - 推理引擎", model.getReasoningEngine());
        assertNotNull("模型组件初始化 - 最终层归一化", model.getFinalLayerNorm());
        assertNotNull("模型组件初始化 - 推理输出头", model.getReasoningOutputHead());
        
        // 检查Transformer块数量
        assertEquals("组件初始化 - Transformer块数量", config.getNumLayers(), model.getTransformerBlocks().size());
    }
    
    @Test
    public void testForwardPass() {
        // 测试基本前向传播
        Variable input = new Variable(testInput);
        Variable output = model.layerForward(input);
        
        assertNotNull("前向传播 - 输出结果", output);
        assertNotNull("前向传播 - 输出值", output.getValue());
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals("输出应该3维张量", 3, outputShape.getDimNum());
        assertEquals("批大小应该保持一致", testInput.getShape().getDimension(0), outputShape.getDimension(0));
        assertEquals("序列长度应该保持一致", testInput.getShape().getDimension(1), outputShape.getDimension(1));
        assertEquals("最后一维应该是词汇表大小", config.getVocabSize(), outputShape.getDimension(2));
    }
    
    @Test
    public void testReasoningModeToggle() {
        // 默认应该启用推理模式
        assertTrue("默认应该启用推理模式", model.isReasoningMode());
        
        // 测试关闭推理模式
        model.setReasoningMode(false);
        assertFalse("应该能够关闭推理模式", model.isReasoningMode());
        
        // 测试重新启用推理模式
        model.setReasoningMode(true);
        assertTrue("应该能够重新启用推理模式", model.isReasoningMode());
    }
    
    @Test
    public void testReasoningCapability() {
        // 执行推理
        ReasoningResult result = model.performReasoning(testInput);
        
        assertNotNull("推理结果不应为空", result);
        
        if (result.isError()) {
            // 如果有错误，至少应该有错误信息
            assertNotNull("错误情况下应该有错误信息", result.getErrorMessage());
        } else {
            // 如果成功，检查结果完整性
            assertNotNull("成功推理应该有最终答案", result.getFinalAnswer());
            assertTrue("置信度应该在0-1之间", 
                      result.getConfidenceScore() >= 0.0 && result.getConfidenceScore() <= 1.0);
            assertTrue("推理步骤数应该非负", result.getNumReasoningSteps() >= 0);
            assertNotNull("推理步骤列表不应为空", result.getReasoningSteps());
        }
    }
    
    @Test
    public void testReasoningStepsRecording() {
        // 启用推理模式
        model.setReasoningMode(true);
        
        // 执行推理
        ReasoningResult result = model.performReasoning(testInput);
        
        if (!result.isError()) {
            // 检查推理步骤记录
            assertNotNull("推理步骤不应为空", result.getReasoningSteps());
            assertEquals("推理步骤数量应该一致", result.getNumReasoningSteps(), result.getReasoningSteps().size());
            
            // 检查模型内部的推理记录
            assertNotNull("模型应该记录推理步骤", model.getReasoningSteps());
        }
    }
    
    @Test
    public void testReasoningContextManagement() {
        // 添加推理上下文
        model.updateReasoningContext("test_key", "test_value");
        
        // 检查推理上下文
        assertNotNull("推理上下文不应为空", model.getReasoningContext());
        assertTrue("应该包含添加的上下文键", model.getReasoningContext().containsKey("test_key"));
        assertEquals("上下文值应该正确", "test_value", model.getReasoningContext().get("test_key"));
    }
    
    @Test
    public void testInputValidation() {
        // 测试过长的输入序列
        int maxSeqLen = model.getMaxSeqLength();
        NdArray longInput = NdArray.of(Shape.of(1, maxSeqLen + 10));
        
        // 填充数据
        for (int i = 0; i < maxSeqLen + 10; i++) {
            longInput.set(i + 1, 0, i);
        }
        
        Variable longInputVar = new Variable(longInput);
        
        // 应该抛出异常
        try {
            model.layerForward(longInputVar);
            fail("过长的输入应该抛出异常");
        } catch (IllegalArgumentException e) {
            // 期望的异常
        }
    }
    
    @Test
    public void testModelConfigRetrieval() {
        String configStr = model.getModelConfig();
        
        assertNotNull("模型配置字符串不应为空", configStr);
        assertTrue("配置应该包含模型名称", configStr.contains("DeepSeek-R1"));
        assertTrue("配置应该包含词汇表大小信息", configStr.contains("Vocab Size"));
        assertTrue("配置应该包含模型维度信息", configStr.contains("Model Dim"));
    }
    
    @Test
    public void testModelInfoPrinting() {
        // 这个测试主要验证方法不会抛出异常
        try {
            model.printModelInfo();
        } catch (Exception e) {
            fail("打印模型信息不应该抛出异常");
        }
    }
    
    @Test
    public void testDifferentInputSizes() {
        int[] seqLengths = {1, 5, 10, 20};
        
        for (int seqLen : seqLengths) {
            if (seqLen <= model.getMaxSeqLength()) {
                NdArray input = NdArray.of(Shape.of(1, seqLen));
                
                // 填充测试数据
                for (int i = 0; i < seqLen; i++) {
                    input.set(i + 1, 0, i);
                }
                
                Variable inputVar = new Variable(input);
                
                try {
                    Variable output = model.layerForward(inputVar);
                    assertNotNull("输出不应为空", output);
                    assertEquals("输出序列长度应该与输入一致", seqLen, output.getValue().getShape().getDimension(1));
                } catch (Exception e) {
                    fail(String.format("序列长度 %d 的输入处理不应该失败", seqLen));
                }
            }
        }
    }
}