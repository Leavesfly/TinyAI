package io.leavesfly.tinyai.gpt;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.block.transformer.GPT2Block;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * GPT2Model单元测试
 * 
 * @author leavesfly
 */
public class GPT2ModelTest {
    
    private GPT2Model smallModel;
    private GPT2Model mediumModel;
    
    @Before
    public void setUp() {
        // 创建小型测试模型
        smallModel = new GPT2Model("test_gpt2_small", 1000, 64, 2, 4, 128, 32, 0.1);
        
        // 创建中等测试模型
        mediumModel = new GPT2Model("test_gpt2_medium", 5000, 256, 4, 8, 1024, 128, 0.1);
    }
    
    @Test
    public void testModelInitialization() {
        // 测试模型初始化
        assertNotNull("Token嵌入层应该被初始化", smallModel.getTokenEmbedding());
        assertNotNull("Transformer块列表应该被初始化", smallModel.getTransformerBlocks());
        assertNotNull("最终层归一化应该被初始化", smallModel.getFinalLayerNorm());
        assertNotNull("输出头应该被初始化", smallModel.getOutputHead());
        
        // 验证Transformer块数量
        assertEquals("应该有2个Transformer块", 2, smallModel.getTransformerBlocks().size());
        assertEquals("应该有4个Transformer块", 4, mediumModel.getTransformerBlocks().size());
    }
    
    @Test 
    public void testModelConfiguration() {
        // 测试模型配置
        assertEquals(1000, smallModel.getVocabSize());
        assertEquals(64, smallModel.getDModel());
        assertEquals(2, smallModel.getNumLayers());
        assertEquals(4, smallModel.getNumHeads());
        assertEquals(128, smallModel.getDFF());
        assertEquals(32, smallModel.getMaxSeqLength());
        assertEquals(0.1, smallModel.getDropoutRate(), 0.01);
        
        // 测试配置信息字符串
        String config = smallModel.getModelConfig();
        assertNotNull(config);
        assertTrue(config.contains("Vocab Size: 1000"));
        assertTrue(config.contains("Model Dim: 64"));
    }
    
    @Test
    public void testForwardPassBasic() {
        // 测试基本前向传播
        int batchSize = 2;
        int seqLen = 8;
        
        // 创建输入token ID (batch_size, seq_len)
        NdArray input = NdArray.zeros(Shape.of(batchSize, seqLen));
        
        // 填充一些有效的token ID
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                input.set(j % 100, i, j);  // 使用0-99之间的token ID
            }
        }
        
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = smallModel.layerForward(inputVar);
        
        // 验证输出形状
        assertNotNull("输出不应为null", output);
        Shape outputShape = output.getValue().getShape();
        assertEquals("输出应该是3维", 3, outputShape.getDimNum());
        assertEquals("批次维度不匹配", batchSize, outputShape.getDimension(0));
        assertEquals("序列长度维度不匹配", seqLen, outputShape.getDimension(1));
        assertEquals("词汇表维度不匹配", smallModel.getVocabSize(), outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassDifferentSeqLengths() {
        // 测试不同序列长度的前向传播
        int batchSize = 1;
        int[] seqLengths = {1, 4, 16, 32};
        
        for (int seqLen : seqLengths) {
            NdArray input = NdArray.zeros(Shape.of(batchSize, seqLen));
            
            // 填充token ID
            for (int j = 0; j < seqLen; j++) {
                input.set(j % 100, 0, j);
            }
            
            Variable inputVar = new Variable(input);
            Variable output = smallModel.layerForward(inputVar);
            
            // 验证输出形状
            Shape outputShape = output.getValue().getShape();
            assertEquals(
                "序列长度" + seqLen + "的输出维度不匹配", 
                seqLen, outputShape.getDimension(1));
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSequenceLengthValidation() {
        // 测试序列长度验证
        int batchSize = 1;
        int tooLongSeqLen = smallModel.getMaxSeqLength() + 1;
        
        NdArray input = NdArray.zeros(Shape.of(batchSize, tooLongSeqLen));
        Variable inputVar = new Variable(input);
        
        // 应该抛出异常
        smallModel.layerForward(inputVar);
    }
    
    @Test
    public void testTransformerBlockAccess() {
        // 测试Transformer块访问
        assertEquals(2, smallModel.getTransformerBlocks().size());
        
        // 测试正常索引
        assertNotNull(smallModel.getTransformerBlock(0));
        assertNotNull(smallModel.getTransformerBlock(1));
        
        // 测试越界索引
        try {
            smallModel.getTransformerBlock(2);
            fail("应该抛出IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            // 期望的异常
        }
        
        try {
            smallModel.getTransformerBlock(-1);
            fail("应该抛出IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            // 期望的异常
        }
    }
    
    @Test
    public void testDefaultConstructors() {
        // 测试默认构造函数
        GPT2Model defaultModel1 = new GPT2Model("default1", 1000, 128, 6, 256);
        assertEquals(1000, defaultModel1.getVocabSize());
        assertEquals(128, defaultModel1.getDModel());
        assertEquals(6, defaultModel1.getNumLayers());
        assertEquals(8, defaultModel1.getNumHeads());  // 默认头数
        
        GPT2Model defaultModel2 = new GPT2Model("default2", 2000, 512);
        assertEquals(2000, defaultModel2.getVocabSize());
        assertEquals(768, defaultModel2.getDModel());  // 默认模型维度
        assertEquals(12, defaultModel2.getNumLayers()); // 默认层数
        
        // 测试兼容性构造函数
        GPT2Model compatModel = new GPT2Model("compat", Shape.of(-1, 100));
        assertEquals(50257, compatModel.getVocabSize()); // GPT-2默认词汇表大小
        assertEquals(768, compatModel.getDModel());
    }
    
    @Test
    public void testModelComponents() {
        // 测试各个组件的基本属性
        
        // Token嵌入层
        assertEquals(smallModel.getVocabSize(), smallModel.getTokenEmbedding().getVocabSize());
        assertEquals(smallModel.getDModel(), smallModel.getTokenEmbedding().getDModel());
        assertEquals(smallModel.getMaxSeqLength(), smallModel.getTokenEmbedding().getMaxSeqLength());
        
        // Transformer块
        for (int i = 0; i < smallModel.getNumLayers(); i++) {
            GPT2Block block = smallModel.getTransformerBlock(i);
            assertNotNull(block.getLayerNorm1());
            assertNotNull(block.getLayerNorm2());
            assertNotNull(block.getAttention());
            assertNotNull(block.getFeedForward());
        }
        
        // 输出头
        assertEquals(smallModel.getDModel(), smallModel.getOutputHead().getDModel());
        assertEquals(smallModel.getVocabSize(), smallModel.getOutputHead().getVocabSize());
        assertFalse(smallModel.getOutputHead().isUseBias()); // GPT-2通常不使用输出偏置
    }
    
    @Test
    public void testOutputProbabilityDistribution() {
        // 测试输出是否为有效的概率分布
        int batchSize = 1;
        int seqLen = 4;
        
        NdArray input = NdArray.zeros(Shape.of(batchSize, seqLen));
        for (int j = 0; j < seqLen; j++) {
            input.set(j + 1, 0, j);  // token ID: 1, 2, 3, 4
        }
        
        Variable inputVar = new Variable(input);
        Variable output = smallModel.layerForward(inputVar);
        
        // 检查输出值的合理性
        NdArray outputData = output.getValue();
        
        // 检查是否有NaN或无穷大值
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                for (int k = 0; k < smallModel.getVocabSize(); k++) {
                    float value = outputData.get(i, j, k);
                    assertFalse("输出包含NaN值", Float.isNaN(value));
                    assertFalse("输出包含无穷大值", Float.isInfinite(value));
                }
            }
        }
    }
    
    @Test
    public void testParameterCount() {
        // 测试参数数量（简单验证）
        java.util.Map<String, io.leavesfly.tinyai.nnet.Parameter> allParams = smallModel.getAllParams();
        assertNotNull(allParams);
        assertTrue("模型应该有参数", allParams.size() > 0);
        
        // 打印参数名称用于调试
        System.out.println("参数列表:");
        for (String key : allParams.keySet()) {
            System.out.println("  - " + key);
        }
        
        // 验证参数包含期望的组件
        boolean hasTokenEmbedding = allParams.keySet().stream()
            .anyMatch(key -> key.contains("token_embedding"));
        boolean hasTransformerParams = allParams.keySet().stream()
            .anyMatch(key -> key.contains("block") || key.contains("ln") || key.contains("attention"));
        boolean hasOutputParams = allParams.keySet().stream()
            .anyMatch(key -> key.contains("output_head"));
        
        assertTrue("应该包含token嵌入参数", hasTokenEmbedding);
        assertTrue("应该包含transformer块参数", hasTransformerParams);
        assertTrue("应该包含输出头参数", hasOutputParams);
    }
}