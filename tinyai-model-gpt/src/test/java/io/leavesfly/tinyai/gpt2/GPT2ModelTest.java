package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * GPT-2模型单元测试
 * 
 * 测试GPT-2模型的核心功能和组件
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2ModelTest {
    
    private GPT2Config config;
    private GPT2Model model;
    
    @Before
    public void setUp() {
        // 创建小型测试配置
        config = new GPT2Config();
        config.setVocabSize(100);
        config.setNPositions(32);
        config.setNEmbd(64);
        config.setNLayer(2);
        config.setNHead(4);
        config.setNInner(256);
        
        model = new GPT2Model("test_gpt2", config);
    }
    
    @Test
    public void testConfigValidation() {
        assertNotNull("配置不应该为null", config);
        config.validate(); // 不应该抛出异常
        
        assertEquals("词汇表大小应该正确", 100, config.getVocabSize());
        assertEquals("嵌入维度应该正确", 64, config.getNEmbd());
        assertEquals("层数应该正确", 2, config.getNLayer());
        assertEquals("注意力头数应该正确", 4, config.getNHead());
    }
    
    @Test
    public void testModelCreation() {
        assertNotNull("模型不应该为null", model);
        assertNotNull("GPT2Block不应该为null", model.getGPT2Block());
        assertNotNull("Token嵌入层不应该为null", model.getTokenEmbedding());
        assertNotNull("输出头不应该为null", model.getOutputHead());
        
        assertEquals("模型名称应该正确", "test_gpt2", model.getName());
        assertTrue("参数数量应该大于0", model.getGPT2Block().getParameterCount() > 0);
    }
    
    @Test
    public void testStaticFactoryMethods() {
        GPT2Model smallModel = GPT2Model.createSmallModel("small");
        assertNotNull("小型模型不应该为null", smallModel);
        
        GPT2Model mediumModel = GPT2Model.createMediumModel("medium");
        assertNotNull("中型模型不应该为null", mediumModel);
        
        GPT2Model largeModel = GPT2Model.createLargeModel("large");
        assertNotNull("大型模型不应该为null", largeModel);
        
        // 验证不同规模模型的参数数量
        assertTrue("中型模型参数应该多于小型模型", 
                  mediumModel.getGPT2Block().getParameterCount() > smallModel.getGPT2Block().getParameterCount());
        assertTrue("大型模型参数应该多于中型模型", 
                  largeModel.getGPT2Block().getParameterCount() > mediumModel.getGPT2Block().getParameterCount());
    }
    
    @Test
    public void testForwardPropagation() {
        // 创建测试输入：批次大小=2，序列长度=8
        float[][] tokenIds = {{1, 5, 10, 2, 7, 3, 8, 4}, {3, 8, 15, 7, 12, 6, 9, 1}};
        NdArray input = NdArray.of(tokenIds);
        
        Variable output = model.predict(input);
        
        assertNotNull("输出不应该为null", output);
        Shape outputShape = output.getValue().getShape();
        assertEquals("输出应该是三维的", 3, outputShape.getDimNum());
        assertEquals("批次维度应该正确", 2, outputShape.getDimension(0));
        assertEquals("序列维度应该正确", 8, outputShape.getDimension(1));
        assertEquals("词汇表维度应该正确", 100, outputShape.getDimension(2));
    }
    
    @Test
    public void testNextTokenPrediction() {
        // 创建测试输入：批次大小=1，序列长度=5
        float[][] tokenIds = {{1, 5, 10, 2, 7}};
        NdArray input = NdArray.of(tokenIds);
        
        int predictedToken = model.predictNextToken(input);
        
        assertTrue("预测的token应该在词汇表范围内", 
                  predictedToken >= 0 && predictedToken < config.getVocabSize());
    }
    
    @Test
    public void testSequenceGeneration() {
        // 创建起始序列
        float[][] startTokens = {{1, 5}};
        NdArray startSequence = NdArray.of(startTokens);
        
        int maxLength = 5;
        NdArray generatedSequence = model.generateSequence(startSequence, maxLength);
        
        assertNotNull("生成的序列不应该为null", generatedSequence);
        Shape genShape = generatedSequence.getShape();
        assertEquals("生成序列应该是二维的", 2, genShape.getDimNum());
        assertEquals("批次维度应该正确", 1, genShape.getDimension(0));
        assertTrue("序列长度应该增加", genShape.getDimension(1) > 2);
        assertTrue("序列长度不应该超过最大长度", genShape.getDimension(1) <= 2 + maxLength);
    }
    
    @Test
    public void testInputValidation() {
        // 测试无效维度
        try {
            NdArray invalidInput = NdArray.of(Shape.of(5));  // 一维数组
            model.validateInput(invalidInput);
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息应该包含维度相关描述", 
                      e.getMessage().contains("二维数组"));
        }
        
        // 测试序列过长
        try {
            NdArray longSequence = NdArray.of(Shape.of(1, 50));  // 超过最大长度32
            model.validateInput(longSequence);
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息应该包含长度相关描述", 
                      e.getMessage().contains("长度"));
        }
    }
    
    @Test
    public void testModelComponents() {
        // 测试Token嵌入层
        GPT2TokenEmbedding tokenEmbedding = model.getTokenEmbedding();
        assertNotNull("Token嵌入层不应该为null", tokenEmbedding);
        assertEquals("Token嵌入词汇表大小应该正确", 100, tokenEmbedding.getVocabSize());
        assertEquals("Token嵌入维度应该正确", 64, tokenEmbedding.getDModel());
        
        // 测试Transformer块
        for (int i = 0; i < config.getNLayer(); i++) {
            GPT2TransformerBlock block = model.getTransformerBlock(i);
            assertNotNull("Transformer块不应该为null", block);
            assertEquals("Transformer块层索引应该正确", i, block.getLayerIdx());
        }
        
        // 测试输出头
        GPT2OutputHead outputHead = model.getOutputHead();
        assertNotNull("输出头不应该为null", outputHead);
        assertEquals("输出头词汇表大小应该正确", 100, outputHead.getVocabSize());
        assertEquals("输出头模型维度应该正确", 64, outputHead.getNEmbd());
    }
    
    @Test
    public void testModelInfo() {
        String configSummary = model.getConfigSummary();
        assertNotNull("配置摘要不应该为null", configSummary);
        assertTrue("配置摘要应该包含词汇表大小", configSummary.contains("100"));
        assertTrue("配置摘要应该包含嵌入维度", configSummary.contains("64"));
        
        // 测试打印模型信息（不会抛出异常）
        model.printModelInfo();
    }
    
    @Test
    public void testConfigTypes() {
        // 测试默认配置
        GPT2Config defaultConfig = new GPT2Config();
        assertEquals("默认词汇表大小应该正确", 50257, defaultConfig.getVocabSize());
        
        // 测试小型配置
        GPT2Config smallConfig = GPT2Config.createSmallConfig();
        assertEquals("小型配置词汇表大小应该正确", 5000, smallConfig.getVocabSize());
        assertEquals("小型配置嵌入维度应该正确", 256, smallConfig.getNEmbd());
        
        // 测试中型配置
        GPT2Config mediumConfig = GPT2Config.createMediumConfig();
        assertEquals("中型配置嵌入维度应该正确", 1024, mediumConfig.getNEmbd());
        
        // 测试大型配置
        GPT2Config largeConfig = GPT2Config.createLargeConfig();
        assertEquals("大型配置嵌入维度应该正确", 1280, largeConfig.getNEmbd());
    }
}