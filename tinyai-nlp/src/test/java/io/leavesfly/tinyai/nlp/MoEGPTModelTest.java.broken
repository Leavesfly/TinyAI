package io.leavesfly.tinyai.modality.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.block.transformer.MoETransformerBlock;
import io.leavesfly.tinyai.nnet.layer.moe.MoELayer;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * MoE-GPT模型单元测试
 * <p>
 * 测试MoE-GPT模型的各个功能，包括：
 * 1. 模型构造和初始化
 * 2. 前向传播
 * 3. 输入输出形状验证
 * 4. MoE专家激活
 * 5. 负载均衡统计
 * 6. 参数计数
 * 7. 配置验证
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoEGPTModelTest {
    
    private MoEGPTModel smallModel;
    private MoEGPTModel mediumModel;
    private MoEGPTModel defaultModel;
    
    @Before
    public void setUp() {
        // 创建小型测试模型
        smallModel = new MoEGPTModel(
            "test_moe_gpt_small", 
            1000,  // vocabSize
            64,    // dModel  
            3,     // numLayers
            4,     // numHeads
            4,     // numExperts
            2,     // topK
            32     // maxSeqLength
        );
        
        // 创建中等测试模型
        mediumModel = new MoEGPTModel(
            "test_moe_gpt_medium", 
            2000,  // vocabSize
            128,   // dModel
            6,     // numLayers
            8,     // numHeads
            8,     // numExperts
            256,   // dExpert
            2,     // topK
            64,    // maxSeqLength
            0.1,   // dropoutRate
            true,  // useNoise
            0.05   // noiseEpsilon
        );
        
        // 创建默认配置模型
        defaultModel = new MoEGPTModel("test_default", Shape.of(-1, 100));
    }
    
    @Test
    public void testModelConstruction() {
        // 测试基本属性
        assertEquals(1000, smallModel.getVocabSize());
        assertEquals(64, smallModel.getDModel());
        assertEquals(3, smallModel.getNumLayers());
        assertEquals(4, smallModel.getNumHeads());
        assertEquals(4, smallModel.getNumExperts());
        assertEquals(2, smallModel.getTopK());
        assertEquals(32, smallModel.getMaxSeqLength());
        
        // 测试完整构造函数
        assertEquals(2000, mediumModel.getVocabSize());
        assertEquals(128, mediumModel.getDModel());
        assertEquals(6, mediumModel.getNumLayers());
        assertEquals(8, mediumModel.getNumHeads());
        assertEquals(8, mediumModel.getNumExperts());
        assertEquals(256, mediumModel.getDExpert());
        assertEquals(2, mediumModel.getTopK());
        assertEquals(64, mediumModel.getMaxSeqLength());
        assertEquals(0.1, mediumModel.getDropoutRate(), 0.001);
        assertTrue(mediumModel.isUseNoise());
        assertEquals(0.05, mediumModel.getNoiseEpsilon(), 0.001);
        
        // 测试默认配置
        assertEquals(50257, defaultModel.getVocabSize());
        assertEquals(768, defaultModel.getDModel());
        assertEquals(12, defaultModel.getNumLayers());
        assertEquals(8, defaultModel.getNumExperts());
    }
    
    @Test
    public void testModelInitialization() {
        // 验证所有组件都已正确初始化
        assertNotNull(smallModel.getTokenEmbedding());
        assertNotNull(smallModel.getMoeTransformerBlocks());
        assertNotNull(smallModel.getFinalLayerNorm());
        assertNotNull(smallModel.getOutputHead());
        
        // 验证MoE Transformer块数量
        assertEquals(smallModel.getNumLayers(), smallModel.getMoeTransformerBlocks().size());
        
        // 验证每个MoE块的配置
        for (int i = 0; i < smallModel.getNumLayers(); i++) {
            MoETransformerBlock block = smallModel.getMoeTransformerBlock(i);
            assertNotNull(block);
            assertEquals(smallModel.getDModel(), block.getDModel());
            assertEquals(smallModel.getNumHeads(), block.getNumHeads());
            assertEquals(smallModel.getNumExperts(), block.getNumExperts());
            assertEquals(smallModel.getTopK(), block.getTopK());
        }
    }
    
    @Test
    public void testForwardPropagation() {
        int batchSize = 2;
        int seqLen = 8;
        
        // 创建输入数据
        NdArray inputTokens = createSampleTokens(batchSize, seqLen, smallModel.getVocabSize());
        Variable input = new Variable(inputTokens);
        
        // 执行前向传播
        Variable output = smallModel.layerForward(input);
        
        // 验证输出形状
        NdArray outputData = output.getValue();
        Shape outputShape = outputData.getShape();
        
        assertEquals(batchSize, outputShape.getDimension(0));
        assertEquals(seqLen, outputShape.getDimension(1));
        assertEquals(smallModel.getVocabSize(), outputShape.getDimension(2));
        
        // 验证输出值不为NaN或无穷大
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int v = 0; v < Math.min(10, smallModel.getVocabSize()); v++) {
                    float value = outputData.get(b, s, v);
                    assertFalse("输出包含NaN", Float.isNaN(value));
                    assertFalse("输出包含无穷大", Float.isInfinite(value));
                }
            }
        }
    }
    
    @Test
    public void testDifferentSequenceLengths() {
        int[] testLengths = {1, 4, 8, 16, 32};
        
        for (int seqLen : testLengths) {
            if (seqLen <= smallModel.getMaxSeqLength()) {
                // 创建测试输入
                NdArray input = createSampleTokens(1, seqLen, smallModel.getVocabSize());
                Variable inputVar = new Variable(input);
                
                // 执行前向传播
                Variable output = smallModel.layerForward(inputVar);
                
                // 验证输出形状
                Shape outputShape = output.getValue().getShape();
                assertEquals(1, outputShape.getDimension(0));
                assertEquals(seqLen, outputShape.getDimension(1));
                assertEquals(smallModel.getVocabSize(), outputShape.getDimension(2));
            }
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSequenceLengthExceedsMax() {
        // 测试超过最大序列长度时抛出异常
        int tooLongSeqLen = smallModel.getMaxSeqLength() + 1;
        NdArray input = createSampleTokens(1, tooLongSeqLen, smallModel.getVocabSize());
        Variable inputVar = new Variable(input);
        
        smallModel.layerForward(inputVar);
    }
    
    @Test
    public void testMoELoadBalancing() {
        // 重置统计信息
        smallModel.resetAllMoEStats();
        
        // 执行多次前向传播以收集统计信息
        int numBatches = 5;
        for (int i = 0; i < numBatches; i++) {
            NdArray input = createSampleTokens(2, 8, smallModel.getVocabSize());
            smallModel.layerForward(new Variable(input));
        }
        
        // 验证统计信息收集
        List<MoELayer.LoadBalancingStats> allStats = smallModel.getAllMoEStats();
        assertEquals(smallModel.getNumLayers(), allStats.size());
        
        // 验证每层都有统计信息
        for (MoELayer.LoadBalancingStats stats : allStats) {
            assertTrue(\"应该有token被处理\", stats.totalTokens > 0);
            assertTrue(\"平均使用率应该大于0\", stats.averageUsage > 0);
            assertNotNull(\"专家使用计数不应为null\", stats.expertUsageCount);
            assertEquals(\"专家使用计数数组长度应该等于专家数量\", 
                        smallModel.getNumExperts(), stats.expertUsageCount.length);
        }
        
        // 验证负载均衡报告生成
        String report = smallModel.getLoadBalancingReport();
        assertNotNull(report);
        assertTrue(\"报告应该包含层信息\", report.contains(\"Layer\"));
    }
    
    @Test
    public void testParameterCounting() {
        // 测试参数计数功能
        long totalParams = smallModel.getTotalParameterCount();
        assertTrue(\"模型应该有参数\", totalParams > 0);
        
        // 验证参数增加比例计算
        double increaseRatio = smallModel.getParameterIncreaseRatio();
        assertTrue(\"MoE应该增加参数数量\", increaseRatio > 1.0);
        
        // 验证每个MoE块的参数计数
        for (MoETransformerBlock block : smallModel.getMoeTransformerBlocks()) {
            long blockParams = block.getTotalParameterCount();
            assertTrue(\"每个块应该有参数\", blockParams > 0);
        }
    }
    
    @Test
    public void testModelConfiguration() {
        // 测试模型配置信息生成
        String config = smallModel.getModelConfig();
        assertNotNull(config);
        assertTrue(\"配置应该包含模型名称\", config.contains(\"MoE-GPT\"));
        assertTrue(\"配置应该包含词汇表大小\", config.contains(\"Vocab Size\"));
        assertTrue(\"配置应该包含专家数量\", config.contains(\"Num Experts\"));
        assertTrue(\"配置应该包含参数总数\", config.contains(\"Total Parameters\"));
    }
    
    @Test
    public void testComponentAccess() {
        // 测试组件访问方法
        
        // Token嵌入层
        assertEquals(smallModel.getVocabSize(), smallModel.getTokenEmbedding().getVocabSize());
        assertEquals(smallModel.getDModel(), smallModel.getTokenEmbedding().getDModel());
        assertEquals(smallModel.getMaxSeqLength(), smallModel.getTokenEmbedding().getMaxSeqLength());
        
        // MoE Transformer块
        for (int i = 0; i < smallModel.getNumLayers(); i++) {
            MoETransformerBlock block = smallModel.getMoeTransformerBlock(i);
            assertNotNull(block.getLayerNorm1());
            assertNotNull(block.getLayerNorm2());
            assertNotNull(block.getAttention());
            assertNotNull(block.getMoeLayer());
        }
        
        // 输出头
        assertEquals(smallModel.getDModel(), smallModel.getOutputHead().getDModel());
        assertEquals(smallModel.getVocabSize(), smallModel.getOutputHead().getVocabSize());
        assertFalse(\"MoE-GPT通常不使用输出偏置\", smallModel.getOutputHead().isUseBias());
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testInvalidBlockIndex() {
        // 测试访问无效块索引时抛出异常
        smallModel.getMoeTransformerBlock(smallModel.getNumLayers());
    }
    
    @Test
    public void testStatisticsControl() {
        // 测试统计信息控制
        assertTrue(\"默认应该收集统计信息\", smallModel.isCollectStats());
        
        // 禁用统计收集
        smallModel.setCollectStats(false);
        assertFalse(\"应该禁用统计收集\", smallModel.isCollectStats());
        
        // 重新启用
        smallModel.setCollectStats(true);
        assertTrue(\"应该重新启用统计收集\", smallModel.isCollectStats());
    }
    
    @Test
    public void testExpertActivationPattern() {
        // 重置统计
        smallModel.resetAllMoEStats();
        
        // 使用固定输入模式测试专家激活
        NdArray patternInput = createPatternTokens(2, 8, smallModel.getVocabSize());
        smallModel.layerForward(new Variable(patternInput));
        
        // 验证专家被激活
        List<MoELayer.LoadBalancingStats> stats = smallModel.getAllMoEStats();
        for (MoELayer.LoadBalancingStats layerStats : stats) {
            // 至少有一些专家应该被使用
            long totalUsage = 0;
            for (long usage : layerStats.expertUsageCount) {
                totalUsage += usage;
            }
            assertTrue(\"应该有专家被激活\", totalUsage > 0);
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidTopK() {
        // 测试无效的topK参数
        new MoEGPTModel(
            \"invalid_model\", 
            1000, 128, 3, 4, 4, 
            5,  // topK > numExperts
            32
        );
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidDModelHeadsRatio() {
        // 测试dModel不能被numHeads整除的情况
        new MoEGPTModel(
            \"invalid_model\", 
            1000, 
            65,  // 不能被numHeads(4)整除
            3, 4, 4, 2, 32
        );
    }
    
    @Test
    public void testDefaultConstructors() {
        // 测试默认构造函数
        MoEGPTModel defaultModel1 = new MoEGPTModel(\"default1\", 1000, 64, 6, 8, 4, 2, 128);
        assertEquals(1000, defaultModel1.getVocabSize());
        assertEquals(64, defaultModel1.getDModel());
        assertEquals(6, defaultModel1.getNumLayers());
        assertEquals(8, defaultModel1.getNumHeads());
        assertEquals(4, defaultModel1.getNumExperts());
        
        MoEGPTModel defaultModel2 = new MoEGPTModel(\"default2\", 2000, 256);
        assertEquals(2000, defaultModel2.getVocabSize());
        assertEquals(768, defaultModel2.getDModel()); // 默认模型维度
        assertEquals(12, defaultModel2.getNumLayers()); // 默认层数
        assertEquals(8, defaultModel2.getNumExperts()); // 默认专家数
    }
    
    /**
     * 创建样本token序列
     */
    private NdArray createSampleTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.of(Shape.of(batchSize, seqLen));
        Random random = new Random(42);  // 固定种子确保可重现性
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = random.nextInt(vocabSize);
                tokens.set(tokenId, b, s);
            }
        }
        
        return tokens;
    }
    
    /**
     * 创建有模式的token序列
     */
    private NdArray createPatternTokens(int batchSize, int seqLen, int vocabSize) {
        NdArray tokens = NdArray.of(Shape.of(batchSize, seqLen));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 创建简单的重复模式
                int tokenId = (s % 4) + (b * 10);
                tokens.set(tokenId % vocabSize, b, s);
            }
        }
        
        return tokens;
    }
}"