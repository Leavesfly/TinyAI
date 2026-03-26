package io.leavesfly.tinyai.minimind.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MiniMindBlock单元测试
 * 
 * @author leavesfly
 */
public class MiniMindBlockTest {
    
    private MiniMindBlock block;
    private MiniMindConfig config;
    
    @BeforeEach
    public void setUp() {
        // 创建超小模型配置用于测试
        config = new MiniMindConfig();
        config.setVocabSize(100);
        config.setMaxSeqLen(32);
        config.setHiddenSize(64);
        config.setNumLayers(2);
        config.setNumHeads(4);
        config.setFfnHiddenSize(128);
        config.setDropout(0.0f);
        
        block = new MiniMindBlock(config);
    }
    
    @Test
    public void testBlockCreation() {
        assertNotNull(block, "MiniMindBlock不应为null");
        assertNotNull(block.getConfig(), "配置不应为null");
        assertNotNull(block.getLayers(), "层列表不应为null");
        assertEquals(2, block.getLayers().size(), "应有2个Transformer层");
    }
    
    @Test
    public void testForwardBasic() {
        // 输入: [batch=1, seq_len=5] token IDs
        int batchSize = 1;
        int seqLen = 5;
        
        float[] tokenIds = new float[]{1, 2, 3, 4, 5};
        NdArray input = NdArray.of(tokenIds, Shape.of(batchSize, seqLen));
        Variable inputVar = new Variable(input);
        
        Variable output = block.forward(inputVar);
        
        assertNotNull(output, "输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应保持");
        assertEquals(seqLen, shape[1], "seq_len维度应保持");
        assertEquals(config.getVocabSize(), shape[2], "vocab_size维度应匹配");
    }
    
    @Test
    public void testForwardWithCache() {
        int batchSize = 1;
        int seqLen = 3;
        
        // 创建KV Caches
        List<KVCache> kvCaches = block.createKVCaches(batchSize);
        
        assertNotNull(kvCaches, "KV Caches不应为null");
        assertEquals(config.getNumLayers(), kvCaches.size(), "应有对应数量的Cache");
        
        // 第一次forward
        float[] tokenIds1 = new float[]{1, 2, 3};
        NdArray input1 = NdArray.of(tokenIds1, Shape.of(batchSize, seqLen));
        Variable inputVar1 = new Variable(input1);
        
        Variable output1 = block.forwardWithCache(inputVar1, kvCaches, 0);
        
        assertNotNull(output1, "第一次输出不应为null");
        
        // 验证Cache被更新
        for (KVCache cache : kvCaches) {
            assertEquals(seqLen, cache.getCurrentSeqLen(), "Cache长度应更新");
        }
        
        // 第二次forward (增量)
        int newSeqLen = 1;
        float[] tokenIds2 = new float[]{4};
        NdArray input2 = NdArray.of(tokenIds2, Shape.of(batchSize, newSeqLen));
        Variable inputVar2 = new Variable(input2);
        
        Variable output2 = block.forwardWithCache(inputVar2, kvCaches, seqLen);
        
        assertNotNull(output2, "第二次输出不应为null");
        
        // 验证Cache累加
        for (KVCache cache : kvCaches) {
            assertEquals(seqLen + newSeqLen, cache.getCurrentSeqLen(), "Cache长度应累加");
        }
    }
    
    @Test
    public void testCreateKVCaches() {
        int batchSize = 2;
        List<KVCache> kvCaches = block.createKVCaches(batchSize);
        
        assertNotNull(kvCaches, "KV Caches不应为null");
        assertEquals(config.getNumLayers(), kvCaches.size(), "Cache数量应等于层数");
        
        // 验证每个Cache初始状态
        for (KVCache cache : kvCaches) {
            assertEquals(0, cache.getCurrentSeqLen(), "初始Cache长度应为0");
        }
    }
    
    @Test
    public void testClearKVCaches() {
        int batchSize = 1;
        List<KVCache> kvCaches = block.createKVCaches(batchSize);
        
        // 先使用Cache
        float[] tokenIds = new float[]{1, 2, 3};
        NdArray input = NdArray.of(tokenIds, Shape.of(batchSize, 3));
        Variable inputVar = new Variable(input);
        
        block.forwardWithCache(inputVar, kvCaches, 0);
        
        // 验证Cache有数据
        for (KVCache cache : kvCaches) {
            assertTrue(cache.getCurrentSeqLen() > 0, "Cache应有数据");
        }
        
        // 清空Cache
        block.clearKVCaches(kvCaches);
        
        // 验证Cache已清空
        for (KVCache cache : kvCaches) {
            assertEquals(0, cache.getCurrentSeqLen(), "Cache应被清空");
        }
    }
    
    @Test
    public void testBatchProcessing() {
        // 测试批处理
        int batchSize = 2;
        int seqLen = 5;
        
        float[] tokenIds = new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        };
        NdArray input = NdArray.of(tokenIds, Shape.of(batchSize, seqLen));
        Variable inputVar = new Variable(input);
        
        Variable output = block.forward(inputVar);
        
        assertNotNull(output, "批处理输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为" + batchSize);
        assertEquals(seqLen, shape[1], "seq_len维度应为" + seqLen);
        assertEquals(config.getVocabSize(), shape[2], "vocab_size应匹配");
    }
    
    @Test
    public void testTrainingModeSwitch() {
        // 测试训练模式切换
        block.setTraining(true);
        
        float[] tokenIds = new float[]{1, 2, 3};
        NdArray input = NdArray.of(tokenIds, Shape.of(1, 3));
        Variable inputVar = new Variable(input);
        
        Variable output1 = block.forward(inputVar);
        assertNotNull(output1, "训练模式输出不应为null");
        
        // 切换到推理模式
        block.setTraining(false);
        Variable output2 = block.forward(inputVar);
        assertNotNull(output2, "推理模式输出不应为null");
    }
    
    @Test
    public void testEstimateParameters() {
        long paramCount = block.estimateParameters();
        
        assertTrue(paramCount > 0, "参数数量应大于0");
        // 验证参数数量在合理范围内
        assertTrue(paramCount < 1_000_000, "小模型参数应少于100万");
    }
    
    @Test
    public void testForwardGeneration() {
        // 测试生成模式的前向传播
        int batchSize = 1;
        List<KVCache> kvCaches = block.createKVCaches(batchSize);
        
        float[] tokenId = new float[]{1};
        NdArray input = NdArray.of(tokenId, Shape.of(batchSize, 1));
        Variable inputVar = new Variable(input);
        
        Variable output = block.forwardGeneration(inputVar, kvCaches, 0);
        
        assertNotNull(output, "生成模式输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为1");
        assertEquals(1, shape[1], "seq_len维度应为1");
        assertEquals(config.getVocabSize(), shape[2], "vocab_size应匹配");
    }
}
