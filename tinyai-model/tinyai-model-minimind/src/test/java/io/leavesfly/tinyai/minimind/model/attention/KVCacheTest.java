package io.leavesfly.tinyai.minimind.model.attention;

import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * KVCache单元测试
 * 
 * @author leavesfly
 */
public class KVCacheTest {
    
    private KVCache kvCache;
    private final int batchSize = 2;
    private final int numHeads = 4;
    private final int headDim = 16;
    private final int maxSeqLen = 128;
    
    @BeforeEach
    public void setUp() {
        kvCache = new KVCache(batchSize, numHeads, headDim, maxSeqLen);
    }
    
    @Test
    public void testKVCacheCreation() {
        assertNotNull(kvCache, "KVCache不应为null");
    }
    
    @Test
    public void testInitialState() {
        // 初始状态currentLen应为0
        assertEquals(0, kvCache.getCurrentSeqLen(), "初始currentLen应为0");
    }
    
    @Test
    public void testUpdateCache() {
        // 创建测试数据: [batch, num_heads, seq_len, head_dim]
        int seqLen = 5;
        float[] kData = new float[batchSize * numHeads * seqLen * headDim];
        float[] vData = new float[batchSize * numHeads * seqLen * headDim];
        
        for (int i = 0; i < kData.length; i++) {
            kData[i] = (float) i;
            vData[i] = (float) (i + 1);
        }
        
        NdArray k = NdArray.of(kData, Shape.of(batchSize, numHeads, seqLen, headDim));
        NdArray v = NdArray.of(vData, Shape.of(batchSize, numHeads, seqLen, headDim));
        
        kvCache.update(k, v);
        
        assertEquals(seqLen, kvCache.getCurrentSeqLen(), "更新后currentLen应为seqLen");
    }
    
    @Test
    public void testIncrementalUpdate() {
        // 第一次更新
        int seqLen1 = 3;
        NdArray k1 = NdArray.of(Shape.of(batchSize, numHeads, seqLen1, headDim));
        NdArray v1 = NdArray.of(Shape.of(batchSize, numHeads, seqLen1, headDim));
        
        kvCache.update(k1, v1);
        assertEquals(seqLen1, kvCache.getCurrentSeqLen(), "第一次更新后长度应为3");
        
        // 第二次增量更新
        int seqLen2 = 2;
        NdArray k2 = NdArray.of(Shape.of(batchSize, numHeads, seqLen2, headDim));
        NdArray v2 = NdArray.of(Shape.of(batchSize, numHeads, seqLen2, headDim));
        
        kvCache.update(k2, v2);
        assertEquals(seqLen1 + seqLen2, kvCache.getCurrentSeqLen(), "增量更新后长度应累加");
    }
    
    @Test
    public void testGetCachedKV() {
        // 更新cache
        int seqLen = 4;
        NdArray k = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        NdArray v = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        kvCache.update(k, v);
        
        // 获取缓存的K和V
        NdArray cachedK = kvCache.getCachedK();
        NdArray cachedV = kvCache.getCachedV();
        
        assertNotNull(cachedK, "缓存的K不应为null");
        assertNotNull(cachedV, "缓存的V不应为null");
        
        // 验证shape
        int[] kShape = cachedK.getShape().getShapeDims();
        assertEquals(batchSize, kShape[0], "batch维度应匹配");
        assertEquals(numHeads, kShape[1], "num_heads维度应匹配");
        assertEquals(seqLen, kShape[2], "seq_len应为当前长度");
        assertEquals(headDim, kShape[3], "head_dim应匹配");
    }
    
    @Test
    public void testClear() {
        // 更新cache
        NdArray k = NdArray.of(Shape.of(batchSize, numHeads, 5, headDim));
        NdArray v = NdArray.of(Shape.of(batchSize, numHeads, 5, headDim));
        
        kvCache.update(k, v);
        assertEquals(5, kvCache.getCurrentSeqLen(), "更新后长度应为5");
        
        // 清空cache
        kvCache.clear();
        assertEquals(0, kvCache.getCurrentSeqLen(), "清空后currentLen应为0");
    }
    
    @Test
    public void testMaxLenBoundary() {
        // 测试接近最大长度
        int seqLen = maxSeqLen - 1;
        NdArray k = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        NdArray v = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        kvCache.update(k, v);
        assertEquals(seqLen, kvCache.getCurrentSeqLen(), "应能缓存到最大长度-1");
    }
}
