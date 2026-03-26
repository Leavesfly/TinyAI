package io.leavesfly.tinyai.minimind.model.attention;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.transformer.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.transformer.attention.MultiHeadAttention;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.util.GradientChecker;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MultiHeadAttention单元测试
 * 
 * @author leavesfly
 */
public class MultiHeadAttentionTest {
    
    private MultiHeadAttention attention;
    private final int dimModel = 64;
    private final int numHeads = 4;
    private final int headDim = 16; // dimModel / numHeads
    private final int maxSeqLen = 128;
    
    @BeforeEach
    public void setUp() {
        attention = new MultiHeadAttention(
            "test_attention",
            dimModel,
            numHeads,
            maxSeqLen,
            0.0f // dropout
        );
    }
    
    @Test
    public void testAttentionCreation() {
        assertNotNull(attention, "MultiHeadAttention不应为null");
    }
    
    @Test
    public void testForwardSelfAttention() {
        // Self-attention: Q=K=V
        // 输入: [batch=1, seq_len=5, dim=64]
        int batchSize = 1;
        int seqLen = 5;
        
        float[] data = new float[batchSize * seqLen * dimModel];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable output = attention.forwardWithCache(inputVar, null, 0);
        
        assertNotNull(output, "输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应保持");
        assertEquals(seqLen, shape[1], "seq_len维度应保持");
        assertEquals(dimModel, shape[2], "dim维度应保持");
    }
    
    @Test
    public void testForwardWithCache() {
        int batchSize = 1;
        int seqLen = 3;
        
        // 创建KV Cache
        KVCache kvCache = new KVCache(batchSize, numHeads, headDim, maxSeqLen);
        
        // 第一次forward
        NdArray input1 = NdArray.of(Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar1 = new Variable(input1);
        
        Variable output1 = attention.forwardWithCache(inputVar1, kvCache, 0);
        
        assertNotNull(output1, "第一次输出不应为null");
        assertEquals(seqLen, kvCache.getCurrentSeqLen(), "Cache长度应更新");
        
        // 第二次forward (增量)
        int newSeqLen = 1;
        NdArray input2 = NdArray.of(Shape.of(batchSize, newSeqLen, dimModel));
        Variable inputVar2 = new Variable(input2);
        
        Variable output2 = attention.forwardWithCache(inputVar2, kvCache, seqLen);
        
        assertNotNull(output2, "第二次输出不应为null");
        assertEquals(seqLen + newSeqLen, kvCache.getCurrentSeqLen(), "Cache长度应累加");
    }
    
    @Test
    public void testCausalMask() {
        // 测试因果掩码 (在MultiHeadAttention内部自动应用)
        int batchSize = 1;
        int seqLen = 4;
        
        NdArray input = NdArray.of(Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable output = attention.forwardWithCache(inputVar, null, 0);
        assertNotNull(output, "输出不应为null");
        
        // 验证输出shape正确
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应保持");
        assertEquals(seqLen, shape[1], "seq_len维度应保持");
        assertEquals(dimModel, shape[2], "dim维度应保持");
    }
    
    @Test
    public void testBatchProcessing() {
        // 测试批处理
        int batchSize = 3;
        int seqLen = 5;
        
        float[] data = new float[batchSize * seqLen * dimModel];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable output = attention.forwardWithCache(inputVar, null, 0);
        
        assertNotNull(output, "批处理输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为" + batchSize);
        assertEquals(seqLen, shape[1], "seq_len维度应为" + seqLen);
        assertEquals(dimModel, shape[2], "dim维度应为" + dimModel);
    }

    @Test
    public void testMultiHeadAttentionGradientCheck() {
        MultiHeadAttention attention = new MultiHeadAttention(
            "test_attention",
            dimModel,
            numHeads,
            maxSeqLen,
            0.0f // dropout
        );
        
        // 创建输入 (batch=1, seq_len=5, dim=64)
        NdArray inputData = NdArray.randn(Shape.of(1, 5, dimModel));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(attention, input);
    }
}
