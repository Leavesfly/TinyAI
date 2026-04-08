package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * TokenEmbedding单元测试
 * 
 * @author leavesfly
 */
public class TokenEmbeddingTest {
    
    private TokenEmbedding embedding;
    private final int vocabSize = 100;
    private final int embedDim = 64;
    
    @BeforeEach
    public void setUp() {
        embedding = new TokenEmbedding(vocabSize, embedDim);
    }
    
    @Test
    public void testEmbeddingCreation() {
        assertNotNull(embedding, "TokenEmbedding不应为null");
    }
    
    @Test
    public void testForwardSingleToken() {
        // 输入: [batch=1, seq_len=1]
        NdArray tokenIds = NdArray.of(new float[]{5}, Shape.of(1, 1));
        Variable input = new Variable(tokenIds);
        
        Variable output = embedding.forward(input);
        
        assertNotNull(output, "输出不应为null");
        NdArray result = output.getValue();
        
        int[] shape = result.getShape().getShapeDims();
        assertEquals(1, shape[0], "batch维度应为1");
        assertEquals(1, shape[1], "seq_len维度应为1");
        assertEquals(embedDim, shape[2], "embed维度应匹配");
    }
    
    @Test
    public void testForwardBatchSequence() {
        // 输入: [batch=2, seq_len=5]
        float[] data = new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        };
        NdArray tokenIds = NdArray.of(data, Shape.of(2, 5));
        Variable input = new Variable(tokenIds);
        
        Variable output = embedding.forward(input);
        
        assertNotNull(output, "输出不应为null");
        NdArray result = output.getValue();
        
        int[] shape = result.getShape().getShapeDims();
        assertEquals(2, shape[0], "batch维度应为2");
        assertEquals(5, shape[1], "seq_len维度应为5");
        assertEquals(embedDim, shape[2], "embed维度应为" + embedDim);
    }
    
    @Test
    public void testEmbeddingValues() {
        // 相同的token应产生相同的嵌入
        NdArray tokenIds1 = NdArray.of(new float[]{10}, Shape.of(1, 1));
        NdArray tokenIds2 = NdArray.of(new float[]{10}, Shape.of(1, 1));
        
        Variable output1 = embedding.forward(new Variable(tokenIds1));
        Variable output2 = embedding.forward(new Variable(tokenIds2));
        
        NdArray embed1 = output1.getValue();
        NdArray embed2 = output2.getValue();
        
        float[] data1 = embed1.getArray();
        float[] data2 = embed2.getArray();
        
        assertArrayEquals(data1, data2, 1e-6f, "相同token的嵌入应相同");
    }
    
    @Test
    public void testDifferentTokensDifferentEmbeddings() {
        NdArray tokenIds1 = NdArray.of(new float[]{5}, Shape.of(1, 1));
        NdArray tokenIds2 = NdArray.of(new float[]{10}, Shape.of(1, 1));
        
        Variable output1 = embedding.forward(new Variable(tokenIds1));
        Variable output2 = embedding.forward(new Variable(tokenIds2));
        
        NdArray embed1 = output1.getValue();
        NdArray embed2 = output2.getValue();
        
        float[] data1 = embed1.getArray();
        float[] data2 = embed2.getArray();
        
        // 不同token应有不同的嵌入
        boolean different = false;
        for (int i = 0; i < data1.length; i++) {
            if (Math.abs(data1[i] - data2[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        assertTrue(different, "不同token的嵌入应不同");
    }

    @Test
    public void testTokenEmbeddingGradientCheck() {
        TokenEmbedding embedding = new TokenEmbedding(vocabSize, embedDim);
        
        // 创建输入 (batch=2, seq_len=5)
        float[] data = new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        };
        NdArray tokenIds = NdArray.of(data, Shape.of(2, 5));
        Variable input = new Variable(tokenIds);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(embedding, input);
    }
}
