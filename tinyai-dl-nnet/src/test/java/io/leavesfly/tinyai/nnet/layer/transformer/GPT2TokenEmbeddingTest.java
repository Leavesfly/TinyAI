package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * GPT2TokenEmbedding层的核心单元测试
 */
public class GPT2TokenEmbeddingTest {

    private GPT2TokenEmbedding embedding;
    
    @Before
    public void setUp() {
        embedding = new GPT2TokenEmbedding("embed", 1000, 64, 100, true, 0.1);
    }

    @Test
    public void testParameterInitialization() {
        assertNotNull("token嵌入参数应该被初始化", embedding.getTokenEmbedding());
        assertNotNull("位置嵌入参数应该被初始化", embedding.getPositionEmbedding());
        assertEquals("词汇表大小应该正确", 1000, embedding.getVocabSize());
        assertEquals("嵌入维度应该正确", 64, embedding.getDModel());
    }

    @Test
    public void testRequireInputNum() {
        assertEquals("GPT2TokenEmbedding应该需要1个输入", 1, embedding.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        // 创建token ID输入
        float[][] tokenIds = {{1, 5, 10, 2}, {3, 8, 15, 7}};  // batch_size=2, seq_len=4
        NdArray input = NdArray.of(tokenIds);
        Variable inputVar = new Variable(input);
        
        Variable output = embedding.layerForward(inputVar);
        
        assertEquals("输出形状应该正确", 
                   Shape.of(2, 4, 64), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testInvalidTokenId() {
        float[][] invalidTokenIds = {{-1, 5, 1001, 2}};  // 包含无效的token ID
        NdArray input = NdArray.of(invalidTokenIds);
        Variable inputVar = new Variable(input);
        
        try {
            embedding.layerForward(inputVar);
            fail("应该抛出IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue("异常信息应该包含相关描述", 
                     e.getMessage().contains("out of vocabulary range"));
        }
    }
}