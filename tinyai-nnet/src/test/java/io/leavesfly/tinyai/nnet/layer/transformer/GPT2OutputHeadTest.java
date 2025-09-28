package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * GPT2OutputHead层的核心单元测试
 */
public class GPT2OutputHeadTest {

    private GPT2OutputHead outputHead;
    
    @Before
    public void setUp() {
        outputHead = new GPT2OutputHead("output", 64, 1000, false);
    }

    @Test
    public void testParameterInitialization() {
        assertNotNull("输出权重参数应该被初始化", outputHead.getOutputWeight());
        assertEquals("词汇表大小应该正确", 1000, outputHead.getVocabSize());
        assertEquals("模型维度应该正确", 64, outputHead.getDModel());
        assertFalse("不应该使用偏置", outputHead.isUseBias());
    }

    @Test
    public void testRequireInputNum() {
        assertEquals("GPT2OutputHead应该需要1个输入", 1, outputHead.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        NdArray input = NdArray.likeRandomN(Shape.of(2, 5, 64));  // batch_size=2, seq_len=5, d_model=64
        Variable inputVar = new Variable(input);
        
        Variable output = outputHead.layerForward(inputVar);
        
        assertEquals("输出形状应该正确", 
                   Shape.of(2, 5, 1000), output.getValue().getShape());  // vocab_size=1000
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardMethod() {
        NdArray input = NdArray.likeRandomN(Shape.of(1, 3, 64));
        
        NdArray output = outputHead.forward(input);
        
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", 
                   Shape.of(1, 3, 1000), output.getShape());
    }

    @Test
    public void testBackwardMethod() {
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 4, 1000));
        
        try {
            outputHead.backward(grad);
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }
}