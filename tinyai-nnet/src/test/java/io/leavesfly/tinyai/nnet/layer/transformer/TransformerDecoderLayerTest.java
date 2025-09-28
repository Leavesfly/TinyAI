package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * TransformerDecoderLayer层的核心单元测试
 */
public class TransformerDecoderLayerTest {

    private TransformerDecoderLayer decoderLayerSmall;
    
    @Before
    public void setUp() {
        decoderLayerSmall = new TransformerDecoderLayer("dec_small", 64, 4, 256, 0.0);
    }

    @Test
    public void testParameterInitialization() {
        assertNotNull("带掩码的自注意力层应该被初始化", decoderLayerSmall.getMaskedSelfAttention());
        assertNotNull("交叉注意力层应该被初始化", decoderLayerSmall.getCrossAttention());
        assertNotNull("前馈网络层应该被初始化", decoderLayerSmall.getFeedForward());
    }

    @Test
    public void testRequireInputNum() {
        assertEquals("TransformerDecoderLayer应该需要2个输入", 2, decoderLayerSmall.requireInputNum());
    }

    @Test
    public void testForwardPropagation() {
        NdArray decoderInput = NdArray.likeRandomN(Shape.of(2, 6, 64));   // 解码器输入
        NdArray encoderOutput = NdArray.likeRandomN(Shape.of(2, 8, 64));  // 编码器输出
        
        Variable decoderVar = new Variable(decoderInput);
        Variable encoderVar = new Variable(encoderOutput);
        
        Variable output = decoderLayerSmall.layerForward(decoderVar, encoderVar);
        
        assertEquals("输出形状应该与解码器输入形状相同", 
                   Shape.of(2, 6, 64), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testBackwardMethod() {
        NdArray grad = NdArray.likeRandomN(Shape.of(2, 5, 64));
        
        try {
            decoderLayerSmall.backward(grad);
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }
}