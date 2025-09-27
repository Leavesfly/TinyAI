package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Dropout层的单元测试
 */
public class DropoutTest {

    private Dropout dropout;
    
    @Before
    public void setUp() {
        dropout = new Dropout("dropout", 0.5f, Shape.of(2, 3));
    }

    @Test
    public void testBasicForwardPass() {
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = dropout.layerForward(inputVar);
        
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 3), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testShapePreservation() {
        Shape[] testShapes = {
            Shape.of(1, 5),
            Shape.of(3, 4),
            Shape.of(2, 5, 3)
        };
        
        for (Shape shape : testShapes) {
            NdArray input = NdArray.likeRandomN(shape);
            Variable inputVar = new Variable(input);
            
            Variable output = dropout.layerForward(inputVar);
            
            assertEquals("Dropout应该保持输入形状不变", shape, output.getValue().getShape());
        }
    }

    @Test
    public void testInitialization() {
        try {
            dropout.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
        
        assertTrue("Dropout层不应该有参数", dropout.getParams().isEmpty());
    }

    @Test
    public void testRequiredInputNumber() {
        assertEquals("requireInputNum应该返回1", 1, dropout.requireInputNum());
    }

    @Test
    public void testLayerName() {
        assertEquals("层名称应该正确", "dropout", dropout.getName());
    }

    @Test
    public void testClearGrads() {
        try {
            dropout.clearGrads();
        } catch (Exception e) {
            fail("清除梯度不应该抛出异常");
        }
    }
}