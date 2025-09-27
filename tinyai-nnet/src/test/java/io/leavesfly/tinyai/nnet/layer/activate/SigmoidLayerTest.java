package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Sigmoid激活层的单元测试
 */
public class SigmoidLayerTest {

    private SigmoidLayer sigmoidLayer;
    
    @Before
    public void setUp() {
        sigmoidLayer = new SigmoidLayer("sigmoid");
    }

    @Test
    public void testBasicForwardPass() {
        float[][] inputData = {{0.0f, 1.0f, -1.0f}, {2.0f, -2.0f, 0.5f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = sigmoidLayer.layerForward(inputVar);
        
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 3), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testOutputRange() {
        NdArray input = NdArray.likeRandomN(Shape.of(10, 10));
        // 扩大输入范围
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                input.set(input.get(i, j) * 20 - 10, i, j); // 范围[-10, 10]
            }
        }
        
        Variable inputVar = new Variable(input);
        Variable output = sigmoidLayer.layerForward(inputVar);
        NdArray outputArray = output.getValue();
        
        // 验证所有输出都在[0,1]范围内
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                float value = outputArray.get(i, j);
                assertTrue("Sigmoid输出应该在[0,1]范围内", value >= 0.0f && value <= 1.0f);
            }
        }
    }

    @Test
    public void testZeroInput() {
        float[][] inputData = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = sigmoidLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 零输入的sigmoid值应该是0.5
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals("Sigmoid(0)应该等于0.5", 0.5f, actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testLargePositiveInput() {
        float[][] inputData = {{10.0f, 100.0f}, {50.0f, 20.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = sigmoidLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 大正数的sigmoid值应该接近1
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertTrue("大正数的Sigmoid值应该接近1", actualData[i][j] > 0.99f);
            }
        }
    }

    @Test
    public void testLargeNegativeInput() {
        float[][] inputData = {{-10.0f, -100.0f}, {-50.0f, -20.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = sigmoidLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 大负数的sigmoid值应该接近0
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertTrue("大负数的Sigmoid值应该接近0", actualData[i][j] < 0.01f);
            }
        }
    }

    @Test
    public void testShapePreservation() {
        Shape[] testShapes = {
            Shape.of(1, 1),
            Shape.of(3, 4),
            Shape.of(2, 5, 3)
        };
        
        for (Shape shape : testShapes) {
            NdArray input = NdArray.likeRandomN(shape);
            Variable inputVar = new Variable(input);
            
            Variable output = sigmoidLayer.layerForward(inputVar);
            
            assertEquals("Sigmoid应该保持输入形状不变", shape, output.getValue().getShape());
        }
    }

    @Test
    public void testInitialization() {
        try {
            sigmoidLayer.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
        
        assertTrue("Sigmoid层不应该有参数", sigmoidLayer.getParams().isEmpty());
    }

    @Test
    public void testRequiredInputNumber() {
        assertEquals("requireInputNum应该返回1", 1, sigmoidLayer.requireInputNum());
    }
}