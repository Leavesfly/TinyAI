package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Tanh激活层的单元测试
 */
public class TanhLayerTest {

    private TanhLayer tanhLayer;
    
    @Before
    public void setUp() {
        tanhLayer = new TanhLayer("tanh");
    }

    @Test
    public void testBasicForwardPass() {
        float[][] inputData = {{0.0f, 1.0f, -1.0f}, {2.0f, -2.0f, 0.5f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = tanhLayer.layerForward(inputVar);
        
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
        Variable output = tanhLayer.layerForward(inputVar);
        NdArray outputArray = output.getValue();
        
        // 验证所有输出都在[-1,1]范围内
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                float value = outputArray.get(i, j);
                assertTrue("Tanh输出应该在[-1,1]范围内", value >= -1.0f && value <= 1.0f);
            }
        }
    }

    @Test
    public void testZeroInput() {
        float[][] inputData = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = tanhLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 零输入的tanh值应该是0
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals("Tanh(0)应该等于0", 0.0f, actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testLargePositiveInput() {
        float[][] inputData = {{10.0f, 100.0f}, {50.0f, 20.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = tanhLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 大正数的tanh值应该接近1
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertTrue("大正数的Tanh值应该接近1", actualData[i][j] > 0.99f);
            }
        }
    }

    @Test
    public void testLargeNegativeInput() {
        float[][] inputData = {{-10.0f, -100.0f}, {-50.0f, -20.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = tanhLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 大负数的tanh值应该接近-1
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertTrue("大负数的Tanh值应该接近-1", actualData[i][j] < -0.99f);
            }
        }
    }

    @Test
    public void testOddFunctionProperty() {
        // 测试Tanh的奇函数性质：tanh(-x) = -tanh(x)
        float testValue = 2.0f;
        
        NdArray positiveInput = NdArray.of(new float[][]{{testValue}});
        NdArray negativeInput = NdArray.of(new float[][]{{-testValue}});
        
        Variable positiveVar = new Variable(positiveInput);
        Variable negativeVar = new Variable(negativeInput);
        
        Variable positiveOutput = tanhLayer.layerForward(positiveVar);
        Variable negativeOutput = tanhLayer.layerForward(negativeVar);
        
        float posValue = positiveOutput.getValue().get(0, 0);
        float negValue = negativeOutput.getValue().get(0, 0);
        
        assertEquals("Tanh奇函数性质：tanh(-x) = -tanh(x)", -posValue, negValue, 0.01f);
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
            
            Variable output = tanhLayer.layerForward(inputVar);
            
            assertEquals("Tanh应该保持输入形状不变", shape, output.getValue().getShape());
        }
    }

    @Test
    public void testInitialization() {
        try {
            tanhLayer.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
        
        assertTrue("Tanh层不应该有参数", tanhLayer.getParams().isEmpty());
    }

    @Test
    public void testRequiredInputNumber() {
        assertEquals("requireInputNum应该返回1", 1, tanhLayer.requireInputNum());
    }
}