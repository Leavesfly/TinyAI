package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * SoftMax激活层的单元测试
 */
public class SoftMaxLayerTest {

    private SoftMaxLayer softMaxLayer;
    
    @Before
    public void setUp() {
        softMaxLayer = new SoftMaxLayer("softmax");
    }

    @Test
    public void testBasicForwardPass() {
        float[][] inputData = {{1.0f, 2.0f, 3.0f}, {0.5f, 1.5f, 2.5f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 3), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testProbabilityDistribution() {
        float[][] inputData = {{1.0f, 2.0f, 3.0f, 4.0f}, {-1.0f, 0.0f, 1.0f, 2.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        float[][] outputData = output.getValue().getMatrix();
        
        // 检查每一行是否满足概率分布的条件
        for (int i = 0; i < 2; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 4; j++) {
                // 每个元素应该大于0
                assertTrue("SoftMax输出应该大于0", outputData[i][j] > 0.0f);
                // 每个元素应该小于等于1
                assertTrue("SoftMax输出应该小于等于1", outputData[i][j] <= 1.0f);
                sum += outputData[i][j];
            }
            // 每一行的和应该等于1
            assertEquals("SoftMax输出的和应该等于1", 1.0f, sum, 0.001f);
        }
    }

    @Test
    public void testMonotonicProperty() {
        // 测试SoftMax的单调性：较大的输入对应较大的输出概率
        float[][] inputData = {{1.0f, 2.0f, 3.0f, 4.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        float[][] outputData = output.getValue().getMatrix();
        
        // 验证输出是递增的（对应于递增的输入）
        for (int j = 1; j < 4; j++) {
            assertTrue("较大的输入应该对应较大的SoftMax概率", outputData[0][j] > outputData[0][j-1]);
        }
    }

    @Test
    public void testNumericalStability() {
        // 测试数值稳定性：大数值输入
        float[][] inputData = {{100.0f, 101.0f, 102.0f}, {-100.0f, -99.0f, -98.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        float[][] outputData = output.getValue().getMatrix();
        
        // 验证输出没有NaN或无穷大
        for (int i = 0; i < 2; i++) {
            float sum = 0.0f;
            for (int j = 0; j < 3; j++) {
                assertFalse("SoftMax输出不应该包含NaN", Float.isNaN(outputData[i][j]));
                assertFalse("SoftMax输出不应该包含无穷大", Float.isInfinite(outputData[i][j]));
                assertTrue("SoftMax输出应该大于0", outputData[i][j] > 0.0f);
                sum += outputData[i][j];
            }
            assertEquals("即使对于大数值，SoftMax输出的和也应该等于1", 1.0f, sum, 0.001f);
        }
    }

    @Test
    public void testUniformInput() {
        // 测试均匀输入（所有元素相等）
        float[][] inputData = {{2.0f, 2.0f, 2.0f, 2.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        float[][] outputData = output.getValue().getMatrix();
        
        // 所有输出应该相等且为1/n
        float expected = 1.0f / 4.0f;
        for (int j = 0; j < 4; j++) {
            assertEquals("均匀输入应该产生均匀分布", expected, outputData[0][j], 0.001f);
        }
    }

    @Test
    public void testSingleElementInput() {
        // 测试单元素输入
        float[][] inputData = {{5.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = softMaxLayer.layerForward(inputVar);
        float result = output.getValue().get(0, 0);
        
        // 单元素的SoftMax应该等于1
        assertEquals("单元素的SoftMax应该等于1", 1.0f, result, 0.001f);
    }

    @Test
    public void testShapePreservation() {
        Shape[] testShapes = {
            Shape.of(1, 3),
            Shape.of(2, 5),
            Shape.of(3, 4)
        };
        
        for (Shape shape : testShapes) {
            NdArray input = NdArray.likeRandomN(shape);
            Variable inputVar = new Variable(input);
            
            Variable output = softMaxLayer.layerForward(inputVar);
            
            assertEquals("SoftMax应该保持输入形状不变", shape, output.getValue().getShape());
        }
    }

    @Test
    public void testInitialization() {
        try {
            softMaxLayer.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
        
        assertTrue("SoftMax层不应该有参数", softMaxLayer.getParams().isEmpty());
    }

    @Test
    public void testRequiredInputNumber() {
        assertEquals("requireInputNum应该返回1", 1, softMaxLayer.requireInputNum());
    }
}