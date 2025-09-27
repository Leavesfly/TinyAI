package io.leavesfly.tinyai.nnet.layer.activate;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * ReLU激活层的单元测试
 * 
 * 测试ReLU激活函数层的基本功能：
 * 1. 前向传播计算正确性
 * 2. 形状处理
 * 3. 边界情况
 * 4. 数值范围验证
 */
public class ReLuLayerTest {

    private ReLuLayer reluLayer;
    private ReLuLayer reluWithShape;
    
    @Before
    public void setUp() {
        // 创建不同配置的ReLU层
        reluLayer = new ReLuLayer("relu");
        reluWithShape = new ReLuLayer("relu_shaped", Shape.of(2, 3));
    }

    @Test
    public void testBasicForwardPass() {
        // 测试基本的前向传播
        float[][] inputData = {{-1.0f, 0.0f, 1.0f}, {-2.5f, 2.5f, 0.5f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 3), output.getValue().getShape());
        
        // 验证ReLU函数的计算结果
        float[][] expectedData = {{0.0f, 0.0f, 1.0f}, {0.0f, 2.5f, 0.5f}};
        float[][] actualData = output.getValue().getMatrix();
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("ReLU应该正确处理负值和正值", expectedData[i][j], actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testNegativeInputs() {
        // 测试全负数输入
        float[][] inputData = {{-1.0f, -2.0f, -3.0f}, {-0.5f, -10.0f, -0.001f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 所有输出应该为0
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("ReLU对负输入应该输出0", 0.0f, actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testPositiveInputs() {
        // 测试全正数输入
        float[][] inputData = {{1.0f, 2.0f, 3.0f}, {0.5f, 10.0f, 0.001f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 输出应该等于输入
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("ReLU对正输入应该保持不变", inputData[i][j], actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testZeroInput() {
        // 测试零输入
        float[][] inputData = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 零输入应该输出零
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals("ReLU对零输入应该输出0", 0.0f, actualData[i][j], 0.001f);
            }
        }
    }

    @Test
    public void testShapePreservation() {
        // 测试各种形状的输入
        Shape[] testShapes = {
            Shape.of(1, 1),
            Shape.of(3, 4),
            Shape.of(2, 5, 3),
            Shape.of(1, 2, 3, 4)
        };
        
        for (Shape shape : testShapes) {
            NdArray input = NdArray.likeRandomN(shape);
            Variable inputVar = new Variable(input);
            
            Variable output = reluLayer.layerForward(inputVar);
            
            assertEquals("ReLU应该保持输入形状不变", shape, output.getValue().getShape());
        }
    }

    @Test
    public void testLargeValues() {
        // 测试大数值
        float[][] inputData = {{1000.0f, -1000.0f}, {Float.MAX_VALUE / 2, -Float.MAX_VALUE / 2}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        float[][] actualData = output.getValue().getMatrix();
        
        // 验证大正数保持不变，大负数变为0
        assertEquals("大正数应该保持不变", 1000.0f, actualData[0][0], 0.001f);
        assertEquals("大负数应该变为0", 0.0f, actualData[0][1], 0.001f);
        assertEquals("极大正数应该保持不变", Float.MAX_VALUE / 2, actualData[1][0], 0.001f);
        assertEquals("极大负数应该变为0", 0.0f, actualData[1][1], 0.001f);
    }

    @Test
    public void testRandomInput() {
        // 测试随机输入的属性
        NdArray input = NdArray.likeRandomN(Shape.of(100, 50));
        // 将一部分数据设为负数
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                if ((i + j) % 2 == 0) {
                    input.set(-Math.abs(input.get(i, j)), i, j);
                }
            }
        }
        
        Variable inputVar = new Variable(input);
        Variable output = reluLayer.layerForward(inputVar);
        NdArray outputArray = output.getValue();
        
        // 验证所有输出都是非负的
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 50; j++) {
                float value = outputArray.get(i, j);
                assertTrue("ReLU输出应该总是非负的", value >= 0.0f);
            }
        }
    }

    @Test
    public void testConstructorWithShape() {
        // 测试带形状的构造函数
        assertEquals("层名称应该正确设置", "relu_shaped", reluWithShape.getName());
        assertEquals("输入形状应该正确设置", Shape.of(2, 3), reluWithShape.getInputShape());
        assertEquals("输出形状应该等于输入形状", Shape.of(2, 3), reluWithShape.getOutputShape());
    }

    @Test
    public void testInitialization() {
        // 测试初始化方法
        try {
            reluLayer.init();
            reluWithShape.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
        
        // ReLU层不应该有参数
        assertTrue("ReLU层不应该有参数", reluLayer.getParams().isEmpty());
        assertTrue("ReLU层不应该有参数", reluWithShape.getParams().isEmpty());
    }

    @Test
    public void testRequiredInputNumber() {
        // 测试所需输入数量
        assertEquals("requireInputNum应该返回1", 1, reluLayer.requireInputNum());
    }

    @Test
    public void testSingleElementInput() {
        // 测试单元素输入
        NdArray input = NdArray.of(new float[][]{{-5.0f}});
        Variable inputVar = new Variable(input);
        
        Variable output = reluLayer.layerForward(inputVar);
        float result = output.getValue().get(0, 0);
        
        assertEquals("单个负数应该输出0", 0.0f, result, 0.001f);
        
        // 测试单个正数
        input = NdArray.of(new float[][]{{5.0f}});
        inputVar = new Variable(input);
        output = reluLayer.layerForward(inputVar);
        result = output.getValue().get(0, 0);
        
        assertEquals("单个正数应该保持不变", 5.0f, result, 0.001f);
    }
}