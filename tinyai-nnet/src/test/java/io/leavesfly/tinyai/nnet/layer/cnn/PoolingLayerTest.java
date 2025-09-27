package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * PoolingLayer池化层的单元测试
 * 
 * 测试池化层的基本功能：
 * 1. 不同池化类型的前向传播
 * 2. 输出形状计算
 * 3. 参数初始化
 * 4. 边界情况处理
 */
public class PoolingLayerTest {

    private PoolingLayer maxPool;
    private PoolingLayer avgPool;
    private PoolingLayer adaptivePool;
    
    @Before
    public void setUp() {
        // 创建不同类型的池化层
        maxPool = new PoolingLayer("max_pool", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        avgPool = new PoolingLayer("avg_pool", PoolingLayer.PoolingType.AVERAGE, 2, 2, 0);
        adaptivePool = new PoolingLayer("adaptive_pool", PoolingLayer.PoolingType.ADAPTIVE, 1, 1, 0);
    }

    @Test
    public void testParameterInitialization() {
        // 池化层不应该有可训练参数
        assertTrue("最大池化层不应该有参数", maxPool.getParams().isEmpty());
        assertTrue("平均池化层不应该有参数", avgPool.getParams().isEmpty());
        assertTrue("自适应池化层不应该有参数", adaptivePool.getParams().isEmpty());
    }

    @Test
    public void testMaxPoolingForwardPass() {
        // 测试最大池化前向传播
        // 输入形状: (batch_size=1, channels=1, height=4, width=4)
        float[][][][] inputData = {
            {
                {
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f}
                }
            }
        };
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = maxPool.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 计算期望的输出形状
            // 输入: 4x4, 池化窗口: 2x2, 步长: 2, 填充: 0
            // 输出尺寸: (4 - 2 + 0*2) / 2 + 1 = 2
            Shape expectedShape = Shape.of(1, 1, 2, 2);
            assertEquals("最大池化输出形状应该正确", expectedShape, output.getValue().getShape());
            
            // 验证最大池化结果
            // 第一个池化窗口: [1,2,5,6] -> max=6
            // 第二个池化窗口: [3,4,7,8] -> max=8
            // 第三个池化窗口: [9,10,13,14] -> max=14
            // 第四个池化窗口: [11,12,15,16] -> max=16
            float[][] result = output.getValue().get4dArray()[0][0];
            assertEquals("第一个池化窗口最大值应该正确", 6.0f, result[0][0], 1e-6f);
            assertEquals("第二个池化窗口最大值应该正确", 8.0f, result[0][1], 1e-6f);
            assertEquals("第三个池化窗口最大值应该正确", 14.0f, result[1][0], 1e-6f);
            assertEquals("第四个池化窗口最大值应该正确", 16.0f, result[1][1], 1e-6f);
            
        } catch (Exception e) {
            fail("最大池化前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testAveragePoolingForwardPass() {
        // 测试平均池化前向传播
        // 输入形状: (batch_size=1, channels=1, height=4, width=4)
        float[][][][] inputData = {
            {
                {
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f}
                }
            }
        };
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = avgPool.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 计算期望的输出形状
            Shape expectedShape = Shape.of(1, 1, 2, 2);
            assertEquals("平均池化输出形状应该正确", expectedShape, output.getValue().getShape());
            
            // 验证平均池化结果
            // 第一个池化窗口: [1,2,5,6] -> avg=3.5
            // 第二个池化窗口: [3,4,7,8] -> avg=5.5
            // 第三个池化窗口: [9,10,13,14] -> avg=11.5
            // 第四个池化窗口: [11,12,15,16] -> avg=13.5
            float[][] result = output.getValue().get4dArray()[0][0];
            assertEquals("第一个池化窗口平均值应该正确", 3.5f, result[0][0], 1e-6f);
            assertEquals("第二个池化窗口平均值应该正确", 5.5f, result[0][1], 1e-6f);
            assertEquals("第三个池化窗口平均值应该正确", 11.5f, result[1][0], 1e-6f);
            assertEquals("第四个池化窗口平均值应该正确", 13.5f, result[1][1], 1e-6f);
            
        } catch (Exception e) {
            fail("平均池化前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testAdaptivePoolingForwardPass() {
        // 测试自适应池化前向传播
        // 输入形状: (batch_size=1, channels=1, height=4, width=4)
        float[][][][] inputData = {
            {
                {
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {5.0f, 6.0f, 7.0f, 8.0f},
                    {9.0f, 10.0f, 11.0f, 12.0f},
                    {13.0f, 14.0f, 15.0f, 16.0f}
                }
            }
        };
        
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = adaptivePool.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 计算期望的输出形状 (自适应池化输出1x1)
            Shape expectedShape = Shape.of(1, 1, 1, 1);
            assertEquals("自适应池化输出形状应该正确", expectedShape, output.getValue().getShape());
            
            // 验证自适应池化结果 (整个特征图的平均值)
            float result = output.getValue().get4dArray()[0][0][0][0];
            // 平均值 = (1+2+...+16) / 16 = 136 / 16 = 8.5
            assertEquals("自适应池化结果应该正确", 8.5f, result, 1e-6f);
            
        } catch (Exception e) {
            fail("自适应池化前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testOutputShapeCalculation() {
        // 测试不同输入尺寸下的输出形状计算
        
        // 测试1: 输入6x6，池化窗口2x2，步长2，填充0
        // 输出应该是: (6 - 2 + 0*2) / 2 + 1 = 3
        NdArray input1 = NdArray.likeRandomN(Shape.of(2, 3, 6, 6));
        Variable inputVar1 = new Variable(input1);
        
        PoolingLayer pool1 = new PoolingLayer("pool1", PoolingLayer.PoolingType.MAX, 2, 2, 0);
        Variable output1 = pool1.layerForward(inputVar1);
        Shape expectedShape1 = Shape.of(2, 3, 3, 3);
        assertEquals("6x6输入池化输出形状应该正确", expectedShape1, output1.getValue().getShape());
        
        // 测试2: 输入5x5，池化窗口3x3，步长1，填充1
        // 输出应该是: (5 + 2*1 - 3) / 1 + 1 = 5
        NdArray input2 = NdArray.likeRandomN(Shape.of(1, 2, 5, 5));
        Variable inputVar2 = new Variable(input2);
        
        PoolingLayer pool2 = new PoolingLayer("pool2", PoolingLayer.PoolingType.MAX, 3, 1, 1);
        Variable output2 = pool2.layerForward(inputVar2);
        Shape expectedShape2 = Shape.of(1, 2, 5, 5);
        assertEquals("5x5输入池化输出形状应该正确", expectedShape2, output2.getValue().getShape());
    }

    @Test
    public void testRequiredInputNumber() {
        // 测试输入数量要求
        assertEquals("池化层应该只需要1个输入", 1, maxPool.requireInputNum());
        assertEquals("池化层应该只需要1个输入", 1, avgPool.requireInputNum());
        assertEquals("池化层应该只需要1个输入", 1, adaptivePool.requireInputNum());
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "max_pool", maxPool.getName());
        assertEquals("层名称应该正确", "avg_pool", avgPool.getName());
        assertEquals("层名称应该正确", "adaptive_pool", adaptivePool.getName());
    }

    @Test
    public void testInitialization() {
        // 测试初始化方法
        try {
            maxPool.init();
            avgPool.init();
            adaptivePool.init();
        } catch (Exception e) {
            fail("池化层初始化不应该抛出异常");
        }
        
        // 池化层初始化后不应该有参数
        assertTrue("池化层初始化后不应该有参数", maxPool.getParams().isEmpty());
    }

    @Test
    public void test4DInputValidation() {
        // 测试4D输入验证
        // 创建3D输入（错误的维度）
        NdArray wrong3DInput = NdArray.likeRandomN(Shape.of(1, 5, 5));
        Variable inputVar = new Variable(wrong3DInput);
        
        try {
            maxPool.layerForward(inputVar);
            fail("应该抛出4维输入要求的异常");
        } catch (RuntimeException e) {
            assertTrue("异常信息应该包含4维要求", e.getMessage().contains("池化层输入必须是4维的"));
        }
    }

    @Test
    public void testPoolingTypeEnum() {
        // 测试池化类型枚举
        assertEquals("应该有3种池化类型", 3, PoolingLayer.PoolingType.values().length);
        assertNotNull("最大池化类型应该存在", PoolingLayer.PoolingType.MAX);
        assertNotNull("平均池化类型应该存在", PoolingLayer.PoolingType.AVERAGE);
        assertNotNull("自适应池化类型应该存在", PoolingLayer.PoolingType.ADAPTIVE);
    }
}