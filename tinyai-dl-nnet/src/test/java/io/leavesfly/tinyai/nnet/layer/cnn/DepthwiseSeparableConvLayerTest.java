package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * DepthwiseSeparableConvLayer深度可分离卷积层的单元测试
 * 
 * 测试深度可分离卷积层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 子层访问
 * 4. 输出形状计算
 */
public class DepthwiseSeparableConvLayerTest {

    private DepthwiseSeparableConvLayer separableConv;
    
    @Before
    public void setUp() {
        // 创建深度可分离卷积层: 3输入通道，16输出通道，3x3卷积核
        separableConv = new DepthwiseSeparableConvLayer("separable_conv", 3, 16, 3, 1, 1, true);
    }

    @Test
    public void testParameterInitialization() {
        // 深度可分离卷积层本身不应该有参数，参数在子层中
        assertTrue("深度可分离卷积层本身不应该有参数", separableConv.getParams().isEmpty());
        
        // 检查子层是否存在
        assertNotNull("深度卷积层应该存在", separableConv.getDepthwiseConv());
        assertNotNull("逐点卷积层应该存在", separableConv.getPointwiseConv());
    }

    @Test
    public void testSubLayerAccess() {
        // 测试子层访问方法
        ConvLayer depthwise = separableConv.getDepthwiseConv();
        ConvLayer pointwise = separableConv.getPointwiseConv();
        
        assertNotNull("深度卷积层不应该为null", depthwise);
        assertNotNull("逐点卷积层不应该为null", pointwise);
        
        // 验证子层名称
        assertTrue("深度卷积层名称应该正确", depthwise.getName().contains("separable_conv_depthwise"));
        assertTrue("逐点卷积层名称应该正确", pointwise.getName().contains("separable_conv_pointwise"));
    }

    @Test
    public void testBasicForwardPass() {
        // 测试基本前向传播
        // 输入形状: (batch_size=1, channels=3, height=8, width=8)
        NdArray input = NdArray.likeRandomN(Shape.of(1, 3, 8, 8));
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = separableConv.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 计算期望的输出形状
            // 输入: 8x8, 卷积核: 3x3, 步长: 1, 填充: 1
            // 输出尺寸: (8 + 2*1 - 3) / 1 + 1 = 8
            // 输出通道数: 16 (由构造函数指定)
            Shape expectedShape = Shape.of(1, 16, 8, 8);
            assertEquals("输出形状应该正确", expectedShape, output.getValue().getShape());
            
        } catch (Exception e) {
            fail("基本前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testOutputShapeCalculation() {
        // 测试不同配置下的输出形状计算
        
        // 测试1: 输入16x16，卷积核3x3，步长1，填充1
        NdArray input1 = NdArray.likeRandomN(Shape.of(2, 3, 16, 16));
        Variable inputVar1 = new Variable(input1);
        
        Variable output1 = separableConv.layerForward(inputVar1);
        Shape expectedShape1 = Shape.of(2, 16, 16, 16);
        assertEquals("16x16输入输出形状应该正确", expectedShape1, output1.getValue().getShape());
        
        // 测试2: 不同通道数配置
        DepthwiseSeparableConvLayer separableConv2 = new DepthwiseSeparableConvLayer("separable_conv2", 8, 32, 3, 2, 1, true);
        NdArray input2 = NdArray.likeRandomN(Shape.of(1, 8, 10, 10));
        Variable inputVar2 = new Variable(input2);
        
        Variable output2 = separableConv2.layerForward(inputVar2);
        // 输入: 10x10, 卷积核: 3x3, 步长: 2, 填充: 1
        // 输出尺寸: (10 + 2*1 - 3) / 2 + 1 = 5
        Shape expectedShape2 = Shape.of(1, 32, 5, 5);
        assertEquals("不同通道数配置输出形状应该正确", expectedShape2, output2.getValue().getShape());
    }

    @Test
    public void testRequiredInputNumber() {
        // 测试输入数量要求
        assertEquals("深度可分离卷积层应该只需要1个输入", 1, separableConv.requireInputNum());
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "separable_conv", separableConv.getName());
    }

    @Test
    public void testInitialization() {
        // 测试初始化方法
        try {
            separableConv.init();
        } catch (Exception e) {
            fail("深度可分离卷积层初始化不应该抛出异常");
        }
        
        // 初始化后子层应该存在
        assertNotNull("初始化后深度卷积层应该存在", separableConv.getDepthwiseConv());
        assertNotNull("初始化后逐点卷积层应该存在", separableConv.getPointwiseConv());
    }

    @Test
    public void testClearGrads() {
        // 测试梯度清理方法
        try {
            separableConv.clearGrads();
            // 不应该抛出异常
        } catch (Exception e) {
            fail("梯度清理不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testFromShapeConstructor() {
        // 测试从输入形状构造
        Shape inputShape = Shape.of(1, 6, 32, 32);
        DepthwiseSeparableConvLayer shapeConv = new DepthwiseSeparableConvLayer("shape_conv", inputShape);
        
        // 验证从形状推断的参数
        assertNotNull("从形状构造的深度卷积层应该存在", shapeConv.getDepthwiseConv());
        assertNotNull("从形状构造的逐点卷积层应该存在", shapeConv.getPointwiseConv());
        
        // 输入通道数应该是6，输出通道数应该是默认的12(输入的两倍)
        ConvLayer pointwise = shapeConv.getPointwiseConv();
        // 这里我们验证子层的存在，具体的参数验证需要访问子层的内部状态
    }

    @Test
    public void test4DInputValidation() {
        // 测试4D输入验证
        // 创建3D输入（错误的维度）
        NdArray wrong3DInput = NdArray.likeRandomN(Shape.of(1, 5, 5));
        Variable inputVar = new Variable(wrong3DInput);
        
        try {
            separableConv.layerForward(inputVar);
            fail("应该抛出4维输入要求的异常");
        } catch (RuntimeException e) {
            assertTrue("异常信息应该包含4维要求", e.getMessage().contains("深度可分离卷积层输入必须是4维的"));
        }
    }
}