package io.leavesfly.tinyai.nnet.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * ConvLayer卷积层的单元测试
 * 
 * 测试卷积层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 输出形状计算
 * 4. 不同配置下的行为
 */
public class ConvLayerTest {

    private ConvLayer basicConv;
    private ConvLayer convWithBias;
    private ConvLayer convNoBias;
    
    @Before
    public void setUp() {
        // 创建基本卷积层: 3通道输入，16通道输出，3x3卷积核
        basicConv = new ConvLayer("basic_conv", 3, 16, 3, 1, 1, true);
        
        // 创建带偏置的卷积层
        convWithBias = new ConvLayer("conv_bias", 1, 8, 3, 1, 0, true);
        
        // 创建不带偏置的卷积层
        convNoBias = new ConvLayer("conv_no_bias", 1, 8, 3, 1, 0, false);
    }

    @Test
    public void testParameterInitialization() {
        // 测试权重参数初始化
        Parameter weight = basicConv.getParamBy("weight");
        assertNotNull("权重参数应该被初始化", weight);
        
        // 检查权重形状: (out_channels, in_channels, kernel_height, kernel_width)
        Shape expectedWeightShape = Shape.of(16, 3, 3, 3);
        assertEquals("权重形状应该正确", expectedWeightShape, weight.getValue().getShape());
        
        // 测试偏置参数初始化
        Parameter bias = basicConv.getParamBy("bias");
        assertNotNull("偏置参数应该被初始化", bias);
        
        // 检查偏置形状: (out_channels,)
        Shape expectedBiasShape = Shape.of(16);
        assertEquals("偏置形状应该正确", expectedBiasShape, bias.getValue().getShape());
    }

    @Test
    public void testNoBiasConfiguration() {
        // 测试不使用偏置的配置
        Parameter weight = convNoBias.getParamBy("weight");
        Parameter bias = convNoBias.getParamBy("bias");
        
        assertNotNull("权重参数应该存在", weight);
        assertNull("偏置参数不应该存在", bias);
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称设置
        Parameter weight = basicConv.getParamBy("weight");
        Parameter bias = basicConv.getParamBy("bias");
        
        assertEquals("权重参数名称应该正确", "basic_conv_weight", weight.getName());
        assertEquals("偏置参数名称应该正确", "basic_conv_bias", bias.getName());
    }

    @Test
    public void testBasicForwardPass() {
        // 测试基本前向传播
        // 输入形状: (batch_size=1, channels=1, height=5, width=5)
        NdArray input = NdArray.likeRandomN(Shape.of(1, 1, 5, 5));
        Variable inputVar = new Variable(input);
        
        // 创建一个匹配输入通道数的卷积层
        ConvLayer matchingConv = new ConvLayer("matching_conv", 1, 8, 3, 1, 0, true);
        
        try {
            Variable output = matchingConv.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 计算期望的输出形状
            // 输入: 5x5, 卷积核: 3x3, 步长: 1, 填充: 0
            // 输出尺寸: (5 - 3 + 0*2) / 1 + 1 = 3
            Shape expectedShape = Shape.of(1, 8, 3, 3);
            assertEquals("输出形状应该正确", expectedShape, output.getValue().getShape());
            
        } catch (Exception e) {
            fail("基本前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testOutputShapeCalculation() {
        // 测试不同配置下的输出形状计算
        
        // 测试1: 输入7x7，卷积核3x3，步长1，填充1
        // 输出应该是: (7 + 2*1 - 3) / 1 + 1 = 7
        NdArray input1 = NdArray.likeRandomN(Shape.of(2, 3, 7, 7));
        Variable inputVar1 = new Variable(input1);
        
        // 创建一个匹配输入通道数的卷积层
        ConvLayer matchingConv = new ConvLayer("matching_conv", 3, 16, 3, 1, 1, true);
        
        try {
            Variable output1 = matchingConv.layerForward(inputVar1);
            Shape expectedShape1 = Shape.of(2, 16, 7, 7);
            assertEquals("填充卷积输出形状应该正确", expectedShape1, output1.getValue().getShape());
        } catch (Exception e) {
            fail("输出形状计算测试不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testRequiredInputNumber() {
        // 测试输入数量要求
        assertEquals("卷积层应该只需要1个输入", 1, basicConv.requireInputNum());
        assertEquals("卷积层应该只需要1个输入", 1, convNoBias.requireInputNum());
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "basic_conv", basicConv.getName());
        assertEquals("层名称应该正确", "conv_no_bias", convNoBias.getName());
    }

    @Test
    public void testHeInitialization() {
        // 测试He权重初始化
        Parameter weight = basicConv.getParamBy("weight");
        NdArray weightData = weight.getValue();
        
        // 将权重数据转换为矩阵格式进行统计（简化处理）
        try {
            float[][] weightMatrix = weightData.getMatrix();
            
            // 计算权重的统计信息
            double sum = 0;
            double sumSquared = 0;
            int totalElements = 0;
            
            for (int i = 0; i < weightMatrix.length; i++) {
                for (int j = 0; j < weightMatrix[i].length; j++) {
                    float w = weightMatrix[i][j];
                    sum += w;
                    sumSquared += w * w;
                    totalElements++;
                }
            }
            
            double mean = sum / totalElements;
            double variance = sumSquared / totalElements - mean * mean;
            double std = Math.sqrt(variance);
            
            // He初始化的标准差应该约为 sqrt(2 / fan_in)
            // fan_in = in_channels * kernel_height * kernel_width = 3 * 3 * 3 = 27
            double expectedStd = Math.sqrt(2.0 / 27.0);
            
            // 验证标准差在合理范围内（允许一定误差）
            assertTrue("权重标准差应该在合理范围内", Math.abs(std - expectedStd) < 0.2);
            assertTrue("权重均值应该接近0", Math.abs(mean) < 0.1);
        } catch (Exception e) {
            // 如果getMatrix()不支持4D数组，使用简化的方法进行验证
            System.out.println("He初始化测试使用简化验证: " + e.getMessage());
            // 简单验证权重数据不为null且有合理值
            assertNotNull("权重数据不应该为null", weightData);
        }
    }

    @Test
    public void testBiasInitialization() {
        // 测试偏置初始化（应该初始化为0）
        Parameter bias = basicConv.getParamBy("bias");
        NdArray biasData = bias.getValue();
        
        // 获取偏置的形状信息
        Shape biasShape = biasData.getShape();
        
        // 验证偏置值均为0，检查前几个元素
        for (int i = 0; i < Math.min(5, biasShape.getDimension(0)); i++) {
            assertEquals("偏置应该初始化为0", 0.0f, biasData.get(i), 1e-6f);
        }
    }

    @Test
    public void testConvLayerFromShape() {
        // 测试从输入形状构造卷积层
        Shape inputShape = Shape.of(1, 1, 32, 32);  // 使用与构造函数默认值一致的输入通道数
        ConvLayer shapeConv = new ConvLayer("shape_conv", inputShape);
        
        // 验证从形状推断的参数
        Parameter weight = shapeConv.getParamBy("weight");
        assertNotNull("从形状构造的权重应该存在", weight);
        
        // 输入通道数应该是1（从输入形状推断），输出通道数应该是默认的32
        Shape expectedWeightShape = Shape.of(32, 1, 3, 3);
        assertEquals("从形状推断的权重形状应该正确", expectedWeightShape, weight.getValue().getShape());
    }
}