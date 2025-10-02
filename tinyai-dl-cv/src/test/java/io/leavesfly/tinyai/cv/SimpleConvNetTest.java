package io.leavesfly.tinyai.cv;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.LayerAble;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * SimpleConvNet的单元测试类
 * 
 * @author leavesfly
 * @version 0.01
 */
public class SimpleConvNetTest {
    
    private SimpleConvNet convNet;
    private Shape inputShape;
    private int numClasses;
    
    @Before
    public void setUp() {
        // 设置测试参数
        int batchSize = 2;
        int channels = 3;  // RGB图像
        int height = 32;   // 图像高度
        int width = 32;    // 图像宽度
        numClasses = 10;   // 分类数
        
        inputShape = Shape.of(batchSize, channels, height, width);
        
        // 创建SimpleConvNet实例
        convNet = new SimpleConvNet("test_conv_net", inputShape, numClasses);
    }
    
    @Test
    public void testNetworkConstruction() {
        assertNotNull(convNet);
        assertEquals("test_conv_net", convNet.getName());
    }
    
    @Test
    public void testNetworkInfo() {
        String info = convNet.getNetworkInfo();
        assertNotNull(info);
        assertTrue(info.contains("SimpleConvNet配置"));
        assertTrue(info.contains("输出类别数: " + numClasses));
    }
    
    @Test
    public void testPrintArchitecture() {
        // 测试打印网络架构不会抛出异常
        try {
            convNet.printArchitecture();
        } catch (Exception e) {
            fail("打印网络架构时发生异常: " + e.getMessage());
        }
    }
    
    @Test
    public void testNetworkInitialization() {
        // 测试网络初始化
        try {
            convNet.init();
        } catch (Exception e) {
            fail("网络初始化时发生异常: " + e.getMessage());
        }
        
        // 验证网络有层结构
        assertTrue(convNet.getLayersCount() > 0);
    }
    
    @Test
    public void testForwardPass() {
        // 创建测试输入数据
        NdArray inputData = NdArray.likeRandomN(inputShape);
        Variable input = new Variable(inputData);
        
        // 初始化网络
        convNet.init();
        
        // 执行前向传播
        Variable output = convNet.layerForward(input);
        
        assertNotNull(output);
        
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum());
        assertEquals(inputShape.getDimension(0), outputShape.getDimension(0));
        assertEquals(numClasses, outputShape.getDimension(1));
    }
    
    @Test
    public void testDifferentConfigurations() {
        // 测试不使用批量归一化的配置
        SimpleConvNet convNetNoBN = new SimpleConvNet("no_bn", inputShape, numClasses, false, 0.0f);
        assertNotNull(convNetNoBN);
        
        // 测试不同dropout率的配置
        SimpleConvNet convNetDropout = new SimpleConvNet("with_dropout", inputShape, numClasses, true, 0.3f);
        assertNotNull(convNetDropout);
    }
    
    @Test
    public void testInvalidInputShape() {
        // 测试无效的输入形状
        Shape invalidShape = Shape.of(32, 32); // 只有2维
        
        try {
            new SimpleConvNet("invalid", invalidShape, numClasses);
            fail("应该抛出无效输入形状的异常");
        } catch (IllegalArgumentException e) {
            // 期望的异常
        }
    }
    
    @Test
    public void testParameterCount() {
        convNet.init();
        
        // 验证网络有参数
        assertFalse(convNet.getAllParams().isEmpty());
        
        // 打印参数数量信息
        int paramCount = convNet.getAllParams().size();
        System.out.println("网络参数数量: " + paramCount);
        assertTrue(paramCount > 0);
    }
    
    @Test
    public void testClearGradients() {
        convNet.init();
        
        // 测试清除梯度不会抛出异常
        try {
            convNet.clearGrads();
        } catch (Exception e) {
            fail("清除梯度时发生异常: " + e.getMessage());
        }
    }
    
    @Test
    public void testMultipleBatchSizes() {
        // 测试不同批次大小的输入
        int[] batchSizes = {1, 4, 8};
        
        convNet.init();
        
        for (int batchSize : batchSizes) {
            Shape testInputShape = Shape.of(batchSize, 3, 32, 32);
            NdArray testInputData = NdArray.likeRandomN(testInputShape);
            Variable testInput = new Variable(testInputData);
            
            Variable output = convNet.layerForward(testInput);
            
            assertEquals(batchSize, output.getValue().getShape().getDimension(0));
            assertEquals(numClasses, output.getValue().getShape().getDimension(1));
        }
    }
    
    @Test
    public void testNewAccessorMethods() {
        // 测试新增的getter方法
        assertEquals(numClasses, convNet.getNumClasses());
        assertTrue(convNet.isUseBatchNorm());
        assertEquals(0.5f, convNet.getDropoutRate(), 0.001f);
        
        // 测试层数统计
        convNet.init();
        assertTrue(convNet.getLayersCount() > 0);
        
        // 测试获取指定层
        LayerAble firstLayer = convNet.getLayer(0);
        assertNotNull(firstLayer);
        
        // 测试边界条件
        try {
            convNet.getLayer(-1);
            fail("应该抛出索引越界异常");
        } catch (IndexOutOfBoundsException e) {
            // 期望的异常
        }
        
        try {
            convNet.getLayer(convNet.getLayersCount());
            fail("应该抛出索引越界异常");
        } catch (IndexOutOfBoundsException e) {
            // 期望的异常
        }
    }
    
    @Test
    public void testNetworkSummary() {
        // 测试网络摘要
        String summary = convNet.getSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("SimpleConvNet"));
        assertTrue(summary.contains(String.valueOf(numClasses)));
        assertTrue(summary.contains("true")); // BatchNorm启用
        assertTrue(summary.contains("0.50")); // Dropout比例
        
        System.out.println("网络摘要: " + summary);
    }
}