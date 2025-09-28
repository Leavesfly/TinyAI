package io.leavesfly.tinyai.modality.cv;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.func.Variable;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

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
    
    @BeforeEach
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
        assertNotNull(convNet, "网络创建失败");
        assertEquals("test_conv_net", convNet.name, "网络名称不匹配");
    }
    
    @Test
    public void testNetworkInfo() {
        String info = convNet.getNetworkInfo();
        assertNotNull(info, "网络信息获取失败");
        assertTrue(info.contains("SimpleConvNet配置"), "网络信息格式不正确");
        assertTrue(info.contains("输出类别数: " + numClasses), "输出类别数信息不正确");
    }
    
    @Test
    public void testPrintArchitecture() {
        // 测试打印网络架构不会抛出异常
        assertDoesNotThrow(() -> {
            convNet.printArchitecture();
        }, "打印网络架构时发生异常");
    }
    
    @Test
    public void testNetworkInitialization() {
        // 测试网络初始化
        assertDoesNotThrow(() -> {
            convNet.init();
        }, "网络初始化时发生异常");
        
        // 验证网络有层结构
        assertTrue(convNet.layers.size() > 0, "网络应该包含多个层");
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
        
        assertNotNull(output, "前向传播输出为空");
        
        Shape outputShape = output.getValue().getShape();
        assertEquals(2, outputShape.getDimNum(), "输出应该是2维的");
        assertEquals(inputShape.getDimension(0), outputShape.getDimension(0), 
                    "批次大小应该保持不变");
        assertEquals(numClasses, outputShape.getDimension(1), 
                    "输出维度应该等于类别数");
    }
    
    @Test
    public void testDifferentConfigurations() {
        // 测试不使用批量归一化的配置
        SimpleConvNet convNetNoBN = new SimpleConvNet("no_bn", inputShape, numClasses, false, 0.0f);
        assertNotNull(convNetNoBN, "无批量归一化的网络创建失败");
        
        // 测试不同dropout率的配置
        SimpleConvNet convNetDropout = new SimpleConvNet("with_dropout", inputShape, numClasses, true, 0.3f);
        assertNotNull(convNetDropout, "带dropout的网络创建失败");
    }
    
    @Test
    public void testInvalidInputShape() {
        // 测试无效的输入形状
        Shape invalidShape = Shape.of(32, 32); // 只有2维
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SimpleConvNet("invalid", invalidShape, numClasses);
        }, "应该抛出无效输入形状的异常");
    }
    
    @Test
    public void testParameterCount() {
        convNet.init();
        
        // 验证网络有参数
        assertFalse(convNet.getAllParams().isEmpty(), "网络应该包含可训练参数");
        
        // 打印参数数量信息
        int paramCount = convNet.getAllParams().size();
        System.out.println("网络参数数量: " + paramCount);
        assertTrue(paramCount > 0, "参数数量应该大于0");
    }
    
    @Test
    public void testClearGradients() {
        convNet.init();
        
        // 测试清除梯度不会抛出异常
        assertDoesNotThrow(() -> {
            convNet.clearGrads();
        }, "清除梯度时发生异常");
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
            
            assertEquals(batchSize, output.getValue().getShape().getDimension(0),
                        "批次大小 " + batchSize + " 的输出形状不正确");
            assertEquals(numClasses, output.getValue().getShape().getDimension(1),
                        "批次大小 " + batchSize + " 的输出类别数不正确");
        }
    }
}