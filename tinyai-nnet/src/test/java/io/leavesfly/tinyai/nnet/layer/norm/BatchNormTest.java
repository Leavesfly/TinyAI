package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * BatchNorm层的单元测试
 * 
 * 测试批量归一化层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 形状处理
 * 4. 边界情况
 */
public class BatchNormTest {

    private BatchNorm batchNorm2D;
    private BatchNorm batchNorm4D;
    
    @Before
    public void setUp() {
        // 创建2D输入的BatchNorm层
        batchNorm2D = new BatchNorm("bn2d", Shape.of(4, 10), 10, 1e-5);
        
        // 创建4D输入的BatchNorm层（用于卷积）
        batchNorm4D = new BatchNorm("bn4d", Shape.of(2, 8, 16, 16), 8, 1e-5);
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数是否正确初始化
        Parameter gamma = batchNorm2D.getParamBy("gamma");
        Parameter beta = batchNorm2D.getParamBy("beta");
        
        assertNotNull("gamma参数应该被初始化", gamma);
        assertNotNull("beta参数应该被初始化", beta);
        
        // 检查gamma初始化为1
        NdArray gammaValue = gamma.getValue();
        assertEquals("gamma形状应该是(10,)", Shape.of(10), gammaValue.getShape());
        
        // 检查beta初始化为0
        NdArray betaValue = beta.getValue();
        assertEquals("beta形状应该是(10,)", Shape.of(10), betaValue.getShape());
        
        // 验证初始值（gamma应该接近1，beta应该接近0）
        float gammaFirst = gammaValue.get(0);
        float betaFirst = betaValue.get(0);
        
        assertEquals("gamma初始值应该是1", 1.0f, gammaFirst, 0.001f);
        assertEquals("beta初始值应该是0", 0.0f, betaFirst, 0.001f);
    }

    @Test
    public void test2DForwardPass() {
        // 测试2D输入的前向传播
        // 创建测试输入 (batch_size=2, features=10)
        NdArray input = NdArray.likeRandomN(Shape.of(2, 10));
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = batchNorm2D.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 10), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void test4DForwardPass() {
        // 测试4D输入的前向传播（卷积输入）
        // 创建测试输入 (batch_size=1, channels=8, height=4, width=4)
        NdArray input = NdArray.likeRandomN(Shape.of(1, 8, 4, 4));
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = batchNorm4D.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(1, 8, 4, 4), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testDefaultConstructor() {
        // 测试默认构造函数
        BatchNorm defaultBN = new BatchNorm("default", Shape.of(3, 5), 5, 1e-5);
        
        // 验证参数已初始化
        assertNotNull("默认构造的gamma参数应该存在", defaultBN.getParamBy("gamma"));
        assertNotNull("默认构造的beta参数应该存在", defaultBN.getParamBy("beta"));
        
        // 验证特征数量推断正确
        Parameter gamma = defaultBN.getParamBy("gamma");
        assertEquals("默认构造应该推断特征数为5", Shape.of(5), gamma.getValue().getShape());
    }

    @Test
    public void testNullInputShape() {
        // 测试null输入形状的处理
        BatchNorm nullShapeBN = new BatchNorm("null_shape", null, 10, 1e-5);
        
        // 应该能正常创建，使用默认特征数
        assertNotNull("null形状输入应该能正常创建gamma参数", nullShapeBN.getParamBy("gamma"));
        assertNotNull("null形状输入应该能正常创建beta参数", nullShapeBN.getParamBy("beta"));
    }

    @Test
    public void testSmallBatchNormalization() {
        // 测试小批量的归一化效果
        // 创建简单的测试数据
        float[][] data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        NdArray input = NdArray.of(data);
        Variable inputVar = new Variable(input);
        
        // 创建对应的BatchNorm层
        BatchNorm smallBN = new BatchNorm("small", Shape.of(2, 3), 3, 1e-5);
        
        // 执行前向传播
        Variable output = smallBN.layerForward(inputVar);
        
        // 验证输出形状正确
        assertEquals("小批量输出形状应该正确", Shape.of(2, 3), output.getValue().getShape());
        
        // 验证输出数值存在（非NaN或无穷大）
        NdArray outputArray = output.getValue();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float val = outputArray.get(i, j);
                assertFalse("输出不应该包含NaN", Float.isNaN(val));
                assertFalse("输出不应该包含无穷大", Float.isInfinite(val));
            }
        }
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称是否正确设置
        Parameter gamma = batchNorm2D.getParamBy("gamma");
        Parameter beta = batchNorm2D.getParamBy("beta");
        
        assertEquals("gamma参数名称应该正确", "bn2d_gamma", gamma.getName());
        assertEquals("beta参数名称应该正确", "bn2d_beta", beta.getName());
    }

    @Test
    public void testEpsilonParameter() {
        // 测试不同epsilon值的处理
        BatchNorm largeBN = new BatchNorm("large_eps", Shape.of(2, 3), 3, 1e-3);
        BatchNorm smallBN = new BatchNorm("small_eps", Shape.of(2, 3), 3, 1e-8);
        
        // 两者都应该能正常工作
        NdArray testInput = NdArray.likeRandomN(Shape.of(2, 3));
        Variable inputVar = new Variable(testInput);
        
        Variable output1 = largeBN.layerForward(inputVar);
        Variable output2 = smallBN.layerForward(inputVar);
        
        assertNotNull("大epsilon值应该产生有效输出", output1.getValue());
        assertNotNull("小epsilon值应该产生有效输出", output2.getValue());
    }

    @Test
    public void testBasicNormalizationProperty() {
        // 测试基本的归一化属性
        // 使用固定值输入进行测试
        float[][] fixedData = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        NdArray input = NdArray.of(fixedData);
        Variable inputVar = new Variable(input);
        
        BatchNorm fixedBN = new BatchNorm("fixed", Shape.of(2, 2), 2, 1e-5);
        Variable output = fixedBN.layerForward(inputVar);
        
        // 验证输出形状保持不变
        assertEquals("归一化后形状应该保持不变", Shape.of(2, 2), output.getValue().getShape());
        
        // 验证输出数值的有效性
        NdArray outputArray = output.getValue();
        assertNotNull("归一化输出应该存在", outputArray);
        
        // 检查输出不是原始输入（应该经过了归一化变换）
        float[][] outputData = outputArray.getMatrix();
        boolean hasChanged = false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (Math.abs(outputData[i][j] - fixedData[i][j]) > 1e-6) {
                    hasChanged = true;
                    break;
                }
            }
        }
        assertTrue("BatchNorm应该改变输入值", hasChanged);
    }

    @Test
    public void testSingleBatchInput() {
        // 测试单个样本输入
        float[][] singleBatch = {{1.0f, 2.0f, 3.0f}};
        NdArray input = NdArray.of(singleBatch);
        Variable inputVar = new Variable(input);
        
        BatchNorm singleBN = new BatchNorm("single", Shape.of(1, 3), 3, 1e-5);
        
        try {
            Variable output = singleBN.layerForward(inputVar);
            assertNotNull("单样本输入应该产生有效输出", output.getValue());
            assertEquals("单样本输出形状应该正确", Shape.of(1, 3), output.getValue().getShape());
        } catch (Exception e) {
            fail("单样本输入不应该抛出异常: " + e.getMessage());
        }
    }
}