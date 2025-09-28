package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * LayerNorm层的单元测试
 * 
 * 测试层归一化层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 不同构造函数
 * 4. 形状处理
 * 5. 数值稳定性
 */
public class LayerNormTest {

    private LayerNorm layerNorm2D;
    private LayerNorm layerNorm3D;
    private LayerNorm layerNormWithShape;
    
    @Before
    public void setUp() {
        // 创建2D输入的LayerNorm层
        layerNorm2D = new LayerNorm("ln2d", Shape.of(4, 10));
        
        // 创建3D输入的LayerNorm层
        layerNorm3D = new LayerNorm("ln3d", Shape.of(2, 8, 16));
        
        // 使用normalizedShape构造函数
        layerNormWithShape = new LayerNorm("ln_shape", 512, 1e-6);
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数是否正确初始化
        Parameter gamma = layerNorm2D.getParamBy("gamma");
        Parameter beta = layerNorm2D.getParamBy("beta");
        
        assertNotNull("gamma参数应该被初始化", gamma);
        assertNotNull("beta参数应该被初始化", beta);
        
        // 检查gamma初始化为1
        NdArray gammaValue = gamma.getValue();
        assertEquals("gamma形状应该是(10,)", Shape.of(10), gammaValue.getShape());
        
        // 检查beta初始化为0
        NdArray betaValue = beta.getValue();
        assertEquals("beta形状应该是(10,)", Shape.of(10), betaValue.getShape());
        
        // 验证初始值
        float gammaFirst = gammaValue.get(0);
        float betaFirst = betaValue.get(0);
        
        assertEquals("gamma初始值应该是1", 1.0f, gammaFirst, 0.001f);
        assertEquals("beta初始值应该是0", 0.0f, betaFirst, 0.001f);
    }

    @Test
    public void test2DForwardPass() {
        // 测试2D输入的前向传播
        float[][] inputData = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                               {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = layerNorm2D.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 10), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void test3DForwardPass() {
        // 测试3D输入的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 8, 16));
        Variable inputVar = new Variable(input);
        
        // 执行前向传播
        Variable output = layerNorm3D.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该与输入形状相同", Shape.of(2, 8, 16), output.getValue().getShape());
        
        // 验证输出不为null
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testNormalizedShapeConstructor() {
        // 测试使用normalizedShape的构造函数
        Parameter gamma = layerNormWithShape.getParamBy("gamma");
        Parameter beta = layerNormWithShape.getParamBy("beta");
        
        assertNotNull("gamma参数应该存在", gamma);
        assertNotNull("beta参数应该存在", beta);
        
        // 验证参数形状
        assertEquals("gamma形状应该是(512,)", Shape.of(512), gamma.getValue().getShape());
        assertEquals("beta形状应该是(512,)", Shape.of(512), beta.getValue().getShape());
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称是否正确设置
        Parameter gamma = layerNorm2D.getParamBy("gamma");
        Parameter beta = layerNorm2D.getParamBy("beta");
        
        assertEquals("gamma参数名称应该正确", "ln2d_gamma", gamma.getName());
        assertEquals("beta参数名称应该正确", "ln2d_beta", beta.getName());
    }

    @Test
    public void testNullInputShape() {
        // 测试null输入形状的处理
        LayerNorm nullShapeLN = new LayerNorm("null_shape", 64, 1e-6);
        
        // 应该能正常创建，使用默认特征数
        assertNotNull("null形状输入应该能正常创建gamma参数", nullShapeLN.getParamBy("gamma"));
        assertNotNull("null形状输入应该能正常创建beta参数", nullShapeLN.getParamBy("beta"));
    }

    @Test
    public void testEpsilonParameter() {
        // 测试不同epsilon值的LayerNorm
        LayerNorm largeEpsLN = new LayerNorm("large_eps", 10, 1e-3);
        LayerNorm smallEpsLN = new LayerNorm("small_eps", 10, 1e-8);
        
        // 创建测试输入
        NdArray testInput = NdArray.likeRandomN(Shape.of(3, 10));
        Variable inputVar = new Variable(testInput);
        
        // 两者都应该能正常工作
        Variable output1 = largeEpsLN.layerForward(inputVar);
        Variable output2 = smallEpsLN.layerForward(inputVar);
        
        assertNotNull("大epsilon值应该产生有效输出", output1.getValue());
        assertNotNull("小epsilon值应该产生有效输出", output2.getValue());
    }

    @Test
    public void testBasicNormalizationProperty() {
        // 测试基本的归一化属性
        float[][] fixedData = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        NdArray input = NdArray.of(fixedData);
        Variable inputVar = new Variable(input);
        
        LayerNorm fixedLN = new LayerNorm("fixed", Shape.of(2, 3));
        Variable output = fixedLN.layerForward(inputVar);
        
        // 验证输出形状保持不变
        assertEquals("归一化后形状应该保持不变", Shape.of(2, 3), output.getValue().getShape());
        
        // 验证输出数值的有效性
        NdArray outputArray = output.getValue();
        assertNotNull("LayerNorm输出应该存在", outputArray);
        
        // 检查输出不是原始输入（应该经过了归一化变换）
        float[][] outputData = outputArray.getMatrix();
        boolean hasChanged = false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                if (Math.abs(outputData[i][j] - fixedData[i][j]) > 1e-6) {
                    hasChanged = true;
                    break;
                }
            }
        }
        assertTrue("LayerNorm应该改变输入值", hasChanged);
    }

    @Test
    public void testForwardBackwardMethods() {
        // 测试forward和backward方法
        NdArray input = NdArray.likeRandomN(Shape.of(4, 10));  // 修正为与构造函数中的形状一致
        
        // 测试forward方法
        NdArray output = layerNorm2D.forward(input);
        assertNotNull("forward方法应该返回有效输出", output);
        assertEquals("forward输出形状应该正确", input.getShape(), output.getShape());
        
        // 测试backward方法
        NdArray grad = NdArray.likeRandomN(Shape.of(4, 10));  // 修正为与输出形状一致
        try {
            layerNorm2D.backward(grad);
            // backward方法应该能正常执行而不抛出异常
        } catch (Exception e) {
            fail("backward方法不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testRequireInputNum() {
        // 测试requireInputNum方法
        assertEquals("LayerNorm应该需要1个输入", 1, layerNorm2D.requireInputNum());
    }

    @Test
    public void testNumericalStability() {
        // 测试数值稳定性，使用可能导致除零的极端值
        float[][] extremeData = {{0.0f, 0.0f, 0.0f}, {1e-8f, 1e-8f, 1e-8f}};
        NdArray input = NdArray.of(extremeData);
        Variable inputVar = new Variable(input);
        
        LayerNorm extremeLN = new LayerNorm("extreme", Shape.of(2, 3));
        
        try {
            Variable output = extremeLN.layerForward(inputVar);
            assertNotNull("极端值输入应该产生有效输出", output.getValue());
            
            // 检查输出是否包含NaN或无穷大
            NdArray outputArray = output.getValue();
            float[][] outputData = outputArray.getMatrix();
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    float val = outputData[i][j];
                    assertFalse("输出不应该包含NaN", Float.isNaN(val));
                    assertFalse("输出不应该包含无穷大", Float.isInfinite(val));
                }
            }
        } catch (Exception e) {
            fail("极端值输入不应该导致异常: " + e.getMessage());
        }
    }

    @Test
    public void testSingleElementInput() {
        // 测试单元素输入的处理
        float[][] singleElement = {{5.0f}};
        NdArray input = NdArray.of(singleElement);
        Variable inputVar = new Variable(input);
        
        LayerNorm singleLN = new LayerNorm("single", Shape.of(1, 1));
        
        try {
            Variable output = singleLN.layerForward(inputVar);
            assertNotNull("单元素输入应该产生有效输出", output.getValue());
            assertEquals("单元素输出形状应该正确", Shape.of(1, 1), output.getValue().getShape());
        } catch (Exception e) {
            fail("单元素输入不应该导致异常: " + e.getMessage());
        }
    }
}