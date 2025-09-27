package io.leavesfly.tinyai.nnet.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * LinearLayer的单元测试
 * 
 * 测试线性层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 形状变换
 * 4. 有无偏置的情况
 */
public class LinearLayerTest {

    private LinearLayer linearWithBias;
    private LinearLayer linearWithoutBias;
    
    @Before
    public void setUp() {
        // 创建带偏置的线性层：输入5维 -> 输出3维
        linearWithBias = new LinearLayer("linear_bias", 5, 3, true);
        
        // 创建不带偏置的线性层：输入4维 -> 输出2维
        linearWithoutBias = new LinearLayer("linear_no_bias", 4, 2, false);
    }

    @Test
    public void testParameterInitializationWithBias() {
        // 测试带偏置层的参数初始化
        Parameter wParam = linearWithBias.getParamBy("w");
        Parameter bParam = linearWithBias.getParamBy("b");
        
        assertNotNull("权重参数应该被初始化", wParam);
        assertNotNull("偏置参数应该被初始化", bParam);
        
        // 检查权重形状: (input_size, output_size) = (5, 3)
        assertEquals("权重形状应该正确", Shape.of(5, 3), wParam.getValue().getShape());
        
        // 检查偏置形状: (1, output_size) = (1, 3)
        assertEquals("偏置形状应该正确", Shape.of(1, 3), bParam.getValue().getShape());
        
        // 验证偏置初始化为0
        NdArray biasValue = bParam.getValue();
        for (int i = 0; i < 3; i++) {
            assertEquals("偏置应该初始化为0", 0.0f, biasValue.get(0, i), 0.001f);
        }
    }

    @Test
    public void testParameterInitializationWithoutBias() {
        // 测试不带偏置层的参数初始化
        Parameter wParam = linearWithoutBias.getParamBy("w");
        Parameter bParam = linearWithoutBias.getParamBy("b");
        
        assertNotNull("权重参数应该被初始化", wParam);
        assertNull("偏置参数不应该被初始化", bParam);
        
        // 检查权重形状: (input_size, output_size) = (4, 2)
        assertEquals("权重形状应该正确", Shape.of(4, 2), wParam.getValue().getShape());
    }

    @Test
    public void testForwardPassWithBias() {
        // 测试带偏置的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 5));
        Variable inputVar = new Variable(input);
        
        Variable output = linearWithBias.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该正确", Shape.of(2, 3), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardPassWithoutBias() {
        // 测试不带偏置的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(3, 4));
        Variable inputVar = new Variable(input);
        
        Variable output = linearWithoutBias.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该正确", Shape.of(3, 2), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testShapeConsistency() {
        // 测试输入输出形状的一致性
        assertEquals("输入形状应该正确设置", Shape.of(-1, 5), linearWithBias.getInputShape());
        assertEquals("输出形状应该正确设置", Shape.of(-1, 3), linearWithBias.getOutputShape());
        
        assertEquals("输入形状应该正确设置", Shape.of(-1, 4), linearWithoutBias.getInputShape());
        assertEquals("输出形状应该正确设置", Shape.of(-1, 2), linearWithoutBias.getOutputShape());
    }

    @Test
    public void testBatchProcessing() {
        // 测试批量处理
        NdArray input = NdArray.likeRandomN(Shape.of(32, 5)); // batch_size=32
        Variable inputVar = new Variable(input);
        
        Variable output = linearWithBias.layerForward(inputVar);
        
        // 批量大小应该保持不变，只改变特征维度
        assertEquals("批量处理的输出形状应该正确", Shape.of(32, 3), output.getValue().getShape());
    }

    @Test
    public void testSimpleLinearTransformation() {
        // 测试简单的线性变换
        // 创建一个简单的1x1线性层
        LinearLayer simpleLinear = new LinearLayer("simple", 1, 1, false);
        
        // 手动设置权重为3
        Parameter wParam = simpleLinear.getParamBy("w");
        wParam.getValue().set(3.0f, 0, 0);
        
        // 输入为2
        NdArray input = NdArray.of(new float[][]{{2.0f}});
        Variable inputVar = new Variable(input);
        
        Variable output = simpleLinear.layerForward(inputVar);
        
        // 输出应该是 2 * 3 = 6
        assertEquals("简单线性变换应该正确", 6.0f, output.getValue().get(0, 0), 0.001f);
    }

    @Test
    public void testSimpleLinearWithBiasTransformation() {
        // 测试简单的线性变换（包含偏置）
        LinearLayer simpleLinear = new LinearLayer("simple_bias", 1, 1, true);
        
        // 手动设置权重为3，偏置为2
        Parameter wParam = simpleLinear.getParamBy("w");
        Parameter bParam = simpleLinear.getParamBy("b");
        wParam.getValue().set(3.0f, 0, 0);
        bParam.getValue().set(2.0f, 0, 0);
        
        // 输入为2
        NdArray input = NdArray.of(new float[][]{{2.0f}});
        Variable inputVar = new Variable(input);
        
        Variable output = simpleLinear.layerForward(inputVar);
        
        // 输出应该是 2 * 3 + 2 = 8
        assertEquals("简单线性变换（含偏置）应该正确", 8.0f, output.getValue().get(0, 0), 0.001f);
    }

    @Test
    public void testXavierInitialization() {
        // 测试Xavier初始化
        Parameter wParam = linearWithBias.getParamBy("w");
        NdArray weights = wParam.getValue();
        
        // Xavier初始化应该产生合理范围内的权重
        // 权重应该不全为0，且在合理范围内
        boolean hasNonZero = false;
        float sum = 0;
        int count = 0;
        
        for (int i = 0; i < weights.getShape().getRow(); i++) {
            for (int j = 0; j < weights.getShape().getColumn(); j++) {
                float value = weights.get(i, j);
                if (Math.abs(value) > 0.001f) {
                    hasNonZero = true;
                }
                sum += value;
                count++;
                assertTrue("Xavier初始化的权重应该在合理范围内", Math.abs(value) < 1.0f);
            }
        }
        
        assertTrue("权重不应该全为0", hasNonZero);
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称
        assertEquals("权重参数名称应该正确", "w", linearWithBias.getParamBy("w").getName());
        assertEquals("偏置参数名称应该正确", "b", linearWithBias.getParamBy("b").getName());
    }

    @Test
    public void testMultipleForwardPasses() {
        // 测试多次前向传播的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(3, 5));
        Variable inputVar1 = new Variable(input);
        Variable inputVar2 = new Variable(input);
        
        Variable output1 = linearWithBias.layerForward(inputVar1);
        Variable output2 = linearWithBias.layerForward(inputVar2);
        
        // 相同输入应该产生相同输出
        NdArray out1 = output1.getValue();
        NdArray out2 = output2.getValue();
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals("相同输入应该产生相同输出", 
                           out1.get(i, j), out2.get(i, j), 0.001f);
            }
        }
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "linear_bias", linearWithBias.getName());
        assertEquals("层名称应该正确", "linear_no_bias", linearWithoutBias.getName());
    }

    @Test
    public void testInitialization() {
        try {
            linearWithBias.init();
            linearWithoutBias.init();
        } catch (Exception e) {
            fail("初始化不应该抛出异常");
        }
    }

    @Test
    public void testClearGrads() {
        // 测试梯度清零功能
        try {
            linearWithBias.clearGrads();
            linearWithoutBias.clearGrads();
        } catch (Exception e) {
            fail("清除梯度不应该抛出异常");
        }
    }

    @Test
    public void testMatrixMultiplication() {
        // 测试矩阵乘法的正确性
        // 创建2x3的输入，3x2的权重，期望2x2的输出
        LinearLayer testLinear = new LinearLayer("test", 3, 2, false);
        
        // 手动设置权重
        Parameter wParam = testLinear.getParamBy("w");
        float[][] weightData = {{1, 2}, {3, 4}, {5, 6}};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                wParam.getValue().set(weightData[i][j], i, j);
            }
        }
        
        // 输入矩阵
        float[][] inputData = {{1, 0, 1}, {2, 1, 0}};
        NdArray input = NdArray.of(inputData);
        Variable inputVar = new Variable(input);
        
        Variable output = testLinear.layerForward(inputVar);
        
        // 验证矩阵乘法结果
        // 第一行: [1,0,1] * [[1,2],[3,4],[5,6]] = [6,8]
        // 第二行: [2,1,0] * [[1,2],[3,4],[5,6]] = [5,8]
        assertEquals("矩阵乘法第一行第一列应该正确", 6.0f, output.getValue().get(0, 0), 0.001f);
        assertEquals("矩阵乘法第一行第二列应该正确", 8.0f, output.getValue().get(0, 1), 0.001f);
        assertEquals("矩阵乘法第二行第一列应该正确", 5.0f, output.getValue().get(1, 0), 0.001f);
        assertEquals("矩阵乘法第二行第二列应该正确", 8.0f, output.getValue().get(1, 1), 0.001f);
    }
}