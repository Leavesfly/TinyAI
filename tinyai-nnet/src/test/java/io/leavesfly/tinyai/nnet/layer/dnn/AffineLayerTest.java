package io.leavesfly.tinyai.nnet.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * AffineLayer的单元测试
 * 
 * 测试仿射层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 形状变换
 * 4. 有无偏置的情况
 */
public class AffineLayerTest {

    private AffineLayer affineWithBias;
    private AffineLayer affineWithoutBias;
    
    @Before
    public void setUp() {
        // 创建带偏置的仿射层：输入(2,3) -> 输出(2,4)
        affineWithBias = new AffineLayer("affine_bias", Shape.of(2, 3), 4, true);
        
        // 创建不带偏置的仿射层：输入(3,5) -> 输出(3,2)
        affineWithoutBias = new AffineLayer("affine_no_bias", Shape.of(3, 5), 2, false);
    }

    @Test
    public void testParameterInitializationWithBias() {
        // 测试带偏置层的参数初始化
        Parameter wParam = affineWithBias.getParamBy("w");
        Parameter bParam = affineWithBias.getParamBy("b");
        
        assertNotNull("权重参数应该被初始化", wParam);
        assertNotNull("偏置参数应该被初始化", bParam);
        
        // 检查权重形状: (input_col, output_col) = (3, 4)
        assertEquals("权重形状应该正确", Shape.of(3, 4), wParam.getValue().getShape());
        
        // 检查偏置形状: (1, output_col) = (1, 4)
        assertEquals("偏置形状应该正确", Shape.of(1, 4), bParam.getValue().getShape());
        
        // 验证偏置初始化为0
        NdArray biasValue = bParam.getValue();
        for (int i = 0; i < 4; i++) {
            assertEquals("偏置应该初始化为0", 0.0f, biasValue.get(0, i), 0.001f);
        }
    }

    @Test
    public void testParameterInitializationWithoutBias() {
        // 测试不带偏置层的参数初始化
        Parameter wParam = affineWithoutBias.getParamBy("w");
        Parameter bParam = affineWithoutBias.getParamBy("b");
        
        assertNotNull("权重参数应该被初始化", wParam);
        assertNull("偏置参数不应该被初始化", bParam);
        
        // 检查权重形状: (input_col, output_col) = (5, 2)
        assertEquals("权重形状应该正确", Shape.of(5, 2), wParam.getValue().getShape());
    }

    @Test
    public void testForwardPassWithBias() {
        // 测试带偏置的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = affineWithBias.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该正确", Shape.of(2, 4), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testForwardPassWithoutBias() {
        // 测试不带偏置的前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(3, 5));
        Variable inputVar = new Variable(input);
        
        Variable output = affineWithoutBias.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals("输出形状应该正确", Shape.of(3, 2), output.getValue().getShape());
        assertNotNull("输出不应该为null", output.getValue());
    }

    @Test
    public void testShapeConsistency() {
        // 测试输入输出形状的一致性
        assertEquals("输入形状应该正确设置", Shape.of(2, 3), affineWithBias.getInputShape());
        assertEquals("输出形状应该正确设置", Shape.of(2, 4), affineWithBias.getOutputShape());
        
        assertEquals("输入形状应该正确设置", Shape.of(3, 5), affineWithoutBias.getInputShape());
        assertEquals("输出形状应该正确设置", Shape.of(3, 2), affineWithoutBias.getOutputShape());
    }

    @Test
    public void testBatchProcessing() {
        // 测试批量处理
        NdArray input = NdArray.likeRandomN(Shape.of(10, 3)); // batch_size=10
        Variable inputVar = new Variable(input);
        
        Variable output = affineWithBias.layerForward(inputVar);
        
        // 批量大小应该保持不变，只改变特征维度
        assertEquals("批量处理的输出形状应该正确", Shape.of(10, 4), output.getValue().getShape());
    }

    @Test
    public void testSimpleLinearTransformation() {
        // 测试简单的线性变换
        // 创建一个简单的1x1仿射层
        AffineLayer simpleAffine = new AffineLayer("simple", Shape.of(1, 1), 1, false);
        
        // 手动设置权重为2
        Parameter wParam = simpleAffine.getParamBy("w");
        wParam.getValue().set(2.0f, 0, 0);
        
        // 输入为3
        NdArray input = NdArray.of(new float[][]{{3.0f}});
        Variable inputVar = new Variable(input);
        
        Variable output = simpleAffine.layerForward(inputVar);
        
        // 输出应该是 3 * 2 = 6
        assertEquals("简单线性变换应该正确", 6.0f, output.getValue().get(0, 0), 0.001f);
    }

    @Test
    public void testSimpleAffineTransformation() {
        // 测试简单的仿射变换（包含偏置）
        AffineLayer simpleAffine = new AffineLayer("simple_bias", Shape.of(1, 1), 1, true);
        
        // 手动设置权重为2，偏置为1
        Parameter wParam = simpleAffine.getParamBy("w");
        Parameter bParam = simpleAffine.getParamBy("b");
        wParam.getValue().set(2.0f, 0, 0);
        bParam.getValue().set(1.0f, 0, 0);
        
        // 输入为3
        NdArray input = NdArray.of(new float[][]{{3.0f}});
        Variable inputVar = new Variable(input);
        
        Variable output = simpleAffine.layerForward(inputVar);
        
        // 输出应该是 3 * 2 + 1 = 7
        assertEquals("简单仿射变换应该正确", 7.0f, output.getValue().get(0, 0), 0.001f);
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称
        Parameter wParam = affineWithBias.getParamBy("w");
        Parameter bParam = affineWithBias.getParamBy("b");
        
        assertEquals("权重参数名称应该正确", "w", wParam.getName());
        assertEquals("偏置参数名称应该正确", "b", bParam.getName());
    }

    @Test
    public void testMultipleForwardPasses() {
        // 测试多次前向传播的一致性
        NdArray input = NdArray.likeRandomN(Shape.of(2, 3));
        Variable inputVar1 = new Variable(input);
        Variable inputVar2 = new Variable(input);
        
        Variable output1 = affineWithBias.layerForward(inputVar1);
        Variable output2 = affineWithBias.layerForward(inputVar2);
        
        // 相同输入应该产生相同输出
        NdArray out1 = output1.getValue();
        NdArray out2 = output2.getValue();
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals("相同输入应该产生相同输出", 
                           out1.get(i, j), out2.get(i, j), 0.001f);
            }
        }
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "affine_bias", affineWithBias.getName());
        assertEquals("层名称应该正确", "affine_no_bias", affineWithoutBias.getName());
    }

    @Test
    public void testClearGrads() {
        // 测试梯度清零功能
        try {
            affineWithBias.clearGrads();
            affineWithoutBias.clearGrads();
        } catch (Exception e) {
            fail("清除梯度不应该抛出异常");
        }
    }
}