package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * GruLayer门控循环单元层的单元测试
 * 
 * 测试GRU层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 状态重置
 * 4. 输出形状处理
 */
public class GruLayerTest {

    private GruLayer gruLayer;
    
    @Before
    public void setUp() {
        // 创建GRU层: 输入维度4，隐藏维度6
        gruLayer = new GruLayer("gru_test", Shape.of(-1, 4), Shape.of(-1, 6));
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数初始化
        // GRU层应该有9个参数
        assertEquals("GRU层应该有9个参数", 9, gruLayer.getParams().size());
        
        // 检查更新门参数
        assertNotNull("输入到更新门权重应该存在", gruLayer.getParamBy("w_z"));
        assertNotNull("隐藏状态到更新门权重应该存在", gruLayer.getParamBy("u_z"));
        assertNotNull("更新门偏置应该存在", gruLayer.getParamBy("b_z"));
        
        // 检查重置门参数
        assertNotNull("输入到重置门权重应该存在", gruLayer.getParamBy("w_r"));
        assertNotNull("隐藏状态到重置门权重应该存在", gruLayer.getParamBy("u_r"));
        assertNotNull("重置门偏置应该存在", gruLayer.getParamBy("b_r"));
        
        // 检查候选状态参数
        assertNotNull("输入到候选状态权重应该存在", gruLayer.getParamBy("w_h"));
        assertNotNull("隐藏状态到候选状态权重应该存在", gruLayer.getParamBy("u_h"));
        assertNotNull("候选状态偏置应该存在", gruLayer.getParamBy("b_h"));
    }

    @Test
    public void testParameterShapes() {
        // 测试参数形状
        Shape expectedInputShape = Shape.of(4, 6);  // 输入维度到隐藏维度
        Shape expectedHiddenShape = Shape.of(6, 6); // 隐藏维度到隐藏维度
        Shape expectedBiasShape = Shape.of(1, 6);   // 偏置形状
        
        assertEquals("输入到更新门权重形状应该正确", expectedInputShape, 
            gruLayer.getParamBy("w_z").getValue().getShape());
        assertEquals("隐藏状态到更新门权重形状应该正确", expectedHiddenShape, 
            gruLayer.getParamBy("u_z").getValue().getShape());
        assertEquals("更新门偏置形状应该正确", expectedBiasShape, 
            gruLayer.getParamBy("b_z").getValue().getShape());
        
        assertEquals("输入到重置门权重形状应该正确", expectedInputShape, 
            gruLayer.getParamBy("w_r").getValue().getShape());
        assertEquals("隐藏状态到重置门权重形状应该正确", expectedHiddenShape, 
            gruLayer.getParamBy("u_r").getValue().getShape());
        assertEquals("重置门偏置形状应该正确", expectedBiasShape, 
            gruLayer.getParamBy("b_r").getValue().getShape());
        
        assertEquals("输入到候选状态权重形状应该正确", expectedInputShape, 
            gruLayer.getParamBy("w_h").getValue().getShape());
        assertEquals("隐藏状态到候选状态权重形状应该正确", expectedHiddenShape, 
            gruLayer.getParamBy("u_h").getValue().getShape());
        assertEquals("候选状态偏置形状应该正确", expectedBiasShape, 
            gruLayer.getParamBy("b_h").getValue().getShape());
    }

    @Test
    public void testBasicForwardPass() {
        // 测试基本前向传播
        // 输入形状: (batch_size=2, input_size=4)
        NdArray input = NdArray.likeRandomN(Shape.of(2, 4));
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = gruLayer.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 验证输出形状
            Shape expectedShape = Shape.of(2, 6);  // batch_size=2, hidden_size=6
            assertEquals("输出形状应该正确", expectedShape, output.getValue().getShape());
            
        } catch (Exception e) {
            fail("基本前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testSequentialForwardPass() {
        // 测试序列前向传播（多次调用）
        // 输入形状: (batch_size=1, input_size=4)
        NdArray input1 = NdArray.likeRandomN(Shape.of(1, 4));
        NdArray input2 = NdArray.likeRandomN(Shape.of(1, 4));
        Variable inputVar1 = new Variable(input1);
        Variable inputVar2 = new Variable(input2);
        
        try {
            // 第一次前向传播
            Variable output1 = gruLayer.layerForward(inputVar1);
            assertNotNull("第一次输出不应该为null", output1);
            
            // 第二次前向传播（应该使用内部状态）
            Variable output2 = gruLayer.layerForward(inputVar2);
            assertNotNull("第二次输出不应该为null", output2);
            
            // 验证两次输出形状一致
            assertEquals("两次输出形状应该一致", output1.getValue().getShape(), output2.getValue().getShape());
            
        } catch (Exception e) {
            fail("序列前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testResetState() {
        // 测试状态重置
        NdArray input = NdArray.likeRandomN(Shape.of(1, 4));
        Variable inputVar = new Variable(input);
        
        try {
            // 第一次前向传播
            Variable output1 = gruLayer.layerForward(inputVar);
            
            // 重置状态
            gruLayer.resetState();
            
            // 再次前向传播（应该像第一次一样）
            Variable output2 = gruLayer.layerForward(inputVar);
            
            // 两次输出应该有相同的形状
            assertEquals("重置状态后输出形状应该一致", output1.getValue().getShape(), output2.getValue().getShape());
            
        } catch (Exception e) {
            fail("状态重置测试不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "gru_test", gruLayer.getName());
    }

    @Test
    public void testHiddenSize() {
        // 测试隐藏层大小（通过输出形状间接验证）
        NdArray input = NdArray.likeRandomN(Shape.of(1, 4));
        Variable inputVar = new Variable(input);
        
        Variable output = gruLayer.layerForward(inputVar);
        Shape outputShape = output.getValue().getShape();
        
        // 输出形状的最后一维应该是隐藏大小
        assertEquals("隐藏层大小应该正确", 6, outputShape.getDimension(1));
    }

    @Test
    public void testInitialization() {
        // 测试初始化方法
        try {
            gruLayer.init();
            // 不应该抛出异常
        } catch (Exception e) {
            fail("GRU层初始化不应该抛出异常");
        }
    }

    @Test
    public void testDifferentInputSizes() {
        // 测试不同输入尺寸
        GruLayer gruLayer2 = new GruLayer("gru_test2", Shape.of(-1, 3), Shape.of(-1, 5));
        
        NdArray input = NdArray.likeRandomN(Shape.of(3, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = gruLayer2.layerForward(inputVar);
        Shape expectedShape = Shape.of(3, 5);
        assertEquals("不同输入尺寸输出形状应该正确", expectedShape, output.getValue().getShape());
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称
        assertEquals("输入到更新门权重名称应该正确", "w_z", 
            gruLayer.getParamBy("w_z").getName());
        assertEquals("隐藏状态到更新门权重名称应该正确", "u_z", 
            gruLayer.getParamBy("u_z").getName());
        assertEquals("更新门偏置名称应该正确", "b_z", 
            gruLayer.getParamBy("b_z").getName());
    }

    @Test
    public void testZeroInput() {
        // 测试零输入的处理
        float[][] zeroData = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        NdArray zeroInput = NdArray.of(zeroData);
        Variable zeroInputVar = new Variable(zeroInput);
        
        Variable output = gruLayer.layerForward(zeroInputVar);
        
        assertNotNull("零输入应该产生有效输出", output.getValue());
        assertEquals("零输入输出形状应该正确", Shape.of(2, 6), output.getValue().getShape());
    }
}