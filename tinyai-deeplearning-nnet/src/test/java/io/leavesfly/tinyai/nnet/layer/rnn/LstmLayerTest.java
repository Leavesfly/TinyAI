package io.leavesfly.tinyai.nnet.layer.rnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * LstmLayer长短期记忆网络层的单元测试
 * 
 * 测试LSTM层的基本功能：
 * 1. 参数初始化
 * 2. 前向传播计算
 * 3. 状态重置
 * 4. 输出形状处理
 */
public class LstmLayerTest {

    private LstmLayer lstmLayer;
    
    @Before
    public void setUp() {
        // 创建LSTM层: 输入维度10，隐藏维度20
        lstmLayer = new LstmLayer("lstm_test", Shape.of(-1, 10), Shape.of(-1, 20));
    }

    @Test
    public void testParameterInitialization() {
        // 测试参数初始化
        // LSTM层应该有12个参数矩阵和偏置项
        assertEquals("LSTM层应该有12个参数", 12, lstmLayer.getParams().size());
        
        // 检查关键参数是否存在
        assertNotNull("输入到遗忘门权重应该存在", lstmLayer.getParamBy("lstm_test.x2f"));
        assertNotNull("输入到遗忘门偏置应该存在", lstmLayer.getParamBy("lstm_test.x2f-b"));
        assertNotNull("输入到输入门权重应该存在", lstmLayer.getParamBy("lstm_test.x2i"));
        assertNotNull("输入到输入门偏置应该存在", lstmLayer.getParamBy("lstm_test.x2i-b"));
        assertNotNull("输入到输出门权重应该存在", lstmLayer.getParamBy("lstm_test.x2o"));
        assertNotNull("输入到输出门偏置应该存在", lstmLayer.getParamBy("lstm_test.x2o-b"));
        assertNotNull("输入到候选细胞状态权重应该存在", lstmLayer.getParamBy("lstm_test.x2u"));
        assertNotNull("输入到候选细胞状态偏置应该存在", lstmLayer.getParamBy("lstm_test.x2u-b"));
        
        // 检查隐藏状态到门的参数
        assertNotNull("隐藏状态到遗忘门权重应该存在", lstmLayer.getParamBy("lstm_test.h2f"));
        assertNotNull("隐藏状态到输入门权重应该存在", lstmLayer.getParamBy("lstm_test.h2i"));
        assertNotNull("隐藏状态到输出门权重应该存在", lstmLayer.getParamBy("lstm_test.h2o"));
        assertNotNull("隐藏状态到候选细胞状态权重应该存在", lstmLayer.getParamBy("lstm_test.h2u"));
    }

    @Test
    public void testParameterShapes() {
        // 测试参数形状
        Shape expectedX2fShape = Shape.of(10, 20);  // 输入维度到隐藏维度
        Shape expectedH2fShape = Shape.of(20, 20);  // 隐藏维度到隐藏维度
        Shape expectedBiasShape = Shape.of(1, 20);  // 偏置形状
        
        assertEquals("输入到遗忘门权重形状应该正确", expectedX2fShape, 
            lstmLayer.getParamBy("lstm_test.x2f").getValue().getShape());
        assertEquals("隐藏状态到遗忘门权重形状应该正确", expectedH2fShape, 
            lstmLayer.getParamBy("lstm_test.h2f").getValue().getShape());
        assertEquals("遗忘门偏置形状应该正确", expectedBiasShape, 
            lstmLayer.getParamBy("lstm_test.x2f-b").getValue().getShape());
    }

    @Test
    public void testParameterNames() {
        // 测试参数名称
        assertEquals("输入到遗忘门权重名称应该正确", "lstm_test.x2f", 
            lstmLayer.getParamBy("lstm_test.x2f").getName());
        assertEquals("输入到遗忘门偏置名称应该正确", "lstm_test.x2f-b", 
            lstmLayer.getParamBy("lstm_test.x2f-b").getName());
    }

    @Test
    public void testBasicForwardPass() {
        // 测试基本前向传播
        // 输入形状: (batch_size=2, input_size=10)
        NdArray input = NdArray.likeRandomN(Shape.of(2, 10));
        Variable inputVar = new Variable(input);
        
        try {
            Variable output = lstmLayer.layerForward(inputVar);
            
            // 验证输出不为null
            assertNotNull("输出不应该为null", output);
            assertNotNull("输出值不应该为null", output.getValue());
            
            // 验证输出形状
            Shape expectedShape = Shape.of(2, 20);  // batch_size=2, hidden_size=20
            assertEquals("输出形状应该正确", expectedShape, output.getValue().getShape());
            
        } catch (Exception e) {
            fail("基本前向传播不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testSequentialForwardPass() {
        // 测试序列前向传播（多次调用）
        // 输入形状: (batch_size=1, input_size=10)
        NdArray input1 = NdArray.likeRandomN(Shape.of(1, 10));
        NdArray input2 = NdArray.likeRandomN(Shape.of(1, 10));
        Variable inputVar1 = new Variable(input1);
        Variable inputVar2 = new Variable(input2);
        
        try {
            // 第一次前向传播
            Variable output1 = lstmLayer.layerForward(inputVar1);
            assertNotNull("第一次输出不应该为null", output1);
            
            // 第二次前向传播（应该使用内部状态）
            Variable output2 = lstmLayer.layerForward(inputVar2);
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
        NdArray input = NdArray.likeRandomN(Shape.of(1, 10));
        Variable inputVar = new Variable(input);
        
        try {
            // 第一次前向传播
            Variable output1 = lstmLayer.layerForward(inputVar);
            
            // 重置状态
            lstmLayer.resetState();
            
            // 再次前向传播（应该像第一次一样）
            Variable output2 = lstmLayer.layerForward(inputVar);
            
            // 两次输出应该有相同的形状
            assertEquals("重置状态后输出形状应该一致", output1.getValue().getShape(), output2.getValue().getShape());
            
        } catch (Exception e) {
            fail("状态重置测试不应该抛出异常: " + e.getMessage());
        }
    }

    @Test
    public void testLayerName() {
        // 测试层名称
        assertEquals("层名称应该正确", "lstm_test", lstmLayer.getName());
    }

    @Test
    public void testHiddenSize() {
        // 测试隐藏层大小（通过输出形状间接验证）
        NdArray input = NdArray.likeRandomN(Shape.of(1, 10));
        Variable inputVar = new Variable(input);
        
        Variable output = lstmLayer.layerForward(inputVar);
        Shape outputShape = output.getValue().getShape();
        
        // 输出形状的最后一维应该是隐藏大小
        assertEquals("隐藏层大小应该正确", 20, outputShape.getDimension(1));
    }

    @Test
    public void testInitialization() {
        // 测试初始化方法
        try {
            lstmLayer.init();
            // 不应该抛出异常
        } catch (Exception e) {
            fail("LSTM层初始化不应该抛出异常");
        }
    }

    @Test
    public void testDifferentInputSizes() {
        // 测试不同输入尺寸
        LstmLayer lstmLayer2 = new LstmLayer("lstm_test2", Shape.of(-1, 5), Shape.of(-1, 15));
        
        NdArray input = NdArray.likeRandomN(Shape.of(3, 5));
        Variable inputVar = new Variable(input);
        
        Variable output = lstmLayer2.layerForward(inputVar);
        Shape expectedShape = Shape.of(3, 15);
        assertEquals("不同输入尺寸输出形状应该正确", expectedShape, output.getValue().getShape());
    }
}