package io.leavesfly.tinyai.nlp.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * LoraLinearLayer单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraLinearLayerTest {
    
    @Test
    public void testLoraLinearLayerCreation() {
        // 测试LoRA线性层创建
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 100, 50, config, true);
        
        assertNotNull("层应该成功创建", layer);
        assertEquals("层名称应该正确", "test_layer", layer.getName());
        assertEquals("输入形状应该正确", Shape.of(-1, 100), layer.getInputShape());
        assertEquals("输出形状应该正确", Shape.of(-1, 50), layer.getOutputShape());
        assertNotNull("应该有冻结权重", layer.getFrozenWeight());
        assertNotNull("应该有偏置", layer.getBias());
        assertNotNull("应该有LoRA适配器", layer.getLoraAdapter());
    }
    
    @Test
    public void testFromPretrainedWeights() {
        // 测试从预训练权重创建LoRA层
        NdArray pretrainedWeight = NdArray.ones(Shape.of(64, 32));
        NdArray pretrainedBias = NdArray.zeros(Shape.of(1, 32));
        LoraConfig config = new LoraConfig(4, 8.0);
        
        LoraLinearLayer layer = new LoraLinearLayer(
            "pretrained_layer", pretrainedWeight, pretrainedBias, config);
        
        assertNotNull("层应该成功创建", layer);
        assertEquals("冻结权重应该与预训练权重相同", 
                    pretrainedWeight.getShape(), layer.getFrozenWeight().getValue().getShape());
        assertEquals("偏置应该与预训练偏置相同", 
                    pretrainedBias.getShape(), layer.getBias().getValue().getShape());
    }
    
    @Test
    public void testForwardPass() {
        // 测试前向传播
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 20, 10, config, true);
        
        NdArray input = NdArray.ones(Shape.of(5, 20)); // batch_size=5
        Variable output = layer.layerForward(new Variable(input));
        
        assertNotNull("输出不应为null", output);
        assertEquals("输出批次大小应该正确", 5, output.getValue().getShape().getDimension(0));
        assertEquals("输出特征维度应该正确", 10, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testLoraEnableDisable() {
        // 测试LoRA启用/禁用功能
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 10, 5, config, false);
        
        NdArray input = NdArray.ones(Shape.of(2, 10));
        Variable inputVar = new Variable(input);
        
        // 启用LoRA
        layer.enableLora();
        assertTrue("LoRA应该启用", layer.isLoraEnabled());
        Variable outputWithLora = layer.layerForward(inputVar);
        
        // 禁用LoRA
        layer.disableLora();
        assertFalse("LoRA应该禁用", layer.isLoraEnabled());
        Variable outputWithoutLora = layer.layerForward(inputVar);
        
        // 两次输出应该不同（假设LoRA参数非零）
        assertNotNull("两次输出都不应为null", outputWithLora);
        assertNotNull("两次输出都不应为null", outputWithoutLora);
    }
    
    @Test
    public void testWeightFreezing() {
        // 测试权重冻结功能
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 10, 5, config, true);
        
        // 默认应该冻结原始权重
        assertTrue("原始权重应该默认冻结", layer.isOriginalWeightsFrozen());
        assertFalse("冻结权重不应该需要梯度", layer.getFrozenWeight().isRequireGrad());
        
        // 解冻权重
        layer.unfreezeOriginalWeights();
        assertFalse("原始权重应该解冻", layer.isOriginalWeightsFrozen());
        assertTrue("解冻权重应该需要梯度", layer.getFrozenWeight().isRequireGrad());
        
        // 重新冻结权重
        layer.freezeOriginalWeights();
        assertTrue("原始权重应该重新冻结", layer.isOriginalWeightsFrozen());
        assertFalse("重新冻结权重不应该需要梯度", layer.getFrozenWeight().isRequireGrad());
    }
    
    @Test
    public void testParameterCounting() {
        // 测试参数计数
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 100, 50, config, true);
        
        // 计算期望的参数数量
        int loraParams = 8 * (100 + 50); // rank * (input_dim + output_dim)
        int biasParams = 50; // output_dim
        int expectedTrainableParams = loraParams + biasParams;
        
        int frozenParams = 100 * 50; // input_dim * output_dim
        int expectedTotalParams = expectedTrainableParams + frozenParams;
        
        assertEquals("可训练参数数量应该正确", expectedTrainableParams, layer.getTrainableParameterCount());
        assertEquals("总参数数量应该正确", expectedTotalParams, layer.getTotalParameterCount());
        
        double expectedReduction = 1.0 - (double)expectedTrainableParams / expectedTotalParams;
        assertEquals("参数减少比例应该正确", expectedReduction, layer.getParameterReduction(), 1e-6);
    }
    
    @Test
    public void testParameterCountingWithoutBias() {
        // 测试无偏置情况下的参数计数
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 50, 25, config, false);
        
        int expectedTrainableParams = 4 * (50 + 25); // 只有LoRA参数
        assertEquals("无偏置时可训练参数数量应该正确", expectedTrainableParams, layer.getTrainableParameterCount());
    }
    
    @Test
    public void testLoraParameterRetrieval() {
        // 测试LoRA参数获取
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 20, 10, config, true);
        
        Map<String, io.leavesfly.tinyai.nnet.Parameter> loraParams = layer.getLoraParameters();
        assertEquals("应该有2个LoRA参数", 2, loraParams.size());
        assertTrue("应该包含lora_A参数", loraParams.containsKey("test_layer.lora_A"));
        assertTrue("应该包含lora_B参数", loraParams.containsKey("test_layer.lora_B"));
    }
    
    @Test
    public void testWeightMerging() {
        // 测试权重合并
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 10, 5, config, false);
        
        // 获取原始权重
        NdArray originalWeight = layer.getFrozenWeight().getValue();
        
        // 合并权重
        NdArray mergedWeight = layer.mergeLoraWeights();
        
        assertNotNull("合并权重不应为null", mergedWeight);
        assertEquals("合并权重形状应该与原始权重相同", 
                    originalWeight.getShape(), mergedWeight.getShape());
        
        // 如果LoRA被禁用，合并权重应该等于原始权重
        layer.disableLora();
        NdArray mergedWhenDisabled = layer.mergeLoraWeights();
        assertTrue("禁用LoRA时合并权重应该等于原始权重",
                  mergedWhenDisabled.eq(originalWeight).sum().getNumber().floatValue() 
                  == originalWeight.getShape().size());
    }
    
    @Test
    public void testClearGrads() {
        // 测试梯度清除
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 10, 5, config, true);
        
        // 模拟设置梯度
        layer.getLoraAdapter().getMatrixA().setGrad(
            NdArray.ones(layer.getLoraAdapter().getMatrixA().getValue().getShape()));
        layer.getLoraAdapter().getMatrixB().setGrad(
            NdArray.ones(layer.getLoraAdapter().getMatrixB().getValue().getShape()));
        layer.getBias().setGrad(
            NdArray.ones(layer.getBias().getValue().getShape()));
        
        // 验证梯度存在
        assertNotNull("LoRA A梯度应该存在", layer.getLoraAdapter().getMatrixA().getGrad());
        assertNotNull("LoRA B梯度应该存在", layer.getLoraAdapter().getMatrixB().getGrad());
        assertNotNull("偏置梯度应该存在", layer.getBias().getGrad());
        
        // 清除梯度
        layer.clearGrads();
        
        // 验证梯度被清除
        assertNull("LoRA A梯度应该被清除", layer.getLoraAdapter().getMatrixA().getGrad());
        assertNull("LoRA B梯度应该被清除", layer.getLoraAdapter().getMatrixB().getGrad());
        assertNull("偏置梯度应该被清除", layer.getBias().getGrad());
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraLinearLayer layer = new LoraLinearLayer("test_layer", 100, 50, config, true);
        
        String str = layer.toString();
        assertTrue("toString应该包含层名称", str.contains("test_layer"));
        assertTrue("toString应该包含输入维度", str.contains("inputDim=100"));
        assertTrue("toString应该包含输出维度", str.contains("outputDim=50"));
        assertTrue("toString应该包含参数数量信息", str.contains("trainableParams="));
    }
    
    @Test
    public void testConfigValidationInLayer() {
        // 测试层中的配置验证
        LoraConfig invalidConfig = new LoraConfig(100, 16.0); // rank太大
        
        assertThrows("无效配置应该抛出异常", IllegalArgumentException.class, 
                    () -> new LoraLinearLayer("test_layer", 50, 25, invalidConfig, true));
    }
    
    // 辅助方法：断言抛出指定异常
    private void assertThrows(String message, Class<? extends Exception> expectedType, Runnable runnable) {
        try {
            runnable.run();
            fail(message + " - 应该抛出异常");
        } catch (Exception e) {
            assertTrue(message + " - 异常类型错误", expectedType.isInstance(e));
        }
    }
}