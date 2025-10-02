package io.leavesfly.tinyai.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.lora.LoraConfig;
import io.leavesfly.tinyai.lora.LoraLinearLayer;
import io.leavesfly.tinyai.lora.LoraModel;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * LoraModel单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraModelTest {
    
    @Test
    public void testLoraModelCreation() {
        // 测试LoRA模型创建
        int[] layerSizes = {784, 256, 128, 10};
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        assertNotNull("模型应该成功创建", model);
        assertEquals("模型名称应该正确", "test_model", model.getName());
        assertEquals("应该有3个LoRA层", 3, model.getLoraLayers().size());
        assertArrayEquals("层大小应该正确", layerSizes, model.getLayerSizes());
    }
    
    @Test
    public void testFromPretrainedModel() {
        // 测试从预训练模型创建
        List<NdArray> pretrainedWeights = new ArrayList<>();
        pretrainedWeights.add(NdArray.likeRandomN(Shape.of(100, 50)));
        pretrainedWeights.add(NdArray.likeRandomN(Shape.of(50, 10)));
        
        List<NdArray> pretrainedBiases = new ArrayList<>();
        pretrainedBiases.add(NdArray.zeros(Shape.of(1, 50)));
        pretrainedBiases.add(NdArray.zeros(Shape.of(1, 10)));
        
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraModel model = LoraModel.fromPretrained(
            "pretrained_model", pretrainedWeights, pretrainedBiases, config, false);
        
        assertNotNull("预训练模型应该成功创建", model);
        assertEquals("应该有2个LoRA层", 2, model.getLoraLayers().size());
        assertEquals("第一层输入维度应该正确", 100, 
                    model.getLoraLayer(0).getInputShape().getDimension(1));
        assertEquals("最后一层输出维度应该正确", 10, 
                    model.getLoraLayer(1).getOutputShape().getDimension(1));
    }
    
    @Test
    public void testModelForwardPass() {
        // 测试模型前向传播
        int[] layerSizes = {20, 16, 8, 4};
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        NdArray input = NdArray.likeRandomN(Shape.of(5, 20));
        Variable output = model.layerForward(new Variable(input));
        
        assertNotNull("输出不应为null", output);
        assertEquals("输出批次大小应该正确", 5, output.getValue().getShape().getDimension(0));
        assertEquals("输出特征维度应该正确", 4, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testLoraEnableDisableAll() {
        // 测试全体LoRA启用/禁用
        int[] layerSizes = {10, 8, 6};
        LoraConfig config = new LoraConfig(2, 4.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        // 默认应该启用
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertTrue("默认应该启用LoRA", layer.isLoraEnabled());
        }
        
        // 禁用所有LoRA
        model.disableAllLora();
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertFalse("应该禁用所有LoRA", layer.isLoraEnabled());
        }
        
        // 启用所有LoRA
        model.enableAllLora();
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertTrue("应该启用所有LoRA", layer.isLoraEnabled());
        }
    }
    
    @Test
    public void testWeightFreezingAll() {
        // 测试全体权重冻结/解冻
        int[] layerSizes = {10, 8, 6};
        LoraConfig config = new LoraConfig(2, 4.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        // 默认应该冻结原始权重
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertTrue("默认应该冻结原始权重", layer.isOriginalWeightsFrozen());
        }
        
        // 解冻所有权重
        model.unfreezeAllOriginalWeights();
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertFalse("应该解冻所有原始权重", layer.isOriginalWeightsFrozen());
        }
        
        // 重新冻结所有权重
        model.freezeAllOriginalWeights();
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            assertTrue("应该重新冻结所有原始权重", layer.isOriginalWeightsFrozen());
        }
    }
    
    @Test
    public void testParameterCounting() {
        // 测试模型参数计数
        int[] layerSizes = {100, 50, 25, 10};
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        // 计算期望的参数数量
        int expectedLoraParams = 0;
        int expectedTotalParams = 0;
        int expectedBiasParams = 0;
        
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int inputDim = layerSizes[i];
            int outputDim = layerSizes[i + 1];
            
            expectedLoraParams += config.getRank() * (inputDim + outputDim);
            expectedTotalParams += inputDim * outputDim; // 冻结权重
            expectedBiasParams += outputDim; // 偏置
        }
        
        int expectedTrainableParams = expectedLoraParams + expectedBiasParams;
        expectedTotalParams += expectedTrainableParams;
        
        assertEquals("可训练参数数量应该正确", expectedTrainableParams, model.getTrainableParameterCount());
        assertEquals("总参数数量应该正确", expectedTotalParams, model.getTotalParameterCount());
        
        double expectedReduction = 1.0 - (double)expectedTrainableParams / expectedTotalParams;
        assertEquals("参数减少比例应该正确", expectedReduction, model.getParameterReduction(), 1e-6);
    }
    
    @Test
    public void testLoraParameterRetrieval() {
        // 测试LoRA参数获取
        int[] layerSizes = {20, 10, 5};
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        Map<String, Parameter> loraParams = model.getAllLoraParameters();
        assertEquals("应该有4个LoRA参数", 4, loraParams.size()); // 2层 * 2参数/层
        
        assertTrue("应该包含第一层lora_A", loraParams.containsKey("lora_layer_0.lora_A"));
        assertTrue("应该包含第一层lora_B", loraParams.containsKey("lora_layer_0.lora_B"));
        assertTrue("应该包含第二层lora_A", loraParams.containsKey("lora_layer_1.lora_A"));
        assertTrue("应该包含第二层lora_B", loraParams.containsKey("lora_layer_1.lora_B"));
    }
    
    @Test
    public void testWeightMerging() {
        // 测试权重合并
        int[] layerSizes = {15, 10, 5};
        LoraConfig config = new LoraConfig(3, 6.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        List<NdArray> mergedWeights = model.mergeAllLoraWeights();
        assertEquals("合并权重数量应该正确", 2, mergedWeights.size());
        
        assertEquals("第一层合并权重形状应该正确", Shape.of(15, 10), mergedWeights.get(0).getShape());
        assertEquals("第二层合并权重形状应该正确", Shape.of(10, 5), mergedWeights.get(1).getShape());
    }
    
    @Test
    public void testStateSaveAndLoad() {
        // 测试状态保存和加载
        int[] layerSizes = {20, 15, 10};
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, true);
        
        // 保存状态
        Map<String, NdArray> originalState = model.saveLoraState();
        assertTrue("状态应该包含LoRA参数", originalState.size() > 0);
        
        // 修改参数
        for (LoraLinearLayer layer : model.getLoraLayers()) {
            layer.getLoraAdapter().getMatrixA().setValue(
                NdArray.zeros(layer.getLoraAdapter().getMatrixA().getValue().getShape()));
        }
        
        // 恢复状态
        model.loadLoraState(originalState);
        
        // 验证状态恢复
        Map<String, NdArray> restoredState = model.saveLoraState();
        assertEquals("恢复状态的参数数量应该相同", originalState.size(), restoredState.size());
    }
    
    @Test
    public void testGetModelInfo() {
        // 测试模型信息获取
        int[] layerSizes = {784, 128, 10};
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraModel model = new LoraModel("mnist_model", layerSizes, config, false);
        
        String info = model.getModelInfo();
        assertNotNull("模型信息不应为null", info);
        assertTrue("信息应该包含模型名称", info.contains("mnist_model"));
        assertTrue("信息应该包含配置信息", info.contains("rank=8"));
        assertTrue("信息应该包含架构信息", info.contains("[784, 128, 10]"));
        assertTrue("信息应该包含参数统计", info.contains("总参数"));
        assertTrue("信息应该包含层详情", info.contains("层详情"));
    }
    
    @Test
    public void testConfigValidation() {
        // 测试模型配置验证
        int[] layerSizes = {50, 25, 10};
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        // 正常配置应该通过验证
        assertDoesNotThrow(() -> model.validateConfiguration());
        
        // 测试无效配置
        int[] invalidLayerSizes = {8, 4}; // rank >= min(input_dim, output_dim)
        assertThrows("无效配置应该抛出异常", IllegalArgumentException.class, 
                    () -> new LoraModel("invalid_model", invalidLayerSizes, config, false));
    }
    
    @Test
    public void testLayerAccess() {
        // 测试层访问方法
        int[] layerSizes = {30, 20, 10, 5};
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        assertEquals("应该有3个LoRA层", 3, model.getLoraLayers().size());
        
        // 测试按索引访问层
        LoraLinearLayer layer0 = model.getLoraLayer(0);
        assertNotNull("第0层不应为null", layer0);
        assertEquals("第0层输入维度应该正确", 30, layer0.getInputShape().getDimension(1));
        assertEquals("第0层输出维度应该正确", 20, layer0.getOutputShape().getDimension(1));
        
        // 测试索引越界
        assertThrows("索引越界应该抛出异常", IndexOutOfBoundsException.class, 
                    () -> model.getLoraLayer(5));
    }
    
    @Test
    public void testMinimalModel() {
        // 测试最小模型（只有输入层和输出层）
        int[] layerSizes = {10, 5};
        LoraConfig config = new LoraConfig(2, 4.0);
        LoraModel model = new LoraModel("minimal_model", layerSizes, config, false);
        
        assertNotNull("最小模型应该成功创建", model);
        assertEquals("应该有1个LoRA层", 1, model.getLoraLayers().size());
        
        NdArray input = NdArray.ones(Shape.of(3, 10));
        Variable output = model.layerForward(new Variable(input));
        assertEquals("输出维度应该正确", 5, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testInvalidLayerSizes() {
        // 测试无效的层大小配置
        LoraConfig config = new LoraConfig(4, 8.0);
        
        // 空数组
        assertThrows("空层数组应该抛出异常", IllegalArgumentException.class, 
                    () -> new LoraModel("test", new int[]{}, config, false));
        
        // 只有一层
        assertThrows("只有一层应该抛出异常", IllegalArgumentException.class, 
                    () -> new LoraModel("test", new int[]{10}, config, false));
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        int[] layerSizes = {100, 50, 25};
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraModel model = new LoraModel("test_model", layerSizes, config, false);
        
        String str = model.toString();
        assertTrue("toString应该包含模型名称", str.contains("test_model"));
        assertTrue("toString应该包含层数", str.contains("layers=2"));
        assertTrue("toString应该包含参数信息", str.contains("trainableParams="));
        assertTrue("toString应该包含减少比例", str.contains("reduction"));
    }
    
    // 辅助方法：断言不抛出异常
    private void assertDoesNotThrow(Runnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            fail("不应该抛出异常: " + e.getMessage());
        }
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