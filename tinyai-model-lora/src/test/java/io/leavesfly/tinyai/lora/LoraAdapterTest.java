package io.leavesfly.tinyai.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.lora.LoraAdapter;
import io.leavesfly.tinyai.lora.LoraConfig;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * LoraAdapter单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraAdapterTest {
    
    @Test
    public void testLoraAdapterCreation() {
        // 测试LoRA适配器创建
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraAdapter adapter = new LoraAdapter(256, 128, config);
        
        assertNotNull("LoRA适配器应该成功创建", adapter);
        assertEquals("配置应该正确", config, adapter.getConfig());
        assertEquals("缩放因子应该正确", 2.0, adapter.getScaling(), 1e-6);
        assertTrue("适配器应该默认启用", adapter.isEnabled());
    }
    
    @Test
    public void testLoraAdapterForward() {
        // 测试LoRA适配器前向传播
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraAdapter adapter = new LoraAdapter(10, 5, config);
        
        NdArray input = NdArray.ones(Shape.of(3, 10)); // batch_size=3
        Variable inputVar = new Variable(input);
        Variable output = adapter.forward(inputVar);
        
        assertNotNull("输出不应为null", output);
        assertEquals("输出批次大小应该正确", 3, output.getValue().getShape().getDimension(0));
        assertEquals("输出特征维度应该正确", 5, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testLoraAdapterDisable() {
        // 测试LoRA适配器禁用功能
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraAdapter adapter = new LoraAdapter(10, 5, config);
        
        NdArray input = NdArray.ones(Shape.of(2, 10));
        Variable inputVar = new Variable(input);
        
        // 启用状态下的输出
        adapter.enable();
        Variable enabledOutput = adapter.forward(inputVar);
        
        // 禁用状态下的输出
        adapter.disable();
        Variable disabledOutput = adapter.forward(inputVar);
        
        // 禁用时输出应该为0
        NdArray expectedZeros = NdArray.zeros(Shape.of(2, 5));
        assertTrue("禁用LoRA时输出应该为零",
                  disabledOutput.getValue().eq(expectedZeros).sum().getNumber().floatValue() 
                  == expectedZeros.getShape().size());
    }
    
    @Test
    public void testParameterCount() {
        // 测试参数数量计算
        LoraConfig config = new LoraConfig(8, 16.0);
        LoraAdapter adapter = new LoraAdapter(100, 50, config);
        
        int expectedParams = 8 * 100 + 8 * 50; // rank * (input_dim + output_dim)
        assertEquals("参数数量应该正确", expectedParams, adapter.getParameterCount());
        
        int originalParams = 100 * 50;
        double expectedReduction = 1.0 - (double)expectedParams / originalParams;
        assertEquals("参数减少比例应该正确", expectedReduction, 
                    adapter.getParameterReduction(originalParams), 1e-6);
    }
    
    @Test
    public void testLoraInitialization() {
        // 测试LoRA矩阵初始化
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraAdapter adapter = new LoraAdapter(20, 10, config);
        
        // 矩阵A应该用高斯分布初始化
        assertNotNull("矩阵A应该被初始化", adapter.getMatrixA());
        assertEquals("矩阵A形状应该正确", Shape.of(20, 4), adapter.getMatrixA().getValue().getShape());
        
        // 矩阵B应该用零初始化
        assertNotNull("矩阵B应该被初始化", adapter.getMatrixB());
        assertEquals("矩阵B形状应该正确", Shape.of(4, 10), adapter.getMatrixB().getValue().getShape());
        
        // 初始时LoRA输出应该接近零（因为B初始化为0）
        NdArray input = NdArray.ones(Shape.of(1, 20));
        Variable output = adapter.forward(new Variable(input));
        float maxOutput = output.getValue().abs().max();
        assertTrue("初始LoRA输出应该接近零", maxOutput < 1e-6);
    }
    
    @Test
    public void testConfigValidation() {
        // 测试配置验证
        LoraConfig config = new LoraConfig(4, 8.0);
        
        // 正常情况应该通过
        assertDoesNotThrow(() -> config.validate(10, 8));
        
        // rank >= min(input_dim, output_dim) 应该失败
        assertThrows("rank过大应该抛出异常", IllegalArgumentException.class, 
                    () -> config.validate(3, 5));
    }
    
    @Test
    public void testClearGrads() {
        // 测试梯度清除
        LoraConfig config = new LoraConfig(4, 8.0);
        LoraAdapter adapter = new LoraAdapter(10, 5, config);
        
        // 模拟设置梯度
        adapter.getMatrixA().setGrad(NdArray.ones(adapter.getMatrixA().getValue().getShape()));
        adapter.getMatrixB().setGrad(NdArray.ones(adapter.getMatrixB().getValue().getShape()));
        
        assertNotNull("梯度A应该存在", adapter.getMatrixA().getGrad());
        assertNotNull("梯度B应该存在", adapter.getMatrixB().getGrad());
        
        // 清除梯度
        adapter.clearGrads();
        
        assertNull("梯度A应该被清除", adapter.getMatrixA().getGrad());
        assertNull("梯度B应该被清除", adapter.getMatrixB().getGrad());
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