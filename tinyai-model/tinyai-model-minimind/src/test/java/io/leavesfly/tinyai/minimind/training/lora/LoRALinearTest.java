package io.leavesfly.tinyai.minimind.training.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * LoRALinear层的单元测试
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoRALinearTest {
    
    @Test
    public void testLoRALinearCreation() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 512, 256, true, 8, 16.0f);
        
        assertNotNull(loraLinear);
        assertEquals(512, loraLinear.inFeatures);
        assertEquals(256, loraLinear.outFeatures);
        assertEquals(8, loraLinear.getRank());
        assertEquals(16.0f, loraLinear.getAlpha());
        assertEquals(2.0f, loraLinear.getScaling(), 0.001f);
    }
    
    @Test
    public void testLoRAParameters() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 512, 256, true, 8, 16.0f);
        
        // 检查LoRA参数
        Parameter loraA = loraLinear.getLoraA();
        Parameter loraB = loraLinear.getLoraB();
        
        assertNotNull(loraA);
        assertNotNull(loraB);
        
        // 检查形状
        assertArrayEquals(new int[]{8, 512}, loraA.data().getShape().getShapeDims());
        assertArrayEquals(new int[]{256, 8}, loraB.data().getShape().getShapeDims());
    }
    
    @Test
    public void testLoRAForward() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 128, 64, true, 4, 8.0f);
        
        // 创建输入 (batch=16, features=128)
        NdArray inputData = NdArray.randn(Shape.of(16, 128));
        Variable input = new Variable(inputData);
        
        // 前向传播
        Variable output = loraLinear.forward(input);
        
        // 验证输出形状 (batch=16, features=64)
        assertArrayEquals(new int[]{16, 64}, output.getShape().getShapeDims());
    }
    
    @Test
    public void testLoRAInitialization() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 256, 128, false, 8, 16.0f);
        
        // LoRA B应该初始化为0
        Parameter loraB = loraLinear.getLoraB();
        float[] loraBData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) loraB.data()).buffer;
        
        boolean allZero = true;
        for (float val : loraBData) {
            if (Math.abs(val) > 1e-6) {
                allZero = false;
                break;
            }
        }
        
        assertTrue(allZero, "LoRA B应该初始化为0");
    }
    
    @Test
    public void testLoRAFromLinear() {
        // 创建原始权重和偏置
        NdArray weight = NdArray.randn(Shape.of(64, 128));
        NdArray bias = NdArray.randn(Shape.of(64));
        Parameter weightParam = new Parameter(weight);
        Parameter biasParam = new Parameter(bias);
        
        // 从Linear创建LoRALinear
        LoRALinear loraLinear = LoRALinear.fromLinear(
            "lora_fc", weightParam, biasParam, 4, 8.0f, 0.1f
        );
        
        assertNotNull(loraLinear);
        assertEquals(128, loraLinear.inFeatures);
        assertEquals(64, loraLinear.outFeatures);
        assertEquals(4, loraLinear.getRank());
    }
    
    @Test
    public void testLoRAMergeWeights() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 64, 32, false, 4, 8.0f);
        
        // 合并权重
        NdArray mergedWeight = loraLinear.mergeWeights();
        
        assertNotNull(mergedWeight);
        assertArrayEquals(new int[]{32, 64}, mergedWeight.getShape().getShapeDims());
    }
    
    @Test
    public void testLoRAParameterCounts() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 512, 256, true, 8, 16.0f);
        
        int originalParams = loraLinear.getOriginalParams();
        int loraParams = loraLinear.getLoRAParams();
        float compressionRatio = loraLinear.getCompressionRatio();
        
        // 原始参数: 512 * 256 + 256 = 131,328
        assertEquals(131328, originalParams);
        
        // LoRA参数: 8 * 512 + 256 * 8 = 6,144
        assertEquals(6144, loraParams);
        
        // 压缩比应该约为 4.68%
        assertTrue(compressionRatio < 5.0f);
        assertTrue(compressionRatio > 4.0f);
    }
    
    @Test
    public void testLoRATrainMode() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 128, 64, true, 4, 8.0f);
        
        // 训练模式
        loraLinear.train();
        assertTrue(loraLinear.isTraining());
        
        // 评估模式
        loraLinear.eval();
        assertFalse(loraLinear.isTraining());
    }
    
    @Test
    public void testLoRAWithoutBias() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 256, 128, false, 8, 16.0f);
        
        assertNull(loraLinear.getOriginalBias());
        
        // 前向传播应该正常工作
        NdArray inputData = NdArray.randn(Shape.of(8, 256));
        Variable input = new Variable(inputData);
        Variable output = loraLinear.forward(input);
        
        assertNotNull(output);
        assertArrayEquals(new int[]{8, 128}, output.getShape().getShapeDims());
    }
    
    @Test
    public void testLoRAScaling() {
        // 测试不同的alpha和rank组合
        LoRALinear lora1 = new LoRALinear("lora1", 64, 32, false, 4, 8.0f);
        assertEquals(2.0f, lora1.getScaling(), 0.001f);
        
        LoRALinear lora2 = new LoRALinear("lora2", 64, 32, false, 8, 16.0f);
        assertEquals(2.0f, lora2.getScaling(), 0.001f);
        
        LoRALinear lora3 = new LoRALinear("lora3", 64, 32, false, 4, 4.0f);
        assertEquals(1.0f, lora3.getScaling(), 0.001f);
    }
    
    @Test
    public void testLoRABackward() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 32, 16, true, 4, 8.0f);
        
        // 创建输入和目标
        NdArray inputData = NdArray.randn(Shape.of(4, 32));
        Variable input = new Variable(inputData);
        
        // 前向传播
        Variable output = loraLinear.forward(input);
        
        // 计算简单损失(所有输出求和)
        Variable loss = output.sum();
        
        // 反向传播
        loss.backward();
        
        // 检查梯度是否存在
        assertNotNull(loraLinear.getLoraA().getGrad());
        assertNotNull(loraLinear.getLoraB().getGrad());
    }
    
    @Test
    public void testLoRAPrintInfo() {
        LoRALinear loraLinear = new LoRALinear("test_lora", 512, 256, true, 8, 16.0f);
        
        // 应该不会抛出异常
        assertDoesNotThrow(() -> loraLinear.printLoRAInfo());
    }
    
    @Test
    public void testLoRAToString() {
        LoRALinear loraLinear = new LoRALinear("test_lora", 128, 64, true, 4, 8.0f);
        
        String str = loraLinear.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("LoRALinear"));
        assertTrue(str.contains("test_lora"));
        assertTrue(str.contains("128"));
        assertTrue(str.contains("64"));
        assertTrue(str.contains("rank=4"));
    }

    @Test
    public void testLoRALinearGradientCheck() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 128, 64, true, 4, 8.0f);
        
        // 创建输入 (batch=16, features=128)
        NdArray inputData = NdArray.randn(Shape.of(16, 128));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(loraLinear, input);
    }

    @Test
    public void testLoRALinearWithoutBiasGradientCheck() {
        LoRALinear loraLinear = new LoRALinear("lora_fc", 128, 64, false, 4, 8.0f);
        
        // 创建输入 (batch=16, features=128)
        NdArray inputData = NdArray.randn(Shape.of(16, 128));
        Variable input = new Variable(inputData);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(loraLinear, input);
    }
}
