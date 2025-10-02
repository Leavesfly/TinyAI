package io.leavesfly.tinyai.lora;

import io.leavesfly.tinyai.lora.LoraConfig;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * LoraConfig单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraConfigTest {
    
    @Test
    public void testBasicConfigCreation() {
        // 测试基本配置创建
        LoraConfig config = new LoraConfig(8, 16.0);
        
        assertEquals("rank应该正确", 8, config.getRank());
        assertEquals("alpha应该正确", 16.0, config.getAlpha(), 1e-6);
        assertEquals("scaling应该正确", 2.0, config.getScaling(), 1e-6);
        assertEquals("dropout应该为默认值", 0.0, config.getDropout(), 1e-6);
        assertFalse("enableBias应该为默认值", config.isEnableBias());
    }
    
    @Test
    public void testFullConfigCreation() {
        // 测试完整配置创建
        String[] targetModules = {"linear", "attention"};
        LoraConfig config = new LoraConfig(16, 32.0, 0.1, true, targetModules);
        
        assertEquals("rank应该正确", 16, config.getRank());
        assertEquals("alpha应该正确", 32.0, config.getAlpha(), 1e-6);
        assertEquals("scaling应该正确", 2.0, config.getScaling(), 1e-6);
        assertEquals("dropout应该正确", 0.1, config.getDropout(), 1e-6);
        assertTrue("enableBias应该正确", config.isEnableBias());
        assertArrayEquals("targetModules应该正确", targetModules, config.getTargetModules());
    }
    
    @Test
    public void testPresetConfigs() {
        // 测试预设配置
        LoraConfig defaultConfig = LoraConfig.createDefault(16);
        assertEquals("默认配置rank应该正确", 16, defaultConfig.getRank());
        assertEquals("默认配置alpha应该等于rank", 16.0, defaultConfig.getAlpha(), 1e-6);
        
        LoraConfig lowRankConfig = LoraConfig.createLowRank();
        assertEquals("低秩配置rank应该为4", 4, lowRankConfig.getRank());
        
        LoraConfig mediumRankConfig = LoraConfig.createMediumRank();
        assertEquals("中等秩配置rank应该为16", 16, mediumRankConfig.getRank());
        
        LoraConfig highRankConfig = LoraConfig.createHighRank();
        assertEquals("高秩配置rank应该为64", 64, highRankConfig.getRank());
    }
    
    @Test
    public void testConfigValidation() {
        // 测试参数验证
        
        // 正常情况
        assertDoesNotThrow(() -> new LoraConfig(8, 16.0));
        
        // 无效rank
        assertThrows("rank必须为正数", IllegalArgumentException.class, 
                    () -> new LoraConfig(0, 16.0));
        assertThrows("rank必须为正数", IllegalArgumentException.class, 
                    () -> new LoraConfig(-1, 16.0));
        
        // 无效alpha
        assertThrows("alpha必须非负", IllegalArgumentException.class, 
                    () -> new LoraConfig(8, -1.0));
        
        // 无效dropout
        assertThrows("dropout必须在[0,1)范围内", IllegalArgumentException.class, 
                    () -> new LoraConfig(8, 16.0, -0.1, false, null));
        assertThrows("dropout必须在[0,1)范围内", IllegalArgumentException.class, 
                    () -> new LoraConfig(8, 16.0, 1.0, false, null));
    }
    
    @Test
    public void testParameterReduction() {
        // 测试参数减少比例计算
        LoraConfig config = new LoraConfig(8, 16.0);
        
        int inputDim = 100;
        int outputDim = 50;
        int originalParams = inputDim * outputDim;
        int loraParams = config.getRank() * (inputDim + outputDim);
        double expectedReduction = 1.0 - (double)loraParams / originalParams;
        
        assertEquals("参数减少比例应该正确", expectedReduction, 
                    config.getParameterReduction(inputDim, outputDim), 1e-6);
    }
    
    @Test
    public void testDimensionValidation() {
        // 测试维度验证
        LoraConfig config = new LoraConfig(8, 16.0);
        
        // 正常情况
        assertDoesNotThrow(() -> config.validate(20, 15));
        
        // rank过大的情况
        assertThrows("rank不能大于等于min(input_dim, output_dim)", 
                    IllegalArgumentException.class, 
                    () -> config.validate(8, 10));
        assertThrows("rank不能大于等于min(input_dim, output_dim)", 
                    IllegalArgumentException.class, 
                    () -> config.validate(10, 8));
    }
    
    @Test
    public void testTargetModules() {
        // 测试目标模块功能
        String[] targetModules = {"linear", "attention", "feedforward"};
        LoraConfig config = new LoraConfig(8, 16.0, 0.0, false, targetModules);
        
        assertTrue("应该匹配目标模块", config.isTargetModule("linear"));
        assertTrue("应该匹配目标模块", config.isTargetModule("ATTENTION")); // 大小写不敏感
        assertFalse("不应该匹配非目标模块", config.isTargetModule("embedding"));
    }
    
    @Test
    public void testConfigCopy() {
        // 测试配置复制功能
        LoraConfig original = new LoraConfig(8, 16.0, 0.1, true, new String[]{"linear"});
        
        LoraConfig withNewRank = original.withRank(16);
        assertEquals("新rank应该正确", 16, withNewRank.getRank());
        assertEquals("其他参数应该保持不变", 16.0, withNewRank.getAlpha(), 1e-6);
        
        LoraConfig withNewAlpha = original.withAlpha(32.0);
        assertEquals("新alpha应该正确", 32.0, withNewAlpha.getAlpha(), 1e-6);
        assertEquals("其他参数应该保持不变", 8, withNewAlpha.getRank());
    }
    
    @Test
    public void testEqualsAndHashCode() {
        // 测试equals和hashCode方法
        LoraConfig config1 = new LoraConfig(8, 16.0, 0.1, true, new String[]{"linear"});
        LoraConfig config2 = new LoraConfig(8, 16.0, 0.1, true, new String[]{"linear"});
        LoraConfig config3 = new LoraConfig(16, 16.0, 0.1, true, new String[]{"linear"});
        
        assertEquals("相同配置应该相等", config1, config2);
        assertEquals("相同配置的hashCode应该相等", config1.hashCode(), config2.hashCode());
        
        assertNotEquals("不同配置不应该相等", config1, config3);
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        LoraConfig config = new LoraConfig(8, 16.0, 0.1, false, new String[]{"linear", "attention"});
        String str = config.toString();
        
        assertTrue("toString应该包含rank信息", str.contains("rank=8"));
        assertTrue("toString应该包含alpha信息", str.contains("alpha=16.0"));
        assertTrue("toString应该包含scaling信息", str.contains("scaling=2.0000"));
        assertTrue("toString应该包含dropout信息", str.contains("dropout=0.10"));
        assertTrue("toString应该包含enableBias信息", str.contains("enableBias=false"));
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