package io.leavesfly.tinyai.mlearning;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.block.MlpBlock;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ParameterManager测试类
 */
public class ParameterManagerTest {
    
    private Model testModel1;
    private Model testModel2;
    private String testDir = "test-output/parameter/";
    
    @BeforeEach
    public void setUp() {
        // 创建测试模型
        Block block1 = new MlpBlock(2, 3, 1);
        Block block2 = new MlpBlock(2, 3, 1);
        testModel1 = new Model("TestModel1", block1);
        testModel2 = new Model("TestModel2", block2);
        
        // 创建测试目录
        new File(testDir).mkdirs();
    }
    
    @AfterEach
    public void tearDown() {
        // 清理测试文件
        File dir = new File(testDir);
        if (dir.exists()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File file : files) {
                    file.delete();
                }
            }
            dir.delete();
        }
    }
    
    @Test
    public void testSaveAndLoadParameters() {
        String filePath = testDir + "test-params.params";
        
        // 获取参数
        Map<String, Parameter> originalParams = testModel1.getAllParams();
        
        // 保存参数
        ParameterManager.saveParameters(originalParams, filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists());
        
        // 加载参数
        Map<String, Parameter> loadedParams = ParameterManager.loadParameters(filePath);
        
        // 验证参数数量
        assertEquals(originalParams.size(), loadedParams.size());
        
        // 验证每个参数都存在
        for (String paramName : originalParams.keySet()) {
            assertTrue(loadedParams.containsKey(paramName));
        }
    }
    
    @Test
    public void testCopyParameters() {
        // 复制参数（非严格模式）
        int copiedCount = ParameterManager.copyParameters(testModel1, testModel2, false);
        
        // 验证复制的参数数量大于0
        assertTrue(copiedCount > 0);
        
        // 验证参数确实被复制了
        assertTrue(ParameterManager.compareParameters(testModel1, testModel2, 1e-6));
    }
    
    @Test
    public void testCopyParametersStrict() {
        // 在严格模式下复制参数应该成功（因为两个模型结构相同）
        int copiedCount = ParameterManager.copyParameters(testModel1, testModel2, true);
        
        // 验证复制的参数数量大于0
        assertTrue(copiedCount > 0);
        
        // 验证参数确实被复制了
        assertTrue(ParameterManager.compareParameters(testModel1, testModel2, 1e-6));
    }
    
    @Test
    public void testCompareParameters() {
        // 初始状态下参数可能不同
        boolean initiallyEqual = ParameterManager.compareParameters(testModel1, testModel2);
        
        // 复制参数
        ParameterManager.copyParameters(testModel1, testModel2);
        
        // 现在应该相同
        assertTrue(ParameterManager.compareParameters(testModel1, testModel2));
        assertTrue(ParameterManager.compareParameters(testModel1, testModel2, 1e-10));
    }
    
    @Test
    public void testGetParameterStats() {
        Map<String, Parameter> params = testModel1.getAllParams();
        
        // 获取参数统计
        ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(params);
        
        // 验证统计信息
        assertNotNull(stats);
        assertEquals(params.size(), stats.parameterCount);
        assertTrue(stats.totalParameters > 0);
        assertTrue(stats.minValue <= stats.maxValue);
        
        // 输出统计信息
        System.out.println("参数统计信息: " + stats);
    }
    
    @Test
    public void testGetParameterStatsEmpty() {
        // 空参数映射
        ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(new HashMap<>());
        
        assertNotNull(stats);
        assertEquals(0, stats.parameterCount);
        assertEquals(0, stats.totalParameters);
    }
    
    @Test
    public void testDeepCopyParameters() {
        Map<String, Parameter> originalParams = testModel1.getAllParams();
        
        // 深拷贝参数
        Map<String, Parameter> copiedParams = ParameterManager.deepCopyParameters(originalParams);
        
        // 验证拷贝结果
        assertNotNull(copiedParams);
        assertEquals(originalParams.size(), copiedParams.size());
        
        // 验证每个参数都被拷贝了
        for (String paramName : originalParams.keySet()) {
            assertTrue(copiedParams.containsKey(paramName));
            assertNotSame(originalParams.get(paramName), copiedParams.get(paramName));
        }
    }
    
    @Test
    public void testDeepCopyParametersNull() {
        // 传入null应该返回null
        assertNull(ParameterManager.deepCopyParameters(null));
    }
    
    @Test
    public void testFilterParameters() {
        Map<String, Parameter> params = testModel1.getAllParams();
        
        // 测试通配符过滤
        Map<String, Parameter> filteredParams = ParameterManager.filterParameters(params, "*weight*");
        
        // 验证过滤结果
        assertNotNull(filteredParams);
        
        // 所有返回的参数名应该包含"weight"
        for (String paramName : filteredParams.keySet()) {
            assertTrue(paramName.contains("weight") || paramName.matches(".*weight.*"));
        }
    }
    
    @Test
    public void testFilterParametersExact() {
        Map<String, Parameter> params = testModel1.getAllParams();
        
        // 如果我们知道确切的参数名，可以进行精确过滤
        if (!params.isEmpty()) {
            String firstParamName = params.keySet().iterator().next();
            Map<String, Parameter> exactFilter = ParameterManager.filterParameters(params, firstParamName);
            
            assertEquals(1, exactFilter.size());
            assertTrue(exactFilter.containsKey(firstParamName));
        }
    }
    
    @Test
    public void testSaveParameterStats() {
        String filePath = testDir + "param-stats.txt";
        Map<String, Parameter> params = testModel1.getAllParams();
        
        // 保存参数统计
        ParameterManager.saveParameterStats(params, filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists());
        
        // 验证文件大小大于0
        assertTrue(new File(filePath).length() > 0);
    }
    
    @Test
    public void testParameterStatsToString() {
        Map<String, Parameter> params = testModel1.getAllParams();
        ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(params);
        
        String statsString = stats.toString();
        assertNotNull(statsString);
        assertTrue(statsString.contains("ParameterStats"));
        assertTrue(statsString.contains("totalParams"));
        assertTrue(statsString.contains("paramCount"));
    }
}