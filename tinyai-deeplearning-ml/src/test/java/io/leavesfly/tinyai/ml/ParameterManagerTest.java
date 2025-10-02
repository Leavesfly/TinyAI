package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import org.junit.Before;
import org.junit.Test;
import org.junit.After;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * ParameterManager 单元测试
 * 
 * 测试参数管理器的各种功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class ParameterManagerTest {

    private Map<String, Parameter> testParameters;
    private TestModel sourceModel;
    private TestModel targetModel;
    private String testDirectory;
    private File tempDir;

    @Before
    public void setUp() throws IOException {
        // 创建测试参数
        testParameters = new HashMap<>();
        testParameters.put("weight1", new Parameter(NdArray.of(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}})));
        testParameters.put("bias1", new Parameter(NdArray.of(new float[][]{{0.1f, 0.2f}})));
        testParameters.put("weight2", new Parameter(NdArray.of(new float[][]{{5.0f}})));
        
        // 创建测试模型
        sourceModel = new TestModel();
        sourceModel.setParameters(new HashMap<>(testParameters));
        
        targetModel = new TestModel();
        Map<String, Parameter> targetParams = new HashMap<>();
        targetParams.put("weight1", new Parameter(NdArray.of(new float[][]{{0.0f, 0.0f}, {0.0f, 0.0f}})));
        targetParams.put("bias1", new Parameter(NdArray.of(new float[][]{{0.0f, 0.0f}})));
        targetParams.put("weight2", new Parameter(NdArray.of(new float[][]{{0.0f}})));
        targetModel.setParameters(targetParams);
        
        // 创建临时测试目录
        Path tempPath = Files.createTempDirectory("parammanager_test");
        tempDir = tempPath.toFile();
        testDirectory = tempPath.toString();
    }

    @After
    public void tearDown() {
        // 清理测试文件
        if (tempDir != null && tempDir.exists()) {
            deleteDirectory(tempDir);
        }
    }

    @Test
    public void testSaveAndLoadParameters() {
        // 测试保存和加载参数
        String paramsPath = testDirectory + "/test_params.params";
        
        // 保存参数
        ParameterManager.saveParameters(testParameters, paramsPath);
        
        // 验证文件存在
        File paramsFile = new File(paramsPath);
        assertTrue("参数文件应该存在", paramsFile.exists());
        assertTrue("参数文件大小应该大于0", paramsFile.length() > 0);
        
        // 加载参数
        Map<String, Parameter> loadedParams = ParameterManager.loadParameters(paramsPath);
        
        assertNotNull("加载的参数不应为null", loadedParams);
        assertEquals("参数数量应该相同", testParameters.size(), loadedParams.size());
        
        for (String paramName : testParameters.keySet()) {
            assertTrue("应包含参数: " + paramName, loadedParams.containsKey(paramName));
            
            Parameter originalParam = testParameters.get(paramName);
            Parameter loadedParam = loadedParams.get(paramName);
            
            assertEquals("参数形状应该相同: " + paramName,
                       originalParam.getValue().getShape(),
                       loadedParam.getValue().getShape());
        }
    }

    @Test
    public void testCopyParametersSuccess() {
        // 测试成功的参数复制
        int copiedCount = ParameterManager.copyParameters(sourceModel, targetModel);
        
        assertEquals("应该复制3个参数", 3, copiedCount);
        
        // 验证参数已被复制
        Map<String, Parameter> sourceParams = sourceModel.getAllParams();
        Map<String, Parameter> targetParams = targetModel.getAllParams();
        
        for (String paramName : sourceParams.keySet()) {
            Parameter sourceParam = sourceParams.get(paramName);
            Parameter targetParam = targetParams.get(paramName);
            
            // 检查参数值是否相同
            float[][] sourceMatrix = sourceParam.getValue().getMatrix();
            float[][] targetMatrix = targetParam.getValue().getMatrix();
            
            assertArrayEquals("参数值应该相同: " + paramName, 
                            sourceMatrix[0], targetMatrix[0], 1e-6f);
        }
    }

    @Test
    public void testCopyParametersStrict() {
        // 测试严格模式的参数复制
        
        // 创建参数不匹配的模型
        TestModel mismatchModel = new TestModel();
        Map<String, Parameter> mismatchParams = new HashMap<>();
        mismatchParams.put("weight1", new Parameter(NdArray.of(new float[][]{{1.0f}}))); // 形状不匹配
        mismatchParams.put("extra_param", new Parameter(NdArray.of(new float[][]{{1.0f}}))); // 额外参数
        mismatchModel.setParameters(mismatchParams);
        
        // 严格模式应该抛出异常
        try {
            ParameterManager.copyParameters(sourceModel, mismatchModel, true);
            fail("严格模式下形状不匹配应该抛出异常");
        } catch (RuntimeException e) {
            assertTrue("异常消息应包含形状信息", e.getMessage().contains("形状不匹配") || e.getMessage().contains("不存在参数"));
        }
    }

    @Test
    public void testCopyParametersNonStrict() {
        // 测试非严格模式的参数复制
        
        // 创建部分匹配的模型
        TestModel partialModel = new TestModel();
        Map<String, Parameter> partialParams = new HashMap<>();
        partialParams.put("weight1", new Parameter(NdArray.of(new float[][]{{0.0f, 0.0f}, {0.0f, 0.0f}}))); // 匹配
        partialParams.put("different_shape", new Parameter(NdArray.of(new float[][]{{1.0f}}))); // 形状不匹配
        partialModel.setParameters(partialParams);
        
        // 非严格模式应该成功复制匹配的参数
        int copiedCount = ParameterManager.copyParameters(sourceModel, partialModel, false);
        
        assertEquals("应该复制1个匹配的参数", 1, copiedCount);
    }

    @Test
    public void testCopyParametersWithNullModels() {
        // 测试null模型的处理
        try {
            ParameterManager.copyParameters(null, targetModel);
            fail("null源模型应该抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常消息应包含模型为空信息", e.getMessage().contains("模型不能为空"));
        }
        
        try {
            ParameterManager.copyParameters(sourceModel, null);
            fail("null目标模型应该抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue("异常消息应包含模型为空信息", e.getMessage().contains("模型不能为空"));
        }
    }

    @Test
    public void testCompareParameters() {
        // 测试参数比较
        
        // 创建相同的模型
        TestModel sameModel = new TestModel();
        Map<String, Parameter> sameParams = new HashMap<>();
        for (Map.Entry<String, Parameter> entry : testParameters.entrySet()) {
            Parameter originalParam = entry.getValue();
            float[][] originalMatrix = originalParam.getValue().getMatrix();
            
            // 创建相同值的参数
            float[][] copyMatrix = new float[originalMatrix.length][];
            for (int i = 0; i < originalMatrix.length; i++) {
                copyMatrix[i] = originalMatrix[i].clone();
            }
            sameParams.put(entry.getKey(), new Parameter(NdArray.of(copyMatrix)));
        }
        sameModel.setParameters(sameParams);
        
        // 比较相同的模型
        assertTrue("相同参数的模型应该比较相等", 
                  ParameterManager.compareParameters(sourceModel, sameModel));
        
        // 比较不同的模型
        assertFalse("不同参数的模型应该比较不相等", 
                   ParameterManager.compareParameters(sourceModel, targetModel));
        
        // 测试容忍度
        TestModel slightlyDifferentModel = new TestModel();
        Map<String, Parameter> slightlyDifferentParams = new HashMap<>();
        slightlyDifferentParams.put("weight1", 
                new Parameter(NdArray.of(new float[][]{{1.0001f, 2.0001f}, {3.0001f, 4.0001f}})));
        slightlyDifferentParams.put("bias1", 
                new Parameter(NdArray.of(new float[][]{{0.1001f, 0.2001f}})));
        slightlyDifferentParams.put("weight2", 
                new Parameter(NdArray.of(new float[][]{{5.0001f}})));
        slightlyDifferentModel.setParameters(slightlyDifferentParams);
        
        // 使用较大的容忍度应该相等
        assertTrue("小差异在容忍度内应该相等", 
                  ParameterManager.compareParameters(sourceModel, slightlyDifferentModel, 1e-3));
        
        // 使用较小的容忍度应该不相等
        assertFalse("小差异超出容忍度应该不相等", 
                   ParameterManager.compareParameters(sourceModel, slightlyDifferentModel, 1e-6));
    }

    @Test
    public void testGetParameterStats() {
        // 测试参数统计信息
        ParameterManager.ParameterStats stats = ParameterManager.getParameterStats(testParameters);
        
        assertNotNull("统计信息不应为null", stats);
        assertEquals("参数组数量应该正确", 3, stats.parameterCount);
        assertEquals("总参数数量应该正确", 7, stats.totalParameters); // 2*2 + 1*2 + 1*1 = 7
        
        assertTrue("最小值应该是合理的", stats.minValue >= 0.0f);
        assertTrue("最大值应该是合理的", stats.maxValue >= stats.minValue);
        assertTrue("平均值应该在最小值和最大值之间", 
                  stats.meanValue >= stats.minValue && stats.meanValue <= stats.maxValue);
        
        // 测试toString方法
        String statsString = stats.toString();
        assertNotNull("统计信息字符串不应为null", statsString);
        assertTrue("统计信息字符串应包含参数数量", statsString.contains("totalParams"));
    }

    @Test
    public void testGetParameterStatsEmpty() {
        // 测试空参数映射的统计
        ParameterManager.ParameterStats emptyStats = ParameterManager.getParameterStats(new HashMap<>());
        
        assertNotNull("空统计信息不应为null", emptyStats);
        assertEquals("空参数的数量应为0", 0, emptyStats.parameterCount);
        assertEquals("空参数的总数应为0", 0, emptyStats.totalParameters);
        
        // 测试null参数映射
        ParameterManager.ParameterStats nullStats = ParameterManager.getParameterStats(null);
        assertNotNull("null统计信息不应为null", nullStats);
        assertEquals("null参数的数量应为0", 0, nullStats.parameterCount);
    }

    @Test
    public void testDeepCopyParameters() {
        // 测试深拷贝参数
        Map<String, Parameter> copiedParams = ParameterManager.deepCopyParameters(testParameters);
        
        assertNotNull("拷贝的参数不应为null", copiedParams);
        assertEquals("参数数量应该相同", testParameters.size(), copiedParams.size());
        
        // 验证深拷贝：修改原始参数不应影响拷贝
        Parameter originalParam = testParameters.get("weight1");
        Parameter copiedParam = copiedParams.get("weight1");
        
        // 修改原始参数
        originalParam.getValue().set(999.0f, 0, 0);
        
        // 验证拷贝没有被影响
        assertNotEquals("深拷贝应该独立于原始参数", 
                       999.0f, copiedParam.getValue().getMatrix()[0][0]);
        
        // 测试null参数映射
        Map<String, Parameter> nullCopy = ParameterManager.deepCopyParameters(null);
        assertNull("null参数映射的拷贝应为null", nullCopy);
    }

    @Test
    public void testFilterParameters() {
        // 测试参数筛选
        
        // 筛选weight参数
        Map<String, Parameter> weightParams = ParameterManager.filterParameters(testParameters, "weight*");
        assertEquals("应该找到2个weight参数", 2, weightParams.size());
        assertTrue("应包含weight1", weightParams.containsKey("weight1"));
        assertTrue("应包含weight2", weightParams.containsKey("weight2"));
        assertFalse("不应包含bias1", weightParams.containsKey("bias1"));
        
        // 筛选bias参数
        Map<String, Parameter> biasParams = ParameterManager.filterParameters(testParameters, "bias*");
        assertEquals("应该找到1个bias参数", 1, biasParams.size());
        assertTrue("应包含bias1", biasParams.containsKey("bias1"));
        
        // 筛选特定参数
        Map<String, Parameter> specificParams = ParameterManager.filterParameters(testParameters, "weight1");
        assertEquals("应该找到1个特定参数", 1, specificParams.size());
        assertTrue("应包含weight1", specificParams.containsKey("weight1"));
        
        // 筛选不存在的模式
        Map<String, Parameter> noMatchParams = ParameterManager.filterParameters(testParameters, "nonexistent*");
        assertEquals("不匹配的模式应返回空映射", 0, noMatchParams.size());
    }

    @Test
    public void testSaveParameterStats() {
        // 测试保存参数统计信息
        String statsPath = testDirectory + "/param_stats.txt";
        
        ParameterManager.saveParameterStats(testParameters, statsPath);
        
        // 验证文件存在
        File statsFile = new File(statsPath);
        assertTrue("统计文件应该存在", statsFile.exists());
        assertTrue("统计文件大小应该大于0", statsFile.length() > 0);
        
        // 验证文件内容（简单检查）
        try {
            String content = new String(Files.readAllBytes(statsFile.toPath()));
            assertTrue("文件应包含统计标题", content.contains("模型参数统计"));
            assertTrue("文件应包含参数详细信息", content.contains("参数详细信息"));
            assertTrue("文件应包含weight1", content.contains("weight1"));
        } catch (IOException e) {
            fail("读取统计文件失败: " + e.getMessage());
        }
    }

    @Test
    public void testLoadNonExistentParameterFile() {
        // 测试加载不存在的参数文件
        String nonExistentPath = testDirectory + "/non_existent_params.params";
        
        try {
            ParameterManager.loadParameters(nonExistentPath);
            fail("加载不存在的文件应该抛出异常");
        } catch (RuntimeException e) {
            assertTrue("异常消息应包含文件不存在信息", e.getMessage().contains("does not exist") || e.getMessage().contains("不存在"));
        }
    }

    /**
     * 测试用的 Block 实现
     */
    private static class TestBlock extends Block implements java.io.Serializable {
        
        private static final long serialVersionUID = 1L;
        
        public TestBlock() {
            super("TestBlock", Shape.of(2, 2));
        }
        
        // 序列化所需的默认构造函数
        private TestBlock(String name, Shape inputShape) {
            super(name, inputShape);
        }
        
        @Override
        public void init() {
            // 简单初始化
            alreadyInit = true;
        }
    }

    /**
     * 测试用的 Model 实现
     */
    private static class TestModel extends Model {
        
        private Map<String, Parameter> parameters = new HashMap<>();

        public TestModel() {
            super("TestModel", new TestBlock());
        }

        public Variable forward(Variable x) {
            return x; // 简单返回输入
        }

        @Override
        public Map<String, Parameter> getAllParams() {
            // 结合 block 的参数和自定义参数
            Map<String, Parameter> allParams = new HashMap<>();
            allParams.putAll(super.getAllParams()); // block 的参数
            allParams.putAll(parameters); // 自定义参数
            return allParams;
        }

        public void setParameters(Map<String, Parameter> parameters) {
            this.parameters = parameters;
        }
    }

    /**
     * 递归删除目录
     */
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            directory.delete();
        }
    }
}