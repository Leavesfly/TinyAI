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
 * ModelSerializer 单元测试
 * 
 * 测试模型序列化和反序列化功能
 * 
 * @author TinyDL
 * @version 1.0
 */
public class ModelSerializerTest {

    private SimpleTestModel testModel;
    private String testDirectory;
    private File tempDir;

    @Before
    public void setUp() throws IOException {
        // 创建测试模型
        testModel = new SimpleTestModel();
        
        // 创建临时测试目录
        Path tempPath = Files.createTempDirectory("modelserializer_test");
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
    public void testSaveAndLoadModel() {
        // 测试保存和加载完整模型
        String modelPath = testDirectory + "/test_model.model";
        
        // 保存模型
        ModelSerializer.saveModel(testModel, modelPath);
        
        // 验证文件存在
        File modelFile = new File(modelPath);
        assertTrue("模型文件应该存在", modelFile.exists());
        assertTrue("模型文件大小应该大于0", modelFile.length() > 0);
        
        // 加载模型
        Model loadedModel = ModelSerializer.loadModel(modelPath);
        
        assertNotNull("加载的模型不应为null", loadedModel);
        assertTrue("加载的模型应该是SimpleTestModel类型", loadedModel instanceof SimpleTestModel);
        
        // 验证参数是否正确加载
        Map<String, Parameter> originalParams = testModel.getAllParams();
        Map<String, Parameter> loadedParams = loadedModel.getAllParams();
        
        assertEquals("参数数量应该相同", originalParams.size(), loadedParams.size());
        
        for (String paramName : originalParams.keySet()) {
            assertTrue("应包含参数: " + paramName, loadedParams.containsKey(paramName));
        }
    }

    @Test
    public void testSaveAndLoadModelCompressed() {
        // 测试压缩保存和加载
        String modelPath = testDirectory + "/test_model_compressed.model";
        
        // 保存压缩模型
        ModelSerializer.saveModel(testModel, modelPath, true);
        
        // 验证文件存在
        File modelFile = new File(modelPath);
        assertTrue("压缩模型文件应该存在", modelFile.exists());
        
        // 加载压缩模型
        Model loadedModel = ModelSerializer.loadModel(modelPath, true);
        
        assertNotNull("加载的压缩模型不应为null", loadedModel);
        assertTrue("加载的模型应该是SimpleTestModel类型", loadedModel instanceof SimpleTestModel);
    }

    @Test
    public void testAutoDetectCompression() {
        // 测试自动检测压缩格式
        String uncompressedPath = testDirectory + "/uncompressed_model.model";
        String compressedPath = testDirectory + "/compressed_model.model";
        
        // 保存非压缩和压缩版本
        ModelSerializer.saveModel(testModel, uncompressedPath, false);
        ModelSerializer.saveModel(testModel, compressedPath, true);
        
        // 使用自动检测加载
        Model uncompressedModel = ModelSerializer.loadModel(uncompressedPath);
        Model compressedModel = ModelSerializer.loadModel(compressedPath);
        
        assertNotNull("非压缩模型加载应成功", uncompressedModel);
        assertNotNull("压缩模型加载应成功", compressedModel);
    }

    @Test
    public void testSaveAndLoadParameters() {
        // 测试仅保存和加载参数
        String paramsPath = testDirectory + "/test_params.params";
        
        // 保存参数
        ModelSerializer.saveParameters(testModel, paramsPath);
        
        // 验证文件存在
        File paramsFile = new File(paramsPath);
        assertTrue("参数文件应该存在", paramsFile.exists());
        
        // 创建新模型并加载参数
        SimpleTestModel newModel = new SimpleTestModel();
        ModelSerializer.loadParameters(newModel, paramsPath);
        
        // 验证参数加载正确
        Map<String, Parameter> originalParams = testModel.getAllParams();
        Map<String, Parameter> loadedParams = newModel.getAllParams();
        
        for (String paramName : originalParams.keySet()) {
            if (loadedParams.containsKey(paramName)) {
                Parameter originalParam = originalParams.get(paramName);
                Parameter loadedParam = loadedParams.get(paramName);
                
                // 检查参数形状是否相同
                assertEquals("参数形状应该相同: " + paramName,
                           originalParam.getValue().getShape(),
                           loadedParam.getValue().getShape());
            }
        }
    }

    @Test
    public void testSaveAndLoadCheckpoint() {
        // 测试保存和加载检查点
        String checkpointPath = testDirectory + "/test_checkpoint.ckpt";
        int epoch = 10;
        double loss = 0.5;
        
        // 保存检查点
        ModelSerializer.saveCheckpoint(testModel, epoch, loss, checkpointPath);
        
        // 验证文件存在
        File checkpointFile = new File(checkpointPath);
        assertTrue("检查点文件应该存在", checkpointFile.exists());
        
        // 加载检查点
        Map<String, Object> checkpoint = ModelSerializer.loadCheckpoint(checkpointPath);
        
        assertNotNull("检查点不应为null", checkpoint);
        assertTrue("检查点应包含模型", checkpoint.containsKey("model"));
        assertTrue("检查点应包含轮次", checkpoint.containsKey("epoch"));
        assertTrue("检查点应包含损失", checkpoint.containsKey("loss"));
        assertTrue("检查点应包含时间戳", checkpoint.containsKey("timestamp"));
        assertTrue("检查点应包含版本", checkpoint.containsKey("version"));
        
        assertEquals("轮次应该正确", epoch, checkpoint.get("epoch"));
        assertEquals("损失应该正确", loss, (Double) checkpoint.get("loss"), 1e-6);
        
        // 测试从检查点恢复
        Model resumedModel = ModelSerializer.resumeFromCheckpoint(checkpointPath);
        assertNotNull("恢复的模型不应为null", resumedModel);
    }

    @Test
    public void testGetModelSize() {
        // 测试获取模型文件大小
        String modelPath = testDirectory + "/size_test_model.model";
        
        // 保存模型
        ModelSerializer.saveModel(testModel, modelPath);
        
        // 获取文件大小
        long size = ModelSerializer.getModelSize(modelPath);
        assertTrue("模型文件大小应该大于0", size > 0);
        
        // 测试不存在的文件
        long nonExistentSize = ModelSerializer.getModelSize(testDirectory + "/non_existent.model");
        assertEquals("不存在文件的大小应为-1", -1, nonExistentSize);
    }

    @Test
    public void testValidateModelFile() {
        // 测试验证模型文件
        String validModelPath = testDirectory + "/valid_model.model";
        String invalidModelPath = testDirectory + "/invalid_model.txt";
        
        // 保存有效模型
        ModelSerializer.saveModel(testModel, validModelPath);
        
        // 创建无效文件
        try {
            Files.write(new File(invalidModelPath).toPath(), "invalid content".getBytes());
        } catch (IOException e) {
            fail("创建无效文件失败");
        }
        
        // 验证文件
        assertTrue("有效模型文件应该通过验证", ModelSerializer.validateModelFile(validModelPath));
        assertFalse("无效模型文件应该验证失败", ModelSerializer.validateModelFile(invalidModelPath));
        assertFalse("不存在的文件应该验证失败", ModelSerializer.validateModelFile(testDirectory + "/non_existent.model"));
    }

    @Test
    public void testCompareModelParameters() {
        // 测试比较模型参数
        SimpleTestModel model1 = new SimpleTestModel();
        SimpleTestModel model2 = new SimpleTestModel();
        SimpleTestModel model3 = new SimpleTestModel();
        
        // 修改model3的参数
        model3.getAllParams().get("weight").getValue().set(999.0f, 0, 0);
        
        // 比较相同的模型
        assertTrue("相同模型的参数应该相等", ModelSerializer.compareModelParameters(model1, model2));
        
        // 比较不同的模型
        assertFalse("不同模型的参数应该不相等", ModelSerializer.compareModelParameters(model1, model3));
        
        // 测试null输入
        assertFalse("null模型比较应该返回false", ModelSerializer.compareModelParameters(null, model1));
        assertFalse("null模型比较应该返回false", ModelSerializer.compareModelParameters(model1, null));
        assertFalse("两个null模型比较应该返回false", ModelSerializer.compareModelParameters(null, null));
    }

    @Test
    public void testSerializationWithDirectoryCreation() {
        // 测试在不存在的目录中保存模型
        String deepPath = testDirectory + "/deep/nested/directory/model.model";
        
        // 保存模型（应该自动创建目录）
        ModelSerializer.saveModel(testModel, deepPath);
        
        // 验证文件和目录都被创建
        File modelFile = new File(deepPath);
        assertTrue("模型文件应该存在", modelFile.exists());
        assertTrue("父目录应该被创建", modelFile.getParentFile().exists());
    }

    @Test
    public void testLoadNonExistentFile() {
        // 测试加载不存在的文件
        String nonExistentPath = testDirectory + "/non_existent_model.model";
        
        try {
            ModelSerializer.loadModel(nonExistentPath);
            fail("加载不存在的文件应该抛出异常");
        } catch (RuntimeException e) {
            assertTrue("异常消息应包含相关信息", e.getMessage().contains("does not exist") || e.getMessage().contains("不存在") || e.getMessage().contains("Failed to load model"));
        }
    }

    @Test
    public void testParameterLoadWithShapeMismatch() {
        // 测试参数形状不匹配的情况
        String paramsPath = testDirectory + "/mismatch_params.params";
        
        // 保存原始参数
        ModelSerializer.saveParameters(testModel, paramsPath);
        
        // 创建形状不匹配的新模型
        SimpleTestModel mismatchModel = new SimpleTestModel();
        mismatchModel.getAllParams().put("weight", 
                new Parameter(NdArray.of(new float[][]{{1.0f}}))); // 不同形状
        
        // 尝试加载参数（应该跳过不匹配的参数）
        ModelSerializer.loadParameters(mismatchModel, paramsPath);
        
        // 应该不会抛出异常，只是跳过不匹配的参数
    }

    /**
     * 测试用的 Block 实现
     */
    public static class TestBlock extends Block implements java.io.Serializable {
        
        private static final long serialVersionUID = 1L;
        
        // 默认构造函数，序列化所需
        public TestBlock() {
            super("TestBlock", Shape.of(2, 2));
        }
        
        // 带参数的构造函数
        public TestBlock(String name, Shape inputShape) {
            super(name, inputShape);
        }
        
        @Override
        public void init() {
            // 简单初始化
            alreadyInit = true;
        }
    }

    /**
     * 测试用的简单 Model 实现，不依赖 Block
     */
    public static class SimpleTestModel extends Model implements java.io.Serializable {
        
        private static final long serialVersionUID = 1L;
        private Map<String, Parameter> parameters;

        public SimpleTestModel() {
            // 传入 null block，简化测试
            super("SimpleTestModel", null);
            parameters = new HashMap<>();
            
            // 添加测试参数
            parameters.put("weight", new Parameter(NdArray.of(new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}})));
            parameters.put("bias", new Parameter(NdArray.of(new float[][]{{0.1f, 0.2f}})));
        }

        @Override
        public Variable forward(Variable... inputs) {
            // 简单的前向传播实现
            return inputs[0];
        }

        @Override
        public Map<String, Parameter> getAllParams() {
            // 直接返回自定义参数
            return new HashMap<>(parameters);
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