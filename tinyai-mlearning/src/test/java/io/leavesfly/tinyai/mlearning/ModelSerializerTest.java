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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ModelSerializer测试类
 */
public class ModelSerializerTest {
    
    private Model testModel;
    private String testDir = "test-output/serializer/";
    
    @BeforeEach
    public void setUp() {
        // 创建测试模型
        Block mlpBlock = new MlpBlock(2, 3, 1);
        testModel = new Model("TestModel", mlpBlock);
        
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
    public void testSaveAndLoadModel() {
        String filePath = testDir + "test-model.model";
        
        // 保存模型
        ModelSerializer.saveModel(testModel, filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists());
        
        // 加载模型
        Model loadedModel = ModelSerializer.loadModel(filePath);
        
        // 验证模型
        assertNotNull(loadedModel);
        assertEquals(testModel.getName(), loadedModel.getName());
        assertNotNull(loadedModel.getBlock());
    }
    
    @Test
    public void testSaveAndLoadParameters() {
        String filePath = testDir + "test-params.params";
        
        // 保存参数
        ModelSerializer.saveParameters(testModel, filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists());
        
        // 创建新的相同结构模型
        Block newBlock = new MlpBlock(2, 3, 1);
        Model newModel = new Model("NewModel", newBlock);
        
        // 加载参数
        ModelSerializer.loadParameters(newModel, filePath);
        
        // 验证参数数量相同
        Map<String, Parameter> originalParams = testModel.getAllParams();
        Map<String, Parameter> loadedParams = newModel.getAllParams();
        assertEquals(originalParams.size(), loadedParams.size());
    }
    
    @Test
    public void testCompareModelParameters() {
        // 创建相同结构的两个模型
        Block block1 = new MlpBlock(2, 3, 1);
        Block block2 = new MlpBlock(2, 3, 1);
        Model model1 = new Model("Model1", block1);
        Model model2 = new Model("Model2", block2);
        
        // 初始状态下参数可能不同（随机初始化）
        boolean initialCompare = ModelSerializer.compareModelParameters(model1, model2);
        
        // 复制参数使其相同
        ParameterManager.copyParameters(model1, model2);
        
        // 现在应该相同
        assertTrue(ModelSerializer.compareModelParameters(model1, model2));
    }
    
    @Test
    public void testSaveAndLoadCheckpoint() {
        String filePath = testDir + "test-checkpoint.ckpt";
        
        // 保存检查点
        ModelSerializer.saveCheckpoint(testModel, 10, 0.5, filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists());
        
        // 加载检查点
        Map<String, Object> checkpoint = ModelSerializer.loadCheckpoint(filePath);
        
        // 验证检查点数据
        assertNotNull(checkpoint);
        assertEquals(10, checkpoint.get("epoch"));
        assertEquals(0.5, checkpoint.get("loss"));
        assertNotNull(checkpoint.get("model"));
        assertNotNull(checkpoint.get("timestamp"));
        assertEquals("TinyDL-0.01", checkpoint.get("version"));
    }
    
    @Test
    public void testResumeFromCheckpoint() {
        String filePath = testDir + "test-resume.ckpt";
        
        // 保存检查点
        ModelSerializer.saveCheckpoint(testModel, 5, 0.3, filePath);
        
        // 从检查点恢复
        Model resumedModel = ModelSerializer.resumeFromCheckpoint(filePath);
        
        // 验证恢复的模型
        assertNotNull(resumedModel);
        assertEquals(testModel.getName(), resumedModel.getName());
    }
    
    @Test
    public void testValidateModelFile() {
        String validFilePath = testDir + "valid-model.model";
        String invalidFilePath = testDir + "invalid-file.txt";
        
        // 保存有效模型
        ModelSerializer.saveModel(testModel, validFilePath);
        
        // 创建无效文件
        try {
            new File(invalidFilePath).createNewFile();
        } catch (Exception e) {
            fail("Failed to create test file");
        }
        
        // 验证文件
        assertTrue(ModelSerializer.validateModelFile(validFilePath));
        assertFalse(ModelSerializer.validateModelFile(invalidFilePath));
        assertFalse(ModelSerializer.validateModelFile("non-existent-file.model"));
    }
    
    @Test
    public void testGetModelSize() {
        String filePath = testDir + "size-test.model";
        
        // 保存模型
        ModelSerializer.saveModel(testModel, filePath);
        
        // 获取文件大小
        long size = ModelSerializer.getModelSize(filePath);
        
        // 验证大小大于0
        assertTrue(size > 0);
        
        // 不存在的文件应该返回-1
        assertEquals(-1, ModelSerializer.getModelSize("non-existent.model"));
    }
}