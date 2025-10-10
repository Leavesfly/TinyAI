package io.leavesfly.tinyai.cv.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Cifar10DatasetTest {
    
    @Test
    public void testCifar10DatasetConstruction() {
        // 测试CIFAR-10数据集加载器构造
        String datasetPath = "/path/to/cifar10";
        Cifar10Dataset dataset = new Cifar10Dataset(datasetPath);
        
        // 验证数据集加载器不为null
        assertNotNull(dataset);
        
        // 验证类别名称
        assertEquals(10, Cifar10Dataset.CLASS_NAMES.length);
        assertEquals("airplane", Cifar10Dataset.CLASS_NAMES[0]);
        assertEquals("truck", Cifar10Dataset.CLASS_NAMES[9]);
    }
    
    @Test
    public void testCifar10DatasetGetters() {
        // 测试CIFAR-10数据集加载器构造
        String datasetPath = "/path/to/cifar10";
        Cifar10Dataset dataset = new Cifar10Dataset(datasetPath);
        
        // 验证类别数量
        assertEquals(10, dataset.getNumClasses());
        
        // 验证初始样本数量为0
        assertEquals(0, dataset.getTrainSize());
        assertEquals(0, dataset.getTestSize());
        
        // 验证初始数据为null
        assertNull(dataset.getTrainImages());
        assertNull(dataset.getTrainLabels());
        assertNull(dataset.getTestImages());
        assertNull(dataset.getTestLabels());
    }
    
    // 注意：以下测试需要实际的CIFAR-10数据集文件，因此在实际测试中可能需要mock文件系统
    // @Test
    // public void testLoadTrainData() {
    //     // 测试加载训练数据
    //     String datasetPath = "src/test/resources/cifar10"; // 假设测试资源目录中有数据集文件
    //     Cifar10Dataset dataset = new Cifar10Dataset(datasetPath);
    //     
    //     // 加载训练数据
    //     boolean success = dataset.loadTrainData();
    //     
    //     // 验证加载成功
    //     assertTrue(success);
    //     
    //     // 验证训练数据不为null
    //     assertNotNull(dataset.getTrainImages());
    //     assertNotNull(dataset.getTrainLabels());
    //     
    //     // 验证训练数据形状
    //     NdArray trainImages = dataset.getTrainImages();
    //     NdArray trainLabels = dataset.getTrainLabels();
    //     
    //     // 验证图像数据形状 (batch_size, 32, 32, 3)
    //     Shape imageShape = trainImages.getShape();
    //     assertEquals(50000, imageShape.getDimension(0)); // 5个批次，每批次10000张图像
    //     assertEquals(32, imageShape.getDimension(1));
    //     assertEquals(32, imageShape.getDimension(2));
    //     assertEquals(3, imageShape.getDimension(3));
    //     
    //     // 验证标签数据形状 (batch_size, 1)
    //     Shape labelShape = trainLabels.getShape();
    //     assertEquals(50000, labelShape.getDimension(0));
    //     assertEquals(1, labelShape.getDimension(1));
    // }
    // 
    // @Test
    // public void testLoadTestData() {
    //     // 测试加载测试数据
    //     String datasetPath = "src/test/resources/cifar10";
    //     Cifar10Dataset dataset = new Cifar10Dataset(datasetPath);
    //     
    //     // 加载测试数据
    //     boolean success = dataset.loadTestData();
    //     
    //     // 验证加载成功
    //     assertTrue(success);
    //     
    //     // 验证测试数据不为null
    //     assertNotNull(dataset.getTestImages());
    //     assertNotNull(dataset.getTestLabels());
    //     
    //     // 验证测试数据形状
    //     NdArray testImages = dataset.getTestImages();
    //     NdArray testLabels = dataset.getTestLabels();
    //     
    //     // 验证图像数据形状 (batch_size, 32, 32, 3)
    //     Shape imageShape = testImages.getShape();
    //     assertEquals(10000, imageShape.getDimension(0)); // 1个测试批次，10000张图像
    //     assertEquals(32, imageShape.getDimension(1));
    //     assertEquals(32, imageShape.getDimension(2));
    //     assertEquals(3, imageShape.getDimension(3));
    //     
    //     // 验证标签数据形状 (batch_size, 1)
    //     Shape labelShape = testLabels.getShape();
    //     assertEquals(10000, labelShape.getDimension(0));
    //     assertEquals(1, labelShape.getDimension(1));
    // }
}