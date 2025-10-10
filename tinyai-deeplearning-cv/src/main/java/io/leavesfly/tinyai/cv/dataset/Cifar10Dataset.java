package io.leavesfly.tinyai.cv.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * CIFAR-10数据集加载器
 * 
 * CIFAR-10是一个常用的图像分类数据集，包含10个类别的60000张32x32彩色图像：
 * 飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车
 * 
 * 数据集结构：
 * - 训练集：50000张图像，分为5个批次文件
 * - 测试集：10000张图像，1个批次文件
 * - 每个批次文件包含10000张图像
 * - 每张图像为32x32x3的彩色图像
 * - 每个批次文件格式：1字节标签 + 3072字节图像数据（RGB通道顺序）
 */
public class Cifar10Dataset {
    
    // CIFAR-10类别名称
    public static final String[] CLASS_NAMES = {
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    };
    
    // 数据集路径
    private final String datasetPath;
    
    // 训练数据和标签
    private NdArray trainImages;
    private NdArray trainLabels;
    
    // 测试数据和标签
    private NdArray testImages;
    private NdArray testLabels;
    
    /**
     * 构造CIFAR-10数据集加载器
     * @param datasetPath 数据集文件路径
     */
    public Cifar10Dataset(String datasetPath) {
        this.datasetPath = datasetPath;
    }
    
    /**
     * 加载训练数据集
     * @return 是否加载成功
     */
    public boolean loadTrainData() {
        try {
            // CIFAR-10训练集包含5个批次文件
            List<NdArray> imageBatches = new ArrayList<>();
            List<NdArray> labelBatches = new ArrayList<>();
            
            for (int i = 1; i <= 5; i++) {
                String batchFile = datasetPath + "/data_batch_" + i + ".bin";
                Object[] batchData = loadBatchFile(batchFile);
                imageBatches.add((NdArray) batchData[0]);
                labelBatches.add((NdArray) batchData[1]);
            }
            
            // 合并所有批次数据
            trainImages = concatenateBatches(imageBatches);
            trainLabels = concatenateBatches(labelBatches);
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * 加载测试数据集
     * @return 是否加载成功
     */
    public boolean loadTestData() {
        try {
            String testFile = datasetPath + "/test_batch.bin";
            Object[] testData = loadBatchFile(testFile);
            testImages = (NdArray) testData[0];
            testLabels = (NdArray) testData[1];
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * 加载单个批次文件
     * @param batchFile 批次文件路径
     * @return 包含图像和标签的数组
     * @throws IOException 文件读取异常
     */
    private Object[] loadBatchFile(String batchFile) throws IOException {
        File file = new File(batchFile);
        if (!file.exists()) {
            throw new FileNotFoundException("Batch file not found: " + batchFile);
        }
        
        try (DataInputStream dis = new DataInputStream(new FileInputStream(file))) {
            int batchSize = 10000; // 每个批次包含10000张图像
            int imageBytes = 32 * 32 * 3; // 每张图像的字节数
            
            // 创建存储图像和标签的数组
            float[][][] images = new float[batchSize][32][32][3];
            float[] labels = new float[batchSize];
            
            // 读取批次数据
            for (int i = 0; i < batchSize; i++) {
                // 读取标签（1字节）
                labels[i] = dis.readByte() & 0xFF;
                
                // 读取图像数据（3072字节）
                byte[] imageData = new byte[imageBytes];
                dis.readFully(imageData);
                
                // 将图像数据转换为3D数组（高度、宽度、通道）
                // CIFAR-10的图像数据按通道顺序存储：R、G、B
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        // R通道
                        images[i][h][w][0] = (imageData[h * 32 + w] & 0xFF) / 255.0f;
                        // G通道
                        images[i][h][w][1] = (imageData[1024 + h * 32 + w] & 0xFF) / 255.0f;
                        // B通道
                        images[i][h][w][2] = (imageData[2048 + h * 32 + w] & 0xFF) / 255.0f;
                    }
                }
            }
            
            // 转换为NdArray
            NdArray imageArray = NdArray.of(images);
            NdArray labelArray = NdArray.of(labels, Shape.of(batchSize, 1));
            
            return new Object[]{imageArray, labelArray};
        }
    }
    
    /**
     * 合并多个批次数据
     * @param batches 批次数据列表
     * @return 合并后的数据
     */
    private NdArray concatenateBatches(List<NdArray> batches) {
        if (batches.isEmpty()) {
            return null;
        }
        
        // 计算总样本数
        int totalSamples = 0;
        for (NdArray batch : batches) {
            totalSamples += batch.getShape().getDimension(0);
        }
        
        // 获取单个样本的形状
        Shape sampleShape = batches.get(0).getShape();
        int[] dimensions = new int[sampleShape.getDimNum() + 1];
        dimensions[0] = totalSamples;
        for (int i = 0; i < sampleShape.getDimNum(); i++) {
            dimensions[i + 1] = sampleShape.getDimension(i);
        }
        
        // 创建合并后的数组
        Shape mergedShape = Shape.of(dimensions);
        NdArray merged = NdArray.of(mergedShape);
        
        // 复制数据
        int offset = 0;
        for (NdArray batch : batches) {
            int batchSize = batch.getShape().getDimension(0);
            for (int i = 0; i < batchSize; i++) {
                // 复制单个样本
                if (batch.getShape().getDimNum() == 4) {
                    // 图像数据 (batch, height, width, channels)
                    for (int h = 0; h < 32; h++) {
                        for (int w = 0; w < 32; w++) {
                            for (int c = 0; c < 3; c++) {
                                float value = batch.get(i, h, w, c);
                                merged.set(value, offset + i, h, w, c);
                            }
                        }
                    }
                } else if (batch.getShape().getDimNum() == 2) {
                    // 标签数据 (batch, 1)
                    float value = batch.get(i, 0);
                    merged.set(value, offset + i, 0);
                }
            }
            offset += batchSize;
        }
        
        return merged;
    }
    
    /**
     * 获取训练图像
     * @return 训练图像数据 (batch_size, 32, 32, 3)
     */
    public NdArray getTrainImages() {
        return trainImages;
    }
    
    /**
     * 获取训练标签
     * @return 训练标签数据 (batch_size, 1)
     */
    public NdArray getTrainLabels() {
        return trainLabels;
    }
    
    /**
     * 获取测试图像
     * @return 测试图像数据 (batch_size, 32, 32, 3)
     */
    public NdArray getTestImages() {
        return testImages;
    }
    
    /**
     * 获取测试标签
     * @return 测试标签数据 (batch_size, 1)
     */
    public NdArray getTestLabels() {
        return testLabels;
    }
    
    /**
     * 获取类别数量
     * @return 类别数量（10）
     */
    public int getNumClasses() {
        return CLASS_NAMES.length;
    }
    
    /**
     * 获取训练样本数量
     * @return 训练样本数量
     */
    public int getTrainSize() {
        return trainImages != null ? trainImages.getShape().getDimension(0) : 0;
    }
    
    /**
     * 获取测试样本数量
     * @return 测试样本数量
     */
    public int getTestSize() {
        return testImages != null ? testImages.getShape().getDimension(0) : 0;
    }
}