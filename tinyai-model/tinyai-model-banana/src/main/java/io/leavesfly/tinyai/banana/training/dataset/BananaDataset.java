package io.leavesfly.tinyai.banana.training.dataset;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Banana多模态数据集
 * 
 * 负责加载文本-图像配对数据，用于多模态预训练
 * 支持以下任务:
 * - 文本到图像生成
 * - 图像描述生成
 * - 多模态对比学习
 * 
 * @author TinyAI
 * @since 2024
 */
public class BananaDataset implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final int maxTextLen;    // 最大文本长度
    private final int imageSize;     // 图像大小
    private final int batchSize;
    
    // 存储所有训练样本
    private List<Sample> samples;
    
    // 批次数据
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * 构造函数
     * 
     * @param maxTextLen 最大文本长度
     * @param imageSize 图像大小(正方形)
     * @param batchSize 批次大小
     */
    public BananaDataset(int maxTextLen, int imageSize, int batchSize) {
        this.maxTextLen = maxTextLen;
        this.imageSize = imageSize;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从CSV文件加载文本-图像对数据
     * 
     * 文件格式: text,image_path
     * 
     * @param filePath CSV文件路径
     * @throws IOException IO异常
     */
    public void loadFromCSV(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        
        // 跳过标题行
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            
            String[] parts = line.split(",", 2);
            if (parts.length != 2) {
                continue;
            }
            
            String text = parts[0].trim();
            String imagePath = parts[1].trim();
            
            // 模拟图像加载(实际应用中需要真实加载图像)
            float[] imageData = simulateImageLoad(imagePath);
            
            // 简单分词(实际应用中应使用真实Tokenizer)
            int[] textTokens = simpleTokenize(text);
            
            samples.add(new Sample(textTokens, imageData, text, imagePath));
        }
        
        if (samples.isEmpty()) {
            System.err.println("警告: 未从文件中加载到有效样本: " + filePath);
        }
        
        System.out.println("数据加载完成,共 " + samples.size() + " 个训练样本");
    }
    
    /**
     * 从合成数据加载(用于演示和测试)
     * 
     * @param sampleCount 样本数量
     */
    public void loadSyntheticData(int sampleCount) {
        if (sampleCount <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0: " + sampleCount);
        }
        
        samples.clear();
        Random random = new Random(42);
        
        String[] templates = {
            "A photo of a cat",
            "A beautiful landscape",
            "A modern building",
            "A portrait of a person",
            "Abstract art with colors"
        };
        
        for (int i = 0; i < sampleCount; i++) {
            String text = templates[random.nextInt(templates.length)];
            int[] textTokens = simpleTokenize(text);
            
            // 生成随机图像数据
            float[] imageData = new float[3 * imageSize * imageSize];
            for (int j = 0; j < imageData.length; j++) {
                imageData[j] = random.nextFloat();
            }
            
            samples.add(new Sample(textTokens, imageData, text, "synthetic_" + i));
        }
        
        System.out.println("合成数据生成完成,共 " + samples.size() + " 个样本");
    }
    
    /**
     * 准备批次数据
     * 
     * @param shuffle 是否打乱数据
     */
    public void prepare(boolean shuffle) {
        if (samples.isEmpty()) {
            throw new IllegalStateException("数据集为空,请先加载数据");
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        // 打乱样本
        List<Sample> workingSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(workingSamples);
        }
        
        // 创建批次
        for (int i = 0; i < workingSamples.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, workingSamples.size());
            List<Sample> batchSamples = workingSamples.subList(i, endIdx);
            batches.add(createBatch(batchSamples));
        }
        
        System.out.println("批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 创建单个批次
     * 
     * @param batchSamples 批次样本列表
     * @return 批次对象
     */
    private Batch createBatch(List<Sample> batchSamples) {
        int actualBatchSize = batchSamples.size();
        
        // 文本数据: [batchSize, maxTextLen]
        float[][] textData = new float[actualBatchSize][maxTextLen];
        
        // 图像数据: [batchSize, channels, height, width]
        float[][][][] imageData = new float[actualBatchSize][3][imageSize][imageSize];
        
        for (int i = 0; i < actualBatchSize; i++) {
            Sample sample = batchSamples.get(i);
            
            // 填充文本
            int[] tokens = sample.getTextTokens();
            for (int j = 0; j < Math.min(tokens.length, maxTextLen); j++) {
                textData[i][j] = (float) tokens[j];
            }
            // 剩余位置用0填充(PAD)
            
            // 填充图像
            float[] image = sample.getImageData();
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < imageSize; h++) {
                    for (int w = 0; w < imageSize; w++) {
                        int idx = c * imageSize * imageSize + h * imageSize + w;
                        imageData[i][c][h][w] = image[idx];
                    }
                }
            }
        }
        
        // 转换为NdArray
        NdArray textArray = createTextNdArray(textData, actualBatchSize);
        NdArray imageArray = createImageNdArray(imageData, actualBatchSize);
        
        return new Batch(textArray, imageArray, actualBatchSize);
    }
    
    /**
     * 创建文本NdArray
     */
    private NdArray createTextNdArray(float[][] data, int batchSize) {
        float[] flatData = new float[batchSize * maxTextLen];
        int idx = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < maxTextLen; j++) {
                flatData[idx++] = data[i][j];
            }
        }
        return NdArray.of(flatData, Shape.of(batchSize, maxTextLen));
    }
    
    /**
     * 创建图像NdArray
     */
    private NdArray createImageNdArray(float[][][][] data, int batchSize) {
        int totalSize = batchSize * 3 * imageSize * imageSize;
        float[] flatData = new float[totalSize];
        int idx = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < imageSize; h++) {
                    for (int w = 0; w < imageSize; w++) {
                        flatData[idx++] = data[i][c][h][w];
                    }
                }
            }
        }
        return NdArray.of(flatData, Shape.of(batchSize, 3, imageSize, imageSize));
    }
    
    /**
     * 模拟图像加载
     */
    private float[] simulateImageLoad(String imagePath) {
        Random random = new Random(imagePath.hashCode());
        float[] data = new float[3 * imageSize * imageSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextFloat();
        }
        return data;
    }
    
    /**
     * 简单分词(实际应使用真实Tokenizer)
     */
    private int[] simpleTokenize(String text) {
        String[] words = text.toLowerCase().split("\\s+");
        int[] tokens = new int[Math.min(words.length, maxTextLen)];
        for (int i = 0; i < tokens.length; i++) {
            // 确保token ID为非负数且在词汇表范围内
            // 使用Math.abs避免负数,然后取模确保在范围内
            tokens[i] = Math.abs(words[i].hashCode()) % 10000;
        }
        return tokens;
    }
    
    /**
     * 是否还有下一个批次
     */
    public boolean hasNextBatch() {
        return currentBatchIndex < batches.size();
    }
    
    /**
     * 获取下一个批次
     */
    public Batch getNextBatch() {
        if (!hasNextBatch()) {
            throw new NoSuchElementException("没有更多批次数据");
        }
        return batches.get(currentBatchIndex++);
    }
    
    /**
     * 重置批次索引
     */
    public void reset() {
        currentBatchIndex = 0;
    }
    
    /**
     * 获取批次总数
     */
    public int getBatchCount() {
        return batches.size();
    }
    
    /**
     * 获取样本总数
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 训练样本类
     */
    public static class Sample {
        private final int[] textTokens;
        private final float[] imageData;
        private final String text;
        private final String imagePath;
        
        public Sample(int[] textTokens, float[] imageData, String text, String imagePath) {
            this.textTokens = textTokens;
            this.imageData = imageData;
            this.text = text;
            this.imagePath = imagePath;
        }
        
        public int[] getTextTokens() {
            return textTokens;
        }
        
        public float[] getImageData() {
            return imageData;
        }
        
        public String getText() {
            return text;
        }
        
        public String getImagePath() {
            return imagePath;
        }
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray textInput;   // [batchSize, maxTextLen]
        private final NdArray imageInput;  // [batchSize, 3, imageSize, imageSize]
        private final int batchSize;
        
        public Batch(NdArray textInput, NdArray imageInput, int batchSize) {
            this.textInput = textInput;
            this.imageInput = imageInput;
            this.batchSize = batchSize;
        }
        
        public NdArray getTextInput() {
            return textInput;
        }
        
        public NdArray getImageInput() {
            return imageInput;
        }
        
        public int getBatchSize() {
            return batchSize;
        }
    }
}