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

    /** 默认词汇表大小，与 {@link io.leavesfly.tinyai.banana.config.BananaConfig#getVocabSize()} 的默认值保持一致。 */
    public static final int DEFAULT_VOCAB_SIZE = 32000;

    /** PAD token 统一为 0，tokenize 时会把正常 token 映射到 {@code [1, vocabSize)} 以避开 PAD。 */
    public static final int PAD_TOKEN_ID = 0;

    private final int maxTextLen;    // 最大文本长度
    private final int imageSize;     // 图像大小
    private final int batchSize;
    private final int vocabSize;     // 词汇表大小，用于约束 tokenize 的取值范围

    // 存储所有训练样本
    private List<Sample> samples;

    // 批次数据
    private List<Batch> batches;
    private int currentBatchIndex;

    /**
     * 兼容旧签名的构造函数，使用 {@link #DEFAULT_VOCAB_SIZE} 作为默认词汇表大小。
     *
     * @param maxTextLen 最大文本长度
     * @param imageSize  图像大小(正方形)
     * @param batchSize  批次大小
     */
    public BananaDataset(int maxTextLen, int imageSize, int batchSize) {
        this(maxTextLen, imageSize, batchSize, DEFAULT_VOCAB_SIZE);
    }

    /**
     * 完整构造函数。
     *
     * @param maxTextLen 最大文本长度
     * @param imageSize  图像大小(正方形)
     * @param batchSize  批次大小
     * @param vocabSize  词汇表大小（必须与 {@code BananaConfig.vocabSize} 一致，用于约束 token id 范围）
     */
    public BananaDataset(int maxTextLen, int imageSize, int batchSize, int vocabSize) {
        if (maxTextLen <= 0 || imageSize <= 0 || batchSize <= 0 || vocabSize <= 1) {
            throw new IllegalArgumentException(String.format(
                    "非法参数: maxTextLen=%d, imageSize=%d, batchSize=%d, vocabSize=%d",
                    maxTextLen, imageSize, batchSize, vocabSize));
        }
        this.maxTextLen = maxTextLen;
        this.imageSize = imageSize;
        this.batchSize = batchSize;
        this.vocabSize = vocabSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从CSV文件加载文本-图像对数据。
     *
     * <p>文件格式：每行 {@code text,image_path}，首行为标题行将被跳过。
     * 第一行起始处的 UTF-8 BOM（{@code \uFEFF}）会被自动剔除；
     * 对于字段数不足 2、text 为空或 image_path 为空的行，将被跳过并记入 {@code badLineCount}。</p>
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
        int badLineCount = 0;

        // 跳过标题行
        for (int i = 1; i < lines.size(); i++) {
            String rawLine = lines.get(i);
            if (rawLine == null) {
                continue;
            }
            // 剔除首行可能带的 UTF-8 BOM
            if (i == 1 && !rawLine.isEmpty() && rawLine.charAt(0) == '\uFEFF') {
                rawLine = rawLine.substring(1);
            }
            String line = rawLine.trim();
            if (line.isEmpty()) {
                continue;
            }

            String[] parts = line.split(",", 2);
            if (parts.length != 2) {
                badLineCount++;
                continue;
            }

            String text = parts[0].trim();
            String imagePath = parts[1].trim();
            if (text.isEmpty() || imagePath.isEmpty()) {
                badLineCount++;
                continue;
            }

            // 模拟图像加载(实际应用中需要真实加载图像)
            float[] imageData = simulateImageLoad(imagePath);

            // 简单分词(实际应用中应使用真实Tokenizer)
            int[] textTokens = simpleTokenize(text);

            samples.add(new Sample(textTokens, imageData, text, imagePath));
        }

        if (badLineCount > 0) {
            System.err.println("警告: CSV 中有 " + badLineCount + " 行格式非法已跳过（字段数不足或空字段）: " + filePath);
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

            // 填充文本：超出 vocabSize 的 token 会被强制夹到 [0, vocabSize-1]，防止 Embedding 查表越界
            int[] tokens = sample.getTextTokens();
            int copyLen = Math.min(tokens.length, maxTextLen);
            for (int j = 0; j < copyLen; j++) {
                int tok = tokens[j];
                if (tok < 0 || tok >= vocabSize) {
                    tok = PAD_TOKEN_ID;
                }
                textData[i][j] = (float) tok;
            }
            // 剩余位置默认 0 即 PAD_TOKEN_ID
            
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
     * 简单分词（仅用于演示；生产中应使用真实 Tokenizer）。
     *
     * <p>映射规则：将 hash 取模到 {@code [1, vocabSize)}，以避开 {@link #PAD_TOKEN_ID}（0）；
     * 这样 {@code createBatch} 中未填满位置的 0 能被模型明确识别为 PAD。</p>
     *
     * @param text 输入文本
     * @return token id 数组，长度不超过 {@link #maxTextLen}
     */
    private int[] simpleTokenize(String text) {
        String[] words = text.toLowerCase().split("\\s+");
        int effectiveLen = Math.min(words.length, maxTextLen);
        int[] tokens = new int[effectiveLen];
        // 用 vocabSize-1 而非 vocabSize 作为模，结果落在 [0, vocabSize-1)；再 +1 得到 [1, vocabSize)。
        // 注意：使用 & 0x7FFFFFFF 而非 Math.abs(hashCode())，因为 Math.abs(Integer.MIN_VALUE)
        // 仍返回 Integer.MIN_VALUE（负数），会导致 % 得到负数或后续 +1 得到 0（PAD 冲突）。
        int modRange = vocabSize - 1;
        for (int i = 0; i < effectiveLen; i++) {
            int h = (words[i].hashCode() & 0x7FFFFFFF) % modRange;
            tokens[i] = h + 1;
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