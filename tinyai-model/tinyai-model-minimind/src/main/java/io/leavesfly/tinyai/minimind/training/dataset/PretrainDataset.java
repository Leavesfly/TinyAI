package io.leavesfly.tinyai.minimind.training.dataset;

import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 预训练数据集
 * 
 * 负责加载文本数据并转换为模型训练所需的Token序列
 * 支持因果语言建模(Causal Language Modeling)任务
 * 
 * @author leavesfly
 * @since 2024
 */
public class PretrainDataset implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    
    // 存储所有训练样本
    private List<int[]> samples;
    
    // 批次数据
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * 构造函数
     * 
     * @param tokenizer 分词器
     * @param maxSeqLen 最大序列长度
     * @param batchSize 批次大小
     */
    public PretrainDataset(MiniMindTokenizer tokenizer, int maxSeqLen, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从文本文件加载数据
     * 
     * @param filePath 文件路径
     * @throws IOException IO异常
     */
    public void loadFromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        loadFromTexts(lines);
    }
    
    /**
     * 从文本列表加载数据
     * 
     * 主流做法：将所有文本编码后拼接成一个超长token序列，然后按固定长度连续切分。
     * 这样每个样本都是满长度的，无需padding，数据利用率最高。
     * 文档之间用EOS token自然分隔，模型可以学到文档边界。
     * 
     * @param texts 文本列表
     */
    public void loadFromTexts(List<String> texts) {
        samples.clear();
        
        // 第一步：将所有文本编码后拼接成一个超长token序列
        List<Integer> allTokenIds = new ArrayList<>();
        for (String text : texts) {
            if (text == null || text.trim().isEmpty()) {
                continue;
            }
            List<Integer> tokenIds = tokenizer.encode(text, true, true);
            allTokenIds.addAll(tokenIds);
        }
        
        // 第二步：按固定长度连续切分为训练样本
        splitIntoSequences(allTokenIds);
        
        System.out.println("数据加载完成,共 " + allTokenIds.size() + " 个token, "
                + samples.size() + " 个训练样本");
    }
    
    /**
     * 将拼接后的超长token序列按固定长度连续切分为训练样本
     * 
     * 每个样本长度为 maxSeqLen+1，其中前 maxSeqLen 个token作为输入，
     * 后 maxSeqLen 个token作为目标（右移一位）。
     * 末尾不足 maxSeqLen+1 的部分直接丢弃，避免引入padding。
     * 
     * @param allTokenIds 拼接后的完整Token ID列表
     */
    private void splitIntoSequences(List<Integer> allTokenIds) {
        int totalLen = allTokenIds.size();
        int sampleLen = maxSeqLen + 1;
        
        if (totalLen < sampleLen) {
            // 数据量太少，无法构建完整样本，降级为单个短样本
            if (totalLen >= 2) {
                int[] sequence = new int[totalLen];
                for (int i = 0; i < totalLen; i++) {
                    sequence[i] = allTokenIds.get(i);
                }
                samples.add(sequence);
            }
            return;
        }
        
        // 连续切分，不重叠，丢弃末尾不足一个完整样本的部分
        for (int i = 0; i + sampleLen <= totalLen; i += sampleLen) {
            int[] sequence = new int[sampleLen];
            for (int j = 0; j < sampleLen; j++) {
                sequence[j] = allTokenIds.get(i + j);
            }
            samples.add(sequence);
        }
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
        List<int[]> workingSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(workingSamples);
        }
        
        // 创建批次
        for (int i = 0; i < workingSamples.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, workingSamples.size());
            List<int[]> batchSamples = workingSamples.subList(i, endIdx);
            batches.add(createBatch(batchSamples));
        }
        
        System.out.println("批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 创建单个批次
     * 
     * 样本已通过拼接+连续切分生成，长度固定为 maxSeqLen+1，无需padding。
     * 
     * @param batchSamples 批次样本列表
     * @return 批次对象
     */
    private Batch createBatch(List<int[]> batchSamples) {
        int actualBatchSize = batchSamples.size();
        int seqLen = maxSeqLen;
        
        // 输入数据: [batchSize, seqLen]
        int[][] inputData = new int[actualBatchSize][seqLen];
        // 目标数据: [batchSize, seqLen]
        int[][] targetData = new int[actualBatchSize][seqLen];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int[] sample = batchSamples.get(i);
            int validLen = Math.min(sample.length - 1, seqLen);
            
            // 输入: sample[0 .. validLen-1]
            System.arraycopy(sample, 0, inputData[i], 0, validLen);
            // 目标: sample[1 .. validLen] (右移一位)
            System.arraycopy(sample, 1, targetData[i], 0, validLen);
            
            // padding部分填0（仅在数据量不足时的降级短样本才会触发）
            for (int j = validLen; j < seqLen; j++) {
                inputData[i][j] = 0;
                targetData[i][j] = 0;
            }
        }
        
        // 转换为NdArray
        NdArray inputArray = createNdArray(inputData, actualBatchSize, seqLen);
        NdArray targetArray = createNdArray(targetData, actualBatchSize, seqLen);
        
        return new Batch(inputArray, targetArray, actualBatchSize, seqLen);
    }
    
    /**
     * 创建NdArray
     * 
     * @param data 二维整数数组
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @return NdArray对象
     */
    private NdArray createNdArray(int[][] data, int batchSize, int seqLen) {
        float[] flatData = new float[batchSize * seqLen];
        int idx = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                flatData[idx++] = (float) data[i][j];
            }
        }
        return NdArray.of(flatData, Shape.of(batchSize, seqLen));
    }
    
    /**
     * 是否还有下一个批次
     * 
     * @return true如果还有批次
     */
    public boolean hasNextBatch() {
        return currentBatchIndex < batches.size();
    }
    
    /**
     * 获取下一个批次
     * 
     * @return 批次对象
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
     * 
     * @return 批次数量
     */
    public int getBatchCount() {
        return batches.size();
    }
    
    /**
     * 获取样本总数
     * 
     * @return 样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray input;      // [batchSize, seqLen]
        private final NdArray target;     // [batchSize, seqLen]
        private final int batchSize;
        private final int seqLen;
        
        public Batch(NdArray input, NdArray target, int batchSize, int seqLen) {
            this.input = input;
            this.target = target;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
        
        public NdArray getInput() {
            return input;
        }
        
        public NdArray getTarget() {
            return target;
        }
        
        public int getBatchSize() {
            return batchSize;
        }
        
        public int getSeqLen() {
            return seqLen;
        }
    }
}
