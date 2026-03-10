package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * GPT-1数据集实现
 * 
 * 支持预训练和微调两种模式的数据加载
 * 实现因果语言建模的数据处理
 * 
 * @author TinyAI
 * @since 2024
 */
public class GPT1Dataset {
    
    private final int maxSeqLen;
    private final int batchSize;
    private final int vocabSize;
    
    // 数据存储
    private List<int[]> samples;
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * 批次数据结构
     */
    public static class Batch {
        private final NdArray inputIds;   // 输入token序列
        private final NdArray targetIds;  // 目标token序列(右移1位)
        private final int batchSize;
        private final int seqLen;
        
        public Batch(NdArray inputIds, NdArray targetIds, int batchSize, int seqLen) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
        
        public NdArray getInputIds() { return inputIds; }
        public NdArray getTargetIds() { return targetIds; }
        public int getBatchSize() { return batchSize; }
        public int getSeqLen() { return seqLen; }
    }
    
    /**
     * 构造函数
     * 
     * @param maxSeqLen 最大序列长度
     * @param batchSize 批次大小
     * @param vocabSize 词汇表大小
     */
    public GPT1Dataset(int maxSeqLen, int batchSize, int vocabSize) {
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.vocabSize = vocabSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从文件加载文本数据
     * 
     * @param filePath 文件路径
     * @param tokenizer 分词器(简化版,实际应使用BPE)
     * @throws IOException 文件读取异常
     */
    public void loadFromFile(String filePath, SimpleTokenizer tokenizer) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        loadFromTexts(lines, tokenizer);
    }
    
    /**
     * 从文本列表加载数据
     * 
     * 主流做法：将所有文本拼接成一个超长token序列，然后按固定长度连续切分。
     * 这样每个样本都是满长度的，无需padding，数据利用率最高。
     * 文档之间用EOS token分隔，模型可以学到文档边界。
     * 
     * @param texts 文本列表
     * @param tokenizer 分词器
     */
    public void loadFromTexts(List<String> texts, SimpleTokenizer tokenizer) {
        samples.clear();
        
        // 第一步：将所有文本编码后拼接成一个超长token序列
        List<Integer> allTokenIds = new ArrayList<>();
        for (String text : texts) {
            if (text == null || text.trim().isEmpty()) {
                continue;
            }
            List<Integer> tokenIds = tokenizer.encode(text);
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
        int sampleLen = maxSeqLen + 1; // 每个样本需要 maxSeqLen+1 个token
        
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
        if (shuffle) {
            Collections.shuffle(samples, new Random(System.currentTimeMillis()));
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        // 分批处理
        for (int i = 0; i < samples.size(); i += batchSize) {
            int end = Math.min(i + batchSize, samples.size());
            List<int[]> batchSamples = samples.subList(i, end);
            
            Batch batch = createBatch(batchSamples);
            batches.add(batch);
        }
        
        System.out.println("批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 创建单个批次
     * 
     * 由于样本已通过拼接+连续切分生成，绝大多数样本长度固定为 maxSeqLen+1，
     * 无需大量padding。对每个样本取前 maxSeqLen 个token作为输入，
     * 取后 maxSeqLen 个token（右移一位）作为目标。
     * 
     * @param batchSamples 批次样本列表
     * @return 批次对象
     */
    private Batch createBatch(List<int[]> batchSamples) {
        int actualBatchSize = batchSamples.size();
        int seqLen = maxSeqLen;
        
        // 初始化数组
        int[][] inputData = new int[actualBatchSize][seqLen];
        int[][] targetData = new int[actualBatchSize][seqLen];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int[] sample = batchSamples.get(i);
            int validLen = Math.min(sample.length - 1, seqLen);
            
            // 输入: sample[0 .. validLen-1]
            System.arraycopy(sample, 0, inputData[i], 0, validLen);
            // 目标: sample[1 .. validLen] (右移一位)
            System.arraycopy(sample, 1, targetData[i], 0, validLen);
            
            // 仅在极端降级情况下（数据不足一个完整样本）才需要padding
            for (int j = validLen; j < seqLen; j++) {
                inputData[i][j] = 0;
                targetData[i][j] = 0;
            }
        }
        
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
                flatData[idx++] = data[i][j];
            }
        }
        return NdArray.of(flatData, Shape.of(batchSize, seqLen));
    }
    
    /**
     * 是否有下一个批次
     */
    public boolean hasNext() {
        return currentBatchIndex < batches.size();
    }
    
    /**
     * 获取下一个批次
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            return null;
        }
        return batches.get(currentBatchIndex++);
    }
    
    /**
     * 重置迭代器
     */
    public void reset() {
        currentBatchIndex = 0;
    }
    
    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return batches.size();
    }
    
    /**
     * 简化的分词器实现
     * 实际应用应使用BPE或WordPiece
     */
    public static class SimpleTokenizer {
        private final Map<String, Integer> word2idx;
        private final Map<Integer, String> idx2word;
        private int nextId;
        private boolean frozen;  // 词汇表是否冻结
        
        public SimpleTokenizer() {
            this.word2idx = new HashMap<>();
            this.idx2word = new HashMap<>();
            this.nextId = 0;
            this.frozen = false;
            
            // 添加特殊token
            addToken("<PAD>");
            addToken("<UNK>");
            addToken("<BOS>");
            addToken("<EOS>");
        }
        
        private void addToken(String token) {
            if (!word2idx.containsKey(token)) {
                word2idx.put(token, nextId);
                idx2word.put(nextId, token);
                nextId++;
            }
        }
        
        /**
         * 冻结词汇表,不再添加新词
         */
        public void freeze() {
            this.frozen = true;
        }
        
        /**
         * 解冻词汇表,允许添加新词
         */
        public void unfreeze() {
            this.frozen = false;
        }
        
        /**
         * 编码文本
         */
        public List<Integer> encode(String text) {
            List<Integer> ids = new ArrayList<>();
            ids.add(word2idx.get("<BOS>"));
            
            // 简单的空格分词
            String[] words = text.toLowerCase().split("\\s+");
            for (String word : words) {
                if (word.isEmpty()) continue;
                
                // 如果词汇表未冻结且词不存在,则添加新词
                if (!word2idx.containsKey(word)) {
                    if (!frozen) {
                        addToken(word);
                    } else {
                        // 词汇表已冻结,使用<UNK>
                        ids.add(word2idx.get("<UNK>"));
                        continue;
                    }
                }
                ids.add(word2idx.get(word));
            }
            
            ids.add(word2idx.get("<EOS>"));
            return ids;
        }
        
        /**
         * 解码ID序列
         */
        public String decode(int[] ids) {
            StringBuilder sb = new StringBuilder();
            for (int id : ids) {
                String token;
                if (idx2word.containsKey(id)) {
                    token = idx2word.get(id);
                } else {
                    token = "<" + id + ">";  // 显示未知token的id
                }
                
                if (!token.startsWith("<") && sb.length() > 0) {
                    sb.append(" ");
                }
                if (!token.equals("<PAD>") && !token.equals("<BOS>") && !token.equals("<EOS>")) {
                    sb.append(token);
                }
            }
            return sb.toString().trim();
        }
        
        /**
         * 获取词汇表大小
         */
        public int getVocabSize() {
            return nextId;
        }
    }
}
