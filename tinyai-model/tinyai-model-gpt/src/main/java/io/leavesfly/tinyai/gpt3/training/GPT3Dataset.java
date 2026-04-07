package io.leavesfly.tinyai.gpt3.training;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * GPT-3数据集实现
 *
 * 支持预训练和微调两种模式的数据加载：
 * - 预训练：将所有文本拼接为超长token序列，连续切分为固定长度样本
 * - 微调（SFT）：每条样本独立处理，Instruction部分屏蔽loss，只在Response部分计算loss
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT3Dataset {

    private final int maxSeqLen;
    private final int batchSize;
    private final int vocabSize;

    // 数据存储
    private List<int[]> samples;
    private List<int[]> lossMasks;       // 微调模式下每个样本的loss掩码
    private boolean finetuneMode;        // 是否为微调模式
    private List<Batch> batches;
    private int currentBatchIndex;

    /**
     * 批次数据结构
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final NdArray lossMask;  // null 表示全部参与 loss 计算
        private final int batchSize;
        private final int seqLen;

        public Batch(NdArray inputIds, NdArray targetIds, int batchSize, int seqLen) {
            this(inputIds, targetIds, null, batchSize, seqLen);
        }

        public Batch(NdArray inputIds, NdArray targetIds, NdArray lossMask, int batchSize, int seqLen) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.lossMask = lossMask;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }

        public NdArray getInputIds()  { return inputIds; }
        public NdArray getTargetIds() { return targetIds; }
        public NdArray getLossMask()  { return lossMask; }
        public int getBatchSize()     { return batchSize; }
        public int getSeqLen()        { return seqLen; }
    }

    /**
     * 构造函数
     *
     * @param maxSeqLen 最大序列长度
     * @param batchSize 批次大小
     * @param vocabSize 词汇表大小
     */
    public GPT3Dataset(int maxSeqLen, int batchSize, int vocabSize) {
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.vocabSize = vocabSize;
        this.samples = new ArrayList<>();
        this.lossMasks = new ArrayList<>();
        this.finetuneMode = false;
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }

    /**
     * 从文件加载文本数据（预训练模式）
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
     * 从文本列表加载预训练数据
     *
     * 主流做法：将所有文本拼接成一个超长token序列，然后按固定长度连续切分。
     * 每个样本满长度，无需padding，数据利用率最高。
     */
    public void loadFromTexts(List<String> texts, SimpleTokenizer tokenizer) {
        samples.clear();
        finetuneMode = false;

        List<Integer> allTokenIds = new ArrayList<>();
        for (String text : texts) {
            if (text == null || text.trim().isEmpty()) {
                continue;
            }
            allTokenIds.addAll(tokenizer.encode(text));
        }

        splitIntoSequences(allTokenIds);

        System.out.println("数据加载完成, 共 " + allTokenIds.size() + " 个token, "
                + samples.size() + " 个训练样本");
    }

    /**
     * 将超长token序列连续切分为训练样本（每条 maxSeqLen+1）
     */
    private void splitIntoSequences(List<Integer> allTokenIds) {
        int totalLen = allTokenIds.size();
        int sampleLen = maxSeqLen + 1;

        if (totalLen < sampleLen) {
            if (totalLen >= 2) {
                int[] sequence = new int[totalLen];
                for (int i = 0; i < totalLen; i++) {
                    sequence[i] = allTokenIds.get(i);
                }
                samples.add(sequence);
            }
            return;
        }

        for (int i = 0; i + sampleLen <= totalLen; i += sampleLen) {
            int[] sequence = new int[sampleLen];
            for (int j = 0; j < sampleLen; j++) {
                sequence[j] = allTokenIds.get(i + j);
            }
            samples.add(sequence);
        }
    }

    /**
     * 从指令-回答格式的文本列表加载微调数据（SFT模式）
     *
     * 行业主流SFT做法：每条样本独立处理，只在Response部分计算loss。
     *
     * @param texts             指令-回答格式的文本列表
     * @param tokenizer         分词器
     * @param responseSeparator Response部分的分隔关键词，如 "Response:"
     */
    public void loadFromInstructionTexts(List<String> texts, SimpleTokenizer tokenizer,
                                          String responseSeparator) {
        samples.clear();
        lossMasks.clear();
        finetuneMode = true;

        int totalTokens = 0;
        int skippedCount = 0;

        for (String text : texts) {
            if (text == null || text.trim().isEmpty()) {
                continue;
            }

            int separatorIndex = text.indexOf(responseSeparator);
            if (separatorIndex < 0) {
                skippedCount++;
                continue;
            }

            String instructionPart = text.substring(0, separatorIndex + responseSeparator.length());
            String responsePart    = text.substring(separatorIndex + responseSeparator.length());

            // 分别编码，拼接完整序列
            List<Integer> instructionIds = new ArrayList<>(tokenizer.encode(instructionPart));
            // encode自动加<BOS>和<EOS>，去掉末尾EOS（因为response紧接）
            instructionIds = instructionIds.subList(0, instructionIds.size() - 1);

            List<Integer> responseIds = new ArrayList<>(tokenizer.encode(responsePart));
            // encode自动加<BOS>和<EOS>，去掉开头BOS
            responseIds = responseIds.subList(1, responseIds.size());

            List<Integer> fullIds = new ArrayList<>();
            fullIds.addAll(instructionIds);
            fullIds.addAll(responseIds);

            int instructionLen = instructionIds.size();
            int totalLen = fullIds.size();

            int sampleLen = maxSeqLen + 1;
            if (totalLen > sampleLen) {
                fullIds = fullIds.subList(0, sampleLen);
                totalLen = sampleLen;
            }

            if (totalLen < 2) {
                skippedCount++;
                continue;
            }

            int[] sequence = new int[totalLen];
            for (int i = 0; i < totalLen; i++) {
                sequence[i] = fullIds.get(i);
            }
            samples.add(sequence);

            // loss掩码：response部分参与loss计算（mask=1），instruction部分屏蔽（mask=0）
            int targetLen = totalLen - 1;
            int[] mask = new int[targetLen];
            for (int i = 0; i < targetLen; i++) {
                mask[i] = (i + 1 >= instructionLen) ? 1 : 0;
            }
            lossMasks.add(mask);

            totalTokens += totalLen;
        }

        if (skippedCount > 0) {
            System.out.println("警告: 跳过了 " + skippedCount + " 条无效数据(缺少分隔符'" + responseSeparator + "')");
        }
        System.out.println("微调数据加载完成, 共 " + totalTokens + " 个token, "
                + samples.size() + " 个训练样本");
    }

    /**
     * 准备批次数据
     *
     * @param shuffle 是否打乱数据
     */
    public void prepare(boolean shuffle) {
        if (shuffle) {
            if (finetuneMode && !lossMasks.isEmpty()) {
                long seed = System.currentTimeMillis();
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < samples.size(); i++) {
                    indices.add(i);
                }
                Collections.shuffle(indices, new Random(seed));

                List<int[]> shuffledSamples = new ArrayList<>();
                List<int[]> shuffledMasks   = new ArrayList<>();
                for (int idx : indices) {
                    shuffledSamples.add(samples.get(idx));
                    shuffledMasks.add(lossMasks.get(idx));
                }
                samples   = shuffledSamples;
                lossMasks = shuffledMasks;
            } else {
                Collections.shuffle(samples, new Random(System.currentTimeMillis()));
            }
        }

        batches.clear();
        currentBatchIndex = 0;

        for (int i = 0; i < samples.size(); i += batchSize) {
            int end = Math.min(i + batchSize, samples.size());
            List<int[]> batchSamples = samples.subList(i, end);
            List<int[]> batchMasks   = finetuneMode ? lossMasks.subList(i, end) : null;
            batches.add(createBatch(batchSamples, batchMasks));
        }
    }

    /**
     * 创建单个批次
     */
    private Batch createBatch(List<int[]> batchSamples, List<int[]> batchMasks) {
        int actualBatchSize = batchSamples.size();
        int seqLen = maxSeqLen;

        int[][] inputData  = new int[actualBatchSize][seqLen];
        int[][] targetData = new int[actualBatchSize][seqLen];
        int[][] maskData   = (batchMasks != null) ? new int[actualBatchSize][seqLen] : null;

        for (int i = 0; i < actualBatchSize; i++) {
            int[] sample   = batchSamples.get(i);
            int validLen   = Math.min(sample.length - 1, seqLen);

            System.arraycopy(sample, 0, inputData[i],  0, validLen);
            System.arraycopy(sample, 1, targetData[i], 0, validLen);

            for (int j = validLen; j < seqLen; j++) {
                inputData[i][j]  = 0;
                targetData[i][j] = 0;
            }

            if (maskData != null && batchMasks != null) {
                int[] sampleMask   = batchMasks.get(i);
                int maskValidLen   = Math.min(sampleMask.length, seqLen);
                System.arraycopy(sampleMask, 0, maskData[i], 0, maskValidLen);
                for (int j = maskValidLen; j < seqLen; j++) {
                    maskData[i][j] = 0;
                }
            }
        }

        NdArray inputArray  = createNdArray(inputData,  actualBatchSize, seqLen);
        NdArray targetArray = createNdArray(targetData, actualBatchSize, seqLen);
        NdArray maskArray   = (maskData != null) ? createNdArray(maskData, actualBatchSize, seqLen) : null;

        return new Batch(inputArray, targetArray, maskArray, actualBatchSize, seqLen);
    }

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

    public boolean hasNext() {
        return currentBatchIndex < batches.size();
    }

    public Batch nextBatch() {
        if (!hasNext()) return null;
        return batches.get(currentBatchIndex++);
    }

    public void reset() {
        currentBatchIndex = 0;
    }

    public int getSampleCount() {
        return samples.size();
    }

    public int getBatchCount() {
        return batches.size();
    }

    // ==================== 简化分词器 ====================

    /**
     * 简化的分词器实现（字符级/词级，实际应使用BPE）
     */
    public static class SimpleTokenizer {
        private final Map<String, Integer> word2idx;
        private final Map<Integer, String> idx2word;
        private int nextId;
        private boolean frozen;

        public SimpleTokenizer() {
            this.word2idx = new HashMap<>();
            this.idx2word = new HashMap<>();
            this.nextId   = 0;
            this.frozen   = false;

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

        public void freeze()   { this.frozen = true; }
        public void unfreeze() { this.frozen = false; }

        public List<Integer> encode(String text) {
            List<Integer> ids = new ArrayList<>();
            ids.add(word2idx.get("<BOS>"));

            String[] words = text.toLowerCase().split("\\s+");
            for (String word : words) {
                if (word.isEmpty()) continue;
                if (!word2idx.containsKey(word)) {
                    if (!frozen) {
                        addToken(word);
                    } else {
                        ids.add(word2idx.get("<UNK>"));
                        continue;
                    }
                }
                ids.add(word2idx.get(word));
            }

            ids.add(word2idx.get("<EOS>"));
            return ids;
        }

        public String decode(int[] ids) {
            StringBuilder sb = new StringBuilder();
            for (int id : ids) {
                String token = idx2word.getOrDefault(id, "<" + id + ">");
                if (!token.startsWith("<") && sb.length() > 0) {
                    sb.append(" ");
                }
                if (!token.equals("<PAD>") && !token.equals("<BOS>") && !token.equals("<EOS>") && !token.equals("<UNK>")) {
                    sb.append(token);
                }
            }
            return sb.toString().trim();
        }

        public int getVocabSize() {
            return nextId;
        }
    }
}
