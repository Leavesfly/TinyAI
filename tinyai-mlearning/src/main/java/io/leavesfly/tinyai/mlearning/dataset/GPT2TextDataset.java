//package io.leavesfly.tinyai.mlearning.dataset;
//
//import io.leavesfly.tinyai.modality.nlp.SimpleTokenizer;
//import io.leavesfly.tinyai.ndarr.NdArray;
//import io.leavesfly.tinyai.ndarr.Shape;
//
//import java.util.*;
//
///**
// * GPT-2文本数据集
// *
// * 专门为GPT-2语言模型设计的数据集类，支持：
// * 1. 文本序列的自回归训练数据生成
// * 2. 输入序列和目标序列的准备
// * 3. 批次数据的生成和管理
// *
// * @author TinyDL
// * @version 1.0
// */
//public class GPT2TextDataset extends DataSet {
//
//    protected String name;
//    protected SimpleTokenizer tokenizer;
//    protected List<String> texts;           // 原始文本列表
//    protected List<int[]> tokenSequences;  // 编码后的token序列
//    protected int maxSeqLength;             // 最大序列长度
//    protected boolean shuffleData;          // 是否打乱数据
//    protected int currentIdx;               // 当前索引
//    protected int dataSize;                 // 数据集大小
//
//    /**
//     * 构造GPT-2文本数据集
//     * @param name 数据集名称
//     * @param texts 原始文本列表
//     * @param tokenizer 分词器
//     * @param maxSeqLength 最大序列长度
//     * @param batchSize 批次大小
//     * @param shuffle 是否打乱数据
//     */
//    public GPT2TextDataset(String name, List<String> texts, SimpleTokenizer tokenizer,
//                          int maxSeqLength, int batchSize, boolean shuffle) {
//        super(batchSize);
//
//        this.name = name;
//        this.texts = new ArrayList<>(texts);
//        this.tokenizer = tokenizer;
//        this.maxSeqLength = maxSeqLength;
//        this.shuffleData = shuffle;
//        this.tokenSequences = new ArrayList<>();
//        this.currentIdx = 0;
//
//        // 预处理文本数据
//        preprocessTexts();
//
//        // 计算数据集大小
//        this.dataSize = tokenSequences.size();
//        System.out.println("GPT-2 Dataset created with " + dataSize + " sequences");
//    }
//
//    /**
//     * 预处理文本数据
//     * 将原始文本转换为token序列，并进行填充或截断处理
//     */
//    private void preprocessTexts() {
//        for (String text : texts) {
//            if (text != null && !text.trim().isEmpty()) {
//                int[] tokens = tokenizer.encode(text, true);
//
//                if (tokens.length > maxSeqLength) {
//                    int stepSize = maxSeqLength / 2;
//                    for (int start = 0; start < tokens.length - maxSeqLength + 1; start += stepSize) {
//                        int[] sequence = new int[maxSeqLength];
//                        System.arraycopy(tokens, start, sequence, 0, maxSeqLength);
//                        tokenSequences.add(sequence);
//                    }
//                } else {
//                    int[] paddedTokens = tokenizer.pad(tokens, maxSeqLength, "post");
//                    tokenSequences.add(paddedTokens);
//                }
//            }
//        }
//
//        if (shuffleData) {
//            Collections.shuffle(tokenSequences);
//        }
//    }
//
//    @Override
//    public List<Batch> getBatches() {
//        List<Batch> batches = new ArrayList<>();
//        int totalBatches = (dataSize + batchSize - 1) / batchSize;
//
//        for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
//            int startIdx = batchIdx * batchSize;
//            int endIdx = Math.min(startIdx + batchSize, dataSize);
//            int actualBatchSize = endIdx - startIdx;
//
//            NdArray inputSequences = new NdArray(new Shape(actualBatchSize, maxSeqLength));
//            NdArray targetSequences = new NdArray(new Shape(actualBatchSize, maxSeqLength));
//
//            for (int i = 0; i < actualBatchSize; i++) {
//                int[] sequence = tokenSequences.get(startIdx + i);
//
//                for (int j = 0; j < maxSeqLength; j++) {
//                    if (j < maxSeqLength - 1) {
//                        inputSequences.set(sequence[j], i, j);
//                        targetSequences.set(sequence[j + 1], i, j);
//                    } else {
//                        inputSequences.set(sequence[j], i, j);
//                        targetSequences.set(sequence[j], i, j);
//                    }
//                }
//            }
//
//            batches.add(new Batch(new NdArray[]{inputSequences}, new NdArray[]{targetSequences}));
//        }
//
//        return batches;
//    }
//
//    @Override
//    public void doPrepare() {
//        // 数据已经在构造函数中预处理了
//    }
//
//    @Override
//    public void shuffle() {
//        Collections.shuffle(tokenSequences);
//    }
//
//    @Override
//    public Map<String, DataSet> splitDataset(float trainRatio, float testRatio, float validationRatio) {
//        Map<String, DataSet> result = new HashMap<>();
//
//        int totalSize = tokenSequences.size();
//        int trainSize = (int) (totalSize * trainRatio);
//        int testSize = (int) (totalSize * testRatio);
//
//        List<int[]> trainSequences = tokenSequences.subList(0, trainSize);
//        List<int[]> testSequences = tokenSequences.subList(trainSize, trainSize + testSize);
//        List<int[]> validSequences = tokenSequences.subList(trainSize + testSize, totalSize);
//
//        // 为了简化，这里直接返回当前数据集
//        result.put(Usage.TRAIN.name(), this);
//        result.put(Usage.TEST.name(), this);
//        result.put(Usage.VALIDATION.name(), this);
//
//        return result;
//    }
//
//    @Override
//    public int getSize() {
//        return dataSize;
//    }
//
//    /**
//     * 创建生成批次
//     * @param prompt 提示文本
//     * @return 生成批次
//     */
//    public Batch createGenerationBatch(String prompt) {
//        int[] tokens = tokenizer.encode(prompt, true);
//        int[] paddedTokens = tokenizer.pad(tokens, maxSeqLength, "post");
//
//        NdArray inputSequence = new NdArray(new Shape(1, maxSeqLength));
//        for (int i = 0; i < maxSeqLength; i++) {
//            inputSequence.set(paddedTokens[i], 0, i);
//        }
//
//        return new Batch(new NdArray[]{inputSequence}, null);
//    }
//
//    /**
//     * 创建示例数据集
//     * @param tokenizer 分词器
//     * @return 示例数据集
//     */
//    public static GPT2TextDataset createSampleDataset(SimpleTokenizer tokenizer) {
//        List<String> sampleTexts = new ArrayList<>();
//        sampleTexts.add("Hello world! This is a simple test.");
//        sampleTexts.add("Machine learning is fascinating and powerful.");
//        sampleTexts.add("GPT models can generate coherent text sequences.");
//        sampleTexts.add("Deep learning revolutionizes artificial intelligence.");
//        sampleTexts.add("Natural language processing enables human-computer interaction.");
//        sampleTexts.add("Transformers have changed the landscape of NLP.");
//        sampleTexts.add("Attention mechanisms help models focus on relevant information.");
//        sampleTexts.add("Language models learn patterns from large text corpora.");
//
//        return new GPT2TextDataset("sample_dataset", sampleTexts, tokenizer, 32, 2, true);
//    }
//
//    // Getters
//    /**
//     * 获取分词器
//     * @return 分词器
//     */
//    public SimpleTokenizer getTokenizer() { return tokenizer; }
//
//    /**
//     * 获取最大序列长度
//     * @return 最大序列长度
//     */
//    public int getMaxSeqLength() { return maxSeqLength; }
//
//    /**
//     * 获取Token序列列表
//     * @return Token序列列表
//     */
//    public List<int[]> getTokenSequences() { return new ArrayList<>(tokenSequences); }
//
//    /**
//     * 获取原始文本列表
//     * @return 原始文本列表
//     */
//    public List<String> getTexts() { return new ArrayList<>(texts); }
//}