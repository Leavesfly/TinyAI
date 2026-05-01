package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-R1 RLVR数据集
 * 
 * RLVR (Reinforcement Learning from Verifiable Rewards) 数据集
 * 继承 DeepSeekBaseDataset，复用通用的批次管理和索引打乱逻辑。
 * 
 * 与RLHF数据集的区别:
 * - RLHF: 需要人工标注的奖励分数 (0-1连续值)
 * - RLVR: 通过验证器自动获取奖励 (0或1二值)
 * 
 * 注意：RLVR 是强化学习场景，不做 next-token prediction，
 * 因此不使用基类的 createInputTargetData，而是直接将 tokenIds 作为输入。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1RLVRDataset extends DeepSeekBaseDataset<DeepSeekR1RLVRDataset.Batch> {

    /**
     * 词汇表上限（token ID 范围：[0, VOCAB_LIMIT)）
     *
     * <p><b>重要约束：</b>使用本数据集训练的 DeepSeekR1Model 必须满足
     * {@code config.getVocabSize() >= VOCAB_LIMIT}，否则 token embedding 会越界访问。
     * <br>可以通过 {@link #requireCompatibleVocabSize(int)} 做静态校验。
     */
    public static final int VOCAB_LIMIT = 1000;

    /** 单条样本最多保留的 token 数（截断长度） */
    private static final int MAX_TOKENS_PER_SAMPLE = 100;

    /**
     * 词表：word → token id。
     * 自动增长，id 从 1 开始分配，0 保留给 &lt;pad&gt;/未知，
     * 超过 {@link #VOCAB_LIMIT}-1 后退化为 hash 取模（保证不 OOM）。
     */
    private final Map<String, Integer> vocab = new HashMap<>();

    /** 下一个可用的 token id（从 1 开始，0 保留） */
    private int nextTokenId = 1;

    private final List<String> questions;       // 问题文本
    private final List<String> groundTruths;    // 标准答案
    private final List<String> verifierTypes;   // 验证器类型
    
    /**
     * 构造函数
     * 
     * @param batchSize 批次大小
     * @param maxSeqLen 最大序列长度
     */
    public DeepSeekR1RLVRDataset(int batchSize, int maxSeqLen) {
        super(new ArrayList<>(), maxSeqLen, batchSize, true);
        this.questions = new ArrayList<>();
        this.groundTruths = new ArrayList<>();
        this.verifierTypes = new ArrayList<>();
    }
    
    /**
     * 添加样本
     * 
     * @param question 问题
     * @param groundTruth 标准答案
     * @param verifierType 验证器类型 ("math", "code", "logic")
     */
    public void addSample(String question, String groundTruth, String verifierType) {
        int[] tokenIds = simpleTokenize(question);
        sequences.add(tokenIds);
        questions.add(question);
        groundTruths.add(groundTruth);
        verifierTypes.add(verifierType);
    }
    
    /**
     * 添加样本（带Token IDs）
     * 
     * @param tokenIds Token ID数组
     * @param question 问题文本
     * @param groundTruth 标准答案
     * @param verifierType 验证器类型
     */
    public void addSample(int[] tokenIds, String question, String groundTruth, String verifierType) {
        sequences.add(tokenIds);
        questions.add(question);
        groundTruths.add(groundTruth);
        verifierTypes.add(verifierType);
    }
    
    /**
     * 准备数据集
     * 
     * 重写基类的 prepare 方法，在打乱前重新初始化索引列表，
     * 因为 RLVR 支持动态添加样本。
     * 
     * @param shouldShuffle 是否打乱顺序
     */
    @Override
    public void prepare(boolean shouldShuffle) {
        // 重新初始化索引列表（因为可能动态添加了新样本）
        indices.clear();
        for (int i = 0; i < sequences.size(); i++) {
            indices.add(i);
        }
        super.prepare(shouldShuffle);
    }
    
    /**
     * 获取下一批次
     * 
     * RLVR 不做 next-token prediction，直接将 tokenIds 作为输入，
     * 附带 question、groundTruth、verifierType 元数据。
     */
    @Override
    public Batch nextBatch() {
        int actualBatchSize = calculateActualBatchSize();
        
        float[][] inputIds = new float[actualBatchSize][maxSeqLength];
        String[] batchQuestions = new String[actualBatchSize];
        String[] batchGroundTruths = new String[actualBatchSize];
        String[] batchVerifierTypes = new String[actualBatchSize];
        
        List<Integer> batchIndices = getCurrentBatchIndices(actualBatchSize);
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = batchIndices.get(i);
            int[] tokenIds = sequences.get(dataIndex);
            
            // 填充 input IDs，截断或补零
            int copyLen = Math.min(tokenIds.length, maxSeqLength);
            for (int j = 0; j < copyLen; j++) {
                inputIds[i][j] = tokenIds[j];
            }
            
            batchQuestions[i] = questions.get(dataIndex);
            batchGroundTruths[i] = groundTruths.get(dataIndex);
            batchVerifierTypes[i] = verifierTypes.get(dataIndex);
        }
        
        advanceIndex(currentIndex + actualBatchSize);
        
        return new Batch(
            NdArray.of(inputIds),
            batchQuestions,
            batchGroundTruths,
            batchVerifierTypes
        );
    }
    
    /**
     * 基于空格分词 + 动态词表的 tokenization
     *
     * <p>相比原"字符取模"方案：
     * <ul>
     *   <li>字符取模会把 "dog"/"god"/"ogd" 映射到同一组 id，token 语义严重坍缩</li>
     *   <li>中文/数字/标点全都只看低位字节，造成不同 token 碰撞</li>
     * </ul>
     *
     * <p>本实现策略（教学场景够用，不引入 BPE 依赖）：
     * <ol>
     *   <li>先按空格 + 常见标点切分</li>
     *   <li>每个 word 查询词表；不存在则按 {@link #nextTokenId} 自增分配</li>
     *   <li>词表规模达到上限后，退化为 {@code Math.abs(word.hashCode()) % VOCAB_LIMIT} 的稳定映射</li>
     *   <li>截断至 {@link #MAX_TOKENS_PER_SAMPLE}，防止超长 prompt 爆内存</li>
     * </ol>
     *
     * <p>空输入返回 <code>[0]</code>（pad token）。
     *
     * @param text 输入文本
     * @return token ID 数组（元素范围 [0, VOCAB_LIMIT)）
     */
    private int[] simpleTokenize(String text) {
        if (text == null || text.isEmpty()) {
            return new int[]{0};
        }

        // 按空格与常见标点切分；保留非空的词单元
        String[] words = text.toLowerCase().split("[\\s,.;:!?\"'()\\[\\]{}<>]+");
        List<Integer> ids = new ArrayList<>(Math.min(words.length, MAX_TOKENS_PER_SAMPLE));
        for (String w : words) {
            if (w.isEmpty()) continue;
            if (ids.size() >= MAX_TOKENS_PER_SAMPLE) break;
            ids.add(wordToTokenId(w));
        }
        if (ids.isEmpty()) {
            return new int[]{0};
        }
        int[] tokens = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            tokens[i] = ids.get(i);
        }
        return tokens;
    }

    /**
     * 词到 token id 的映射：优先命中词表，已满则 hash 回退。
     */
    private int wordToTokenId(String word) {
        Integer existing = vocab.get(word);
        if (existing != null) {
            return existing;
        }
        if (nextTokenId < VOCAB_LIMIT) {
            int id = nextTokenId++;
            vocab.put(word, id);
            return id;
        }
        // 词表已满，退化为稳定 hash；保证范围 [1, VOCAB_LIMIT-1]，避免与 pad 冲突
        int h = Math.abs(word.hashCode()) % (VOCAB_LIMIT - 1) + 1;
        return h;
    }

    /**
     * 获取当前词表大小（真实登记词数，不含 hash 回退）。
     * 主要用于调试与断言。
     */
    public int vocabSize() {
        return nextTokenId;
    }

    /**
     * 校验模型配置的 vocabSize 是否与本数据集兼容。
     *
     * <p>数据集产出的 token id ∈ [0, {@link #VOCAB_LIMIT}），
     * 因此模型 embedding 层的 vocabSize 不得小于 {@link #VOCAB_LIMIT}。
     *
     * @param modelVocabSize 模型（{@code DeepSeekR1Config.getVocabSize()}）配置的词表大小
     * @throws IllegalArgumentException 当 {@code modelVocabSize < VOCAB_LIMIT} 时抛出
     */
    public static void requireCompatibleVocabSize(int modelVocabSize) {
        if (modelVocabSize < VOCAB_LIMIT) {
            throw new IllegalArgumentException(String.format(
                    "DeepSeekR1RLVRDataset 要求 model.config.vocabSize >= %d，实际为 %d，" +
                            "这会导致 token embedding 越界。请调大 config.setVocabSize() 或缩小 VOCAB_LIMIT。",
                    VOCAB_LIMIT, modelVocabSize));
        }
    }
    
    // ==================== 内部类 ====================
    
    /**
     * 批次数据
     */
    public static class Batch {
        private final NdArray inputIds;
        private final String[] questions;
        private final String[] groundTruths;
        private final String[] verifierTypes;
        
        public Batch(NdArray inputIds, String[] questions, 
                    String[] groundTruths, String[] verifierTypes) {
            this.inputIds = inputIds;
            this.questions = questions;
            this.groundTruths = groundTruths;
            this.verifierTypes = verifierTypes;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public String[] getQuestions() {
            return questions;
        }
        
        public String[] getGroundTruths() {
            return groundTruths;
        }
        
        public String[] getVerifierTypes() {
            return verifierTypes;
        }
        
        public int getBatchSize() {
            return questions.length;
        }
    }
}
