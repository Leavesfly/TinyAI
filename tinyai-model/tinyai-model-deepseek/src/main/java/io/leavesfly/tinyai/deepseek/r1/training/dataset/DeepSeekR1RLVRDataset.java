package io.leavesfly.tinyai.deepseek.r1.training.dataset;

import io.leavesfly.tinyai.deepseek.base.dataset.DeepSeekBaseDataset;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

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
     * 简单的字符级 tokenization
     * 将文本转换为 int[] 以适配基类的 sequences 结构
     * 
     * @param text 输入文本
     * @return token ID 数组
     */
    private int[] simpleTokenize(String text) {
        if (text == null || text.isEmpty()) {
            return new int[]{0};
        }
        
        char[] chars = text.toCharArray();
        int length = Math.min(chars.length, 100);
        int[] tokens = new int[length];
        for (int i = 0; i < length; i++) {
            tokens[i] = chars[i] % 1000;
        }
        return tokens;
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
