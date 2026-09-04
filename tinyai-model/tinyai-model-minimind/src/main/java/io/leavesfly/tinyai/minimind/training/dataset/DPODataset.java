package io.leavesfly.tinyai.minimind.training.dataset;

import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *  数据集
 * 
 * DPO数据格式: (prompt, chosen_response, rejected_response)
 * - prompt: 输入提示
 * - chosen_response: 更好的响应 (preferred)
 * - rejected_response: 较差的响应 (rejected)
 * 
 * DPO不需要奖励模型,直接从偏好数据优化策略
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPODataset {
    
    /**
     * DPO偏好对样本
     */
    public static class PreferencePair {
        private final String prompt;
        private final String chosenResponse;
        private final String rejectedResponse;
        
        public PreferencePair(String prompt, String chosenResponse, String rejectedResponse) {
            this.prompt = prompt;
            this.chosenResponse = chosenResponse;
            this.rejectedResponse = rejectedResponse;
        }
        
        public String getPrompt() { return prompt; }
        public String getChosenResponse() { return chosenResponse; }
        public String getRejectedResponse() { return rejectedResponse; }
        
        public String getChosenFullText() {
            return prompt + chosenResponse;
        }
        
        public String getRejectedFullText() {
            return prompt + rejectedResponse;
        }
    }
    
    /**
     * DPO批次数据
     * <p>
     * chosen 与 rejected 长度不同，必须各自拥有独立的损失掩码：
     * 若共用一份，后写入的序列会覆盖前一份，导致长序列的回复被截断不参与损失、
     * 或短序列在超出自身长度的位置带上默认标签(PAD)而被计入损失。
     */
    public static class Batch {
        private final NdArray chosenInput;        // Chosen序列输入
        private final NdArray chosenLabels;       // Chosen序列标签
        private final NdArray rejectedInput;      // Rejected序列输入
        private final NdArray rejectedLabels;     // Rejected序列标签
        private final NdArray chosenMask;         // Chosen 损失掩码(1=response, 0=prompt/pad)
        private final NdArray rejectedMask;       // Rejected 损失掩码(1=response, 0=prompt/pad)
        
        public Batch(NdArray chosenInput, NdArray chosenLabels,
                     NdArray rejectedInput, NdArray rejectedLabels,
                     NdArray chosenMask, NdArray rejectedMask) {
            this.chosenInput = chosenInput;
            this.chosenLabels = chosenLabels;
            this.rejectedInput = rejectedInput;
            this.rejectedLabels = rejectedLabels;
            this.chosenMask = chosenMask;
            this.rejectedMask = rejectedMask;
        }
        
        public NdArray getChosenInput() { return chosenInput; }
        public NdArray getChosenLabels() { return chosenLabels; }
        public NdArray getRejectedInput() { return rejectedInput; }
        public NdArray getRejectedLabels() { return rejectedLabels; }
        public NdArray getChosenMask() { return chosenMask; }
        public NdArray getRejectedMask() { return rejectedMask; }
        
        /**
         * 兼容旧接口：返回 chosen 侧的损失掩码
         *
         * @deprecated 请改用 {@link #getChosenMask()} / {@link #getRejectedMask()}，
         *             chosen 与 rejected 长度不同时共用一份掩码会造成损失错位
         */
        @Deprecated
        public NdArray getPromptMask() { return chosenMask; }
    }
    
    private final List<PreferencePair> samples;
    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    
    /**
     * 标签忽略位标记（对齐 SoftmaxCE 的 ignore_index 约定）
     * <p>
     * 不使用 pad token id(0)：SoftmaxCE 只跳过负数标签，标签 0 会被当成一个
     * 正常类别计入损失，从而把模型训成去预测 PAD。
     */
    private static final int IGNORE_LABEL = -100;
    
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * 构造函数
     * 
     * @param tokenizer 分词器
     * @param maxSeqLen 最大序列长度
     * @param batchSize 批次大小
     */
    public DPODataset(MiniMindTokenizer tokenizer, int maxSeqLen, int batchSize) {
        this.samples = new ArrayList<>();
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 添加偏好对样本
     * 
     * @param prompt 提示
     * @param chosenResponse 更好的响应
     * @param rejectedResponse 较差的响应
     */
    public void addSample(String prompt, String chosenResponse, String rejectedResponse) {
        samples.add(new PreferencePair(prompt, chosenResponse, rejectedResponse));
    }
    
    /**
     * 准备批次数据
     * 
     * @param shuffle 是否打乱数据
     */
    public void prepare(boolean shuffle) {
        if (shuffle) {
            Collections.shuffle(samples);
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        // 分批处理
        for (int i = 0; i < samples.size(); i += batchSize) {
            int end = Math.min(i + batchSize, samples.size());
            List<PreferencePair> batchSamples = samples.subList(i, end);
            
            Batch batch = createBatch(batchSamples);
            batches.add(batch);
        }
    }
    
    /**
     * 创建单个批次
     */
    private Batch createBatch(List<PreferencePair> batchSamples) {
        int actualBatchSize = batchSamples.size();
        
        // 初始化数组
        float[] chosenInputData = new float[actualBatchSize * maxSeqLen];
        float[] chosenLabelsData = new float[actualBatchSize * maxSeqLen];
        float[] rejectedInputData = new float[actualBatchSize * maxSeqLen];
        float[] rejectedLabelsData = new float[actualBatchSize * maxSeqLen];
        // chosen / rejected 各自一份掩码，互不覆盖
        float[] chosenMaskData = new float[actualBatchSize * maxSeqLen];
        float[] rejectedMaskData = new float[actualBatchSize * maxSeqLen];
        
        // 处理每个样本
        for (int i = 0; i < actualBatchSize; i++) {
            PreferencePair pair = batchSamples.get(i);
            
            // 编码prompt(用于确定掩码位置)
            List<Integer> promptTokens = tokenizer.encode(pair.getPrompt(), false, false);
            int promptLen = Math.min(promptTokens.size(), maxSeqLen);
            
            // 编码chosen序列
            List<Integer> chosenTokens = tokenizer.encode(pair.getChosenFullText(), false, false);
            processSequence(chosenTokens, i, chosenInputData, chosenLabelsData,
                          chosenMaskData, promptLen);
            
            // 编码rejected序列
            List<Integer> rejectedTokens = tokenizer.encode(pair.getRejectedFullText(), false, false);
            processSequence(rejectedTokens, i, rejectedInputData, rejectedLabelsData, 
                          rejectedMaskData, promptLen);
        }
        
        // 创建NdArray
        NdArray chosenInput = NdArray.of(chosenInputData, Shape.of(actualBatchSize, maxSeqLen));
        NdArray chosenLabels = NdArray.of(chosenLabelsData, Shape.of(actualBatchSize, maxSeqLen));
        NdArray rejectedInput = NdArray.of(rejectedInputData, Shape.of(actualBatchSize, maxSeqLen));
        NdArray rejectedLabels = NdArray.of(rejectedLabelsData, Shape.of(actualBatchSize, maxSeqLen));
        NdArray chosenMask = NdArray.of(chosenMaskData, Shape.of(actualBatchSize, maxSeqLen));
        NdArray rejectedMask = NdArray.of(rejectedMaskData, Shape.of(actualBatchSize, maxSeqLen));
        
        return new Batch(chosenInput, chosenLabels, rejectedInput, rejectedLabels,
                        chosenMask, rejectedMask);
    }
    
    /**
     * 处理单个序列
     * <p>
     * 掩码对齐说明：位置 j 的标签是 {@code tokens[j+1]}，因此第一个回复 token
     * （下标 promptLen）对应的位置是 {@code j = promptLen - 1}，掩码条件应为
     * {@code j >= promptLen - 1}。若写成 {@code j >= promptLen}，第一个回复 token
     * 会被错误地排除在损失之外。
     */
    private void processSequence(List<Integer> tokens, int batchIdx, 
                                 float[] inputData, float[] labelsData,
                                 float[] maskData, int promptLen) {
        int seqLen = Math.min(tokens.size(), maxSeqLen);
        int offset = batchIdx * maxSeqLen;
        
        // 填充输入和标签
        for (int j = 0; j < seqLen - 1; j++) {
            inputData[offset + j] = tokens.get(j).floatValue();
            labelsData[offset + j] = tokens.get(j + 1).floatValue();
            
            // Prompt部分的掩码设为0(不计算损失),Response部分设为1
            maskData[offset + j] = (j >= promptLen - 1) ? 1.0f : 0.0f;
        }
        
        // 最后一个token（无下一个 token 可监督）
        if (seqLen > 0) {
            inputData[offset + seqLen - 1] = tokens.get(seqLen - 1).floatValue();
            labelsData[offset + seqLen - 1] = IGNORE_LABEL;
            maskData[offset + seqLen - 1] = 0.0f;
        }
        
        // 填充剩余部分（标签用 IGNORE_LABEL 而非 PAD(0)，避免被当成正常类别计入损失）
        for (int j = seqLen; j < maxSeqLen; j++) {
            inputData[offset + j] = 0;
            labelsData[offset + j] = IGNORE_LABEL;
            maskData[offset + j] = 0.0f;
        }
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
     * 是否还有下一个批次
     */
    public boolean hasNext() {
        return currentBatchIndex < batches.size();
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
     * 清空数据集
     */
    public void clear() {
        samples.clear();
        batches.clear();
        currentBatchIndex = 0;
    }
}
