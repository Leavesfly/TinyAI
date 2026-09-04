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
import org.json.*;

/**
 * SFT(Supervised Fine-Tuning)数据集
 * 
 * 支持指令微调数据格式:
 * {
 *   "instruction": "用户指令",
 *   "input": "可选的输入",
 *   "output": "期望的输出"
 * }
 * 
 * @author leavesfly
 * @since 2024
 */
public class SFTDataset implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    
    // 对话模板
    private static final String CHAT_TEMPLATE = 
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>";
    
    // 训练样本
    private List<SFTSample> samples;
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /** 最近一次 prepare 中因无可监督 token 而被跳过的样本数 */
    private int skippedNoSupervision;
    /** prompt 前缀不一致只告警一次 */
    private boolean prefixMismatchWarned;
    
    /**
     * SFT样本
     */
    public static class SFTSample {
        public final String instruction;
        public final String input;
        public final String output;
        
        public SFTSample(String instruction, String input, String output) {
            this.instruction = instruction;
            this.input = input;
            this.output = output;
        }
        
        public String formatPrompt() {
            if (input != null && !input.isEmpty()) {
                return instruction + "\n" + input;
            }
            return instruction;
        }
    }
    
    /**
     * 构造函数
     */
    public SFTDataset(MiniMindTokenizer tokenizer, int maxSeqLen, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
        this.skippedNoSupervision = 0;
        this.prefixMismatchWarned = false;
        if (maxSeqLen > tokenizer.getMaxSeqLen()) {
            System.err.println("⚠️ SFT: maxSeqLen=" + maxSeqLen
                    + " 大于 tokenizer.maxSeqLen=" + tokenizer.getMaxSeqLen()
                    + "，编码时会先被 tokenizer 截断，实际生效上限为 "
                    + tokenizer.getMaxSeqLen());
        }
    }
    
    /**
     * 从JSONL文件加载数据
     * 
     * @param filePath 文件路径
     * @throws IOException IO异常
     */
    public void loadFromJsonl(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        samples.clear();
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        
        for (String line : lines) {
            if (line.trim().isEmpty()) {
                continue;
            }
            
            try {
                JSONObject json = new JSONObject(line);
                String instruction = json.optString("instruction", "");
                String input = json.optString("input", "");
                String output = json.optString("output", "");
                
                if (!instruction.isEmpty() && !output.isEmpty()) {
                    samples.add(new SFTSample(instruction, input, output));
                }
            } catch (Exception e) {
                System.err.println("解析JSON失败: " + line);
            }
        }
        
        System.out.println("SFT数据加载完成,共 " + samples.size() + " 个样本");
    }
    
    /**
     * 添加单个样本
     */
    public void addSample(String instruction, String input, String output) {
        samples.add(new SFTSample(instruction, input, output));
    }
    
    /**
     * 准备批次数据
     */
    public void prepare(boolean shuffle) {
        if (samples.isEmpty()) {
            throw new IllegalStateException("数据集为空");
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        List<SFTSample> workingSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(workingSamples);
        }
        
        // 先逐样本编码并过滤掉无法监督的样本，再切批；
        // 否则一个批里混入全零掩码行会静默拉低损失，且无任何梯度贡献
        List<EncodedSample> encoded = new ArrayList<>(workingSamples.size());
        int skipped = 0;
        for (SFTSample sample : workingSamples) {
            EncodedSample item = encodeSample(sample);
            if (item == null) {
                skipped++;
            } else {
                encoded.add(item);
            }
        }
        
        if (skipped > 0) {
            System.err.println("⚠️ SFT: " + skipped + "/" + workingSamples.size()
                    + " 个样本被跳过——截断到 maxSeqLen=" + effectiveSeqLen()
                    + " 后没有任何 assistant token 可监督（损失掩码全为 0）。"
                    + "请增大 maxSeqLen / tokenizer.maxSeqLen 或缩短 instruction/input。");
        }
        if (encoded.isEmpty()) {
            throw new IllegalStateException("所有样本在 maxSeqLen=" + effectiveSeqLen()
                    + " 下都没有可监督的 assistant token，无法构造 SFT 批次；"
                    + "空对话模板本身就需要 " + minimumTemplateTokens() + " 个 token"
                    + "（tokenizer.maxSeqLen=" + tokenizer.getMaxSeqLen()
                    + ", 数据集 maxSeqLen=" + maxSeqLen + "）");
        }
        
        // 创建批次
        for (int i = 0; i < encoded.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, encoded.size());
            batches.add(createBatch(encoded.subList(i, endIdx)));
        }
        
        skippedNoSupervision = skipped;
        System.out.println("SFT批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 编码单个样本
     *
     * @return 编码结果；若该样本在当前 maxSeqLen 下没有任何可监督位置，返回 null
     */
    private EncodedSample encodeSample(SFTSample sample) {
        // 构建对话文本
        String prompt = sample.formatPrompt();
        String response = sample.output;
        String fullText = String.format(CHAT_TEMPLATE, prompt, response);
        
        // 编码整个对话
        List<Integer> fullTokenIds = tokenizer.encode(fullText, false, false);
        
        // 编码用户部分(用于确定掩码位置)
        String userPart = String.format(
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
        List<Integer> userTokenIds = tokenizer.encode(userPart, false, false);
        int promptLen = alignPromptLen(fullTokenIds, userTokenIds);
        
        // 截断到最大长度
        if (fullTokenIds.size() > maxSeqLen) {
            fullTokenIds = new ArrayList<>(fullTokenIds.subList(0, maxSeqLen));
        }
        
        int seqLen = fullTokenIds.size();
        // 位置 i 的标签是 fullTokenIds[i+1]，i ∈ [0, seqLen-2]；
        // 要监督到 assistant 内容需要 promptLen-1 <= seqLen-2，即 promptLen <= seqLen-1
        if (seqLen < 2 || promptLen > seqLen - 1) {
            return null;
        }
        
        int supervisedLen = seqLen - 1;
        int[] input = new int[supervisedLen];
        int[] labels = new int[supervisedLen];
        int[] mask = new int[supervisedLen];
        for (int i = 0; i < supervisedLen; i++) {
            input[i] = fullTokenIds.get(i);
            labels[i] = fullTokenIds.get(i + 1);
            // 损失掩码: 只计算 assistant 部分的损失
            mask[i] = (i >= promptLen - 1) ? 1 : 0;
        }
        return new EncodedSample(input, labels, mask);
    }
    
    /**
     * 校准 prompt 长度
     * <p>
     * 字符级分词器下 {@code encode(userPart)} 必然是 {@code encode(fullText)} 的前缀；
     * BPE 分词器则可能在边界处发生跨串合并，使两者在末尾 1~2 个 token 上不一致。
     * 此时取最长公共前缀长度，宁可少监督一个 token，也不要把 prompt 算进损失。
     */
    private int alignPromptLen(List<Integer> fullTokenIds, List<Integer> userTokenIds) {
        int limit = Math.min(fullTokenIds.size(), userTokenIds.size());
        int common = 0;
        while (common < limit && fullTokenIds.get(common).equals(userTokenIds.get(common))) {
            common++;
        }
        if (common < userTokenIds.size() && !prefixMismatchWarned) {
            prefixMismatchWarned = true;
            System.err.println("⚠️ SFT: user 部分编码与完整对话前缀不一致（"
                    + common + "/" + userTokenIds.size()
                    + "），可能是 BPE 跨边界合并；已改用最长公共前缀作为掩码分界。");
        }
        return common;
    }
    
    /** 空对话模板（prompt/response 均为空）所需的 token 数，用于给出可操作的报错提示 */
    private int minimumTemplateTokens() {
        return tokenizer.encode(String.format(CHAT_TEMPLATE, "", ""), false, false).size();
    }
    
    /**
     * 实际生效的序列上限
     * <p>
     * {@code tokenizer.encode} 会先按 tokenizer 自己的 maxSeqLen 截断，因此当
     * {@code maxSeqLen > tokenizer.getMaxSeqLen()} 时，数据集的设置完全无效。
     */
    private int effectiveSeqLen() {
        return Math.min(maxSeqLen, tokenizer.getMaxSeqLen());
    }
    
    /**
     * 创建批次
     */
    private Batch createBatch(List<EncodedSample> batchSamples) {
        int actualBatchSize = batchSamples.size();
        
        int maxLen = 0;
        for (EncodedSample item : batchSamples) {
            maxLen = Math.max(maxLen, item.input.length);
        }
        
        // 填充到批内最长长度
        int padTokenId = tokenizer.getVocabulary().getPadTokenId();
        int paddedLen = maxLen;
        
        int[][] inputData = new int[actualBatchSize][paddedLen];
        int[][] labelData = new int[actualBatchSize][paddedLen];
        int[][] maskData = new int[actualBatchSize][paddedLen];
        
        for (int i = 0; i < actualBatchSize; i++) {
            EncodedSample item = batchSamples.get(i);
            
            System.arraycopy(item.input, 0, inputData[i], 0, item.input.length);
            System.arraycopy(item.labels, 0, labelData[i], 0, item.labels.length);
            System.arraycopy(item.mask, 0, maskData[i], 0, item.mask.length);
            
            // 填充部分
            for (int j = item.input.length; j < paddedLen; j++) {
                inputData[i][j] = padTokenId;
                labelData[i][j] = padTokenId;
                maskData[i][j] = 0;
            }
        }
        
        // 转换为NdArray
        NdArray inputArray = createNdArray(inputData, actualBatchSize, paddedLen);
        NdArray labelArray = createNdArray(labelData, actualBatchSize, paddedLen);
        NdArray maskArray = createNdArray(maskData, actualBatchSize, paddedLen);
        
        return new Batch(inputArray, labelArray, maskArray, actualBatchSize, paddedLen);
    }
    
    /**
     * 编码后的单条样本（input/labels/mask 已按 seqLen-1 对齐，尚未填充）
     */
    private static final class EncodedSample {
        final int[] input;
        final int[] labels;
        final int[] mask;
        
        EncodedSample(int[] input, int[] labels, int[] mask) {
            this.input = input;
            this.labels = labels;
            this.mask = mask;
        }
    }
    
    /**
     * 创建NdArray
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
    
    public boolean hasNextBatch() {
        return currentBatchIndex < batches.size();
    }
    
    public Batch getNextBatch() {
        if (!hasNextBatch()) {
            throw new NoSuchElementException("没有更多批次数据");
        }
        return batches.get(currentBatchIndex++);
    }
    
    public void reset() {
        currentBatchIndex = 0;
    }
    
    public int getBatchCount() {
        return batches.size();
    }
    
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 最近一次 {@link #prepare(boolean)} 中因“截断后无可监督 assistant token”而被跳过的样本数
     * <p>
     * 大于 0 说明 maxSeqLen 偏小，训练实际上丢弃了这部分数据。
     */
    public int getSkippedNoSupervisionCount() {
        return skippedNoSupervision;
    }
    
    /**
     * SFT批次数据
     */
    public static class Batch {
        private final NdArray input;
        private final NdArray labels;
        private final NdArray lossMask;  // 损失掩码,1表示计算损失,0表示忽略
        private final int batchSize;
        private final int seqLen;
        
        public Batch(NdArray input, NdArray labels, NdArray lossMask,
                     int batchSize, int seqLen) {
            this.input = input;
            this.labels = labels;
            this.lossMask = lossMask;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
        
        public NdArray getInput() {
            return input;
        }
        
        public NdArray getLabels() {
            return labels;
        }
        
        public NdArray getLossMask() {
            return lossMask;
        }
        
        public int getBatchSize() {
            return batchSize;
        }
        
        public int getSeqLen() {
            return seqLen;
        }
    }
}
