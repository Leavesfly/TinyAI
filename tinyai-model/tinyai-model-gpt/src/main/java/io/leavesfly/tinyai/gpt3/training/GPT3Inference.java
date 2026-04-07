package io.leavesfly.tinyai.gpt3.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt3.GPT3Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * GPT-3推理引擎
 *
 * 提供多种文本生成策略，并充分利用GPT-3特有的KV Cache机制加速推理：
 * 1. 贪婪解码 (Greedy Decoding) - 最简单，速度最快
 * 2. Temperature采样 - 控制生成随机性
 * 3. Top-K采样 - 截断低概率词汇
 * 4. Top-P采样 (Nucleus Sampling) - 动态截断，业界主流
 * 5. Beam Search - 更高质量但更慢
 * 6. 带KV Cache的加速推理 - GPT-3特有优化，O(n)效率
 *
 * @author TinyAI
 * @since 2024
 */
public class GPT3Inference {

    private final GPT3Model model;
    private final int maxSeqLen;

    /**
     * EOS（End of Sequence）token ID。
     * GPT-2/GPT-3 中 <|endoftext|> 对应 vocabSize - 1（即 50256）。
     * 生成过程中遇到此 token 时立即停止，避免无意义的续写。
     */
    private final int eosTokenId;

    /** 需要在推理时抑制的特殊 token ID（<PAD>=0, <UNK>=1, <BOS>=2） */
    private static final int[] SUPPRESSED_TOKEN_IDS = {0, 1, 2};

    /**
     * 构造函数
     *
     * @param model GPT-3模型
     */
    public GPT3Inference(GPT3Model model) {
        this.model = model;
        this.maxSeqLen = model.getConfig().getNPositions();
        this.eosTokenId = model.getConfig().getVocabSize() - 1;
    }

    /**
     * 将特殊 token（PAD/UNK/BOS）的 logits 设为负无穷，
     * 防止生成过程中选中这些无意义的 token。
     */
    private void suppressSpecialTokens(float[] logits) {
        for (int id : SUPPRESSED_TOKEN_IDS) {
            if (id < logits.length) {
                logits[id] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    /**
     * 贪婪解码生成（每步选最大概率token）
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @return 生成的完整序列（含提示词）
     */
    public int[] generateGreedy(int[] promptIds, int maxNewTokens) {
        List<Integer> generated = toList(promptIds);

        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;

            int[] currentSeq = toArray(generated);
            Variable logits = model.predict(new Variable(createInputArray(currentSeq)));
            NdArray logitsArray = logits.getValue();

            int lastPos = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);

            float[] logitsArr = extractLogits(logitsArray, lastPos, vocabSize);
            suppressSpecialTokens(logitsArr);

            int nextToken = argmaxFromArray(logitsArr);
            generated.add(nextToken);

            if (nextToken == eosTokenId) break;
        }

        return toArray(generated);
    }

    /**
     * 带KV Cache的贪婪解码（GPT-3特有，O(n)效率 vs 无Cache的O(n²)）
     *
     * Phase1: 处理完整Prompt，填充KV Cache
     * Phase2: 每步只输入1个Token，复用历史K/V，显著加速长序列生成
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @return 生成的完整序列（含提示词）
     */
    public int[] generateGreedyWithCache(int[] promptIds, int maxNewTokens) {
        NdArray promptArray = createInputArray(promptIds);
        NdArray resultArray = model.generateWithCache(promptArray, maxNewTokens);

        int batchSize = resultArray.getShape().getDimension(0);
        int seqLen    = resultArray.getShape().getDimension(1);
        int[] result  = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            result[i] = (int) resultArray.get(0, i);
        }
        return result;
    }

    /**
     * Temperature采样生成
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param temperature  温度参数（越小越确定，越大越随机）
     * @return 生成的完整序列
     */
    public int[] generateWithTemperature(int[] promptIds, int maxNewTokens, float temperature) {
        List<Integer> generated = toList(promptIds);
        Random random = new Random();

        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;

            int[] currentSeq = toArray(generated);
            Variable logits = model.predict(new Variable(createInputArray(currentSeq)));
            NdArray logitsArray = logits.getValue();

            int lastPos  = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);

            float[] logitsArr = extractLogits(logitsArray, lastPos, vocabSize);
            suppressSpecialTokens(logitsArr);

            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < vocabSize; j++) {
                logitsArr[j] /= temperature;
                maxLogit = Math.max(maxLogit, logitsArr[j]);
            }

            float[] probs = new float[vocabSize];
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++) {
                probs[j] = (float) Math.exp(logitsArr[j] - maxLogit);
                sum += probs[j];
            }
            for (int j = 0; j < vocabSize; j++) probs[j] /= sum;

            int nextToken = sample(probs, random);
            generated.add(nextToken);

            if (nextToken == eosTokenId) break;
        }

        return toArray(generated);
    }

    /**
     * Top-K采样生成（只保留概率最高的K个token参与采样）
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topK         保留的top-K个token数量
     * @param temperature  温度参数
     * @return 生成的完整序列
     */
    public int[] generateTopK(int[] promptIds, int maxNewTokens, int topK, float temperature) {
        List<Integer> generated = toList(promptIds);
        Random random = new Random();

        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;

            int[] currentSeq = toArray(generated);
            Variable logits = model.predict(new Variable(createInputArray(currentSeq)));
            NdArray logitsArray = logits.getValue();

            int lastPos   = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);

            float[] logitsArr = extractLogits(logitsArray, lastPos, vocabSize);
            suppressSpecialTokens(logitsArr);
            for (int j = 0; j < vocabSize; j++) {
                logitsArr[j] /= temperature;
            }

            int[] topKIndices = getTopKIndices(logitsArr, topK);

            float[] topKProbs = new float[topK];
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < topK; j++) {
                topKProbs[j] = logitsArr[topKIndices[j]];
                maxLogit = Math.max(maxLogit, topKProbs[j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < topK; j++) {
                topKProbs[j] = (float) Math.exp(topKProbs[j] - maxLogit);
                sum += topKProbs[j];
            }
            for (int j = 0; j < topK; j++) topKProbs[j] /= sum;

            int sampledIdx = sample(topKProbs, random);
            int nextToken = topKIndices[sampledIdx];
            generated.add(nextToken);

            if (nextToken == eosTokenId) break;
        }

        return toArray(generated);
    }

    /**
     * Top-P (Nucleus) 采样生成（动态截断累积概率达到topP的最小词汇集）
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topP         累积概率阈值（通常0.9或0.95）
     * @param temperature  温度参数
     * @return 生成的完整序列
     */
    public int[] generateTopP(int[] promptIds, int maxNewTokens, float topP, float temperature) {
        List<Integer> generated = toList(promptIds);
        Random random = new Random();

        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) break;

            int[] currentSeq = toArray(generated);
            Variable logits = model.predict(new Variable(createInputArray(currentSeq)));
            NdArray logitsArray = logits.getValue();

            int lastPos   = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);

            float[] logitsArr = extractLogits(logitsArray, lastPos, vocabSize);
            suppressSpecialTokens(logitsArr);
            for (int j = 0; j < vocabSize; j++) {
                logitsArr[j] /= temperature;
            }

            // 计算softmax概率
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float v : logitsArr) maxLogit = Math.max(maxLogit, v);

            float[] probs = new float[vocabSize];
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++) {
                probs[j] = (float) Math.exp(logitsArr[j] - maxLogit);
                sum += probs[j];
            }
            for (int j = 0; j < vocabSize; j++) probs[j] /= sum;

            // 按概率降序排列索引
            Integer[] indices = new Integer[vocabSize];
            for (int j = 0; j < vocabSize; j++) indices[j] = j;
            Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

            // 找到累积概率达到topP的nucleus
            float cumProb = 0.0f;
            int nucleusSize = 0;
            for (int j = 0; j < vocabSize; j++) {
                cumProb += probs[indices[j]];
                nucleusSize++;
                if (cumProb >= topP) break;
            }

            // 重新归一化nucleus内的概率
            float[] nucleusProbs = new float[nucleusSize];
            sum = 0.0f;
            for (int j = 0; j < nucleusSize; j++) {
                nucleusProbs[j] = probs[indices[j]];
                sum += nucleusProbs[j];
            }
            for (int j = 0; j < nucleusSize; j++) nucleusProbs[j] /= sum;

            int sampledIdx = sample(nucleusProbs, random);
            int nextToken = indices[sampledIdx];
            generated.add(nextToken);

            if (nextToken == eosTokenId) break;
        }

        return toArray(generated);
    }

    /**
     * Beam Search生成（更高质量，适合翻译/摘要等确定性任务）
     *
     * @param promptIds    提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param beamSize     beam宽度
     * @return 得分最高的序列
     */
    public int[] generateBeamSearch(int[] promptIds, int maxNewTokens, int beamSize) {
        List<Beam> beams = new ArrayList<>();
        Beam initialBeam = new Beam();
        for (int id : promptIds) initialBeam.tokens.add(id);
        initialBeam.score = 0.0f;
        beams.add(initialBeam);

        for (int step = 0; step < maxNewTokens; step++) {
            // 所有 beam 都已遇到 EOS，提前结束
            if (beams.stream().allMatch(b -> b.finished)) break;

            List<Beam> candidates = new ArrayList<>();

            for (Beam beam : beams) {
                // 已完成的 beam 直接保留，不再扩展
                if (beam.finished || beam.tokens.size() >= maxSeqLen) {
                    candidates.add(beam);
                    continue;
                }

                int[] currentSeq = toArray(beam.tokens);
                Variable logits = model.predict(new Variable(createInputArray(currentSeq)));
                NdArray logitsArray = logits.getValue();

                int lastPos   = currentSeq.length - 1;
                int vocabSize = logitsArray.getShape().getDimension(2);

                // 计算log概率
                float[] logProbs = extractLogits(logitsArray, lastPos, vocabSize);
                suppressSpecialTokens(logProbs);
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < vocabSize; j++) {
                    maxLogit = Math.max(maxLogit, logProbs[j]);
                }

                float logSumExp = 0.0f;
                for (float v : logProbs) logSumExp += Math.exp(v - maxLogit);
                logSumExp = maxLogit + (float) Math.log(logSumExp);
                for (int j = 0; j < vocabSize; j++) logProbs[j] -= logSumExp;

                int[] topKIndices = getTopKIndices(logProbs, beamSize);
                for (int idx : topKIndices) {
                    Beam newBeam = new Beam();
                    newBeam.tokens.addAll(beam.tokens);
                    newBeam.tokens.add(idx);
                    newBeam.score = beam.score + logProbs[idx];
                    newBeam.finished = (idx == eosTokenId);
                    candidates.add(newBeam);
                }
            }

            candidates.sort((a, b) -> Float.compare(b.score, a.score));
            beams = candidates.subList(0, Math.min(beamSize, candidates.size()));
        }

        return toArray(beams.get(0).tokens);
    }

    // ==================== 辅助方法 ====================

    private static class Beam {
        List<Integer> tokens = new ArrayList<>();
        float score = 0.0f;
        boolean finished = false;
    }

    private NdArray createInputArray(int[] sequence) {
        float[] data = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) data[i] = sequence[i];
        return NdArray.of(data, Shape.of(1, sequence.length));
    }

    /**
     * 委托给 GPT3Model.argmax，消除重复代码
     */
    private int argmax(NdArray logits, int batchIdx, int seqIdx) {
        return GPT3Model.argmax(logits, batchIdx, seqIdx);
    }

    /**
     * 从 NdArray 中提取指定位置的 logits 到 float 数组
     */
    private float[] extractLogits(NdArray logitsArray, int seqPos, int vocabSize) {
        float[] logits = new float[vocabSize];
        for (int j = 0; j < vocabSize; j++) {
            logits[j] = logitsArray.get(0, seqPos, j);
        }
        return logits;
    }

    /**
     * 从 float 数组中找到最大值的索引
     */
    private int argmaxFromArray(float[] values) {
        int bestIdx = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestVal) {
                bestVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * 使用最小堆高效获取 Top-K 索引。
     * 时间复杂度 O(V log K)，远优于全排序的 O(V log V)，
     * 在 vocabSize 很大（如 50257）而 K 较小时效果显著。
     */
    private int[] getTopKIndices(float[] values, int k) {
        int actualK = Math.min(k, values.length);

        // 最小堆：堆顶是当前 Top-K 中最小的，方便淘汰
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(
                actualK, (a, b) -> Float.compare(values[a[0]], values[b[0]]));

        for (int i = 0; i < values.length; i++) {
            if (minHeap.size() < actualK) {
                minHeap.offer(new int[]{i});
            } else if (values[i] > values[minHeap.peek()[0]]) {
                minHeap.poll();
                minHeap.offer(new int[]{i});
            }
        }

        int[] topKIndices = new int[minHeap.size()];
        int idx = topKIndices.length - 1;
        while (!minHeap.isEmpty()) {
            topKIndices[idx--] = minHeap.poll()[0];
        }
        return topKIndices;
    }

    private int sample(float[] probs, Random random) {
        float r = random.nextFloat();
        float cumProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) return i;
        }
        return probs.length - 1;
    }

    private List<Integer> toList(int[] arr) {
        List<Integer> list = new ArrayList<>();
        for (int v : arr) list.add(v);
        return list;
    }

    private int[] toArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }
}
