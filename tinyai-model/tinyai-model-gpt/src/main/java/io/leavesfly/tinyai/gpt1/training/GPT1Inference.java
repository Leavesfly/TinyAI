package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * GPT-1推理引擎
 * 
 * 提供多种文本生成策略:
 * 1. 贪婪解码 (Greedy Decoding)
 * 2. Top-K采样
 * 3. Top-P采样 (Nucleus Sampling)
 * 4. 温度采样 (Temperature Sampling)
 * 5. Beam Search
 * 
 * @author TinyAI
 * @since 2024
 */
public class GPT1Inference {
    
    /** 默认EOS token ID，表示不启用EOS终止 */
    private static final int NO_EOS_TOKEN = -1;
    
    private final GPT1Model model;
    private final int maxSeqLen;
    private final int eosTokenId;
    
    /**
     * 构造函数（不启用EOS终止）
     * 
     * @param model GPT-1模型
     */
    public GPT1Inference(GPT1Model model) {
        this(model, NO_EOS_TOKEN);
    }
    
    /**
     * 构造函数（启用EOS终止）
     * 
     * @param model GPT-1模型
     * @param eosTokenId EOS token的ID，生成到该token时提前终止；传入-1表示不启用
     */
    public GPT1Inference(GPT1Model model, int eosTokenId) {
        this.model = model;
        this.maxSeqLen = model.getConfig().getNPositions();
        this.eosTokenId = eosTokenId;
    }
    
    /**
     * 判断是否应因生成EOS而终止
     */
    private boolean isEosToken(int tokenId) {
        return eosTokenId != NO_EOS_TOKEN && tokenId == eosTokenId;
    }
    
    /**
     * 贪婪解码生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @return 生成的完整序列
     */
    public int[] generateGreedy(int[] promptIds, int maxNewTokens) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) {
                break;
            }
            
            // 准备输入
            int[] currentSeq = toArray(generated);
            NdArray inputArray = createInputArray(currentSeq);
            Variable inputVar = new Variable(inputArray);
            
            // 前向传播
            Variable logits = model.predict(inputVar);
            NdArray logitsArray = logits.getValue();
            
            // 获取最后一个位置的logits
            int lastPos = currentSeq.length - 1;
            int nextToken = argmax(logitsArray, 0, lastPos);
            
            if (isEosToken(nextToken)) {
                break;
            }
            generated.add(nextToken);
        }
        
        return toArray(generated);
    }
    
    /**
     * Temperature采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param temperature 温度参数(越小越确定,越大越随机)
     * @return 生成的完整序列
     */
    public int[] generateWithTemperature(int[] promptIds, int maxNewTokens, float temperature) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        Random random = new Random();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) {
                break;
            }
            
            int[] currentSeq = toArray(generated);
            NdArray inputArray = createInputArray(currentSeq);
            Variable inputVar = new Variable(inputArray);
            
            Variable logits = model.predict(inputVar);
            NdArray logitsArray = logits.getValue();
            
            int lastPos = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);
            
            // 应用温度并计算softmax
            float[] probs = new float[vocabSize];
            float maxLogit = Float.NEGATIVE_INFINITY;
            
            for (int j = 0; j < vocabSize; j++) {
                float logit = logitsArray.get(0, lastPos, j) / temperature;
                probs[j] = logit;
                maxLogit = Math.max(maxLogit, logit);
            }
            
            // 数值稳定的softmax
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++) {
                probs[j] = (float) Math.exp(probs[j] - maxLogit);
                sum += probs[j];
            }
            
            for (int j = 0; j < vocabSize; j++) {
                probs[j] /= sum;
            }
            
            // 采样
            int nextToken = sample(probs, random);
            
            if (isEosToken(nextToken)) {
                break;
            }
            generated.add(nextToken);
        }
        
        return toArray(generated);
    }
    
    /**
     * Top-K采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topK 保留概率最高的K个token
     * @param temperature 温度参数
     * @return 生成的完整序列
     */
    public int[] generateTopK(int[] promptIds, int maxNewTokens, int topK, float temperature) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        Random random = new Random();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) {
                break;
            }
            
            int[] currentSeq = toArray(generated);
            NdArray inputArray = createInputArray(currentSeq);
            Variable inputVar = new Variable(inputArray);
            
            Variable logits = model.predict(inputVar);
            NdArray logitsArray = logits.getValue();
            
            int lastPos = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);
            
            // 获取logits并应用温度
            float[] logitsArr = new float[vocabSize];
            for (int j = 0; j < vocabSize; j++) {
                logitsArr[j] = logitsArray.get(0, lastPos, j) / temperature;
            }
            
            // 获取top-k索引
            int[] topKIndices = getTopKIndices(logitsArr, topK);
            
            // 计算top-k的概率分布
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
            
            for (int j = 0; j < topK; j++) {
                topKProbs[j] /= sum;
            }
            
            // 从top-k中采样
            int sampledIdx = sample(topKProbs, random);
            int nextToken = topKIndices[sampledIdx];
            
            if (isEosToken(nextToken)) {
                break;
            }
            generated.add(nextToken);
        }
        
        return toArray(generated);
    }
    
    /**
     * Top-P (Nucleus) 采样生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param topP 累积概率阈值
     * @param temperature 温度参数
     * @return 生成的完整序列
     */
    public int[] generateTopP(int[] promptIds, int maxNewTokens, float topP, float temperature) {
        List<Integer> generated = new ArrayList<>();
        for (int id : promptIds) {
            generated.add(id);
        }
        
        Random random = new Random();
        
        for (int i = 0; i < maxNewTokens; i++) {
            if (generated.size() >= maxSeqLen) {
                break;
            }
            
            int[] currentSeq = toArray(generated);
            NdArray inputArray = createInputArray(currentSeq);
            Variable inputVar = new Variable(inputArray);
            
            Variable logits = model.predict(inputVar);
            NdArray logitsArray = logits.getValue();
            
            int lastPos = currentSeq.length - 1;
            int vocabSize = logitsArray.getShape().getDimension(2);
            
            // 获取logits并应用温度
            float[] logitsArr = new float[vocabSize];
            for (int j = 0; j < vocabSize; j++) {
                logitsArr[j] = logitsArray.get(0, lastPos, j) / temperature;
            }
            
            // 计算概率分布
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float logit : logitsArr) {
                maxLogit = Math.max(maxLogit, logit);
            }
            
            float[] probs = new float[vocabSize];
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++) {
                probs[j] = (float) Math.exp(logitsArr[j] - maxLogit);
                sum += probs[j];
            }
            
            for (int j = 0; j < vocabSize; j++) {
                probs[j] /= sum;
            }
            
            // 按概率降序排序
            Integer[] indices = new Integer[vocabSize];
            for (int j = 0; j < vocabSize; j++) {
                indices[j] = j;
            }
            Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
            
            // 找到累积概率达到topP的位置
            float cumProb = 0.0f;
            int nucleusSize = 0;
            for (int j = 0; j < vocabSize; j++) {
                cumProb += probs[indices[j]];
                nucleusSize++;
                if (cumProb >= topP) {
                    break;
                }
            }
            
            // 重新归一化nucleus内的概率
            float[] nucleusProbs = new float[nucleusSize];
            sum = 0.0f;
            for (int j = 0; j < nucleusSize; j++) {
                nucleusProbs[j] = probs[indices[j]];
                sum += nucleusProbs[j];
            }
            
            for (int j = 0; j < nucleusSize; j++) {
                nucleusProbs[j] /= sum;
            }
            
            // 采样
            int sampledIdx = sample(nucleusProbs, random);
            int nextToken = indices[sampledIdx];
            
            if (isEosToken(nextToken)) {
                break;
            }
            generated.add(nextToken);
        }
        
        return toArray(generated);
    }
    
    /**
     * Beam Search生成
     * 
     * @param promptIds 提示词token序列
     * @param maxNewTokens 最大生成token数
     * @param beamSize beam大小
     * @return 最佳序列
     */
    public int[] generateBeamSearch(int[] promptIds, int maxNewTokens, int beamSize) {
        // 初始化beam
        List<Beam> beams = new ArrayList<>();
        Beam initialBeam = new Beam();
        for (int id : promptIds) {
            initialBeam.tokens.add(id);
        }
        initialBeam.score = 0.0f;
        beams.add(initialBeam);
        
        // 已完成的beam（生成了EOS的beam）
        List<Beam> finishedBeams = new ArrayList<>();
        
        // Beam search循环
        for (int step = 0; step < maxNewTokens; step++) {
            List<Beam> candidates = new ArrayList<>();
            
            for (Beam beam : beams) {
                // 已完成的beam直接保留，不再扩展
                if (beam.finished || beam.tokens.size() >= maxSeqLen) {
                    candidates.add(beam);
                    continue;
                }
                
                int[] currentSeq = toArray(beam.tokens);
                NdArray inputArray = createInputArray(currentSeq);
                Variable inputVar = new Variable(inputArray);
                
                Variable logits = model.predict(inputVar);
                NdArray logitsArray = logits.getValue();
                
                int lastPos = currentSeq.length - 1;
                int vocabSize = logitsArray.getShape().getDimension(2);
                
                // 计算log概率
                float[] logProbs = new float[vocabSize];
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < vocabSize; j++) {
                    float logit = logitsArray.get(0, lastPos, j);
                    logProbs[j] = logit;
                    maxLogit = Math.max(maxLogit, logit);
                }
                
                float logSumExp = 0.0f;
                for (int j = 0; j < vocabSize; j++) {
                    logSumExp += Math.exp(logProbs[j] - maxLogit);
                }
                logSumExp = maxLogit + (float) Math.log(logSumExp);
                
                for (int j = 0; j < vocabSize; j++) {
                    logProbs[j] -= logSumExp;
                }
                
                // 扩展beam
                int[] topKIndices = getTopKIndices(logProbs, beamSize);
                for (int idx : topKIndices) {
                    Beam newBeam = new Beam();
                    newBeam.tokens.addAll(beam.tokens);
                    newBeam.tokens.add(idx);
                    newBeam.score = beam.score + logProbs[idx];
                    // 生成EOS的beam标记为已完成
                    if (isEosToken(idx)) {
                        newBeam.finished = true;
                        finishedBeams.add(newBeam);
                    }
                    candidates.add(newBeam);
                }
            }
            
            // 选择top-k beams
            candidates.sort((a, b) -> Float.compare(b.score, a.score));
            beams = new ArrayList<>(candidates.subList(0, Math.min(beamSize, candidates.size())));
            
            // 如果所有活跃beam都已完成，提前终止
            boolean allFinished = true;
            for (Beam beam : beams) {
                if (!beam.finished) {
                    allFinished = false;
                    break;
                }
            }
            if (allFinished) {
                break;
            }
        }
        
        // 合并已完成和未完成的beam，选择得分最高的序列
        List<Beam> allBeams = new ArrayList<>(finishedBeams);
        for (Beam beam : beams) {
            if (!finishedBeams.contains(beam)) {
                allBeams.add(beam);
            }
        }
        allBeams.sort((a, b) -> Float.compare(b.score, a.score));
        
        return toArray(allBeams.get(0).tokens);
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * Beam结构
     */
    private static class Beam {
        List<Integer> tokens = new ArrayList<>();
        float score = 0.0f;
        boolean finished = false;
    }
    
    /**
     * 创建输入数组
     */
    private NdArray createInputArray(int[] sequence) {
        float[] data = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            data[i] = sequence[i];
        }
        return NdArray.of(data, Shape.of(1, sequence.length));
    }
    
    /**
     * 获取最大值索引
     */
    private int argmax(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 0;
        float maxVal = logits.get(batchIdx, seqIdx, 0);
        
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * 获取Top-K索引
     */
    private int[] getTopKIndices(float[] values, int k) {
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < values.length; i++) {
            indices[i] = i;
        }
        
        Arrays.sort(indices, (a, b) -> Float.compare(values[b], values[a]));
        
        int[] topK = new int[Math.min(k, indices.length)];
        for (int i = 0; i < topK.length; i++) {
            topK[i] = indices[i];
        }
        
        return topK;
    }
    
    /**
     * 根据概率分布采样
     */
    private int sample(float[] probs, Random random) {
        float r = random.nextFloat();
        float cumProb = 0.0f;
        
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) {
                return i;
            }
        }
        
        return probs.length - 1;
    }
    
    /**
     * List转数组
     */
    private int[] toArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
