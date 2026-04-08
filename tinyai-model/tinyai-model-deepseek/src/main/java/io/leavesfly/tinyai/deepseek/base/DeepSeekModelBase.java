package io.leavesfly.tinyai.deepseek.base;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.core.Module;

/**
 * DeepSeek 模型基类
 *
 * 提取 V3 和 R1 共享的通用方法，避免代码重复：
 * 1. formatParamCount - 格式化参数数量
 * 2. generateSequence - 贪婪解码序列生成
 * 3. argmax - 查找最大值索引
 *
 * @author leavesfly
 * @version 1.0
 */
public abstract class DeepSeekModelBase extends Model {

    /**
     * 构造函数
     *
     * @param name   模型名称
     * @param module 模型的神经网络结构
     */
    protected DeepSeekModelBase(String name, Module module) {
        super(name, module);
    }

    /**
     * 获取模型配置
     */
    public abstract DeepSeekBaseConfig getBaseConfig();

    /**
     * 标准预测方法
     *
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }

    /**
     * 格式化参数数量为可读字符串
     *
     * @param count 参数数量
     * @return 格式化后的字符串（如 "1.23B"、"45.67M"）
     */
    protected static String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%,d", count);
        }
    }

    /**
     * 生成序列（贪婪解码）
     *
     * @param promptIds    提示词token ID序列 [batch_size, prompt_len]
     * @param maxNewTokens 最大生成token数量
     * @return 生成的完整序列 [batch_size, prompt_len + maxNewTokens]
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);

        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];

        // 复制提示词
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }

        // 自回归生成（预分配最大长度数组，避免每步创建新数组）
        int totalLen = promptLen + maxNewTokens;
        float[][] currentInput = new float[batchSize][totalLen];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, promptLen);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;

            // 创建当前长度的视图用于预测
            float[][] inputView = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(currentInput[b], 0, inputView[b], 0, currentLen);
            }

            // 预测下一个token
            Variable logits = predict(new Variable(NdArray.of(inputView)));
            NdArray logitsArray = logits.getValue();

            // 贪婪选择（选择概率最大的token）
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
                currentInput[b][currentLen] = nextToken;
            }
        }

        return NdArray.of(generatedSeq);
    }

    /**
     * 查找最大值的索引（argmax）
     *
     * @param logits   logits张量 [batch_size, seq_len, vocab_size]
     * @param batchIdx 批次索引
     * @param seqIdx   序列位置索引
     * @return 最大值对应的词汇表索引
     */
    protected static int argmax(NdArray logits, int batchIdx, int seqIdx) {
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
}
