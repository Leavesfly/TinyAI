package io.leavesfly.tinyai.deepseek.v3.training.demo;

import java.util.*;

/**
 * DeepSeek-V3 简单分词器工具类
 *
 * 提供文本编码和解码功能，支持词汇表构建和冻结。
 * 基于空格分词，将标点符号替换为空格，适用于教学演示场景。
 *
 * 注意：id=0 保留给 PAD token，避免与词汇 ID 冲突；词汇 ID 从 1 开始。
 *
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3TokenizerUtil {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private int nextId;
    private boolean frozen;

    /** PAD token 的 ID，用于序列填充 */
    public static final int PAD_TOKEN_ID = 0;

    public DeepSeekV3TokenizerUtil() {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        // id=0 保留给 PAD，词汇从 1 开始
        this.nextId = 1;
        this.frozen = false;
        // 预注册 PAD token
        this.vocab.put("<PAD>", PAD_TOKEN_ID);
        this.reverseVocab.put(PAD_TOKEN_ID, "<PAD>");
    }

    /**
     * 将文本编码为 token ID 序列
     *
     * 先将文本转小写并将非字母数字字符替换为空格，然后按空格分词。
     * 训练阶段（未冻结）会将新词加入词汇表；推理阶段（已冻结）未知词映射为 UNK（id=1）。
     *
     * @param text 输入文本
     * @return token ID 列表
     */
    public List<Integer> encode(String text) {
        String[] words = text.toLowerCase()
            .replaceAll("[^a-z0-9\\s]", " ")
            .split("\\s+");

        List<Integer> tokens = new ArrayList<>();
        for (String word : words) {
            if (word.isEmpty()) continue;

            if (!vocab.containsKey(word)) {
                if (!frozen) {
                    vocab.put(word, nextId);
                    reverseVocab.put(nextId, word);
                    nextId++;
                } else {
                    // 冻结后使用 UNK token (id=1，避免与 PAD 冲突)
                    tokens.add(1);
                    continue;
                }
            }
            tokens.add(vocab.get(word));
        }
        return tokens;
    }

    /**
     * 将 token ID 序列解码为文本
     *
     * 自动跳过 PAD token，用空格连接各词。
     *
     * @param tokens token ID 数组
     * @return 解码后的文本
     */
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            // 跳过 PAD token
            if (token == PAD_TOKEN_ID) continue;
            if (reverseVocab.containsKey(token)) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(reverseVocab.get(token));
            }
        }
        return sb.toString();
    }

    /**
     * 获取当前词汇表大小（含 PAD token）
     *
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return nextId;
    }

    /**
     * 冻结词汇表，冻结后不再接收新词
     */
    public void freeze() {
        this.frozen = true;
    }

    /**
     * 移除文本开头的任务标签（如 [REASONING]、[CODING]、[REWARD:0.9] 等）
     *
     * @param text 原始文本
     * @return 去除任务标签后的文本
     */
    public static String removeTaskLabel(String text) {
        return text.replaceFirst("^\\[\\w+[:\\w.]*\\]\\s*", "");
    }
}
