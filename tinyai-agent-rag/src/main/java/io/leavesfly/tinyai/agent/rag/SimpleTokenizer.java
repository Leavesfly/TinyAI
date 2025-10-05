package io.leavesfly.tinyai.agent.rag;

import java.util.*;
import java.util.regex.Pattern;

/**
 * 简单的分词器类
 * 支持中英文分词，构建词汇表，并将词汇转换为ID
 */
public class SimpleTokenizer {
    private Map<String, Integer> vocab;     // 词汇表：词汇 -> ID
    private int vocabSize;                  // 词汇表大小
    private Pattern chinesePattern;         // 中文字符正则模式
    private Pattern punctuationPattern;     // 标点符号正则模式

    /**
     * 构造函数
     */
    public SimpleTokenizer() {
        this.vocab = new HashMap<>();
        this.vocabSize = 0;
        // 中文字符范围
        this.chinesePattern = Pattern.compile("[\\u4e00-\\u9fff]");
        // 标点符号模式（保留中英文字符和数字）
        this.punctuationPattern = Pattern.compile("[^\\w\\s\\u4e00-\\u9fff]");
    }

    /**
     * 分词方法
     * @param text 输入文本
     * @return 分词结果列表
     */
    public List<String> tokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }

        // 转换为小写
        text = text.toLowerCase();
        
        // 移除标点符号，保留中英文字符和数字
        text = punctuationPattern.matcher(text).replaceAll(" ");
        
        List<String> result = new ArrayList<>();
        
        // 使用正则表达式分离中英文
        // 匹配连续的英文单词或单个中文字符
        Pattern tokenPattern = Pattern.compile("[a-zA-Z]+|[\\u4e00-\\u9fff]");
        java.util.regex.Matcher matcher = tokenPattern.matcher(text);
        
        while (matcher.find()) {
            String token = matcher.group().trim();
            if (!token.isEmpty()) {
                result.add(token);
            }
        }
        
        return result;
    }

    /**
     * 构建词汇表
     * @param texts 文本列表
     */
    public void buildVocab(List<String> texts) {
        Map<String, Integer> wordCounts = new HashMap<>();
        
        // 统计词频
        for (String text : texts) {
            List<String> tokens = tokenize(text);
            for (String token : tokens) {
                wordCounts.put(token, wordCounts.getOrDefault(token, 0) + 1);
            }
        }
        
        // 按频率排序，构建词汇表
        List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<>(wordCounts.entrySet());
        sortedEntries.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        vocab.clear();
        int idx = 0;
        for (Map.Entry<String, Integer> entry : sortedEntries) {
            vocab.put(entry.getKey(), idx++);
        }
        
        vocabSize = vocab.size();
        System.out.println("构建词汇表完成，共 " + vocabSize + " 个词汇");
    }

    /**
     * 将词汇列表转换为ID列表
     * @param tokens 词汇列表
     * @return ID列表
     */
    public List<Integer> tokensToIds(List<String> tokens) {
        List<Integer> ids = new ArrayList<>();
        for (String token : tokens) {
            ids.add(vocab.getOrDefault(token, 0)); // 未知词汇使用ID 0
        }
        return ids;
    }

    /**
     * 获取词汇表
     * @return 词汇表映射
     */
    public Map<String, Integer> getVocab() {
        return new HashMap<>(vocab);
    }

    /**
     * 获取词汇表大小
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 检查是否包含指定词汇
     * @param token 词汇
     * @return 是否存在
     */
    public boolean hasToken(String token) {
        return vocab.containsKey(token);
    }

    /**
     * 获取词汇的ID
     * @param token 词汇
     * @return 词汇ID，不存在返回0
     */
    public int getTokenId(String token) {
        return vocab.getOrDefault(token, 0);
    }

    @Override
    public String toString() {
        return "SimpleTokenizer{" +
                "vocabSize=" + vocabSize +
                ", sampleVocab=" + vocab.entrySet().stream()
                .limit(5)
                .collect(HashMap::new, 
                        (m, e) -> m.put(e.getKey(), e.getValue()),
                        HashMap::putAll) +
                '}';
    }
}