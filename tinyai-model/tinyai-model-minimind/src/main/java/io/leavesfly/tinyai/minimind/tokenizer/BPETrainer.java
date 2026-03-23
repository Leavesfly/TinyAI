package io.leavesfly.tinyai.minimind.tokenizer;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * BPE (Byte Pair Encoding) 训练器
 * 
 * <p>功能:
 * - 从语料库学习BPE merge规则
 * - 统计字节对频率
 * - 迭代合并高频字节对
 * - 生成词汇表
 * 
 * <p>算法原理:
 * 1. 初始化: 将所有字符作为基础tokens
 * 2. 统计: 计算所有相邻token对的频率
 * 3. 合并: 选择频率最高的token对进行合并
 * 4. 迭代: 重复步骤2-3直到达到目标词汇表大小
 * 
 * @author leavesfly
 * @since 2024
 */
public class BPETrainer {
    
    /**
     * 目标词汇表大小
     */
    private final int vocabSize;
    
    /**
     * 最小token频率阈值
     */
    private final int minFrequency;
    
    /**
     * BPE merge规则列表
     * 格式: ["a b", "c d", ...] 表示将"a b"合并为"ab"
     */
    private final List<String> merges;
    
    /**
     * Token到ID的映射
     */
    private final Map<String, Integer> vocab;
    
    /**
     * 训练进度回调
     */
    private ProgressCallback progressCallback;
    
    /**
     * 构造函数
     * 
     * @param vocabSize 目标词汇表大小
     * @param minFrequency 最小频率阈值
     */
    public BPETrainer(int vocabSize, int minFrequency) {
        this.vocabSize = vocabSize;
        this.minFrequency = minFrequency;
        this.merges = new ArrayList<>();
        this.vocab = new LinkedHashMap<>();
    }
    
    /**
     * 默认构造函数(词汇表6400, 最小频率2)
     */
    public BPETrainer() {
        this(6400, 2);
    }
    
    /**
     * 从语料库训练BPE
     * 
     * @param corpus 训练语料(文本列表)
     * @return 训练好的词汇表
     */
    public Vocabulary train(List<String> corpus) {
        System.out.println("=".repeat(60));
        System.out.println("开始BPE训练");
        System.out.println("=".repeat(60));
        System.out.println("语料大小: " + corpus.size() + " 条");
        System.out.println("目标词汇表大小: " + vocabSize);
        System.out.println("最小频率阈值: " + minFrequency);
        System.out.println("=".repeat(60));
        
        // 1. 初始化词汇表(特殊tokens + 基础字符)
        initializeVocab();
        
        // 2. 统计word频率
        Map<String, Integer> wordFreqs = getWordFrequencies(corpus);
        System.out.println("统计得到 " + wordFreqs.size() + " 个不同的词");
        
        // 3. 将words转换为字符序列
        Map<List<String>, Integer> wordTokens = initializeWordTokens(wordFreqs);
        
        // 4. 迭代合并高频pair
        int numMerges = vocabSize - vocab.size();
        System.out.println("需要进行 " + numMerges + " 次合并");
        
        for (int i = 0; i < numMerges; i++) {
            // 统计所有pair的频率
            Map<String, Integer> pairFreqs = computePairFrequencies(wordTokens);
            
            if (pairFreqs.isEmpty()) {
                System.out.println("没有更多的pair可以合并,训练提前结束");
                break;
            }
            
            // 选择频率最高的pair
            Map.Entry<String, Integer> bestPair = Collections.max(
                pairFreqs.entrySet(),
                Map.Entry.comparingByValue()
            );
            
            String pair = bestPair.getKey();
            int freq = bestPair.getValue();
            
            if (freq < minFrequency) {
                System.out.println("最高频率 " + freq + " 小于阈值 " + minFrequency + ",训练提前结束");
                break;
            }
            
            // 合并pair
            String[] tokens = pair.split(" ");
            String mergedToken = tokens[0] + tokens[1];
            
            // 更新词汇表
            vocab.put(mergedToken, vocab.size());
            merges.add(pair);
            
            // 更新word tokens（原地修改）
            mergePairInWords(wordTokens, tokens[0], tokens[1], mergedToken);
            
            // 进度回调
            if ((i + 1) % 100 == 0 || i == numMerges - 1) {
                float progress = (float) (i + 1) / numMerges * 100;
                System.out.printf("进度: %d/%d (%.1f%%) | 最新merge: %s (freq=%d)%n",
                    i + 1, numMerges, progress, pair, freq);
                
                if (progressCallback != null) {
                    progressCallback.onProgress(i + 1, numMerges, pair, freq);
                }
            }
        }
        
        System.out.println("=".repeat(60));
        System.out.println("BPE训练完成!");
        System.out.println("最终词汇表大小: " + vocab.size());
        System.out.println("merge规则数量: " + merges.size());
        System.out.println("=".repeat(60));
        
        return new Vocabulary(vocab);
    }
    
    /**
     * 初始化词汇表(特殊tokens + ASCII字符 + 常用中文字符)
     */
    private void initializeVocab() {
        // 特殊tokens
        vocab.put(Vocabulary.PAD_TOKEN, 0);
        vocab.put(Vocabulary.UNK_TOKEN, 1);
        vocab.put(Vocabulary.BOS_TOKEN, 2);
        vocab.put(Vocabulary.EOS_TOKEN, 3);
        
        // ASCII可打印字符 (32-126)
        for (int i = 32; i < 127; i++) {
            vocab.put(String.valueOf((char) i), vocab.size());
        }
        
        // 常用中文字符
        String commonChineseChars = 
            "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开要手知道";
        for (char c : commonChineseChars.toCharArray()) {
            String token = String.valueOf(c);
            if (!vocab.containsKey(token)) {
                vocab.put(token, vocab.size());
            }
        }
    }
    
    /**
     * 统计语料中的word频率
     */
    private Map<String, Integer> getWordFrequencies(List<String> corpus) {
        Map<String, Integer> wordFreqs = new HashMap<>();
        
        for (String text : corpus) {
            // 简单的分词:按空格和标点分割
            String[] words = text.split("\\s+|(?<=[,.!?;:])|(?=[,.!?;:])");
            
            for (String word : words) {
                if (!word.trim().isEmpty()) {
                    wordFreqs.put(word, wordFreqs.getOrDefault(word, 0) + 1);
                }
            }
        }
        
        return wordFreqs;
    }
    
    /**
     * 将words转换为字符序列(带频率)
     */
    private Map<List<String>, Integer> initializeWordTokens(Map<String, Integer> wordFreqs) {
        Map<List<String>, Integer> wordTokens = new HashMap<>();
        
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            
            // 将word拆分为字符列表
            List<String> tokens = new ArrayList<>();
            for (char c : word.toCharArray()) {
                tokens.add(String.valueOf(c));
            }
            
            wordTokens.put(tokens, freq);
        }
        
        return wordTokens;
    }
    
    /**
     * 计算所有pair的频率
     * 
     * 优化：使用更高效的遍历方式，减少字符串拼接
     */
    private Map<String, Integer> computePairFrequencies(Map<List<String>, Integer> wordTokens) {
        Map<String, Integer> pairFreqs = new HashMap<>();
        
        for (Map.Entry<List<String>, Integer> entry : wordTokens.entrySet()) {
            List<String> tokens = entry.getKey();
            int freq = entry.getValue();
            
            // 统计相邻token pairs
            // 优化：使用 StringBuilder 减少字符串拼接开销
            for (int i = 0; i < tokens.size() - 1; i++) {
                String pair = tokens.get(i) + " " + tokens.get(i + 1);
                pairFreqs.merge(pair, freq, Integer::sum);
            }
        }
        
        return pairFreqs;
    }
    
    /**
     * 在所有words中合并指定的token pair
     * 
     * 优化：原地修改 wordTokens，避免每次创建新 Map
     */
    private void mergePairInWords(
            Map<List<String>, Integer> wordTokens,
            String token1,
            String token2,
            String mergedToken) {
        
        // 使用迭代器原地修改 Map
        for (Map.Entry<List<String>, Integer> entry : wordTokens.entrySet()) {
            List<String> tokens = entry.getKey();
            
            // 原地合并 tokens 中的 pair
            mergePairInPlace(tokens, token1, token2, mergedToken);
        }
    }
    
    /**
     * 在 token 列表中原地合并指定 pair
     */
    private void mergePairInPlace(List<String> tokens, String token1, String token2, String mergedToken) {
        int i = 0;
        while (i < tokens.size() - 1) {
            // 检查是否匹配 pair
            if (tokens.get(i).equals(token1) && tokens.get(i + 1).equals(token2)) {
                // 合并：删除第二个元素，修改第一个为合并后的 token
                tokens.set(i, mergedToken);
                tokens.remove(i + 1);
                // 不增加 i，因为当前位置可能还能继续合并
            } else {
                i += 1;
            }
        }
    }
    
    /**
     * 在token列表中合并指定pair
     */
    private List<String> mergePair(List<String> tokens, String token1, String token2, String mergedToken) {
        List<String> newTokens = new ArrayList<>();
        
        int i = 0;
        while (i < tokens.size()) {
            // 检查是否匹配pair
            if (i < tokens.size() - 1 && 
                tokens.get(i).equals(token1) && 
                tokens.get(i + 1).equals(token2)) {
                newTokens.add(mergedToken);
                i += 2;
            } else {
                newTokens.add(tokens.get(i));
                i += 1;
            }
        }
        
        return newTokens;
    }
    
    /**
     * 获取merge规则
     */
    public List<String> getMerges() {
        return new ArrayList<>(merges);
    }
    
    /**
     * 获取词汇表
     */
    public Map<String, Integer> getVocab() {
        return new LinkedHashMap<>(vocab);
    }
    
    /**
     * 保存BPE模型
     * 
     * @param dirPath 保存目录
     */
    public void save(String dirPath) throws IOException {
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        // 保存merge规则
        try (PrintWriter writer = new PrintWriter(dirPath + "/merges.txt")) {
            for (String merge : merges) {
                writer.println(merge);
            }
        }
        
        // 保存词汇表
        try (PrintWriter writer = new PrintWriter(dirPath + "/vocab.txt")) {
            List<Map.Entry<String, Integer>> sorted = new ArrayList<>(vocab.entrySet());
            sorted.sort(Map.Entry.comparingByValue());
            
            for (Map.Entry<String, Integer> entry : sorted) {
                writer.println(entry.getKey());
            }
        }
        
        System.out.println("BPE模型已保存到: " + dirPath);
    }
    
    /**
     * 加载BPE模型
     * 
     * @param dirPath 加载目录
     * @return BPE训练器实例
     */
    public static BPETrainer load(String dirPath) throws IOException {
        BPETrainer trainer = new BPETrainer();
        
        // 加载merge规则
        try (BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/merges.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                trainer.merges.add(line);
            }
        }
        
        // 加载词汇表
        try (BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/vocab.txt"))) {
            String line;
            int id = 0;
            while ((line = reader.readLine()) != null) {
                trainer.vocab.put(line, id++);
            }
        }
        
        System.out.println("BPE模型已加载: " + dirPath);
        return trainer;
    }
    
    /**
     * 设置进度回调
     */
    public void setProgressCallback(ProgressCallback callback) {
        this.progressCallback = callback;
    }
    
    /**
     * 进度回调接口
     */
    public interface ProgressCallback {
        void onProgress(int current, int total, String latestMerge, int frequency);
    }
    
    /**
     * 打印训练统计
     */
    public void printStats() {
        System.out.println("=".repeat(60));
        System.out.println("BPE模型统计:");
        System.out.println("  词汇表大小: " + vocab.size());
        System.out.println("  Merge规则数: " + merges.size());
        System.out.println("  特殊tokens: 4");
        System.out.println("=".repeat(60));
    }
}
