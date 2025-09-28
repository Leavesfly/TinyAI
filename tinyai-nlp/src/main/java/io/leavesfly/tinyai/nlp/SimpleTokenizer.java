package io.leavesfly.tinyai.nlp;

import java.util.*;
import java.util.regex.Pattern;

/**
 * 简单的文本分词器
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * SimpleTokenizer类提供了基本的文本分词和编码功能，用于GPT-2模型的数据预处理。
 * 支持词汇表构建、文本到token ID的转换、token ID到文本的转换以及特殊token处理。
 */
public class SimpleTokenizer {
    
    /**
     * 填充token
     */
    public static final String PAD_TOKEN = "<pad>";
    
    /**
     * 未知token
     */
    public static final String UNK_TOKEN = "<unk>";
    
    /**
     * 序列开始token
     */
    public static final String BOS_TOKEN = "<bos>";
    
    /**
     * 序列结束token
     */
    public static final String EOS_TOKEN = "<eos>";
    
    /**
     * 填充token ID
     */
    public static final int PAD_ID = 0;
    
    /**
     * 未知token ID
     */
    public static final int UNK_ID = 1;
    
    /**
     * 序列开始token ID
     */
    public static final int BOS_ID = 2;
    
    /**
     * 序列结束token ID
     */
    public static final int EOS_ID = 3;
    
    /**
     * 词汇到ID的映射
     */
    private Map<String, Integer> vocab;
    
    /**
     * ID到词汇的映射
     */
    private Map<Integer, String> reverseVocab;
    
    /**
     * 词汇表大小
     */
    private int vocabSize;
    
    /**
     * 分词正则表达式
     */
    private Pattern tokenPattern;
    
    /**
     * 构造分词器
     */
    public SimpleTokenizer() {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        this.vocabSize = 0;
        
        // 简单的分词模式：按空格和标点符号分割
        this.tokenPattern = Pattern.compile("\\s+|(?=[.,!?;:])|(?<=[.,!?;:])");
        
        // 添加特殊token
        addSpecialTokens();
    }
    
    /**
     * 添加特殊token到词汇表
     */
    private void addSpecialTokens() {
        addToken(PAD_TOKEN, PAD_ID);
        addToken(UNK_TOKEN, UNK_ID);
        addToken(BOS_TOKEN, BOS_ID);
        addToken(EOS_TOKEN, EOS_ID);
        vocabSize = 4;
    }
    
    /**
     * 从文本集合构建词汇表
     * 
     * @param texts 文本集合
     * @param minFreq 最小词频阈值
     * @param maxVocabSize 最大词汇表大小
     */
    public void buildVocab(List<String> texts, int minFreq, int maxVocabSize) {
        // 统计词频
        Map<String, Integer> tokenFreq = new HashMap<>();
        
        for (String text : texts) {
            List<String> tokens = tokenize(text);
            for (String token : tokens) {
                tokenFreq.put(token, tokenFreq.getOrDefault(token, 0) + 1);
            }
        }
        
        // 过滤低频词并按频率排序
        List<Map.Entry<String, Integer>> sortedTokens = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : tokenFreq.entrySet()) {
            if (entry.getValue() >= minFreq) {
                sortedTokens.add(entry);
            }
        }
        
        // 按频率降序排序
        sortedTokens.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        // 限制词汇表大小
        int actualMaxSize = Math.min(maxVocabSize - 4, sortedTokens.size()); // 减去特殊token数量
        
        // 添加token到词汇表
        for (int i = 0; i < actualMaxSize; i++) {
            String token = sortedTokens.get(i).getKey();
            if (!vocab.containsKey(token)) {  // 避免重复添加特殊token
                addToken(token, vocabSize++);
            }
        }
        
        System.out.println("Vocabulary built with " + vocabSize + " tokens");
        System.out.println("Most frequent tokens: " + 
            sortedTokens.subList(0, Math.min(10, sortedTokens.size())));
    }
    
    /**
     * 添加token到词汇表
     * 
     * @param token 要添加的token
     * @param id token对应的ID
     */
    private void addToken(String token, int id) {
        vocab.put(token, id);
        reverseVocab.put(id, token);
    }
    
    /**
     * 将文本分词为token列表
     * 
     * @param text 输入文本
     * @return token列表
     */
    public List<String> tokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        // 转换为小写并分词
        String processedText = text.toLowerCase().trim();
        String[] tokens = tokenPattern.split(processedText);
        
        List<String> result = new ArrayList<>();
        for (String token : tokens) {
            token = token.trim();
            if (!token.isEmpty()) {
                result.add(token);
            }
        }
        
        return result;
    }
    
    /**
     * 将文本编码为token ID数组
     * 
     * @param text 输入文本
     * @param addSpecialTokens 是否添加特殊token
     * @return token ID数组
     */
    public int[] encode(String text, boolean addSpecialTokens) {
        List<String> tokens = tokenize(text);
        List<Integer> tokenIds = new ArrayList<>();
        
        // 添加开始token
        if (addSpecialTokens) {
            tokenIds.add(BOS_ID);
        }
        
        // 转换token为ID
        for (String token : tokens) {
            int tokenId = vocab.getOrDefault(token, UNK_ID);
            tokenIds.add(tokenId);
        }
        
        // 添加结束token
        if (addSpecialTokens) {
            tokenIds.add(EOS_ID);
        }
        
        // 转换为数组
        return tokenIds.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * 将文本编码为token ID数组（默认添加特殊token）
     * 
     * @param text 输入文本
     * @return token ID数组
     */
    public int[] encode(String text) {
        return encode(text, true);
    }
    
    /**
     * 将token ID数组解码为文本
     * 
     * @param tokenIds token ID数组
     * @param skipSpecialTokens 是否跳过特殊token
     * @return 解码后的文本
     */
    public String decode(int[] tokenIds, boolean skipSpecialTokens) {
        List<String> tokens = new ArrayList<>();
        
        for (int tokenId : tokenIds) {
            String token = reverseVocab.get(tokenId);
            if (token != null) {
                // 跳过特殊token（如果需要的话）
                if (skipSpecialTokens && isSpecialToken(token)) {
                    continue;
                }
                tokens.add(token);
            }
        }
        
        return String.join(" ", tokens);
    }
    
    /**
     * 将token ID数组解码为文本（默认跳过特殊token）
     * 
     * @param tokenIds token ID数组
     * @return 解码后的文本
     */
    public String decode(int[] tokenIds) {
        return decode(tokenIds, true);
    }
    
    /**
     * 检查是否为特殊token
     * 
     * @param token 要检查的token
     * @return 如果是特殊token返回true，否则返回false
     */
    private boolean isSpecialToken(String token) {
        return token.equals(PAD_TOKEN) || token.equals(UNK_TOKEN) || 
               token.equals(BOS_TOKEN) || token.equals(EOS_TOKEN);
    }
    
    /**
     * 将token ID列表填充到指定长度
     * 
     * @param tokenIds 原始token ID列表
     * @param maxLength 目标长度
     * @param padding 填充方式（"pre"或"post"）
     * @return 填充后的数组
     */
    public int[] pad(int[] tokenIds, int maxLength, String padding) {
        if (tokenIds.length >= maxLength) {
            // 如果长度超过maxLength，截断
            int[] result = new int[maxLength];
            System.arraycopy(tokenIds, 0, result, 0, maxLength);
            return result;
        }
        
        int[] padded = new int[maxLength];
        int padCount = maxLength - tokenIds.length;
        
        if ("pre".equals(padding)) {
            // 前置填充
            Arrays.fill(padded, 0, padCount, PAD_ID);
            System.arraycopy(tokenIds, 0, padded, padCount, tokenIds.length);
        } else {
            // 后置填充（默认）
            System.arraycopy(tokenIds, 0, padded, 0, tokenIds.length);
            Arrays.fill(padded, tokenIds.length, maxLength, PAD_ID);
        }
        
        return padded;
    }
    
    /**
     * 获取词汇表大小
     * 
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }
    
    /**
     * 获取token对应的ID
     * 
     * @param token 要查询的token
     * @return token对应的ID，如果不存在则返回UNK_ID
     */
    public int getTokenId(String token) {
        return vocab.getOrDefault(token, UNK_ID);
    }
    
    /**
     * 获取ID对应的token
     * 
     * @param id 要查询的ID
     * @return ID对应的token，如果不存在则返回UNK_TOKEN
     */
    public String getToken(int id) {
        return reverseVocab.getOrDefault(id, UNK_TOKEN);
    }
    
    /**
     * 检查词汇表是否包含指定token
     * 
     * @param token 要检查的token
     * @return 如果包含返回true，否则返回false
     */
    public boolean containsToken(String token) {
        return vocab.containsKey(token);
    }
    
    /**
     * 打印词汇表信息
     */
    public void printVocabInfo() {
        System.out.println("=== Tokenizer Information ===");
        System.out.println("Vocabulary Size: " + vocabSize);
        System.out.println("Special Tokens:");
        System.out.println("  PAD: " + PAD_TOKEN + " (" + PAD_ID + ")");
        System.out.println("  UNK: " + UNK_TOKEN + " (" + UNK_ID + ")");
        System.out.println("  BOS: " + BOS_TOKEN + " (" + BOS_ID + ")");
        System.out.println("  EOS: " + EOS_TOKEN + " (" + EOS_ID + ")");
        
        // 显示一些示例token
        System.out.println("Sample tokens:");
        int count = 0;
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            if (count >= 10) break;
            if (!isSpecialToken(entry.getKey())) {
                System.out.println("  " + entry.getKey() + " (" + entry.getValue() + ")");
                count++;
            }
        }
        System.out.println("=============================");
    }
}