package io.leavesfly.tinyai.nlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * 中文分词器 - 专门用于古诗词文本处理
 * 
 * 提供以下功能：
 * 1. 中文字符级分词（适合古诗词）
 * 2. 词汇表构建和管理
 * 3. 文本编码和解码
 * 4. 特殊token处理（PAD, UNK, BOS, EOS）
 * 
 * @author 山泽
 * @version 1.0
 */
public class ChineseTokenizer {
    
    /** 特殊token定义 */
    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final String BOS_TOKEN = "<BOS>";
    public static final String EOS_TOKEN = "<EOS>";
    
    /** 特殊token ID */
    public static final int PAD_TOKEN_ID = 0;
    public static final int UNK_TOKEN_ID = 1;
    public static final int BOS_TOKEN_ID = 2;
    public static final int EOS_TOKEN_ID = 3;
    
    /** 词汇表：token -> id */
    private Map<String, Integer> tokenToId;
    
    /** 反向词汇表：id -> token */
    private Map<Integer, String> idToToken;
    
    /** 词汇表大小 */
    private int vocabSize;
    
    /**
     * 默认构造函数
     */
    public ChineseTokenizer() {
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        this.vocabSize = 0;
        
        // 添加特殊token
        addSpecialTokens();
    }
    
    /**
     * 添加特殊token到词汇表
     */
    private void addSpecialTokens() {
        addToken(PAD_TOKEN, PAD_TOKEN_ID);
        addToken(UNK_TOKEN, UNK_TOKEN_ID);
        addToken(BOS_TOKEN, BOS_TOKEN_ID);
        addToken(EOS_TOKEN, EOS_TOKEN_ID);
        vocabSize = 4;
    }
    
    /**
     * 从文件构建词汇表
     * 
     * @param filePath 文件路径
     * @throws IOException IO异常
     */
    public void buildVocabFromFile(String filePath) throws IOException {
        Set<String> uniqueTokens = new HashSet<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // 跳过空行和标题行
                if (line.trim().isEmpty() || isTitle(line)) {
                    continue;
                }
                
                // 中文字符级分词
                List<String> tokens = tokenizeToChars(line.trim());
                uniqueTokens.addAll(tokens);
            }
        }
        
        // 按字符顺序排序，确保词汇表的一致性
        List<String> sortedTokens = new ArrayList<>(uniqueTokens);
        Collections.sort(sortedTokens);
        
        // 添加到词汇表（跳过已存在的特殊token）
        for (String token : sortedTokens) {
            if (!tokenToId.containsKey(token)) {
                addToken(token, vocabSize);
                vocabSize++;
            }
        }
        
        System.out.println("词汇表构建完成，共 " + vocabSize + " 个token");
        System.out.println("包含特殊token: " + Arrays.asList(PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN));
    }
    
    /**
     * 判断是否为标题行
     * 
     * @param line 文本行
     * @return 是否为标题
     */
    private boolean isTitle(String line) {
        // 简单规则：没有标点符号且长度较短的行通常是标题
        return line.length() <= 10 && !line.contains("，") && !line.contains("。") 
               && !line.contains("？") && !line.contains("！");
    }
    
    /**
     * 中文字符级分词
     * 
     * @param text 输入文本
     * @return token列表
     */
    public List<String> tokenizeToChars(String text) {
        List<String> tokens = new ArrayList<>();
        
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            
            // 跳过空白字符
            if (Character.isWhitespace(ch)) {
                continue;
            }
            
            // 添加字符token
            tokens.add(String.valueOf(ch));
        }
        
        return tokens;
    }
    
    /**
     * 词级分词（简单版本）
     * 
     * @param text 输入文本
     * @return token列表
     */
    public List<String> tokenizeToWords(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder currentWord = new StringBuilder();
        
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            
            if (Character.isWhitespace(ch)) {
                if (currentWord.length() > 0) {
                    tokens.add(currentWord.toString());
                    currentWord.setLength(0);
                }
            } else if (isPunctuation(ch)) {
                if (currentWord.length() > 0) {
                    tokens.add(currentWord.toString());
                    currentWord.setLength(0);
                }
                tokens.add(String.valueOf(ch));
            } else {
                currentWord.append(ch);
            }
        }
        
        if (currentWord.length() > 0) {
            tokens.add(currentWord.toString());
        }
        
        return tokens;
    }
    
    /**
     * 判断是否为标点符号
     * 
     * @param ch 字符
     * @return 是否为标点符号
     */
    private boolean isPunctuation(char ch) {
        return "，。？！；：「」『』《》【】\"\"''（）".indexOf(ch) >= 0;
    }
    
    /**
     * 编码文本为token ID序列
     * 
     * @param text 输入文本
     * @param addSpecialTokens 是否添加特殊token（BOS/EOS）
     * @return token ID列表
     */
    public List<Integer> encode(String text, boolean addSpecialTokens) {
        List<Integer> tokenIds = new ArrayList<>();
        
        if (addSpecialTokens) {
            tokenIds.add(BOS_TOKEN_ID);
        }
        
        List<String> tokens = tokenizeToChars(text);
        for (String token : tokens) {
            int tokenId = tokenToId.getOrDefault(token, UNK_TOKEN_ID);
            tokenIds.add(tokenId);
        }
        
        if (addSpecialTokens) {
            tokenIds.add(EOS_TOKEN_ID);
        }

        
        return tokenIds;
    }
    
    /**
     * 编码文本为token ID序列（默认添加特殊token）
     * 
     * @param text 输入文本
     * @return token ID列表
     */
    public List<Integer> encode(String text) {
        return encode(text, true);
    }
    
    /**
     * 解码token ID序列为文本
     * 
     * @param tokenIds token ID列表
     * @param skipSpecialTokens 是否跳过特殊token
     * @return 解码后的文本
     */
    public String decode(List<Integer> tokenIds, boolean skipSpecialTokens) {
        StringBuilder text = new StringBuilder();
        
        for (int tokenId : tokenIds) {
            String token = idToToken.get(tokenId);
            if (token == null) {
                token = UNK_TOKEN;
            }
            
            // 跳过特殊token
            if (skipSpecialTokens && isSpecialToken(token)) {
                continue;
            }
            
            text.append(token);
        }
        
        return text.toString();
    }
    
    /**
     * 解码token ID序列为文本（默认跳过特殊token）
     * 
     * @param tokenIds token ID列表
     * @return 解码后的文本
     */
    public String decode(List<Integer> tokenIds) {
        return decode(tokenIds, false);
    }
    
    /**
     * 判断是否为特殊token
     * 
     * @param token token字符串
     * @return 是否为特殊token
     */
    private boolean isSpecialToken(String token) {
        return token.equals(PAD_TOKEN) || token.equals(UNK_TOKEN) || 
               token.equals(BOS_TOKEN) || token.equals(EOS_TOKEN);
    }
    
    /**
     * 添加token到词汇表
     * 
     * @param token token字符串
     * @param id token ID
     */
    private void addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }
    
    /**
     * 从文件加载训练数据并编码
     * 
     * @param filePath 文件路径
     * @return 编码后的序列列表
     * @throws IOException IO异常
     */
    public List<List<Integer>> loadAndEncodeData(String filePath) throws IOException {
        List<List<Integer>> encodedSequences = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // 跳过空行和标题行
                if (line.trim().isEmpty() || isTitle(line)) {
                    continue;
                }

                // 编码每一行为序列
                List<Integer> encoded = encode(line.trim());
                if (!encoded.isEmpty()) {
                    encodedSequences.add(encoded);
                }
            }
        }
        
        return encodedSequences;
    }
    
    /**
     * 创建训练数据对（输入-目标对）
     * 
     * @param sequences 原始序列列表
     * @param maxLength 最大序列长度
     * @return 输入-目标对列表
     */
    public List<TrainingPair> createTrainingPairs(List<List<Integer>> sequences, int maxLength) {
        List<TrainingPair> pairs = new ArrayList<>();
        
        for (List<Integer> sequence : sequences) {
            // 如果序列太长，进行截断
            if (sequence.size() > maxLength) {
                sequence = sequence.subList(0, maxLength);
            }
            
            // 创建输入-目标对（语言模型的标准做法）
            if (sequence.size() >= 2) {
                List<Integer> input = sequence.subList(0, sequence.size() - 1);
                List<Integer> target = sequence.subList(1, sequence.size());
                
                pairs.add(new TrainingPair(input, target));
            }
        }
        
        return pairs;
    }
    
    /**
     * 填充序列到指定长度
     * 
     * @param sequence 原始序列
     * @param targetLength 目标长度
     * @return 填充后的序列
     */
    public List<Integer> padSequence(List<Integer> sequence, int targetLength) {
        List<Integer> padded = new ArrayList<>(sequence);
        
        while (padded.size() < targetLength) {
            padded.add(PAD_TOKEN_ID);
        }
        
        return padded;
    }
    
    /**
     * 显示词汇表统计信息
     */
    public void printVocabStats() {
        System.out.println("=== 词汇表统计信息 ===");
        System.out.println("总词汇数: " + vocabSize);
        System.out.println("特殊token数: 4");
        System.out.println("普通token数: " + (vocabSize - 4));
        
        // 显示部分词汇表内容
        System.out.println("\n前20个token:");
        for (int i = 0; i < Math.min(20, vocabSize); i++) {
            String token = idToToken.get(i);
            System.out.printf("ID %d: %s\n", i, token);
        }
        
        if (vocabSize > 20) {
            System.out.println("...");
            System.out.println("最后几个token:");
            for (int i = Math.max(20, vocabSize - 5); i < vocabSize; i++) {
                String token = idToToken.get(i);
                System.out.printf("ID %d: %s\n", i, token);
            }
        }
    }
    
    // ==================== Getter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public Map<String, Integer> getTokenToId() {
        return tokenToId;
    }
    
    public Map<Integer, String> getIdToToken() {
        return idToToken;
    }
    
    /**
     * 训练数据对类
     */
    public static class TrainingPair {
        private final List<Integer> input;
        private final List<Integer> target;
        
        public TrainingPair(List<Integer> input, List<Integer> target) {
            this.input = new ArrayList<>(input);
            this.target = new ArrayList<>(target);
        }
        
        public List<Integer> getInput() {
            return input;
        }
        
        public List<Integer> getTarget() {
            return target;
        }
        
        @Override
        public String toString() {
            return String.format("TrainingPair{input=%s, target=%s}", input, target);
        }
    }
}