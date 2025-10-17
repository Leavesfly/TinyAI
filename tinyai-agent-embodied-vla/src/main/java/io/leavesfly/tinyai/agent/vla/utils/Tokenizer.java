package io.leavesfly.tinyai.agent.vla.utils;

import java.util.*;

/**
 * 简化的文本分词器
 * 用于将自然语言指令转换为Token ID序列
 * 
 * @author TinyAI
 */
public class Tokenizer {
    
    private Map<String, Integer> vocab;
    private Map<Integer, String> inverseVocab;
    private int maxVocabSize;
    
    // 特殊Token
    private static final String PAD_TOKEN = "<PAD>";
    private static final String UNK_TOKEN = "<UNK>";
    private static final String CLS_TOKEN = "<CLS>";
    private static final String SEP_TOKEN = "<SEP>";
    
    private static final int PAD_ID = 0;
    private static final int UNK_ID = 1;
    private static final int CLS_ID = 2;
    private static final int SEP_ID = 3;
    
    /**
     * 构造函数 - 使用默认词汇表
     */
    public Tokenizer() {
        this(10000);
    }
    
    /**
     * 构造函数 - 指定词汇表大小
     */
    public Tokenizer(int maxVocabSize) {
        this.maxVocabSize = maxVocabSize;
        this.vocab = new HashMap<>();
        this.inverseVocab = new HashMap<>();
        
        initializeVocab();
    }
    
    /**
     * 初始化基础词汇表
     */
    private void initializeVocab() {
        // 添加特殊Token
        vocab.put(PAD_TOKEN, PAD_ID);
        vocab.put(UNK_TOKEN, UNK_ID);
        vocab.put(CLS_TOKEN, CLS_ID);
        vocab.put(SEP_TOKEN, SEP_ID);
        
        // 添加常用英文单词
        String[] commonWords = {
            "pick", "place", "move", "grasp", "release", "navigate", "go", "to",
            "the", "a", "an", "red", "blue", "green", "yellow", "black", "white",
            "cube", "ball", "box", "object", "target", "left", "right", "up", "down",
            "forward", "backward", "open", "close", "push", "pull", "rotate", "turn",
            "drawer", "door", "bottle", "can", "container", "table", "floor", "shelf"
        };
        
        int id = 4;
        for (String word : commonWords) {
            vocab.put(word, id);
            inverseVocab.put(id, word);
            id++;
        }
    }
    
    /**
     * 编码文本为Token ID序列
     * 
     * @param text 输入文本
     * @return Token ID数组
     */
    public int[] encode(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new int[]{CLS_ID, SEP_ID};
        }
        
        // 简单的空格分词 + 转小写
        String[] tokens = text.toLowerCase().split("\\s+");
        List<Integer> tokenIds = new ArrayList<>();
        
        // 添加CLS token
        tokenIds.add(CLS_ID);
        
        // 编码每个单词
        for (String token : tokens) {
            int id = vocab.getOrDefault(token, UNK_ID);
            tokenIds.add(id);
        }
        
        // 添加SEP token
        tokenIds.add(SEP_ID);
        
        // 转换为数组
        int[] result = new int[tokenIds.size()];
        for (int i = 0; i < tokenIds.size(); i++) {
            result[i] = tokenIds.get(i);
        }
        
        return result;
    }
    
    /**
     * 解码Token ID序列为文本
     * 
     * @param tokenIds Token ID数组
     * @return 解码后的文本
     */
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        
        for (int id : tokenIds) {
            if (id == PAD_ID || id == CLS_ID || id == SEP_ID) {
                continue;
            }
            
            String token = inverseVocab.getOrDefault(id, UNK_TOKEN);
            sb.append(token).append(" ");
        }
        
        return sb.toString().trim();
    }
    
    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocab.size();
    }
    
    /**
     * 添加新词到词汇表
     */
    public void addWord(String word) {
        if (!vocab.containsKey(word) && vocab.size() < maxVocabSize) {
            int id = vocab.size();
            vocab.put(word, id);
            inverseVocab.put(id, word);
        }
    }
}
