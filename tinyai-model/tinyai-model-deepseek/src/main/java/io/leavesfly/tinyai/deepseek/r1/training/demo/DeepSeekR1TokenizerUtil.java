package io.leavesfly.tinyai.deepseek.r1.training.demo;

import java.util.*;

/**
 * DeepSeek-R1简单分词器工具类
 * 
 * 提供文本编码和解码功能，支持词汇表构建和冻结。
 * 包含行业标准的 Chat Template 特殊 Token 定义，
 * 用于结构化标记用户指令、助手回复和推理过程。
 * 
 * Chat Template 格式（对齐 DeepSeek-R1 论文）:
 * <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>{reasoning}<|end_of_thought|>\n{answer}<|im_end|>
 * 
 * @author leavesfly
 */
public class DeepSeekR1TokenizerUtil {
    
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private int nextId;
    private boolean frozen;
    
    // ========== 特殊 Token ID 常量 ==========
    public static final int PAD_TOKEN_ID = 0;
    public static final int UNK_TOKEN_ID = 1;
    public static final int IM_START_TOKEN_ID = 2;
    public static final int IM_END_TOKEN_ID = 3;
    public static final int BEGIN_OF_THOUGHT_TOKEN_ID = 4;
    public static final int END_OF_THOUGHT_TOKEN_ID = 5;
    
    // ========== 特殊 Token 文本 ==========
    public static final String PAD_TOKEN = "<PAD>";
    public static final String UNK_TOKEN = "<UNK>";
    public static final String IM_START_TOKEN = "<|im_start|>";
    public static final String IM_END_TOKEN = "<|im_end|>";
    public static final String BEGIN_OF_THOUGHT_TOKEN = "<|begin_of_thought|>";
    public static final String END_OF_THOUGHT_TOKEN = "<|end_of_thought|>";
    
    // ========== 角色标识 ==========
    public static final String ROLE_USER = "user";
    public static final String ROLE_ASSISTANT = "assistant";
    
    private static final int FIRST_NORMAL_TOKEN_ID = 6;
    
    public DeepSeekR1TokenizerUtil() {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        this.frozen = false;
        
        // 预注册所有特殊 Token
        registerSpecialToken(PAD_TOKEN, PAD_TOKEN_ID);
        registerSpecialToken(UNK_TOKEN, UNK_TOKEN_ID);
        registerSpecialToken(IM_START_TOKEN, IM_START_TOKEN_ID);
        registerSpecialToken(IM_END_TOKEN, IM_END_TOKEN_ID);
        registerSpecialToken(BEGIN_OF_THOUGHT_TOKEN, BEGIN_OF_THOUGHT_TOKEN_ID);
        registerSpecialToken(END_OF_THOUGHT_TOKEN, END_OF_THOUGHT_TOKEN_ID);
        
        this.nextId = FIRST_NORMAL_TOKEN_ID;
    }
    
    private void registerSpecialToken(String token, int tokenId) {
        vocab.put(token, tokenId);
        reverseVocab.put(tokenId, token);
    }
    
    /**
     * 编码文本为token ID序列
     * 
     * 支持在文本中嵌入特殊 Token（如 <|im_start|>），
     * 会先按特殊 Token 分段，再对普通文本做空格分词。
     */
    public List<Integer> encode(String text) {
        List<Integer> tokens = new ArrayList<>();
        
        // 按特殊 Token 分段处理
        List<String> segments = splitBySpecialTokens(text);
        for (String segment : segments) {
            if (vocab.containsKey(segment)) {
                tokens.add(vocab.get(segment));
            } else {
                tokens.addAll(encodeNormalText(segment));
            }
        }
        return tokens;
    }
    
    /**
     * 将文本按特殊 Token 分割为段落列表
     * 
     * 例如 "<|im_start|>user hello<|im_end|>" 
     * 分割为 ["<|im_start|>", "user hello", "<|im_end|>"]
     */
    private List<String> splitBySpecialTokens(String text) {
        List<String> segments = new ArrayList<>();
        String[] specialTokens = {
            IM_START_TOKEN, IM_END_TOKEN,
            BEGIN_OF_THOUGHT_TOKEN, END_OF_THOUGHT_TOKEN
        };
        
        int pos = 0;
        while (pos < text.length()) {
            int earliestIdx = text.length();
            String matchedToken = null;
            
            for (String special : specialTokens) {
                int idx = text.indexOf(special, pos);
                if (idx >= 0 && idx < earliestIdx) {
                    earliestIdx = idx;
                    matchedToken = special;
                }
            }
            
            if (matchedToken != null && earliestIdx < text.length()) {
                if (earliestIdx > pos) {
                    segments.add(text.substring(pos, earliestIdx));
                }
                segments.add(matchedToken);
                pos = earliestIdx + matchedToken.length();
            } else {
                if (pos < text.length()) {
                    segments.add(text.substring(pos));
                }
                break;
            }
        }
        return segments;
    }
    
    /**
     * 编码普通文本（空格分词）
     */
    private List<Integer> encodeNormalText(String text) {
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
                    tokens.add(UNK_TOKEN_ID);
                    continue;
                }
            }
            tokens.add(vocab.get(word));
        }
        return tokens;
    }
    
    /**
     * 解码token ID序列为文本
     */
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            if (token == PAD_TOKEN_ID) continue;
            if (reverseVocab.containsKey(token)) {
                String word = reverseVocab.get(token);
                if (isSpecialToken(token)) {
                    sb.append(word);
                } else {
                    if (sb.length() > 0 && !endsWithSpecialToken(sb)) {
                        sb.append(" ");
                    }
                    sb.append(word);
                }
            }
        }
        return sb.toString();
    }
    
    private boolean isSpecialToken(int tokenId) {
        return tokenId >= PAD_TOKEN_ID && tokenId < FIRST_NORMAL_TOKEN_ID;
    }
    
    private boolean endsWithSpecialToken(StringBuilder sb) {
        String str = sb.toString();
        return str.endsWith(">") || str.endsWith("\n");
    }
    
    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return nextId;
    }
    
    /**
     * 冻结词汇表，不再增加新词
     */
    public void freeze() {
        this.frozen = true;
    }
    
    // ========== Chat Template 构建方法 ==========
    
    /**
     * 构建后训练（SFT）格式的完整对话文本
     * 
     * 格式: <|im_start|>user {question}<|im_end|><|im_start|>assistant <|begin_of_thought|>{reasoning}<|end_of_thought|> {answer}<|im_end|>
     * 
     * @param question 用户问题
     * @param reasoning 推理过程（Chain-of-Thought）
     * @param answer 最终答案
     * @return 格式化后的完整对话文本
     */
    public static String buildSFTText(String question, String reasoning, String answer) {
        StringBuilder sb = new StringBuilder();
        sb.append(IM_START_TOKEN).append(ROLE_USER).append(" ");
        sb.append(question);
        sb.append(IM_END_TOKEN);
        sb.append(IM_START_TOKEN).append(ROLE_ASSISTANT).append(" ");
        sb.append(BEGIN_OF_THOUGHT_TOKEN);
        sb.append(reasoning);
        sb.append(END_OF_THOUGHT_TOKEN);
        sb.append(" ").append(answer);
        sb.append(IM_END_TOKEN);
        return sb.toString();
    }
    
    /**
     * 计算 Loss Mask：标记哪些 token 位置应参与 loss 计算
     * 
     * 行业标准做法：只对 assistant 回复部分（包括推理和答案）计算 loss，
     * user 指令部分不参与梯度更新。
     * 
     * @param tokenIds 完整序列的 token ID 数组
     * @return loss mask 数组，1.0f 表示参与 loss 计算，0.0f 表示忽略
     */
    public static float[] computeLossMask(int[] tokenIds) {
        float[] mask = new float[tokenIds.length];
        boolean inAssistantRegion = false;
        
        for (int i = 0; i < tokenIds.length; i++) {
            if (tokenIds[i] == PAD_TOKEN_ID) {
                mask[i] = 0.0f;
                continue;
            }
            
            if (tokenIds[i] == IM_START_TOKEN_ID) {
                // 检查下一个 token 是否为 "assistant" 角色
                // 在 assistant 区域开始后，后续 token 参与 loss
                // 先标记 <|im_start|> 本身不参与 loss
                mask[i] = 0.0f;
                
                if (i + 1 < tokenIds.length && isAssistantRoleToken(tokenIds[i + 1])) {
                    inAssistantRegion = true;
                } else {
                    inAssistantRegion = false;
                }
                continue;
            }
            
            if (tokenIds[i] == IM_END_TOKEN_ID) {
                // <|im_end|> 在 assistant 区域内参与 loss（模型需要学会何时停止）
                mask[i] = inAssistantRegion ? 1.0f : 0.0f;
                inAssistantRegion = false;
                continue;
            }
            
            mask[i] = inAssistantRegion ? 1.0f : 0.0f;
        }
        
        return mask;
    }
    
    /**
     * 判断 token ID 是否对应 "assistant" 角色标识
     * 
     * 由于分词器会将 "assistant" 编码为一个普通 word token，
     * 这里通过反查词汇表来判断。使用静态方法兼容不同实例。
     */
    private static boolean isAssistantRoleToken(int tokenId) {
        // assistant 角色 token 的 ID 在词汇表构建时动态分配，
        // 但 "assistant" 这个词一定会被注册。
        // 这里用一个简单的标记：在 SFT 文本中，<|im_start|> 后紧跟角色名，
        // 我们通过 token ID 范围判断（非特殊 token 且 >= FIRST_NORMAL_TOKEN_ID）
        return tokenId >= FIRST_NORMAL_TOKEN_ID;
    }
    
    /**
     * 计算 Loss Mask（增强版）：使用分词器实例精确判断 assistant 角色
     * 
     * @param tokenIds 完整序列的 token ID 数组
     * @return loss mask 数组
     */
    public float[] computeLossMaskWithTokenizer(int[] tokenIds) {
        float[] mask = new float[tokenIds.length];
        boolean inAssistantRegion = false;
        
        // 获取 "assistant" 对应的 token ID
        Integer assistantTokenId = vocab.get(ROLE_ASSISTANT);
        
        for (int i = 0; i < tokenIds.length; i++) {
            if (tokenIds[i] == PAD_TOKEN_ID) {
                mask[i] = 0.0f;
                continue;
            }
            
            if (tokenIds[i] == IM_START_TOKEN_ID) {
                mask[i] = 0.0f;
                if (assistantTokenId != null && i + 1 < tokenIds.length 
                        && tokenIds[i + 1] == assistantTokenId) {
                    inAssistantRegion = true;
                } else {
                    inAssistantRegion = false;
                }
                continue;
            }
            
            if (tokenIds[i] == IM_END_TOKEN_ID) {
                mask[i] = inAssistantRegion ? 1.0f : 0.0f;
                inAssistantRegion = false;
                continue;
            }
            
            mask[i] = inAssistantRegion ? 1.0f : 0.0f;
        }
        
        return mask;
    }
    
    // ========== 旧格式兼容方法 ==========
    
    /**
     * 移除文本中的标签（任务类型、奖励等）
     */
    public static String removeLabels(String text) {
        return text.replaceFirst("^\\[REWARD:[\\d.]+\\]\\s*", "")
                   .replaceFirst("^\\[TYPE:\\w+\\]\\s*", "")
                   .replaceFirst("^\\[\\w+\\]\\s*", "");
    }
    
    /**
     * 提取RLHF奖励值
     */
    public static float extractReward(String text) {
        if (text.startsWith("[REWARD:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 8) {
                try {
                    return Float.parseFloat(text.substring(8, endIdx));
                } catch (NumberFormatException e) {
                    return 0.5f;
                }
            }
        }
        return 0.5f;
    }
    
    /**
     * 提取RLVR验证器类型
     */
    public static String extractVerifierType(String text) {
        if (text.startsWith("[TYPE:")) {
            int endIdx = text.indexOf("]");
            if (endIdx > 6) {
                return text.substring(6, endIdx).trim();
            }
        }
        return "math";
    }
    
    /**
     * 从旧格式后训练数据中解析出 question、reasoning、answer 三部分
     * 
     * 旧格式: [TASK] Question: {q} {reasoning_process} Answer: {a}
     * 
     * @param text 旧格式文本
     * @return [question, reasoning, answer]，解析失败时 reasoning 和 answer 可能为空
     */
    public static String[] parseOldFormatQA(String text) {
        String cleanText = removeLabels(text);
        
        String question = "";
        String reasoning = "";
        String answer = "";
        
        // 提取 Question 部分
        int questionIdx = cleanText.indexOf("Question:");
        int answerIdx = cleanText.lastIndexOf("Answer:");
        
        if (questionIdx >= 0 && answerIdx > questionIdx) {
            question = cleanText.substring(questionIdx + "Question:".length(), answerIdx).trim();
            answer = cleanText.substring(answerIdx + "Answer:".length()).trim();
            
            // 从 question 中分离推理过程
            // 查找推理标记词（Think/Reasoning/Steps/Deduction 等）
            String[] reasoningMarkers = {
                "Let me think", "Think:", "Reasoning:", "Steps:", 
                "Deduction:", "Using modus", "By ", "Since "
            };
            
            int reasoningStart = -1;
            for (String marker : reasoningMarkers) {
                int idx = question.indexOf(marker);
                if (idx >= 0 && (reasoningStart < 0 || idx < reasoningStart)) {
                    reasoningStart = idx;
                }
            }
            
            if (reasoningStart > 0) {
                reasoning = question.substring(reasoningStart).trim();
                question = question.substring(0, reasoningStart).trim();
                // 移除末尾的标点
                if (question.endsWith("?") || question.endsWith(".")) {
                    question = question.substring(0, question.length() - 1).trim();
                }
            }
        } else {
            question = cleanText;
        }
        
        return new String[] { question, reasoning, answer };
    }
}
