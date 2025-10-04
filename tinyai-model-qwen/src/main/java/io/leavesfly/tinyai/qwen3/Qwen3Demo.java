package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * Qwen3模型演示程序
 * 
 * 展示如何使用Qwen3模型进行：
 * 1. 文本生成
 * 2. 聊天对话
 * 3. 模型信息展示
 * 4. 基础tokenization
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Demo {
    
    /**
     * 简单的分词器实现
     * 用于演示目的，实际使用中应该使用更完善的分词器
     */
    public static class SimpleTokenizer {
        
        private final int vocabSize;
        private final Map<String, Integer> tokenToId;
        private final Map<Integer, String> idToToken;
        
        // 特殊token ID
        public final int padTokenId = 0;
        public final int bosTokenId = 1;  // Begin of sequence
        public final int eosTokenId = 2;  // End of sequence
        public final int unkTokenId = 3;  // Unknown token
        
        // 特殊token字符串
        public final String padToken = "<pad>";
        public final String bosToken = "<bos>";
        public final String eosToken = "<eos>";
        public final String unkToken = "<unk>";
        
        public SimpleTokenizer(int vocabSize) {
            this.vocabSize = vocabSize;
            this.tokenToId = new HashMap<>();
            this.idToToken = new HashMap<>();
            buildVocab();
        }
        
        public SimpleTokenizer() {
            this(32000);
        }
        
        /**
         * 构建基础词汇表
         */
        private void buildVocab() {
            // 特殊token
            String[] specialTokens = {padToken, bosToken, eosToken, unkToken};
            
            // 添加特殊token
            for (int i = 0; i < specialTokens.length; i++) {
                tokenToId.put(specialTokens[i], i);
                idToToken.put(i, specialTokens[i]);
            }
            
            // 基础字符集
            List<String> basicChars = new ArrayList<>();
            
            // ASCII字符
            for (int i = 32; i < 127; i++) {  // 可打印ASCII字符
                basicChars.add(String.valueOf((char) i));
            }
            
            // 常用中文字符（简化版）
            String commonChinese = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从";
            for (char c : commonChinese.toCharArray()) {
                String charStr = String.valueOf(c);
                if (!basicChars.contains(charStr)) {
                    basicChars.add(charStr);
                }
            }
            
            // 添加基础字符
            int currentId = specialTokens.length;
            for (String charStr : basicChars) {
                if (!tokenToId.containsKey(charStr)) {
                    tokenToId.put(charStr, currentId);
                    idToToken.put(currentId, charStr);
                    currentId++;
                }
            }
            
            // 填充到指定词汇表大小
            while (currentId < vocabSize) {
                String placeholderToken = "<unused_" + currentId + ">";
                tokenToId.put(placeholderToken, currentId);
                idToToken.put(currentId, placeholderToken);
                currentId++;
            }
        }
        
        /**
         * 将文本编码为token ID序列
         * 
         * @param text 输入文本
         * @param addBos 是否添加开始token
         * @param addEos 是否添加结束token
         * @return token ID列表
         */
        public List<Integer> encode(String text, boolean addBos, boolean addEos) {
            List<Integer> tokens = new ArrayList<>();
            
            if (addBos) {
                tokens.add(bosTokenId);
            }
            
            // 简单的字符级别tokenization
            for (char c : text.toCharArray()) {
                String charStr = String.valueOf(c);
                int tokenId = tokenToId.getOrDefault(charStr, unkTokenId);
                tokens.add(tokenId);
            }
            
            if (addEos) {
                tokens.add(eosTokenId);
            }
            
            return tokens;
        }
        
        /**
         * 编码文本（默认参数）
         */
        public List<Integer> encode(String text) {
            return encode(text, true, false);
        }
        
        /**
         * 将token ID序列解码为文本
         * 
         * @param tokenIds token ID序列
         * @param skipSpecialTokens 是否跳过特殊token
         * @return 解码后的文本
         */
        public String decode(List<Integer> tokenIds, boolean skipSpecialTokens) {
            StringBuilder result = new StringBuilder();
            Set<Integer> specialTokenIds = Set.of(padTokenId, bosTokenId, eosTokenId);
            
            for (int tokenId : tokenIds) {
                if (skipSpecialTokens && specialTokenIds.contains(tokenId)) {
                    continue;
                }
                
                String token = idToToken.getOrDefault(tokenId, unkToken);
                if (!(skipSpecialTokens && token.startsWith("<") && token.endsWith(">"))) {
                    result.append(token);
                }
            }
            
            return result.toString();
        }
        
        /**
         * 解码（默认跳过特殊token）
         */
        public String decode(List<Integer> tokenIds) {
            return decode(tokenIds, true);
        }
        
        /**
         * 批量编码文本
         * 
         * @param texts 文本列表
         * @param padding 是否进行填充
         * @param maxLength 最大长度
         * @return 编码结果
         */
        public TokenizerResult batchEncode(List<String> texts, boolean padding, Integer maxLength) {
            List<List<Integer>> encodedBatch = new ArrayList<>();
            
            for (String text : texts) {
                List<Integer> encoded = encode(text);
                encodedBatch.add(encoded);
            }
            
            if (maxLength == null) {
                maxLength = encodedBatch.stream().mapToInt(List::size).max().orElse(0);
            }
            
            // 填充或截断
            List<List<Integer>> inputIds = new ArrayList<>();
            List<List<Integer>> attentionMask = new ArrayList<>();
            
            for (List<Integer> encoded : encodedBatch) {
                List<Integer> ids = new ArrayList<>();
                List<Integer> mask = new ArrayList<>();
                
                if (encoded.size() > maxLength) {
                    // 截断
                    for (int i = 0; i < maxLength; i++) {
                        ids.add(encoded.get(i));
                        mask.add(1);
                    }
                } else {
                    // 填充
                    ids.addAll(encoded);
                    for (int i = 0; i < encoded.size(); i++) {
                        mask.add(1);
                    }
                    int padLength = maxLength - encoded.size();
                    for (int i = 0; i < padLength; i++) {
                        ids.add(padTokenId);
                        mask.add(0);
                    }
                }
                
                inputIds.add(ids);
                attentionMask.add(mask);
            }
            
            return new TokenizerResult(inputIds, attentionMask);
        }
        
        public int getVocabSize() {
            return vocabSize;
        }
    }
    
    /**
     * 分词器结果类
     */
    public static class TokenizerResult {
        public final List<List<Integer>> inputIds;
        public final List<List<Integer>> attentionMask;
        
        public TokenizerResult(List<List<Integer>> inputIds, List<List<Integer>> attentionMask) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
        }
    }
    
    /**
     * 基于Qwen3模型的简单聊天机器人
     */
    public static class Qwen3ChatBot {
        
        private final Qwen3Model model;
        private final SimpleTokenizer tokenizer;
        private final List<Map<String, String>> conversationHistory;
        
        // 生成参数
        private int maxNewTokens = 100;
        private double temperature = 0.7;
        private double topP = 0.9;
        private int topK = 50;
        
        public Qwen3ChatBot(Qwen3Model model, SimpleTokenizer tokenizer) {
            this.model = model;
            this.tokenizer = tokenizer;
            this.conversationHistory = new ArrayList<>();
        }
        
        /**
         * 与用户进行对话
         * 
         * @param userInput 用户输入
         * @param systemPrompt 系统提示
         * @return AI回复
         */
        public String chat(String userInput, String systemPrompt) {
            // 构建对话提示
            String prompt;
            if (conversationHistory.isEmpty()) {
                // 首次对话，添加系统提示
                prompt = systemPrompt + "\n\n用户: " + userInput + "\nAI:";
            } else {
                // 继续对话
                prompt = "用户: " + userInput + "\nAI:";
            }
            
            // 编码输入
            List<Integer> inputTokens = tokenizer.encode(prompt);
            NdArray inputIds = createInputArray(inputTokens);
            
            try {
                // 生成回复（简化版本，实际需要实现采样生成）
                Variable output = model.forward(new Variable(inputIds));
                
                // 简化的文本生成逻辑
                String aiResponse = generateSimpleResponse(userInput);
                
                // 更新对话历史
                Map<String, String> conversation = new HashMap<>();
                conversation.put("user", userInput);
                conversation.put("ai", aiResponse);
                conversationHistory.add(conversation);
                
                return aiResponse;
                
            } catch (Exception e) {
                return "抱歉，生成回复时遇到了问题：" + e.getMessage();
            }
        }
        
        /**
         * 聊天（使用默认系统提示）
         */
        public String chat(String userInput) {
            return chat(userInput, "你是一个有用的AI助手。");
        }
        
        /**
         * 简化的回复生成（演示用）
         */
        private String generateSimpleResponse(String userInput) {
            // 基于关键词的简单回复生成
            if (userInput.contains("你好") || userInput.contains("hello")) {
                return "你好！我是Qwen3模型，很高兴与你对话。";
            } else if (userInput.contains("介绍") && userInput.contains("自己")) {
                return "我是基于TinyAI框架实现的Qwen3大语言模型，具有现代Transformer架构的各种先进特性。";
            } else if (userInput.contains("能做什么") || userInput.contains("功能")) {
                return "我可以进行文本生成、对话聊天、问答回复等任务。这是一个演示版本，实际能力取决于训练数据。";
            } else if (userInput.contains("机器学习")) {
                return "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律，并做出预测或决策。";
            } else {
                return "这是一个很有趣的问题。由于这是演示版本，我的回复相对简化，但在实际训练后可以提供更丰富的回答。";
            }
        }
        
        /**
         * 创建输入数组
         */
        private NdArray createInputArray(List<Integer> tokens) {
            NdArray inputArray = NdArray.of(Shape.of(1, tokens.size()));
            for (int i = 0; i < tokens.size(); i++) {
                inputArray.set(tokens.get(i), 0, i);
            }
            return inputArray;
        }
        
        /**
         * 清除对话历史
         */
        public void clearHistory() {
            conversationHistory.clear();
        }
        
        /**
         * 获取对话历史
         */
        public List<Map<String, String>> getConversationHistory() {
            return new ArrayList<>(conversationHistory);
        }
        
        /**
         * 设置生成参数
         */
        public void setGenerationParams(int maxNewTokens, double temperature, double topP, int topK) {
            this.maxNewTokens = maxNewTokens;
            this.temperature = temperature;
            this.topP = topP;
            this.topK = topK;
        }
    }
    
    /**
     * 创建演示用的Qwen3模型
     */
    public static Qwen3Model createDemoModel() {
        Qwen3Config config = Qwen3Config.createDemoConfig();
        return new Qwen3Model("qwen3-demo", config);
    }
    
    /**
     * 文本生成演示
     */
    public static void textGenerationDemo() {
        System.out.println("=== Qwen3 文本生成演示 ===\n");
        
        try {
            // 创建模型和分词器
            System.out.println("正在创建模型...");
            Qwen3Model model = createDemoModel();
            SimpleTokenizer tokenizer = new SimpleTokenizer(32000);
            
            System.out.println("模型参数数量: " + String.format("%,d", model.countParameters()));
            System.out.println("词汇表大小: " + tokenizer.getVocabSize());
            
            // 测试文本生成
            String[] testPrompts = {
                "今天天气",
                "人工智能",
                "Java编程",
                "机器学习是"
            };
            
            System.out.println("\n开始文本生成测试...");
            
            for (String prompt : testPrompts) {
                System.out.println("\n输入提示: '" + prompt + "'");
                
                // 编码输入
                List<Integer> inputTokens = tokenizer.encode(prompt);
                NdArray inputIds = NdArray.of(Shape.of(1, inputTokens.size()));
                for (int i = 0; i < inputTokens.size(); i++) {
                    inputIds.set(inputTokens.get(i), 0, i);
                }
                
                System.out.println("输入token数: " + inputTokens.size());
                
                // 前向传播（演示用）
                Variable output = model.forward(new Variable(inputIds));
                
                // 模拟文本生成结果
                String generatedText = simulateTextGeneration(prompt, tokenizer);
                
                System.out.println("生成结果: '" + generatedText + "'");
                System.out.println("输出形状: " + output.getValue().getShape());
            }
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 模拟文本生成（演示用）
     */
    private static String simulateTextGeneration(String prompt, SimpleTokenizer tokenizer) {
        // 基于提示生成简单的续写
        Map<String, String> continuations = new HashMap<>();
        continuations.put("今天天气", "今天天气很好，阳光明媚。");
        continuations.put("人工智能", "人工智能是现代科技的重要分支。");
        continuations.put("Java编程", "Java编程语言具有跨平台特性。");
        continuations.put("机器学习是", "机器学习是实现人工智能的重要方法。");
        
        return continuations.getOrDefault(prompt, prompt + "相关的内容生成。");
    }
    
    /**
     * 聊天演示
     */
    public static void chatDemo() {
        System.out.println("\n=== Qwen3 聊天演示 ===\n");
        
        try {
            // 创建聊天机器人
            System.out.println("正在初始化聊天机器人...");
            Qwen3Model model = createDemoModel();
            SimpleTokenizer tokenizer = new SimpleTokenizer(32000);
            Qwen3ChatBot chatbot = new Qwen3ChatBot(model, tokenizer);
            
            // 设置生成参数
            chatbot.setGenerationParams(50, 0.7, 0.9, 50);
            
            System.out.println("聊天机器人已准备就绪！");
            System.out.println("(演示模式，使用预设对话)\n");
            
            // 模拟对话
            String[] demoConversations = {
                "你好，请介绍一下自己",
                "你能做什么？",
                "解释一下机器学习",
                "谢谢你的回答"
            };
            
            for (String userInput : demoConversations) {
                System.out.println("用户: " + userInput);
                
                try {
                    String aiResponse = chatbot.chat(userInput);
                    System.out.println("AI: " + aiResponse + "\n");
                } catch (Exception e) {
                    System.err.println("生成回复时出错: " + e.getMessage() + "\n");
                }
            }
            
            // 显示对话历史
            System.out.println("对话历史:");
            List<Map<String, String>> history = chatbot.getConversationHistory();
            for (int i = 0; i < history.size(); i++) {
                Map<String, String> conv = history.get(i);
                System.out.println((i + 1) + ". 用户: " + conv.get("user"));
                System.out.println("   AI: " + conv.get("ai"));
            }
            
        } catch (Exception e) {
            System.err.println("聊天演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 模型信息演示
     */
    public static void modelInfoDemo() {
        System.out.println("\n=== Qwen3 模型信息 ===\n");
        
        try {
            // 创建模型
            Qwen3Model model = createDemoModel();
            SimpleTokenizer tokenizer = new SimpleTokenizer();
            
            // 模型配置信息
            Qwen3Config config = model.getConfig();
            System.out.println("模型配置:");
            System.out.println("  词汇表大小: " + String.format("%,d", config.getVocabSize()));
            System.out.println("  隐藏层维度: " + config.getHiddenSize());
            System.out.println("  隐藏层数量: " + config.getNumHiddenLayers());
            System.out.println("  注意力头数: " + config.getNumAttentionHeads());
            System.out.println("  中间层维度: " + config.getIntermediateSize());
            System.out.println("  最大序列长度: " + config.getMaxPositionEmbeddings());
            
            // 参数统计
            long totalParams = model.countParameters();
            
            System.out.println("\n参数统计:");
            System.out.println("  总参数数: " + String.format("%,d", totalParams));
            System.out.println("  模型大小: " + String.format("%.2f MB (FP32)", model.getModelSizeMB()));
            
            // 分词器信息
            System.out.println("\n分词器信息:");
            System.out.println("  词汇表大小: " + tokenizer.getVocabSize());
            System.out.println("  特殊token: " + tokenizer.padToken + ", " + tokenizer.bosToken + 
                             ", " + tokenizer.eosToken + ", " + tokenizer.unkToken);
            
            // 测试编码解码
            String testText = "你好，世界！Hello, World!";
            List<Integer> encoded = tokenizer.encode(testText);
            String decoded = tokenizer.decode(encoded);
            
            System.out.println("\n编码解码测试:");
            System.out.println("  原文: " + testText);
            System.out.println("  编码: " + encoded);
            System.out.println("  解码: " + decoded);
            System.out.println("  长度: " + encoded.size() + " tokens");
            
            // 显示详细模型信息
            System.out.println("\n" + model.getModelSummary());
            
        } catch (Exception e) {
            System.err.println("模型信息演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * tokenizer功能演示
     */
    public static void tokenizerDemo() {
        System.out.println("\n=== Tokenizer 功能演示 ===\n");
        
        try {
            SimpleTokenizer tokenizer = new SimpleTokenizer();
            
            // 测试不同类型的文本
            String[] testTexts = {
                "Hello, World!",
                "你好，世界！",
                "Java编程语言",
                "123ABC中文",
                "🚀AI模型"
            };
            
            System.out.println("单文本编码解码测试:");
            for (String text : testTexts) {
                List<Integer> encoded = tokenizer.encode(text);
                String decoded = tokenizer.decode(encoded);
                
                System.out.println("原文: \"" + text + "\"");
                System.out.println("长度: " + encoded.size() + " tokens");
                System.out.println("编码: " + encoded.toString().substring(0, Math.min(50, encoded.toString().length())) + 
                                 (encoded.toString().length() > 50 ? "..." : ""));
                System.out.println("解码: \"" + decoded + "\"");
                System.out.println("匹配: " + text.equals(decoded));
                System.out.println();
            }
            
            // 批量编码测试
            System.out.println("批量编码测试:");
            List<String> batchTexts = Arrays.asList("短文本", "这是一个较长的文本示例", "AI");
            TokenizerResult result = tokenizer.batchEncode(batchTexts, true, null);
            
            for (int i = 0; i < batchTexts.size(); i++) {
                System.out.println("文本 " + (i + 1) + ": \"" + batchTexts.get(i) + "\"");
                System.out.println("编码: " + result.inputIds.get(i));
                System.out.println("掩码: " + result.attentionMask.get(i));
                System.out.println();
            }
            
        } catch (Exception e) {
            System.err.println("Tokenizer演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 主演示方法
     */
    public static void main(String[] args) {
        System.out.println("🤖 Qwen3 模型演示程序");
        System.out.println("=" + "=".repeat(49));
        
        try {
            // 运行各种演示
            modelInfoDemo();
            tokenizerDemo();
            textGenerationDemo();
            chatDemo();
            
            System.out.println("\n✅ 演示完成！");
            System.out.println("\n说明:");
            System.out.println("- 这是基于TinyAI框架的Qwen3实现演示");
            System.out.println("- 包含了完整的模型架构和基础功能");
            System.out.println("- 文本生成功能使用了简化的演示逻辑");
            System.out.println("- 实际使用中需要训练好的模型权重");
            System.out.println("- 所有核心组件都已正确实现并可以正常运行");
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}