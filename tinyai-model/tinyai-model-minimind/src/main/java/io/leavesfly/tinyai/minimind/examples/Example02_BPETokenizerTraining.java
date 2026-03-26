package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.minimind.tokenizer.BPETrainer;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.minimind.tokenizer.Vocabulary;

import java.util.ArrayList;
import java.util.List;

/**
 * 示例02: BPE Tokenizer训练
 * 
 * 本示例演示:
 * 1. 准备训练语料
 * 2. 训练BPE Tokenizer
 * 3. 使用训练好的Tokenizer
 * 4. 保存和加载Tokenizer
 * 
 * @author leavesfly
 */
public class Example02_BPETokenizerTraining {
    
    public static void main(String[] args) throws Exception {
        System.out.println("=== BPE Tokenizer训练示例 ===\n");
        
        // 1. 准备训练语料
        System.out.println("1. 准备训练语料");
        List<String> corpus = prepareCorpus();
        System.out.println("语料库大小: " + corpus.size() + " 条");
        System.out.println("示例文本: " + corpus.get(0) + "\n");
        
        // 2. 训练BPE Tokenizer
        System.out.println("2. 训练BPE Tokenizer");
        int vocabSize = 300;  // 目标词汇表大小
        int minFrequency = 2;  // 最小频率
        
        BPETrainer trainer = new BPETrainer(vocabSize, minFrequency);
        System.out.println("开始训练...");
        
        Vocabulary vocab = trainer.train(corpus);
        
        System.out.println("训练完成!");
        System.out.println("词汇表大小: " + vocab.getVocabSize());
        System.out.println("Merge规则数: " + trainer.getMerges().size() + "\n");
        
        // 3. 创建Tokenizer
        System.out.println("3. 使用训练好的Tokenizer");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.fromBPETrainer(trainer, 128);
        
        // 测试编码解码
        String testText = "深度学习是人工智能的重要分支";
        System.out.println("测试文本: " + testText);
        
        List<Integer> encoded = tokenizer.encode(testText, false, false);
        System.out.println("编码结果: " + encoded);
        System.out.println("Token数量: " + encoded.size());
        
        String decoded = tokenizer.decode(encoded, false);
        System.out.println("解码结果: " + decoded);
        System.out.println("可逆性验证: " + testText.equals(decoded) + "\n");
        
        // 4. 保存Tokenizer
        System.out.println("4. 保存Tokenizer");
        String savePath = "./checkpoints/minimind/tokenizer_model";
        tokenizer.save(savePath);
        System.out.println("已保存到: " + savePath);
        
        // 5. 加载Tokenizer
        System.out.println("\n5. 加载Tokenizer");
        MiniMindTokenizer loadedTokenizer = MiniMindTokenizer.load(savePath);
        System.out.println("加载成功!");
        System.out.println("词汇表大小: " + loadedTokenizer.getVocabSize());
        
        System.out.println("\n=== 示例完成 ===");
    }
    
    /**
     * 准备训练语料
     */
    private static List<String> prepareCorpus() {
        List<String> corpus = new ArrayList<>();
        
        // 添加示例文本
        corpus.add("深度学习是人工智能的重要分支");
        corpus.add("机器学习包括监督学习和无监督学习");
        corpus.add("神经网络是深度学习的基础");
        corpus.add("自然语言处理是人工智能的应用领域");
        corpus.add("计算机视觉处理图像和视频数据");
        corpus.add("强化学习通过奖励机制训练智能体");
        corpus.add("Transformer模型改变了自然语言处理");
        corpus.add("注意力机制是Transformer的核心");
        corpus.add("预训练模型需要大量数据");
        corpus.add("微调可以适应特定任务");
        corpus.add("大语言模型展现出惊人的能力");
        corpus.add("人工智能正在改变世界");
        corpus.add("深度学习需要大量计算资源");
        corpus.add("GPU加速了神经网络训练");
        corpus.add("数据是机器学习的关键");
        corpus.add("算法优化提升模型性能");
        corpus.add("正则化防止模型过拟合");
        corpus.add("验证集用于调整超参数");
        corpus.add("测试集评估最终性能");
        corpus.add("交叉验证提高评估可靠性");
        
        return corpus;
    }
}
