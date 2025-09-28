package io.leavesfly.tinyai.example.embedd;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;

/**
 * Embedding层完整示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 该示例演示如何在实际场景中使用Embedding层，包括：
 * 1. 创建Embedding层
 * 2. 将词汇索引转换为词向量
 * 3. 与全连接层和激活函数组合使用
 */
public class EmbeddingFullExample {

    /**
     * 主函数，执行Embedding层完整示例
     *
     * @param args 命令行参数
     */
    public static void main(String[] args) {
        System.out.println("=== Embedding层完整示例 ===");

        // 定义参数
        int vocabSize = 100;     // 词汇表大小
        int embedSize = 16;      // 嵌入维度
        int hiddenSize = 32;     // 隐藏层维度
        int outputSize = 5;      // 输出维度

        // 创建Embedding层
        Embedding embedding = new Embedding("embedding", vocabSize, embedSize);
        System.out.println("1. 创建Embedding层:");
        System.out.println("   - 词汇表大小: " + embedding.getVocabSize());
        System.out.println("   - 嵌入维度: " + embedding.getEmbedSize());
        System.out.println("   - 权重矩阵形状: " + embedding.getWeight().getValue().getShape());

        // 创建全连接层和激活函数层
        LinearLayer linear = new LinearLayer("linear", embedSize, hiddenSize, true);
        ReLuLayer relu = new ReLuLayer("relu");
        LinearLayer outputLayer = new LinearLayer("output", hiddenSize, outputSize, true);

        System.out.println("2. 创建网络结构:");
        System.out.println("   - Embedding层: " + embedSize + " -> " + embedSize);
        System.out.println("   - 线性层1: " + embedSize + " -> " + hiddenSize);
        System.out.println("   - ReLU激活函数");
        System.out.println("   - 线性层2: " + hiddenSize + " -> " + outputSize);

        // 创建输入数据（模拟文本序列）
        // 假设我们有一个包含5个词的句子，每个词用一个索引表示
        float[][] sentenceData = {{10, 25, 3, 55, 78}};
        NdArray sentence = NdArray.of(sentenceData);
        System.out.println("\n3. 输入数据:");
        System.out.println("   - 句子词汇索引: " + java.util.Arrays.toString(sentenceData[0]));
        System.out.println("   - 输入形状: " + sentence.getShape());

        try {
            // 前向传播过程
            System.out.println("\n4. 前向传播过程:");

            // 1. Embedding层：将词汇索引转换为词向量
            Variable embedded = embedding.layerForward(new Variable(sentence));
            System.out.println("   - Embedding输出形状: " + embedded.getValue().getShape());

            // 2. 第一个全连接层
            Variable linear1Out = linear.layerForward(embedded);
            System.out.println("   - 线性层1输出形状: " + linear1Out.getValue().getShape());

            // 3. ReLU激活函数
            Variable reluOut = relu.layerForward(linear1Out);
            System.out.println("   - ReLU输出形状: " + reluOut.getValue().getShape());

            // 4. 第二个全连接层（输出层）
            Variable output = outputLayer.layerForward(reluOut);
            System.out.println("   - 最终输出形状: " + output.getValue().getShape());

            System.out.println("\n5. 最终输出结果:");
            System.out.println(output.getValue());

            System.out.println("\n=== Embedding层完整示例执行成功！ ===");

        } catch (Exception e) {
            System.err.println("Embedding层完整示例执行失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}