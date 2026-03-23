package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.minimind.model.MiniMindBlock;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;

/**
 * MiniMind MoE 模型使用示例
 * <p>
 * 演示如何使用 MoE 架构的 MiniMind 模型进行文本生成
 * 
 * 功能展示:
 * 1. 创建 MoE 模型
 * 2. 文本生成
 * 3. 负载均衡损失计算
 * 4. 专家使用统计
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindMoEExample {

    public static void main(String[] args) {
        System.out.println("========== MiniMind MoE 示例 ==========\n");

        // 1. 创建 MoE 模型
        System.out.println("1. 创建 MoE 模型...");
        MiniMindModel model = MiniMindModel.create("minimind-moe", "moe");
        model.printModelInfo();
        System.out.println();

        // 2. 创建 Tokenizer
        System.out.println("2. 创建 Tokenizer...");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(6400, 512);
        System.out.println("Tokenizer 创建成功\n");

        // 3. 文本生成示例
        System.out.println("3. 文本生成示例:");
        demoTextGeneration(model, tokenizer);
        System.out.println();

        // 4. 负载均衡损失示例
        System.out.println("4. 负载均衡损失示例:");
        demoLoadBalanceLoss(model, tokenizer);
        System.out.println();

        // 5. 专家使用统计
        System.out.println("5. 专家使用统计:");
        System.out.println(model.getExpertUsageStats());

        System.out.println("\n========== 示例结束 ==========");
    }

    /**
     * 演示文本生成
     */
    private static void demoTextGeneration(MiniMindModel model, MiniMindTokenizer tokenizer) {
        String prompt = "你好，世界！";
        System.out.println("提示词: " + prompt);

        List<Integer> tokenIds = tokenizer.encode(prompt);
        int[] promptTokens = tokenIds.stream().mapToInt(Integer::intValue).toArray();

        System.out.println("Token IDs: " + tokenIds);
        System.out.println("生成中...");

        int[] generated = model.generate(promptTokens, 50);

        List<Integer> generatedList = new java.util.ArrayList<>();
        for (int id : generated) {
            generatedList.add(id);
        }
        String output = tokenizer.decode(generatedList);

        System.out.println("生成结果: " + output);
        System.out.println("总 Token 数: " + generated.length);
    }

    /**
     * 演示负载均衡损失计算
     */
    private static void demoLoadBalanceLoss(MiniMindModel model, MiniMindTokenizer tokenizer) {
        String text = "测试负载均衡损失计算";
        System.out.println("输入文本: " + text);

        List<Integer> tokenIds = tokenizer.encode(text);
        
        NdArray inputArray = NdArray.of(Shape.of(1, tokenIds.size()));
        float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) inputArray).buffer;
        for (int i = 0; i < tokenIds.size(); i++) {
            buffer[i] = tokenIds.get(i);
        }

        MiniMindBlock.MoEOutput output = model.predictWithLoss(inputArray);

        System.out.println("Output Shape: " + output.getOutput().getShape());
        System.out.printf("负载均衡损失: %.6f\n", output.getBalanceLoss());
    }

    /**
     * 演示不同采样策略
     */
    public static void demoSamplingStrategies() {
        System.out.println("\n========== 采样策略对比 ==========\n");

        MiniMindModel model = MiniMindModel.create("minimind-moe", "moe");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(6400, 512);

        String prompt = "深度学习";
        List<Integer> tokenIds = tokenizer.encode(prompt);
        int[] promptTokens = tokenIds.stream().mapToInt(Integer::intValue).toArray();

        System.out.println("1. 贪婪采样 (temperature=0.0):");
        int[] greedy = model.generate(promptTokens, 30, 0.0f, 0, 0.0f);
        System.out.println(tokenizer.decode(toList(greedy)));
        System.out.println();

        System.out.println("2. 温度采样 (temperature=0.8):");
        int[] temp = model.generate(promptTokens, 30, 0.8f, 0, 0.0f);
        System.out.println(tokenizer.decode(toList(temp)));
        System.out.println();

        System.out.println("3. Top-K 采样 (topK=40):");
        int[] topK = model.generate(promptTokens, 30, 1.0f, 40, 0.0f);
        System.out.println(tokenizer.decode(toList(topK)));
        System.out.println();

        System.out.println("4. Top-P 采样 (topP=0.9):");
        int[] topP = model.generate(promptTokens, 30, 1.0f, 0, 0.9f);
        System.out.println(tokenizer.decode(toList(topP)));
        System.out.println();

        System.out.println("5. 组合采样 (temp=0.8, topK=40, topP=0.9):");
        int[] combined = model.generate(promptTokens, 30, 0.8f, 40, 0.9f);
        System.out.println(tokenizer.decode(toList(combined)));
    }

    /**
     * 演示专家使用统计分析
     */
    public static void demoExpertAnalysis() {
        System.out.println("\n========== 专家使用分析 ==========\n");

        MiniMindModel model = MiniMindModel.create("minimind-moe", "moe");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(6400, 512);

        String[] testSamples = {
            "深度学习是人工智能的重要分支",
            "机器学习包括监督学习和无监督学习",
            "神经网络模拟人脑神经元的工作方式",
            "Transformer 是现代 NLP 的基础架构",
            "注意力机制可以捕获长距离依赖"
        };

        model.resetStats();

        System.out.println("处理 " + testSamples.length + " 个样本...\n");

        for (int i = 0; i < testSamples.length; i++) {
            String text = testSamples[i];
            System.out.println("样本 " + (i + 1) + ": " + text);

            List<Integer> tokenIds = tokenizer.encode(text);
            NdArray inputArray = NdArray.of(Shape.of(1, tokenIds.size()));
            float[] buffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) inputArray).buffer;
            for (int j = 0; j < tokenIds.size(); j++) {
                buffer[j] = tokenIds.get(j);
            }

            MiniMindBlock.MoEOutput output = model.predictWithLoss(inputArray);
            System.out.printf("  负载均衡损失: %.6f\n", output.getBalanceLoss());
        }

        System.out.println("\n专家使用统计:");
        System.out.println(model.getExpertUsageStats());
    }

    /**
     * int[] 转 List<Integer>
     */
    private static List<Integer> toList(int[] array) {
        List<Integer> list = new java.util.ArrayList<>();
        for (int val : array) {
            list.add(val);
        }
        return list;
    }

    /**
     * 演示完整流程
     */
    public static void demoCompleteWorkflow() {
        System.out.println("\n========== 完整工作流程 ==========\n");

        System.out.println("步骤 1: 创建模型");
        MiniMindConfig config = MiniMindConfig.createMoEConfig();
        MiniMindModel model = new MiniMindModel("my-moe-model", config);
        model.printModelInfo();

        System.out.println("\n步骤 2: 创建 Tokenizer");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), 
            config.getMaxSeqLen()
        );

        System.out.println("\n步骤 3: 文本编码");
        String prompt = "MiniMind 是一个轻量级语言模型";
        List<Integer> tokens = tokenizer.encode(prompt);
        System.out.println("原文: " + prompt);
        System.out.println("Token IDs: " + tokens);

        System.out.println("\n步骤 4: 模型推理");
        int[] promptArray = tokens.stream().mapToInt(Integer::intValue).toArray();
        int[] generated = model.generate(promptArray, 50, 0.8f, 40, 0.9f);

        System.out.println("\n步骤 5: 文本解码");
        String output = tokenizer.decode(toList(generated));
        System.out.println("生成结果: " + output);

        System.out.println("\n步骤 6: 统计分析");
        System.out.println(model.getExpertUsageStats());

        System.out.println("\n工作流程完成!");
    }
}
