package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.util.PlantUML;
import io.leavesfly.tinyai.util.UmlPrinter;

import java.util.Arrays;
import java.util.List;

/**
 * 示例01: 模型创建与推理
 * <p>
 * 本示例演示如何:
 * 1. 创建不同规模的MiniMind模型配置
 * 2. 初始化模型和Tokenizer
 * 3. 执行单次推理
 * 4. 批量推理
 * 5. 查看模型信息
 *
 * @author leavesfly
 */
public class Example01_ModelCreationAndInference {

    public static void main(String[] args) {
        System.out.println("=== MiniMind 模型创建与推理示例 ===\n");

        // 1. 创建模型配置
        System.out.println("1. 创建模型配置");
        MiniMindConfig config = createMicroConfig(100);
        printModelConfig(config);

        // 2. 创建模型
        System.out.println("\n2. 创建模型");
        MiniMindModel model = new MiniMindModel("minimind-small", config);
        System.out.println("模型创建成功!");
        System.out.println("参数数量: " + config.estimateParameters() + " (~" +
                (config.estimateParameters() / 1_000_000) + "M)");

        // 3. 创建Tokenizer
        System.out.println("\n3. 创建Tokenizer");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
                config.getVocabSize(),
                config.getMaxSeqLen()
        );
        System.out.println("Tokenizer创建成功!");
        System.out.println("词汇表大小: " + tokenizer.getVocabSize());

        // 4. 单次推理示例
        System.out.println("\n4. 单次推理示例");
        String inputText = "深度学习";
        singleInference(model, tokenizer, inputText);

        // 5. 批量推理示例
        System.out.println("\n5. 批量推理示例");
        String[] batchTexts = {
                "人工智能",
                "机器学习",
                "神经网络"
        };

        batchInference(model, tokenizer, batchTexts);

        // 6. 不同配置示例
        System.out.println("\n6. 不同规模模型配置对比");
        compareModelConfigs();
        System.out.println("\n=== 示例完成 ===");


    }


    /**
     * 创建超小型配置（用于快速演示）
     */
    public static MiniMindConfig createMicroConfig(int vocabSize) {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(vocabSize);
        config.setMaxSeqLen(64);          // 序列长度
        config.setHiddenSize(128);        // 隐藏维度
        config.setNumLayers(2);           // 层数
        config.setNumHeads(2);            // 注意力头数
        config.setFfnHiddenSize(256);     // FFN隐藏维度
        config.setDropout(0.1f);
        config.setEpsilon(1e-5f);
        return config;
    }

    /**
     * 创建小型模型配置
     */
    private static MiniMindConfig createSmallModelConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(6400);      // 词汇表大小
        config.setMaxSeqLen(512);       // 最大序列长度
        config.setHiddenSize(512);      // 隐藏层维度
        config.setNumLayers(8);         // Transformer层数
        config.setNumHeads(16);         // 注意力头数
        config.setFfnHiddenSize(1024);  // FFN隐藏层维度
        config.setDropout(0.1f);        // Dropout比例
        return config;
    }

    /**
     * 单次推理
     */
    private static void singleInference(MiniMindModel model, MiniMindTokenizer tokenizer, String text) {
        System.out.println("输入文本: " + text);

        // 1. 编码
        List<Integer> tokenIds = tokenizer.encode(text, false, false);
        System.out.println("Token IDs: " + tokenIds);

        // 2. 转换为NdArray
        int[] ids = tokenIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));

        // 3. 模型推理
        Variable inputVar = new Variable(inputArray);
        Variable output = model.predict(inputVar);

        // 4. 输出信息
        int[] outputShape = output.getValue().getShape().getShapeDims();
        System.out.println("输出形状: [" + outputShape[0] + ", " +
                outputShape[1] + ", " + outputShape[2] + "]");
        System.out.println("说明: [batch_size, seq_len, vocab_size]");
    }

    /**
     * 批量推理
     */
    private static void batchInference(MiniMindModel model, MiniMindTokenizer tokenizer, String[] texts) {
        System.out.println("批量输入 " + texts.length + " 个文本:");
        for (int i = 0; i < texts.length; i++) {
            System.out.println("  [" + i + "] " + texts[i]);
        }

        // 1. 批量编码
        List<String> textList = Arrays.asList(texts);
        MiniMindTokenizer.EncodingResult result = tokenizer.batchEncode(textList, true, true);

        List<List<Integer>> inputIds = result.getInputIds();
        System.out.println("批量编码完成, 序列数: " + inputIds.size());
        System.out.println("序列长度(填充后): " + inputIds.get(0).size());

        // 2. 转换为NdArray (简化示例,仅展示第一个)
        List<Integer> firstIds = inputIds.get(0);
        int[] ids = firstIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));

        // 3. 批量推理
        Variable inputVar = new Variable(inputArray);
        Variable output = model.predict(inputVar);

        int[] outputShape = output.getValue().getShape().getShapeDims();
        System.out.println("批量输出形状: [" + outputShape[0] + ", " +
                outputShape[1] + ", " + outputShape[2] + "]");


//        System.out.printf(PlantUML.generateModuleGraph(model.getModule()));
        System.out.printf(PlantUML.generateSimpleGraph(output));

    }

    /**
     * 打印模型配置
     */
    private static void printModelConfig(MiniMindConfig config) {
        System.out.println("模型配置:");
        System.out.println("  - 词汇表大小: " + config.getVocabSize());
        System.out.println("  - 最大序列长度: " + config.getMaxSeqLen());
        System.out.println("  - 隐藏层维度: " + config.getHiddenSize());
        System.out.println("  - Transformer层数: " + config.getNumLayers());
        System.out.println("  - 注意力头数: " + config.getNumHeads());
        System.out.println("  - 每个头维度: " + (config.getHiddenSize() / config.getNumHeads()));
        System.out.println("  - FFN隐藏层维度: " + config.getFfnHiddenSize());
        System.out.println("  - Dropout: " + config.getDropout());
        System.out.println("  - 预估参数量: " + config.estimateParameters() + " (~" +
                (config.estimateParameters() / 1_000_000) + "M)");
    }

    /**
     * 对比不同规模模型配置
     */
    private static void compareModelConfigs() {
        // Small配置
        MiniMindConfig small = MiniMindConfig.createSmallConfig();
        System.out.println("\nSmall模型 (~26M参数):");
        System.out.println("  层数: " + small.getNumLayers() + ", 隐藏维度: " + small.getHiddenSize());

        // Medium配置
        MiniMindConfig medium = MiniMindConfig.createMediumConfig();
        System.out.println("\nMedium模型 (~108M参数):");
        System.out.println("  层数: " + medium.getNumLayers() + ", 隐藏维度: " + medium.getHiddenSize());

        // 自定义Tiny配置(用于快速测试)
        MiniMindConfig tiny = new MiniMindConfig();
        tiny.setVocabSize(512);
        tiny.setHiddenSize(64);
        tiny.setNumLayers(2);
        tiny.setNumHeads(2);
        tiny.setFfnHiddenSize(128);
        System.out.println("\nTiny模型 (用于测试):");
        System.out.println("  层数: " + tiny.getNumLayers() + ", 隐藏维度: " + tiny.getHiddenSize());
        System.out.println("  预估参数量: ~" + (tiny.estimateParameters() / 1000) + "K");
    }
}
