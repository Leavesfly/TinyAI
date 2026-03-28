package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.util.PlantUML;
import org.junit.Test;

/**
 * DeepSeek-R1 Micro 模型 PlantUML 图生成测试
 *
 * 使用 PlantUML 工具类生成两种图：
 * 1. 模块图（generateModuleGraph）- 展示 R1 Micro 模型的模块层级结构
 * 2. 计算图（generateSimpleGraph）- 展示 R1 Micro 模型前向传播的算子级数据流
 *
 * R1 与 V3 的核心区别：
 * - 架构完全相同（共享 MoE 基础架构）
 * - R1 无 MTP Head（MTP 是 V3 的训练辅助机制）
 * - R1 通过纯 RL 训练使推理能力自然涌现
 *
 * @author leavesfly
 */
public class DeepSeekR1MicroPlantUMLTest {

    /**
     * 生成 R1 Micro 模型的模块图
     * 展示模块层级结构：TokenEmbedding → TransformerBlock(Attention+MoE) × nLayer → FinalRMSNorm → OutputProjection
     * 注意：R1 无 MTP Head，这是与 V3 的关键区别
     */
    @Test
    public void generateR1MicroModuleDiagram() {
        DeepSeekR1Config config = DeepSeekR1Config.createMicroConfig();
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Micro", config);

        DeepSeekR1Block r1Block = model.getR1Block();
        r1Block.eval();

        System.out.println("========== DeepSeek-R1 Micro 模块图 (PlantUML) ==========");
        System.out.println("配置: vocabSize=" + config.getVocabSize()
                + ", nEmbd=" + config.getNEmbd()
                + ", nLayer=" + config.getNLayer()
                + ", nHead=" + config.getNHead()
                + ", numExperts=" + config.getNumExperts()
                + ", topK=" + config.getTopK());
        System.out.println("总参数量: " + config.estimateParameterCount()
                + " | 激活参数量: " + config.estimateActiveParameterCount()
                + String.format(" (%.1f%%)", config.getActivationRatio()));
        System.out.println("训练方式: 纯RL | 无MTP Head");
        System.out.println();
        System.out.println(PlantUML.generateModuleGraph(r1Block));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }

    /**
     * 生成 R1 Micro 模型的计算图
     * 通过前向传播构建计算图，展示 Variable → Function → Variable 的数据流
     */
    @Test
    public void generateR1MicroComputationDiagram() {
        DeepSeekR1Config config = DeepSeekR1Config.createMicroConfig();
        DeepSeekR1Model model = new DeepSeekR1Model("DeepSeek-R1-Micro", config);
        model.getR1Block().eval();

        // 构造输入: [batch_size=1, seq_len=4]
        float[][] inputData = {{1, 2, 3, 4}};
        Variable input = new Variable(NdArray.of(inputData), "input_token_ids");

        // 前向传播，构建计算图
        Variable output = model.predict(input);
        output.setName("logits_output");

        System.out.println("========== DeepSeek-R1 Micro 计算图 (PlantUML) ==========");
        System.out.println("输入形状: " + input.getValue().getShape());
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println("训练方式: 纯RL | 无MTP Head");
        System.out.println();
        System.out.println(PlantUML.generateSimpleGraph(output));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }
}
