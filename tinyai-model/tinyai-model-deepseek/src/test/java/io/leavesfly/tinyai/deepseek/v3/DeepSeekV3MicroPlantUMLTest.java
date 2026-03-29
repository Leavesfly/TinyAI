package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.util.PlantUML;
import org.junit.Test;

/**
 * DeepSeek-V3 Micro 模型 PlantUML 图生成测试
 *
 * 使用 PlantUML 工具类生成两种图：
 * 1. 模块图（generateModuleGraph）- 展示 V3 Micro 模型的模块层级结构
 * 2. 计算图（generateSimpleGraph）- 展示 V3 Micro 模型前向传播的算子级数据流
 *
 * @author leavesfly
 */
public class DeepSeekV3MicroPlantUMLTest {

    /**
     * 生成 V3 Micro 模型的模块图
     * 展示模块层级结构：TokenEmbedding → TransformerBlock(Attention+MoE) × nLayer → FinalRMSNorm → OutputProjection + MTPHead
     */
    @Test
    public void generateV3MicroModuleDiagram() {
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);

        DeepSeekV3Block v3Block = model.getV3Block();
        v3Block.eval();

        System.out.println("========== DeepSeek-V3 Micro 模块图 (PlantUML) ==========");
        System.out.println("配置: vocabSize=" + config.getVocabSize()
                + ", nEmbd=" + config.getNEmbd()
                + ", nLayer=" + config.getNLayer()
                + ", nHead=" + config.getNHead()
                + ", numExperts=" + config.getNumExperts()
                + ", topK=" + config.getTopK());
        System.out.println("总参数量: " + config.estimateParameterCount()
                + " | 激活参数量: " + config.estimateActiveParameterCount()
                + String.format(" (%.1f%%)", config.getActivationRatio()));
        System.out.println();
        System.out.println(PlantUML.generateModuleGraph(v3Block));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }

    /**
     * 生成 V3 Micro 模型的计算图
     * 通过前向传播构建计算图，展示 Variable → Function → Variable 的数据流
     */
    @Test
    public void generateV3MicroComputationDiagram() {
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3Model model = new DeepSeekV3Model("DeepSeek-V3-Micro", config);
        model.getV3Block().eval();

        // 构造输入: [batch_size=1, seq_len=4]
        float[][] inputData = {{1, 2, 3, 4}};
        Variable input = new Variable(NdArray.of(inputData), "input");

        // 前向传播，构建计算图
        Variable output = model.predict(input);
        output.setName("output");

        System.out.println("========== DeepSeek-V3 Micro 计算图 (PlantUML) ==========");
        System.out.println("输入形状: " + input.getValue().getShape());
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println();
        System.out.println(PlantUML.generateSimpleGraph(output));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }
}
