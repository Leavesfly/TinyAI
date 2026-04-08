package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.util.PlantUML;
import org.junit.Test;

/**
 * DeepSeek-V3 MoE Layer PlantUML 图生成测试
 *
 * 使用 PlantUML 工具类生成两种图：
 * 1. 模块图（generateModuleGraph）- 展示 MoE 层的模块层级结构（门控网络、共享专家、路由专家等）
 * 2. 计算图（generateSimpleGraph）- 展示 MoE 层前向传播的算子级数据流
 *
 * MoE 层核心组件：
 * - GatingNetwork: 门控网络，计算路由 logits
 * - ExpertBias: 无辅助损失负载均衡的可学习偏置
 * - SharedExperts: 共享专家（每次必激活）
 * - RoutedExperts: 路由专家（Top-K 选择）
 * - ExpertDropout: 专家 Dropout
 *
 * @author leavesfly
 */
public class DeepSeekV3MoELayerPlantUMLTest {

    /**
     * 生成 MoE Layer 的模块图
     * 展示门控网络、共享专家、路由专家、ExpertBias 等子模块的层级结构
     */
    @Test
    public void generateMoELayerModuleDiagram() {
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3MoEBlock moeLayer = new DeepSeekV3MoEBlock("moe_layer", config);

        System.out.println("========== DeepSeek-V3 MoE Layer 模块图 (PlantUML) ==========");
        System.out.println("配置: nEmbd=" + config.getNEmbd()
                + ", numExperts=" + config.getNumExperts()
                + ", topK=" + config.getTopK()
                + ", sharedExperts=" + config.getNumSharedExperts()
                + ", expertHiddenDim=" + config.getExpertHiddenDim());
        System.out.println();
        System.out.println(PlantUML.generateModuleGraph(moeLayer));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }

    /**
     * 生成 MoE Layer 的计算图
     * 通过前向传播构建计算图，展示 Sigmoid 路由、Top-K 选择、专家加权组合等数据流
     */
    @Test
    public void generateMoELayerComputationDiagram() {
        DeepSeekV3Config config = DeepSeekV3Config.createMicroConfig();
        DeepSeekV3MoEBlock moeLayer = new DeepSeekV3MoEBlock("moe_layer", config);
        moeLayer.eval();

        // 构造输入: [batch_size=1, seq_len=4, nEmbd=64]
        int batchSize = 1;
        int seqLen = 4;
        int nEmbd = config.getNEmbd();
        float[] inputData = new float[batchSize * seqLen * nEmbd];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (Math.random() * 0.1);
        }
        Variable input = new Variable(NdArray.of(inputData, Shape.of(batchSize, seqLen, nEmbd)), "moe_input");

        // 前向传播，构建计算图
        Variable output = moeLayer.forward(input);
        output.setName("moe_output");

        System.out.println("========== DeepSeek-V3 MoE Layer 计算图 (PlantUML) ==========");
        System.out.println("输入形状: " + input.getValue().getShape());
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println("配置: numExperts=" + config.getNumExperts()
                + ", topK=" + config.getTopK()
                + ", sharedExperts=" + config.getNumSharedExperts());
        System.out.println();
        System.out.println(PlantUML.generateSimpleGraph(output));
        System.out.println();
        System.out.println("=== 可复制到 http://www.plantuml.com/plantuml/uml 渲染查看 ===");
    }
}
