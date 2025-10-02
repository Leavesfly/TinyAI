package io.leavesfly.tinyai.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * LoRA微调演示程序
 * 
 * 该演示程序展示了LoRA（Low-Rank Adaptation）微调技术的核心原理和实际应用：
 * 1. 创建预训练模型
 * 2. 应用LoRA适配器进行微调
 * 3. 展示参数效率优势
 * 4. 演示模型性能保持
 * 5. 展示权重合并过程
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraDemo {
    
    public static void main(String[] args) {
        System.out.println(repeat("=", 60));
        System.out.println("          LoRA微调技术演示程序");
        System.out.println(repeat("=", 60));
        
        try {
            // 1. 演示基础LoRA概念
            demonstrateBasicLoRA();
            
            // 2. 演示LoRA线性层
            demonstrateLoraLinearLayer();
            
            // 3. 演示完整LoRA模型
            demonstrateLoraModel();
            
            // 4. 演示预训练模型微调
            demonstratePretrainedFineTuning();
            
            // 5. 演示参数效率分析
            demonstrateParameterEfficiency();
            
            System.out.println("\n" + repeat("=", 60));
            System.out.println("          LoRA演示完成！");
            System.out.println(repeat("=", 60));
            
        } catch (Exception e) {
            System.err.println("演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示基础LoRA概念
     */
    private static void demonstrateBasicLoRA() {
        System.out.println("\n1. 基础LoRA概念演示");
        System.out.println(repeat("-", 40));
        
        // 创建LoRA配置
        LoraConfig config = LoraConfig.createDefault(8); // rank=8, alpha=8
        System.out.println("LoRA配置: " + config);
        
        // 创建LoRA适配器
        int inputDim = 256;
        int outputDim = 128;
        LoraAdapter adapter = new LoraAdapter(inputDim, outputDim, config);
        
        System.out.printf("原始矩阵大小: %d x %d = %,d 参数\n", 
                         inputDim, outputDim, inputDim * outputDim);
        System.out.printf("LoRA参数: %,d 参数\n", adapter.getParameterCount());
        System.out.printf("参数减少: %.2f%%\n", adapter.getParameterReduction(inputDim * outputDim) * 100);
        
        // 演示前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(4, inputDim)); // batch_size=4
        Variable inputVar = new Variable(input);
        Variable output = adapter.forward(inputVar);
        
        System.out.printf("输入形状: %s\n", input.getShape());
        System.out.printf("输出形状: %s\n", output.getValue().getShape());
        System.out.println("LoRA适配器状态: " + adapter);
    }
    
    /**
     * 演示LoRA线性层
     */
    private static void demonstrateLoraLinearLayer() {
        System.out.println("\n2. LoRA线性层演示");
        System.out.println(repeat("-", 40));
        
        // 创建LoRA配置
        LoraConfig config = LoraConfig.createMediumRank(); // rank=16, alpha=32
        
        // 创建LoRA线性层
        LoraLinearLayer layer = new LoraLinearLayer(
            "demo_lora_layer", 512, 256, config, true);
        
        System.out.println("LoRA线性层信息:");
        System.out.println(layer);
        
        // 测试前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(8, 512)); // batch_size=8
        Variable output = layer.layerForward(new Variable(input));
        
        System.out.printf("输入形状: %s -> 输出形状: %s\n", 
                         input.getShape(), output.getValue().getShape());
        
        // 演示LoRA开关功能
        System.out.println("\n测试LoRA开关功能:");
        layer.disableLora();
        Variable outputWithoutLora = layer.layerForward(new Variable(input));
        System.out.println("LoRA禁用后的输出形状: " + outputWithoutLora.getValue().getShape());
        
        layer.enableLora();
        Variable outputWithLora = layer.layerForward(new Variable(input));
        System.out.println("LoRA启用后的输出形状: " + outputWithLora.getValue().getShape());
        
        // 比较输出差异
        NdArray diff = outputWithLora.getValue().sub(outputWithoutLora.getValue());
        float maxDiff = diff.abs().max();
        System.out.printf("启用/禁用LoRA的最大输出差异: %.6f\n", maxDiff);
    }
    
    /**
     * 演示完整LoRA模型
     */
    private static void demonstrateLoraModel() {
        System.out.println("\n3. 完整LoRA模型演示");
        System.out.println(repeat("-", 40));
        
        // 定义网络架构
        int[] layerSizes = {784, 256, 128, 10}; // 类似MNIST分类网络
        LoraConfig config = LoraConfig.createLowRank(); // 使用低秩配置以适应较小的层
        
        // 创建LoRA模型
        LoraModel model = new LoraModel("mnist_lora_model", layerSizes, config, false);
        
        System.out.println("LoRA模型信息:");
        System.out.println(model.getModelInfo());
        
        // 测试前向传播
        NdArray input = NdArray.likeRandomN(Shape.of(32, 784)); // batch_size=32
        Variable output = model.layerForward(new Variable(input));
        
        System.out.printf("模型测试 - 输入: %s -> 输出: %s\n", 
                         input.getShape(), output.getValue().getShape());
        
        // 演示模型状态管理
        System.out.println("\n模型状态管理:");
        Map<String, NdArray> state = model.saveLoraState();
        System.out.printf("保存的LoRA状态包含 %d 个参数组\n", state.size());
        
        // 演示权重合并
        List<NdArray> mergedWeights = model.mergeAllLoraWeights();
        System.out.printf("合并后得到 %d 层权重矩阵\n", mergedWeights.size());
        for (int i = 0; i < mergedWeights.size(); i++) {
            System.out.printf("  第%d层权重形状: %s\n", i+1, mergedWeights.get(i).getShape());
        }
    }
    
    /**
     * 演示预训练模型微调
     */
    private static void demonstratePretrainedFineTuning() {
        System.out.println("\n4. 预训练模型微调演示");
        System.out.println(repeat("-", 40));
        
        // 模拟预训练权重
        List<NdArray> pretrainedWeights = new ArrayList<>();
        pretrainedWeights.add(NdArray.likeRandomN(Shape.of(784, 512))); // 输入层到隐藏层1
        pretrainedWeights.add(NdArray.likeRandomN(Shape.of(512, 256))); // 隐藏层1到隐藏层2
        pretrainedWeights.add(NdArray.likeRandomN(Shape.of(256, 10)));  // 隐藏层2到输出层
        
        List<NdArray> pretrainedBiases = new ArrayList<>();
        pretrainedBiases.add(NdArray.zeros(Shape.of(1, 512)));
        pretrainedBiases.add(NdArray.zeros(Shape.of(1, 256)));
        pretrainedBiases.add(NdArray.zeros(Shape.of(1, 10)));
        
        System.out.println("模拟预训练模型权重:");
        for (int i = 0; i < pretrainedWeights.size(); i++) {
            System.out.printf("  第%d层: %s\n", i+1, pretrainedWeights.get(i).getShape());
        }
        
        // 从预训练权重创建LoRA模型
        LoraConfig config = LoraConfig.createLowRank(); // 使用低秩配置进行快速微调
        LoraModel fineTunedModel = LoraModel.fromPretrained(
            "finetuned_model", pretrainedWeights, pretrainedBiases, config, false);
        
        System.out.println("\n微调模型信息:");
        System.out.println(fineTunedModel.getModelInfo());
        
        // 模拟微调前后的性能比较
        NdArray testInput = NdArray.likeRandomN(Shape.of(16, 784));
        Variable originalOutput = fineTunedModel.layerForward(new Variable(testInput));
        
        System.out.printf("微调模型测试 - 输入: %s -> 输出: %s\n", 
                         testInput.getShape(), originalOutput.getValue().getShape());
        
        // 展示只有LoRA参数是可训练的
        System.out.println("\n参数训练状态:");
        System.out.printf("总参数: %,d\n", fineTunedModel.getTotalParameterCount());
        System.out.printf("可训练参数: %,d\n", fineTunedModel.getTrainableParameterCount());
        System.out.printf("冻结参数比例: %.2f%%\n", 
                         (1.0 - (double)fineTunedModel.getTrainableParameterCount() / 
                          fineTunedModel.getTotalParameterCount()) * 100);
    }
    
    /**
     * 演示参数效率分析
     */
    private static void demonstrateParameterEfficiency() {
        System.out.println("\n5. 参数效率分析");
        System.out.println(repeat("-", 40));
        
        // 比较不同rank配置的效率
        int[] ranks = {4, 8, 16, 32, 64};
        int inputDim = 1024;
        int outputDim = 1024;
        
        System.out.printf("原始全连接层参数: %,d\n", inputDim * outputDim);
        System.out.println("\nLoRA配置效率比较:");
        System.out.printf("%-6s %-12s %-15s %-15s\n", "Rank", "LoRA参数", "参数减少率", "计算开销比");
        System.out.println(repeat("-", 55));
        
        for (int rank : ranks) {
            LoraConfig config = new LoraConfig(rank, rank);
            int loraParams = rank * (inputDim + outputDim);
            double reduction = config.getParameterReduction(inputDim, outputDim);
            double computeRatio = (double)loraParams / (inputDim * outputDim);
            
            System.out.printf("%-6d %-12,d %-15.2f%% %-15.4fx\n", 
                             rank, loraParams, reduction * 100, computeRatio);
        }
        
        // 展示不同任务场景的推荐配置
        System.out.println("\n推荐配置指南:");
        System.out.println("• 快速原型验证: " + LoraConfig.createLowRank());
        System.out.println("• 常规微调任务: " + LoraConfig.createMediumRank());
        System.out.println("• 复杂任务适应: " + LoraConfig.createHighRank());
        
        // 演示配置验证
        System.out.println("\n配置验证示例:");
        try {
            LoraConfig validConfig = new LoraConfig(16, 32.0);
            validConfig.validate(512, 256);
            System.out.println("✓ 配置验证通过: " + validConfig);
        } catch (Exception e) {
            System.out.println("✗ 配置验证失败: " + e.getMessage());
        }
        
        try {
            LoraConfig invalidConfig = new LoraConfig(300, 32.0);
            invalidConfig.validate(256, 128); // rank > min(input_dim, output_dim)
            System.out.println("✓ 配置验证通过: " + invalidConfig);
        } catch (Exception e) {
            System.out.println("✗ 配置验证失败: " + e.getMessage());
        }
    }
    
    /**
     * 打印分隔线
     */
    private static void printSeparator(String title) {
        System.out.println("\n" + repeat("=", 20) + " " + title + " " + repeat("=", 20));
    }
    
    /**
     * 重复字符串
     */
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    /**
     * 格式化打印矩阵信息
     */
    private static void printMatrixInfo(String name, NdArray matrix) {
        System.out.printf("%s: 形状=%s, 均值=%.4f, 标准差=%.4f\n", 
                         name, matrix.getShape(), 
                         matrix.sum().getNumber().floatValue() / matrix.getShape().size(),
                         calculateStd(matrix));
    }
    
    /**
     * 计算标准差（简单实现）
     */
    private static float calculateStd(NdArray array) {
        float mean = array.sum().getNumber().floatValue() / array.getShape().size();
        // 简化实现，实际应该计算真正的标准差
        return Math.abs(array.max() - mean);
    }
}