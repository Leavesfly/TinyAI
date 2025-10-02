package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Qwen3模型演示类
 * 
 * 展示如何使用基于TinyAI架构实现的Qwen3大语言模型
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Demo {
    
    public static void main(String[] args) {
        System.out.println("=== Qwen3模型演示 ===");
        
        try {
            // 创建小型Qwen3模型用于演示
            demonstrateBasicUsage();
            
            // 展示模型架构
            demonstrateArchitecture();
            
            // 展示文本生成
            demonstrateTextGeneration();
            
        } catch (Exception e) {
            System.err.println("演示出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 基本使用演示
     */
    private static void demonstrateBasicUsage() {
        System.out.println("\n1. 基本使用演示");
        
        // 创建模型
        Qwen3Model model = Qwen3Model.createTinyModel("demo_qwen3");
        System.out.println("✓ 创建了小型Qwen3模型");
        
        // 准备输入数据
        NdArray inputIds = NdArray.of(new float[]{1, 15, 25, 35, 45});
        System.out.println("✓ 准备了输入序列: [1, 15, 25, 35, 45]");
        
        // 前向传播
        Variable logits = model.forwardWithLogits(new Variable(inputIds));
        Shape outputShape = logits.getValue().getShape();
        System.out.println("✓ 前向传播完成，输出形状: " + outputShape);
        
        // 预测下一个token
        int nextToken = model.predictNextToken(inputIds);
        System.out.println("✓ 预测的下一个token ID: " + nextToken);
    }
    
    /**
     * 模型架构演示
     */
    private static void demonstrateArchitecture() {
        System.out.println("\n2. 模型架构演示");
        
        Qwen3Model model = Qwen3Model.createTinyModel("arch_demo");
        
        // 打印模型信息
        model.printModelInfo();
        
        // 展示配置详情
        Qwen3Config config = model.getConfig();
        System.out.println("\n详细配置:");
        System.out.println("- 每个注意力头维度: " + config.getHeadDim());
        System.out.println("- KV头分组数: " + config.getNumKeyValueGroups());
        System.out.println("- RoPE基础频率: " + config.getRopeTheta());
        System.out.println("- RMSNorm epsilon: " + config.getRmsNormEps());
    }
    
    /**
     * 文本生成演示
     */
    private static void demonstrateTextGeneration() {
        System.out.println("\n3. 文本生成演示");
        
        Qwen3Model model = Qwen3Model.createTinyModel("gen_demo");
        
        // 单序列生成
        NdArray prompt = NdArray.of(new float[]{10, 20, 30});
        System.out.println("输入序列: " + arrayToString(prompt));
        
        NdArray generated = model.generate(prompt, 10);
        System.out.println("生成序列: " + arrayToString(generated));
        
        // 批次生成
        NdArray batchPrompt = NdArray.of(Shape.of(2, 3));
        batchPrompt.set(5, 0, 0); batchPrompt.set(10, 0, 1); batchPrompt.set(15, 0, 2);
        batchPrompt.set(7, 1, 0); batchPrompt.set(14, 1, 1); batchPrompt.set(21, 1, 2);
        
        System.out.println("\n批次输入:");
        printBatchArray(batchPrompt);
        
        NdArray batchGenerated = model.generate(batchPrompt, 8);
        System.out.println("批次生成:");
        printBatchArray(batchGenerated);
    }
    
    /**
     * 将NdArray转换为字符串显示
     */
    private static String arrayToString(NdArray array) {
        if (array.getShape().getDimNum() == 1) {
            StringBuilder sb = new StringBuilder("[");
            int len = array.getShape().getDimension(0);
            for (int i = 0; i < len; i++) {
                if (i > 0) sb.append(", ");
                sb.append((int) array.get(i));
            }
            sb.append("]");
            return sb.toString();
        } else {
            return array.getShape().toString();
        }
    }
    
    /**
     * 打印批次数组
     */
    private static void printBatchArray(NdArray array) {
        if (array.getShape().getDimNum() == 2) {
            int batchSize = array.getShape().getDimension(0);
            int seqLen = array.getShape().getDimension(1);
            
            for (int b = 0; b < batchSize; b++) {
                System.out.print("  Batch " + b + ": [");
                for (int s = 0; s < seqLen; s++) {
                    if (s > 0) System.out.print(", ");
                    System.out.print((int) array.get(b, s));
                }  
                System.out.println("]");
            }
        }
    }
}
