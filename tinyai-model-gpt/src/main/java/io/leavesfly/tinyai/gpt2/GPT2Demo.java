package io.leavesfly.tinyai.gpt2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-2简单演示程序
 * 
 * 演示如何使用新实现的GPT-2模型
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT2Demo {
    
    public static void main(String[] args) {
        System.out.println("=== GPT-2 模型演示 ===");
        
        try {
            // 1. 创建小型GPT-2模型进行测试
            System.out.println("1. 创建小型GPT-2模型...");
            GPT2Model model = GPT2Model.createSmallModel("demo_gpt2");
            
            // 2. 打印模型信息
            System.out.println("2. 模型信息:");
            model.printModelInfo();
            
            // 3. 测试前向传播
            System.out.println("3. 测试前向传播...");
            
            // 创建测试输入 (batch_size=1, seq_len=10)
            float[][] testInput = {{1, 5, 10, 2, 7, 3, 8, 4, 9, 6}};
            NdArray inputTokens = NdArray.of(testInput);
            
            System.out.println("输入形状: " + inputTokens.getShape());
            
            // 验证输入
            model.validateInput(inputTokens);
            System.out.println("✓ 输入验证通过");
            
            // 前向传播
            Variable output = model.predict(inputTokens);
            System.out.println("输出形状: " + output.getValue().getShape());
            System.out.println("✓ 前向传播成功");
            
            // 4. 测试下一个token预测
            System.out.println("4. 测试下一个token预测...");
            int predictedToken = model.predictNextToken(inputTokens);
            System.out.println("预测的下一个token ID: " + predictedToken);
            System.out.println("✓ token预测成功");
            
            // 5. 测试序列生成
            System.out.println("5. 测试序列生成...");
            float[][] startSeq = {{1, 5}};
            NdArray startTokens = NdArray.of(startSeq);
            
            NdArray generatedSeq = model.generateSequence(startTokens, 5);
            System.out.println("生成序列形状: " + generatedSeq.getShape());
            
            // 打印生成的序列
            System.out.print("生成的序列: [");
            for (int i = 0; i < generatedSeq.getShape().getDimension(1); i++) {
                System.out.print((int)generatedSeq.get(0, i));
                if (i < generatedSeq.getShape().getDimension(1) - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");
            System.out.println("✓ 序列生成成功");
            
            // 6. 测试模型组件
            System.out.println("6. 测试模型组件...");
            
            GPT2Config config = model.getConfig();
            System.out.println("模型配置验证通过: " + config.getNLayer() + " 层");
            
            GPT2TokenEmbedding tokenEmbedding = model.getTokenEmbedding();
            System.out.println("Token嵌入层: 词汇表大小=" + tokenEmbedding.getVocabSize() + 
                             ", 嵌入维度=" + tokenEmbedding.getDModel());
            
            GPT2OutputHead outputHead = model.getOutputHead();
            System.out.println("输出头: 词汇表大小=" + outputHead.getVocabSize() + 
                             ", 模型维度=" + outputHead.getNEmbd());
            
            System.out.println("✓ 组件测试通过");
            
            System.out.println("\n=== 演示完成！GPT-2模型工作正常 ===");
            
        } catch (Exception e) {
            System.err.println("❌ 演示过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}