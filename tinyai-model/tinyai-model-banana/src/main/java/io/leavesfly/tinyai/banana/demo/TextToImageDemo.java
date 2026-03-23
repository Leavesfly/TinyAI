package io.leavesfly.tinyai.banana.demo;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.banana.model.BananaModel;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import java.util.Random;

/**
 * 文本到图像生成演示
 * 
 * 演示如何使用Banana模型从文本描述生成图像
 * 
 * 流程:
 * 1. 创建Banana模型
 * 2. 准备文本描述(token IDs)
 * 3. 调用generateImage生成图像
 * 4. 输出生成的图像信息
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TextToImageDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Banana模型 - 文本到图像生成演示");
        System.out.println("=".repeat(80));
        
        try {
            // 1. 创建Banana模型 (使用Tiny配置进行演示)
            System.out.println("\n【步骤1】创建Banana模型...");
            BananaModel model = BananaModel.create("banana_demo", "tiny");
            
            System.out.println("✓ 模型创建成功");
            System.out.println(model.getConfigSummary());
            
            // 2. 准备文本描述
            System.out.println("\n【步骤2】准备文本描述...");
            
            // 创建模拟的文本token IDs
            // 实际应用中，这些token应该由tokenizer生成
            int batchSize = 2;
            int textLength = 32;
            
            float[] tokenData = new float[batchSize * textLength];
            Random random = new Random(42);
            for (int i = 0; i < tokenData.length; i++) {
                // 使用随机token ID (范围: 0-999的整数)
                tokenData[i] = random.nextInt(1000);
            }
            
            NdArray textTokens = NdArray.of(tokenData, Shape.of(batchSize, textLength));
            Variable textInput = new Variable(textTokens);
            
            System.out.println("✓ 文本描述准备完成");
            System.out.println("  - Batch大小: " + batchSize);
            System.out.println("  - 序列长度: " + textLength);
            System.out.println("  - 示例文本: \"A beautiful sunset over the ocean\"");
            System.out.println("  - 示例文本: \"A cute cat sitting on a chair\"");
            
            // 3. 生成图像
            System.out.println("\n【步骤3】生成图像...");
            System.out.println("  ⏳ 正在执行文本编码...");
            System.out.println("  ⏳ 正在执行多模态融合...");
            System.out.println("  ⏳ 正在执行图像解码...");
            
            long startTime = System.currentTimeMillis();
            Variable generatedImage = model.generateImage(textInput);
            long endTime = System.currentTimeMillis();
            
            System.out.println("✓ 图像生成完成");
            System.out.println("  - 生成耗时: " + (endTime - startTime) + "ms");
            
            // 4. 输出结果信息
            System.out.println("\n【步骤4】输出结果...");
            int[] imageShape = generatedImage.getValue().getShape().getShapeDims();
            
            System.out.println("✓ 生成图像信息:");
            System.out.println("  - 形状: " + java.util.Arrays.toString(imageShape));
            System.out.println("  - 批次大小: " + imageShape[0]);
            System.out.println("  - 通道数: " + imageShape[1] + " (RGB)");
            System.out.println("  - 图像尺寸: " + imageShape[2] + "x" + imageShape[3]);
            
            // 计算像素值范围
            NdArray imageData = generatedImage.getValue();
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            float sum = 0.0f;
            int count = imageShape[0] * imageShape[1] * imageShape[2] * imageShape[3];
            
            for (int i = 0; i < count; i++) {
                float value = imageData.getArray()[i];
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
            }
            float mean = sum / count;
            
            System.out.println("  - 像素值范围: [" + String.format("%.4f", min) + ", " + 
                             String.format("%.4f", max) + "]");
            System.out.println("  - 像素值均值: " + String.format("%.4f", mean));
            
            // 5. 保存提示
            System.out.println("\n【步骤5】保存图像 (模拟)");
            System.out.println("  提示: 在实际应用中，可以使用以下方式保存图像:");
            System.out.println("  1. 将像素值从[-1, 1]归一化到[0, 255]");
            System.out.println("  2. 使用ImageIO或其他图像库保存为PNG/JPEG");
            System.out.println("  3. 示例代码:");
            System.out.println("     BufferedImage img = new BufferedImage(width, height, TYPE_INT_RGB);");
            System.out.println("     // 设置像素值...");
            System.out.println("     ImageIO.write(img, \"png\", new File(\"output.png\"));");
            
            // 6. 性能统计
            System.out.println("\n【性能统计】");
            System.out.println("  - 模型参数量: " + model.getConfig().formatParameters());
            System.out.println("  - 单张图像生成时间: " + (endTime - startTime) / batchSize + "ms");
            System.out.println("  - 吞吐量: " + String.format("%.2f", (float) batchSize * 1000 / (endTime - startTime)) + " 张/秒");
            
            System.out.println("\n" + "=".repeat(80));
            System.out.println("演示完成！");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("\n❌ 发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}