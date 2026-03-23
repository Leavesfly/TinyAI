package io.leavesfly.tinyai.minimind.cli;

import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.Command;
import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.ArgParser;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.io.File;
import java.util.List;

/**
 * 文本生成命令
 * 
 * @author leavesfly
 * @since 2024
 */
public class GenerateCommand implements Command {
    
    @Override
    public void execute(String[] args) throws Exception {
        ArgParser parser = new ArgParser(args);
        
        if (parser.has("help")) {
            printHelp();
            return;
        }
        
        String modelPath = parser.get("model", "model/trained.pt");
        String prompt = parser.get("prompt", "你好");
        int maxLength = parser.getInt("max-length", 100);
        float temperature = parser.getFloat("temperature", 1.0f);
        int topK = parser.getInt("top-k", 50);
        
        System.out.println("=".repeat(60));
        System.out.println("MiniMind 文本生成");
        System.out.println("=".repeat(60));
        System.out.println("模型路径: " + modelPath);
        System.out.println("提示词: " + prompt);
        System.out.println("最大长度: " + maxLength);
        System.out.println("温度: " + temperature);
        System.out.println("Top-K: " + topK);
        System.out.println("=".repeat(60));
        System.out.println();
        
        // 实际生成逻辑
        try {
            // 1. 加载或创建模型
            ModelLoader.ModelLoadResult loadResult = ModelLoader.loadModel(modelPath, "minimind-generate");
            MiniMindModel model = loadResult.getModel();
            MiniMindTokenizer tokenizer = loadResult.getTokenizer();
            
            // 2. 编码输入
            List<Integer> promptIds = tokenizer.encode(prompt, false, false);
            int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
            
            System.out.println("开始生成...");
            long startTime = System.currentTimeMillis();
            
            // 3. 调用模型生成
            int[] generated = model.generate(
                promptArray,
                maxLength,
                temperature,
                topK,
                temperature > 0 ? 0.9f : 0.0f  // topP
            );
            
            long endTime = System.currentTimeMillis();
            
            // 4. 解码输出
            List<Integer> genIds = new java.util.ArrayList<>();
            for (int id : generated) {
                genIds.add(id);
            }
            String result = tokenizer.decode(genIds, true);
            
            // 5. 输出结果
            System.out.println("生成结果:");
            System.out.println("-".repeat(60));
            System.out.println(result);
            System.out.println("-".repeat(60));
            
            // 6. 统计信息
            int generatedTokens = generated.length - promptArray.length;
            double elapsedSeconds = (endTime - startTime) / 1000.0;
            double tokensPerSecond = generatedTokens / elapsedSeconds;
            
            System.out.println("\n统计信息:");
            System.out.println("  - 生成Token数: " + generatedTokens);
            System.out.println("  - 总Token数: " + generated.length);
            System.out.println("  - 耗时: " + String.format("%.2f", elapsedSeconds) + "秒");
            System.out.println("  - 速度: " + String.format("%.2f", tokensPerSecond) + " tokens/秒");
            
            if (!new File(modelPath).exists()) {
                System.out.println("\n[提示] 当前使用随机初始化模型,输出为随机文本");
                System.out.println("       请先训练模型或加载预训练权重以获得有意义的输出");
                System.out.println("       参考: Example05-预训练流程.java");
            }
            
        } catch (Exception e) {
            System.err.println("生成失败: " + e.getMessage());
            e.printStackTrace();
            System.out.println("\n提示: 请参考 Example06-文本生成策略.java");
        }
    }
    
    @Override
    public String getDescription() {
        return "文本生成";
    }
    
    @Override
    public void printHelp() {
        System.out.println("用法: minimind generate [options]");
        System.out.println();
        System.out.println("文本生成");
        System.out.println();
        System.out.println("选项:");
        System.out.println("  --model FILE           模型路径");
        System.out.println("  --prompt TEXT          提示词");
        System.out.println("  --max-length INT       最大生成长度 (default: 100)");
        System.out.println("  --temperature FLOAT    温度参数 (default: 1.0)");
        System.out.println("  --top-k INT            Top-K采样 (default: 50)");
    }
}