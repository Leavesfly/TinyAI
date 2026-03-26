package io.leavesfly.tinyai.minimind.cli;

import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.Command;
import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.ArgParser;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * 交互式对话命令
 * 
 * @author leavesfly
 * @since 2024
 */
public class ChatCommand implements Command {
    
    /** 对话历史最大长度 */
    private static final int MAX_HISTORY_SIZE = 20;
    
    @Override
    public void execute(String[] args) throws Exception {
        ArgParser parser = new ArgParser(args);
        
        if (parser.has("help")) {
            printHelp();
            return;
        }
        
        String modelPath = parser.get("model", "model/chat.pt");
        
        System.out.println("=".repeat(60));
        System.out.println("MiniMind 交互式对话");
        System.out.println("=".repeat(60));
        System.out.println("模型路径: " + modelPath);
        System.out.println("输入 'exit' 或 'quit' 退出");
        System.out.println("=".repeat(60));
        System.out.println();
        
        // 加载模型
        MiniMindModel model;
        MiniMindTokenizer tokenizer;
        
        try {
            ModelLoader.ModelLoadResult loadResult = ModelLoader.loadModel(modelPath, "minimind-chat");
            model = loadResult.getModel();
            tokenizer = loadResult.getTokenizer();
            
            System.out.println("模型加载完成!");
            
            if (!new File(modelPath).exists()) {
                System.out.println("\n[提示] 当前使用随机初始化模型,回复为随机文本");
                System.out.println("       请先训练模型或加载预训练权重\n");
            }
            
        } catch (Exception e) {
            System.err.println("模型加载失败: " + e.getMessage());
            e.printStackTrace();
            return;
        }
        
        // 对话历史
        List<String> conversationHistory = new ArrayList<>();
        
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("用户: ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit")) {
                System.out.println("再见!");
                break;
            }
            
            if (input.isEmpty()) {
                continue;
            }
            
            // 添加到对话历史
            conversationHistory.add("用户: " + input);
            
            // 限制对话历史长度，防止内存泄漏
            while (conversationHistory.size() > MAX_HISTORY_SIZE) {
                conversationHistory.remove(0);
            }
            
            try {
                // 构建对话上下文（保留最近5轮对话）
                StringBuilder context = new StringBuilder();
                int startIdx = Math.max(0, conversationHistory.size() - 10);
                for (int i = startIdx; i < conversationHistory.size(); i++) {
                    context.append(conversationHistory.get(i)).append("\n");
                }
                context.append("助手: ");
                
                // 编码输入
                List<Integer> promptIds = tokenizer.encode(context.toString(), false, false);
                int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
                
                // 生成回复（使用低温度保证连贯性）
                int[] generated = model.generate(
                    promptArray,
                    50,          // maxNewTokens
                    0.7f,        // temperature
                    40,          // topK
                    0.9f         // topP
                );
                
                // 解码回复
                List<Integer> genIds = new ArrayList<>();
                for (int id : generated) {
                    genIds.add(id);
                }
                String fullResponse = tokenizer.decode(genIds, true);
                
                // 提取助手回复部分（移除上下文）
                String response = fullResponse;
                if (fullResponse.contains("助手: ")) {
                    int assistantIdx = fullResponse.lastIndexOf("助手: ");
                    response = fullResponse.substring(assistantIdx + 4).trim();
                }
                
                // 输出回复
                System.out.println("助手: " + response);
                System.out.println();
                
                // 添加到对话历史
                conversationHistory.add("助手: " + response);
                
                // 再次限制对话历史长度
                while (conversationHistory.size() > MAX_HISTORY_SIZE) {
                    conversationHistory.remove(0);
                }
                
            } catch (Exception e) {
                System.err.println("生成回复失败: " + e.getMessage());
                System.out.println("助手: [模型生成失败,请重试]\n");
            }
        }
        
        // 不关闭Scanner，因为它包装了System.in
        // scanner.close();
    }
    
    @Override
    public String getDescription() {
        return "交互式对话模式";
    }
    
    @Override
    public void printHelp() {
        System.out.println("用法: minimind chat [options]");
        System.out.println();
        System.out.println("交互式对话模式");
        System.out.println();
        System.out.println("选项:");
        System.out.println("  --model FILE           模型路径");
    }
}