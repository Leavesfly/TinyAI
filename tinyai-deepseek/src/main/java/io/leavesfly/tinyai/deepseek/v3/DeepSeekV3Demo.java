package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;
import java.util.Scanner;

/**
 * DeepSeek V3 模型演示类
 * 
 * 提供交互式演示界面，展示V3模型的各种功能：
 * 1. 文本生成演示
 * 2. 代码生成演示  
 * 3. 数学推理演示
 * 4. 多任务处理演示
 * 5. 模型性能分析
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Demo {
    
    private static DeepSeekV3Model model;
    private static Scanner scanner;
    
    public static void main(String[] args) {
        System.out.println("=== DeepSeek V3 交互式演示 ===");
        System.out.println("基于TinyAI架构的DeepSeek V3模型演示");
        
        // 初始化
        initializeDemo();
        
        // 主菜单循环
        mainMenuLoop();
        
        // 清理资源
        cleanup();
    }
    
    /**
     * 初始化演示环境
     */
    private static void initializeDemo() {
        System.out.println("\n正在初始化DeepSeek V3模型...");
        
        // 创建小型V3模型用于演示
        model = DeepSeekV3Model.createSmallV3("demo_v3");
        scanner = new Scanner(System.in);
        
        System.out.println("✓ 模型初始化完成");
        System.out.println(model.getV3ModelInfo());
    }
    
    /**
     * 主菜单循环
     */
    private static void mainMenuLoop() {
        boolean running = true;
        
        while (running) {
            showMainMenu();
            
            try {
                int choice = scanner.nextInt();
                scanner.nextLine(); // 消费换行符
                
                switch (choice) {
                    case 1:
                        textGenerationDemo();
                        break;
                    case 2:
                        codeGenerationDemo();
                        break;
                    case 3:
                        mathReasoningDemo();
                        break;
                    case 4:
                        multiTaskDemo();
                        break;
                    case 5:
                        performanceAnalysisDemo();
                        break;
                    case 6:
                        modelInspectionDemo();
                        break;
                    case 7:
                        expertRoutingDemo();
                        break;
                    case 8:
                        reasoningChainDemo();
                        break;
                    case 0:
                        running = false;
                        System.out.println("感谢使用DeepSeek V3演示！");
                        break;
                    default:
                        System.out.println("无效选择，请重试。");
                }
                
                if (running) {
                    System.out.println("\n按回车键继续...");
                    scanner.nextLine();
                }
                
            } catch (Exception e) {
                System.err.println("执行出错: " + e.getMessage());
                scanner.nextLine(); // 清理输入
            }
        }
    }
    
    /**
     * 显示主菜单
     */
    private static void showMainMenu() {
        System.out.println("\n========== DeepSeek V3 演示菜单 ==========");
        System.out.println("1. 文本生成演示");
        System.out.println("2. 代码生成演示");
        System.out.println("3. 数学推理演示");
        System.out.println("4. 多任务处理演示");
        System.out.println("5. 性能分析演示");
        System.out.println("6. 模型结构检视");
        System.out.println("7. 专家路由分析");
        System.out.println("8. 推理链展示");
        System.out.println("0. 退出");
        System.out.println("=========================================");
        System.out.print("请选择功能 (0-8): ");
    }
    
    /**
     * 文本生成演示
     */
    private static void textGenerationDemo() {
        System.out.println("\n=== 文本生成演示 ===");
        System.out.print("请输入提示文本: ");
        String prompt = scanner.nextLine();
        
        // 模拟文本编码（简化实现）
        NdArray inputIds = encodeText(prompt);
        Variable input = new Variable(inputIds);
        
        System.out.println("\n正在生成文本...");
        long startTime = System.currentTimeMillis();
        
        // 前向传播
        Variable output = model.layerForward(input);
        
        long endTime = System.currentTimeMillis();
        
        // 显示结果
        System.out.println("✓ 文本生成完成");
        System.out.println("输入长度: " + inputIds.getShape().getDimension(1) + " tokens");
        System.out.println("输出形状: " + output.getValue().getShape());
        System.out.println("生成时间: " + (endTime - startTime) + "ms");
        System.out.println("当前任务类型: " + model.getCurrentTaskType());
        
        // 显示推理链摘要
        List<V3ReasoningStep> reasoningChain = model.getCurrentReasoningChain();
        if (!reasoningChain.isEmpty()) {
            System.out.println("\n推理过程摘要:");
            for (int i = 0; i < Math.min(3, reasoningChain.size()); i++) {
                V3ReasoningStep step = reasoningChain.get(i);
                System.out.println("  " + (i+1) + ". " + step.getSummary());
            }
        }
    }
    
    /**
     * 代码生成演示
     */
    private static void codeGenerationDemo() {
        System.out.println("\n=== 代码生成演示 ===");
        System.out.print("请输入代码描述: ");
        String description = scanner.nextLine();
        
        // 模拟代码任务编码
        NdArray inputIds = encodeText(description);
        Variable input = new Variable(inputIds);
        
        System.out.println("\n正在分析代码任务...");
        
        // 前向传播（模拟代码任务）
        model.resetState();
        Variable output = model.layerForward(input);
        
        // 显示代码生成分析结果
        System.out.println("✓ 代码分析完成");
        
        CodeGenerationModule.CodeGenerationResult codeInfo = model.getCodeInfo();
        if (codeInfo != null) {
            System.out.println("\n代码分析结果:");
            System.out.println(codeInfo.getSummary());
        } else {
            System.out.println("注意: 代码分析模块需要特定的任务类型触发");
        }
        
        // 显示代码生成模块报告
        System.out.println("\n代码生成模块状态:");
        System.out.println(model.getCodeGenerationReport());
    }
    
    /**
     * 数学推理演示
     */
    private static void mathReasoningDemo() {
        System.out.println("\n=== 数学推理演示 ===");
        System.out.print("请输入数学问题: ");
        String mathProblem = scanner.nextLine();
        
        // 模拟数学任务编码
        NdArray inputIds = encodeText(mathProblem);
        Variable input = new Variable(inputIds);
        
        System.out.println("\n正在进行数学推理...");
        
        // 重置状态确保清洁的推理过程
        model.resetState();
        Variable output = model.layerForward(input);
        
        // 显示数学推理结果
        System.out.println("✓ 数学推理完成");
        
        // 详细展示推理链
        List<V3ReasoningStep> reasoningChain = model.getCurrentReasoningChain();
        if (!reasoningChain.isEmpty()) {
            System.out.println("\n数学推理过程:");
            
            double totalConfidence = 0.0;
            for (int i = 0; i < reasoningChain.size(); i++) {
                V3ReasoningStep step = reasoningChain.get(i);
                totalConfidence += step.getConfidence();
                
                System.out.println(String.format("  步骤 %d: %s", i+1, step.getThought()));
                System.out.println(String.format("          置信度: %.3f, 任务类型: %s", 
                    step.getConfidence(), step.getTaskType().getValue()));
                
                if (step.getSelfCorrection() != null) {
                    System.out.println("          自我纠错: " + step.getSelfCorrection());
                }
            }
            
            double avgConfidence = totalConfidence / reasoningChain.size();
            System.out.println(String.format("\n平均置信度: %.3f", avgConfidence));
            
            if (avgConfidence > 0.8) {
                System.out.println("✓ 高置信度推理结果");
            } else if (avgConfidence > 0.6) {
                System.out.println("! 中等置信度推理结果");
            } else {
                System.out.println("⚠ 低置信度推理结果，建议人工验证");
            }
        }
    }
    
    /**
     * 多任务处理演示
     */
    private static void multiTaskDemo() {
        System.out.println("\n=== 多任务处理演示 ===");
        
        TaskType[] taskTypes = {TaskType.GENERAL, TaskType.REASONING, TaskType.CODING, TaskType.MATH};
        String[] sampleInputs = {
            "Hello, how are you today?",
            "Let me think about this complex problem step by step.",
            "Write a function to calculate fibonacci numbers.",
            "Solve the equation: 2x + 5 = 13"
        };
        
        System.out.println("正在测试不同任务类型的处理能力...\n");
        
        for (int i = 0; i < taskTypes.length; i++) {
            TaskType taskType = taskTypes[i];
            String sampleInput = sampleInputs[i];
            
            System.out.println("--- 任务类型: " + taskType.getValue() + " ---");
            System.out.println("示例输入: " + sampleInput);
            
            // 编码输入
            NdArray inputIds = encodeText(sampleInput);
            Variable input = new Variable(inputIds);
            
            // 重置模型状态
            model.resetState();
            
            // 处理任务
            long startTime = System.currentTimeMillis();
            Variable output = model.layerForward(input);
            long endTime = System.currentTimeMillis();
            
            // 分析结果
            System.out.println("处理时间: " + (endTime - startTime) + "ms");
            System.out.println("推理步骤数: " + model.getCurrentReasoningChain().size());
            System.out.println("MoE负载均衡损失: " + 
                String.format("%.6f", model.computeTotalLoadBalancingLoss()));
            
            // 计算专家使用分布
            List<long[]> expertUsage = model.getAllLayersExpertUsage();
            if (!expertUsage.isEmpty()) {
                long[] firstLayerUsage = expertUsage.get(0);
                long totalUsage = 0;
                for (long usage : firstLayerUsage) {
                    totalUsage += usage;
                }
                System.out.println("专家总使用次数: " + totalUsage);
            }
            
            System.out.println();
        }
        
        System.out.println("✓ 多任务处理演示完成");
        System.out.println("模型展现了对不同任务类型的适应能力");
    }
    
    /**
     * 性能分析演示
     */
    private static void performanceAnalysisDemo() {
        System.out.println("\n=== 性能分析演示 ===");
        
        System.out.print("请输入测试轮数 (建议1-20): ");
        int iterations = Math.max(1, Math.min(20, scanner.nextInt()));
        scanner.nextLine(); // 消费换行符
        
        System.out.println("\n正在进行性能基准测试...");
        
        // 不同序列长度的测试
        int[] sequenceLengths = {8, 16, 32, 64};
        
        for (int seqLen : sequenceLengths) {
            System.out.println("\n--- 序列长度: " + seqLen + " ---");
            
            long totalTime = 0;
            long totalTokens = 0;
            
            for (int i = 0; i < iterations; i++) {
                // 创建测试输入
                NdArray inputIds = createRandomInput(1, seqLen);
                Variable input = new Variable(inputIds);
                
                // 重置模型
                model.resetState();
                
                // 计时执行
                long startTime = System.currentTimeMillis();
                Variable output = model.layerForward(input);
                long endTime = System.currentTimeMillis();
                
                totalTime += (endTime - startTime);
                totalTokens += seqLen;
            }
            
            // 计算性能指标
            double avgTime = (double) totalTime / iterations;
            double tokensPerSecond = (totalTokens * 1000.0) / totalTime;
            
            System.out.println("平均推理时间: " + String.format("%.2f", avgTime) + "ms");
            System.out.println("吞吐量: " + String.format("%.1f", tokensPerSecond) + " tokens/秒");
            
            // 内存效率分析
            long totalParams = model.getTotalParameterCount();
            long activeParams = model.getActiveParameterCount();
            double efficiency = (double) activeParams / totalParams * 100;
            
            System.out.println("参数效率: " + String.format("%.1f", efficiency) + "%");
            System.out.println("激活参数: " + String.format("%,d", activeParams));
        }
        
        System.out.println("\n✓ 性能分析完成");
        System.out.println("注意: 实际性能依赖于硬件配置和Java虚拟机优化");
    }
    
    /**
     * 模型结构检视
     */
    private static void modelInspectionDemo() {
        System.out.println("\n=== 模型结构检视 ===");
        
        // 显示完整模型信息
        System.out.println(model.getV3ModelInfo());
        
        // 显示Transformer层详情
        System.out.println("\n--- Transformer层详情 ---");
        List<V3TransformerBlock> layers = model.getTransformerLayers();
        for (int i = 0; i < layers.size(); i++) {
            V3TransformerBlock layer = layers.get(i);
            System.out.println(String.format("第%d层: 模型维度=%d, 注意力头数=%d, 专家数=%d",
                i+1, layer.getDModel(), layer.getNumHeads(), layer.getNumExperts()));
        }
        
        // 显示模块信息
        System.out.println("\n--- 核心模块信息 ---");
        System.out.println("推理模块: " + model.getReasoningModule().getClass().getSimpleName());
        System.out.println("代码生成模块: " + model.getCodeGeneration().getClass().getSimpleName());
        System.out.println("支持的编程语言: " + model.getCodeGeneration().getSupportedLanguages());
        
        // 参数统计
        System.out.println("\n--- 参数统计 ---");
        System.out.println("总参数数: " + String.format("%,d", model.getTotalParameterCount()));
        System.out.println("激活参数数: " + String.format("%,d", model.getActiveParameterCount()));
        
        double efficiency = (double) model.getActiveParameterCount() / model.getTotalParameterCount() * 100;
        System.out.println("参数效率: " + String.format("%.2f", efficiency) + "%");
        
        System.out.println("\n✓ 模型结构检视完成");
    }
    
    /**
     * 专家路由分析
     */
    private static void expertRoutingDemo() {
        System.out.println("\n=== 专家路由分析 ===");
        
        // 创建测试输入进行分析
        System.out.println("正在分析专家路由模式...");
        
        NdArray inputIds = createRandomInput(2, 16);
        Variable input = new Variable(inputIds);
        
        model.resetState();
        model.layerForward(input);
        
        // 显示负载均衡报告
        System.out.println(model.getLoadBalancingReport());
        
        // 分析专家使用模式
        System.out.println("\n--- 专家使用模式分析 ---");
        List<long[]> expertUsage = model.getAllLayersExpertUsage();
        
        for (int layerIdx = 0; layerIdx < expertUsage.size(); layerIdx++) {
            long[] usage = expertUsage.get(layerIdx);
            
            // 计算使用分布
            long totalUsage = 0;
            int activeExperts = 0;
            for (long count : usage) {
                totalUsage += count;
                if (count > 0) activeExperts++;
            }
            
            double averageUsage = totalUsage > 0 ? (double) totalUsage / usage.length : 0;
            double utilization = (double) activeExperts / usage.length * 100;
            
            System.out.println(String.format("第%d层: 总使用=%d, 激活专家=%d/%d (%.1f%%), 平均使用=%.1f",
                layerIdx + 1, totalUsage, activeExperts, usage.length, utilization, averageUsage));
        }
        
        System.out.println("\n✓ 专家路由分析完成");
    }
    
    /**
     * 推理链展示
     */
    private static void reasoningChainDemo() {
        System.out.println("\n=== 推理链展示 ===");
        System.out.print("请输入需要深度思考的问题: ");
        String complexQuestion = scanner.nextLine();
        
        // 编码复杂问题
        NdArray inputIds = encodeText(complexQuestion);
        Variable input = new Variable(inputIds);
        
        System.out.println("\n正在进行深度推理分析...");
        
        model.resetState();
        Variable output = model.layerForward(input);
        
        // 获取详细推理链报告
        String reasoningReport = model.getReasoningChainReport();
        System.out.println(reasoningReport);
        
        // 分析推理质量
        List<V3ReasoningStep> chain = model.getCurrentReasoningChain();
        if (!chain.isEmpty()) {
            System.out.println("\n--- 推理质量分析 ---");
            
            double totalConfidence = 0;
            int highConfidenceSteps = 0;
            long totalReasoningTime = 0;
            
            for (V3ReasoningStep step : chain) {
                totalConfidence += step.getConfidence();
                totalReasoningTime += step.getExecutionTimeMs();
                
                if (step.getConfidence() > 0.8) {
                    highConfidenceSteps++;
                }
            }
            
            double avgConfidence = totalConfidence / chain.size();
            double highConfidenceRatio = (double) highConfidenceSteps / chain.size() * 100;
            
            System.out.println("推理步骤总数: " + chain.size());
            System.out.println("平均置信度: " + String.format("%.3f", avgConfidence));
            System.out.println("高置信度步骤比例: " + String.format("%.1f", highConfidenceRatio) + "%");
            System.out.println("总推理时间: " + totalReasoningTime + "ms");
            
            // 推理质量评估
            if (avgConfidence > 0.85 && highConfidenceRatio > 70) {
                System.out.println("✓ 推理质量评估: 优秀");
            } else if (avgConfidence > 0.7 && highConfidenceRatio > 50) {
                System.out.println("✓ 推理质量评估: 良好");
            } else {
                System.out.println("! 推理质量评估: 一般，建议进一步验证");
            }
        }
        
        System.out.println("\n✓ 推理链展示完成");
    }
    
    /**
     * 编码文本为数字ID（简化实现）
     */
    private static NdArray encodeText(String text) {
        // 简化的文本编码：将字符转换为ASCII值
        char[] chars = text.toCharArray();
        int maxLen = Math.min(chars.length, 20); // 限制最大长度
        
        double[] encoded = new double[maxLen];
        for (int i = 0; i < maxLen; i++) {
            // 将字符ASCII值映射到词汇表范围
            encoded[i] = Math.min(chars[i] % model.getVocabSize(), model.getVocabSize() - 1);
        }
        
        return NdArray.of(encoded).reshape(Shape.of(1, maxLen));
    }
    
    /**
     * 创建随机输入
     */
    private static NdArray createRandomInput(int batchSize, int seqLen) {
        double[] data = new double[batchSize * seqLen];
        
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.floor(Math.random() * model.getVocabSize());
        }
        
        return NdArray.of(data).reshape(Shape.of(batchSize, seqLen));
    }
    
    /**
     * 清理资源
     */
    private static void cleanup() {
        if (scanner != null) {
            scanner.close();
        }
        
        if (model != null) {
            System.out.println("\n最终模型统计:");
            System.out.println("总处理Token数: " + model.getTotalTokensProcessed());
            System.out.println("模型重置次数: " + "多次"); // 简化统计
        }
        
        System.out.println("\n演示结束，资源已清理。");
    }
}