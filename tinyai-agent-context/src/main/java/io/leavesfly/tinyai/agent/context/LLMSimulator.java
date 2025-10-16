package io.leavesfly.tinyai.agent.context;

import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * LLM模拟器
 * 模拟真实的LLM API调用，为不同类型的Agent生成合适的回复
 * 
 * @author 山泽
 */
public class LLMSimulator {
    
    private final String modelName;
    private final double temperature;
    private final int maxTokens;
    private final Random random;
    
    // 不同Agent类型的回复模板
    private final Map<String, List<String>> responseTemplates;
    
    public LLMSimulator() {
        this("gpt-3.5-turbo", 0.7, 2048);
    }
    
    public LLMSimulator(String modelName, double temperature, int maxTokens) {
        this.modelName = modelName;
        this.temperature = temperature;
        this.maxTokens = maxTokens;
        this.random = new Random();
        this.responseTemplates = initializeResponseTemplates();
    }
    
    /**
     * 异步聊天完成
     */
    public CompletableFuture<String> chatCompletionAsync(List<Map<String, String>> messages, String agentType) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 模拟API调用延迟
                Thread.sleep(500 + messages.size() * 100);
                
                // 获取最后一条用户消息
                String lastMessage = "";
                if (!messages.isEmpty()) {
                    lastMessage = messages.get(messages.size() - 1).getOrDefault("content", "");
                }
                
                // 根据Agent类型选择回复模板
                List<String> templates = responseTemplates.getOrDefault(agentType, 
                    Arrays.asList("我理解了{}，将会{}。"));
                String template = templates.get(random.nextInt(templates.size()));
                
                // 简单的内容提取和替换
                List<String> keywords = extractKeywords(lastMessage);
                return fillTemplate(template, keywords);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return "抱歉，我在处理您的请求时遇到了问题。";
            }
        });
    }
    
    /**
     * 同步聊天完成
     */
    public String chatCompletion(List<Map<String, String>> messages, String agentType) {
        try {
            return chatCompletionAsync(messages, agentType).get();
        } catch (Exception e) {
            return "抱歉，我遇到了一些技术问题：" + e.getMessage();
        }
    }
    
    /**
     * 提取关键词
     */
    private List<String> extractKeywords(String text) {
        if (text.contains("分析")) {
            return Arrays.asList("市场数据", "上升趋势", "用户体验");
        } else if (text.contains("研究")) {
            return Arrays.asList("人工智能", "显著效果", "深度学习");
        } else if (text.contains("任务") || text.contains("分配")) {
            return Arrays.asList("Alice", "文档编写", "2小时");
        } else if (text.contains("执行")) {
            return Arrays.asList("数据处理", "85%", "30分钟");
        } else if (text.contains("评估") || text.contains("评审")) {
            return Arrays.asList("产品质量", "优秀", "用户界面");
        } else if (text.contains("协调")) {
            return Arrays.asList("团队成员", "高效完成", "明天下午");
        } else {
            return Arrays.asList("项目内容", "进展顺利", "继续推进");
        }
    }
    
    /**
     * 填充模板
     */
    private String fillTemplate(String template, List<String> keywords) {
        try {
            // 统计模板中的占位符数量
            int placeholderCount = 0;
            String temp = template;
            while (temp.contains("{}")) {
                placeholderCount++;
                temp = temp.replaceFirst("\\{\\}", "PLACEHOLDER");
            }
            
            // 确保有足够的关键词
            while (keywords.size() < placeholderCount) {
                keywords.add("相关内容");
            }
            
            // 替换占位符
            String result = template;
            for (int i = 0; i < placeholderCount && i < keywords.size(); i++) {
                result = result.replaceFirst("\\{\\}", keywords.get(i));
            }
            
            return result;
        } catch (Exception e) {
            return template.replace("{}", "相关内容");
        }
    }
    
    /**
     * 初始化回复模板
     */
    private Map<String, List<String>> initializeResponseTemplates() {
        Map<String, List<String>> templates = new HashMap<>();
        
        // 分析师模板
        templates.put("analyst", Arrays.asList(
            "根据数据分析，我发现{}的关键指标显示{}趋势。建议重点关注{}方面的改进。",
            "从分析结果来看，{}表现出{}的特征。我建议采取{}策略来优化。",
            "数据显示{}，这表明{}。为了改善情况，建议实施{}措施。",
            "通过深入分析{}，可以看出{}的明显特点。推荐在{}领域加强投入。"
        ));
        
        // 研究员模板
        templates.put("researcher", Arrays.asList(
            "通过深入研究，我发现{}领域存在{}的现象。相关文献表明{}。",
            "我的研究表明{}具有{}的特性。基于现有研究，我认为{}。",
            "研究发现{}与{}之间存在关联。建议进一步探索{}方向。",
            "根据最新的研究成果，{}显示了{}的潜力。值得在{}方面深入探讨。"
        ));
        
        // 协调员模板
        templates.put("coordinator", Arrays.asList(
            "根据团队情况，我建议{}负责{}任务。预计完成时间为{}。",
            "为了提高效率，我重新分配任务：{}。请各位按照新的安排执行。",
            "项目进度更新：{}已完成，{}正在进行中，{}需要加快速度。",
            "经过协调，{}将专注于{}工作。整体计划预计在{}完成。"
        ));
        
        // 执行员模板
        templates.put("executor", Arrays.asList(
            "我已经完成了{}任务，结果是{}。下一步建议执行{}。",
            "正在执行{}操作，当前进度{}。预计还需要{}时间完成。",
            "任务执行遇到{}问题，已采取{}措施，现在状态是{}。",
            "{}任务已启动，目前完成度达到{}。计划{}内交付结果。"
        ));
        
        // 评审员模板
        templates.put("critic", Arrays.asList(
            "从质量角度看，{}存在{}问题。建议在{}方面进行改进。",
            "评估结果表明{}达到了{}标准，但在{}方面仍有提升空间。",
            "这个方案的优点是{}，但需要注意{}风险。建议调整{}。",
            "经过仔细评审，{}整体表现{}，不过{}部分需要优化。"
        ));
        
        // 通用模板
        templates.put("general", Arrays.asList(
            "我理解了{}的需求，将会{}来满足期望。",
            "关于{}，我认为{}是最佳方案，因为{}。",
            "针对{}的情况，建议{}，这样可以确保{}。",
            "基于当前的{}状况，我会{}，预期能够{}。"
        ));
        
        return templates;
    }
    
    /**
     * 生成系统提示
     */
    public String generateSystemPrompt(String agentType, String agentName, String role) {
        Map<String, String> systemPrompts = new HashMap<>();
        
        systemPrompts.put("analyst", 
            String.format("你是%s，一位专业的%s。你擅长数据分析和趋势预测，会仔细分析数据，发现关键模式，并提供有价值的洞察和建议。", agentName, role));
        
        systemPrompts.put("researcher", 
            String.format("你是%s，一位严谨的%s。你擅长文献调研、实验设计和理论分析，会深入研究问题，寻找科学依据，提供基于证据的结论。", agentName, role));
        
        systemPrompts.put("coordinator", 
            String.format("你是%s，一位高效的%s。你负责任务分配、进度跟踪和团队协调，会根据团队成员的能力合理分配任务，确保项目顺利进行。", agentName, role));
        
        systemPrompts.put("executor", 
            String.format("你是%s，一位专业的%s。你专注于完成分配的具体任务，会认真执行每个步骤，使用合适的工具，并及时报告进度。", agentName, role));
        
        systemPrompts.put("critic", 
            String.format("你是%s，一位严格的%s。你负责评估工作质量和提供改进建议，会从多个角度审查结果，指出问题并提供建设性的改进方案。", agentName, role));
        
        return systemPrompts.getOrDefault(agentType, 
            String.format("你是%s，一位专业的%s。请根据你的专业知识协助用户解决问题。", agentName, role));
    }
    
    // Getter 方法
    public String getModelName() {
        return modelName;
    }
    
    public double getTemperature() {
        return temperature;
    }
    
    public int getMaxTokens() {
        return maxTokens;
    }
}