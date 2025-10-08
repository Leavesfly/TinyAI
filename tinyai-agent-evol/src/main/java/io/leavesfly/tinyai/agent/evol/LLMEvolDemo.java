package io.leavesfly.tinyai.agent.evol;

import java.util.*;

/**
 * LLM增强的自进化智能体演示程序
 * 展示基于大语言模型的智能自进化、深度反思和知识推理能力
 * 
 * @author 山泽
 */
public class LLMEvolDemo {
    
    public static void main(String[] args) {
        if (args.length > 0 && "quickDemo".equals(args[0])) {
            runQuickDemo();
        } else {
            runCompleteDemo();
        }
    }
    
    /**
     * 完整演示程序
     */
    public static void runCompleteDemo() {
        System.out.println("=== LLM增强的自进化智能体完整演示 ===\\n");
        
        // 创建LLM增强的智能体
        LLMSelfEvolvingAgent llmAgent = new LLMSelfEvolvingAgent("LLM智能学习助手", true);
        
        // 设置LLM参数
        llmAgent.setLlmConfidenceThreshold(0.7);
        llmAgent.setEnableAsyncLLM(true);
        
        System.out.println("🤖 创建LLM增强智能体: " + llmAgent.getName());
        System.out.println("🧠 LLM功能状态: " + (llmAgent.isLLMEnabled() ? "启用" : "禁用"));
        System.out.println("⚙️  LLM置信度阈值: " + llmAgent.getLlmConfidenceThreshold());
        System.out.println();
        
        // 执行多样化任务演示
        demonstrateLLMEnhancedTasks(llmAgent);
        
        // 展示LLM增强的反思能力
        demonstrateLLMReflection(llmAgent);
        
        // 展示LLM增强的知识图谱
        demonstrateLLMKnowledgeGraph(llmAgent);
        
        // 展示自进化过程
        demonstrateLLMEvolution(llmAgent);
        
        // 展示性能对比
        demonstratePerformanceComparison();
        
        System.out.println("\\n=== LLM增强智能体演示完成 ===");
    }
    
    /**
     * 快速演示程序
     */
    public static void runQuickDemo() {
        System.out.println("=== LLM增强智能体快速演示 ===\\n");
        
        LLMSelfEvolvingAgent agent = new LLMSelfEvolvingAgent("快速演示智能体");
        
        // 执行几个核心任务
        String[] quickTasks = {
            "分析机器学习趋势",
            "设计创新解决方案", 
            "评估技术风险"
        };
        
        for (String task : quickTasks) {
            System.out.println("📋 执行任务: " + task);
            
            Map<String, Object> context = new HashMap<>();
            context.put("complexity", "medium");
            context.put("domain", "technology");
            
            LLMSelfEvolvingAgent.EnhancedTaskResult result = agent.processTaskWithLLM(task, context);
            
            System.out.println("  ✅ 执行动作: " + result.getAction());
            System.out.println("  📊 执行结果: " + (result.isSuccess() ? "成功" : "失败"));
            System.out.println("  🎯 奖励值: " + String.format("%.2f", result.getReward()));
            System.out.println("  🧠 LLM洞察: " + truncateString(result.getLlmReflection(), 100));
            System.out.println();
        }
        
        // 显示学习成果
        Map<String, Object> performance = agent.getEnhancedPerformanceSummary();
        System.out.println("📈 学习成果:");
        System.out.println("  总任务数: " + performance.get("total_tasks"));
        System.out.println("  成功率: " + String.format("%.1f%%", 
                         ((Number) performance.get("current_success_rate")).doubleValue() * 100));
        System.out.println("  LLM辅助率: " + String.format("%.1f%%", 
                         ((Number) performance.get("llm_assistance_rate")).doubleValue() * 100));
        
        System.out.println("\\n=== 快速演示完成 ===");
    }
    
    /**
     * 演示LLM增强的任务处理
     */
    private static void demonstrateLLMEnhancedTasks(LLMSelfEvolvingAgent agent) {
        System.out.println("🎯 === LLM增强任务处理演示 ===\\n");
        
        // 定义多样化的任务
        List<TaskDemo> tasks = Arrays.asList(
            new TaskDemo("深度学习模型优化", Map.of(
                "difficulty", "hard",
                "domain", "AI",
                "urgency", "high"
            )),
            new TaskDemo("用户体验设计分析", Map.of(
                "difficulty", "medium", 
                "domain", "UX",
                "creativity_required", true
            )),
            new TaskDemo("市场趋势预测研究", Map.of(
                "difficulty", "medium",
                "domain", "business",
                "data_driven", true
            )),
            new TaskDemo("代码架构重构建议", Map.of(
                "difficulty", "hard",
                "domain", "engineering", 
                "technical_depth", "high"
            ))
        );
        
        for (int i = 0; i < tasks.size(); i++) {
            TaskDemo task = tasks.get(i);
            System.out.println(String.format("--- 任务 %d: %s ---", i + 1, task.name));
            
            LLMSelfEvolvingAgent.EnhancedTaskResult result = agent.processTaskWithLLM(task.name, task.context);
            
            System.out.println("🎯 选择的行动: " + result.getAction());
            System.out.println("📊 执行结果: " + (result.isSuccess() ? "✅ 成功" : "❌ 失败"));
            System.out.println("🎖️  奖励值: " + String.format("%.2f", result.getReward()));
            System.out.println("🧠 基础反思: " + truncateString(result.getLearningInsights(), 120));
            System.out.println("🚀 LLM深度洞察: " + truncateString(result.getLlmReflection(), 150));
            
            if (!result.getImprovementAdvice().isEmpty()) {
                System.out.println("💡 改进建议: " + truncateString(result.getImprovementAdvice(), 100));
            }
            
            if (!result.getDiscoveredPatterns().isEmpty()) {
                System.out.println("🔍 发现的模式: " + String.join(", ", result.getDiscoveredPatterns()));
            }
            
            System.out.println();
        }
    }
    
    /**
     * 演示LLM增强的反思能力
     */
    private static void demonstrateLLMReflection(LLMSelfEvolvingAgent agent) {
        System.out.println("🧠 === LLM增强反思能力演示 ===\\n");
        
        // 获取LLM反思模块
        ReflectionModule reflectionModule = agent.getReflectionModule();
        
        if (reflectionModule instanceof LLMReflectionModule) {
            LLMReflectionModule llmReflectionModule = (LLMReflectionModule) reflectionModule;
            
            // 演示智能洞察提取
            System.out.println("🔮 智能洞察提取演示:");
            List<Experience> experiences = agent.getExperiences();
            if (!experiences.isEmpty()) {
                List<String> insights = llmReflectionModule.extractIntelligentInsights(experiences, "学习效率");
                insights.forEach(insight -> System.out.println("  💡 " + insight));
            } else {
                System.out.println("  📝 暂无足够经验数据进行洞察分析");
            }
            System.out.println();
            
            // 演示元学习分析
            System.out.println("🎓 元学习分析演示:");
            if (!experiences.isEmpty()) {
                String metaAnalysis = llmReflectionModule.generateMetaLearningAnalysis(experiences);
                System.out.println("  🧭 元学习洞察: " + truncateString(metaAnalysis, 200));
            } else {
                System.out.println("  📝 暂无足够经验数据进行元学习分析");
            }
            System.out.println();
            
            // 演示LLM模式识别
            System.out.println("🔍 LLM模式识别演示:");
            if (experiences.size() >= 3) {
                List<LLMReflectionModule.LLMPattern> llmPatterns = llmReflectionModule.identifyLLMPatterns(experiences);
                for (LLMReflectionModule.LLMPattern pattern : llmPatterns.stream().limit(3).collect(java.util.stream.Collectors.toList())) {
                    System.out.println(String.format("  📈 模式: %s (强度: %.2f)", 
                                     pattern.getDescription(), pattern.getStrength()));
                    System.out.println("    🤖 LLM解释: " + truncateString(pattern.getLlmInterpretation(), 100));
                    System.out.println("    🔮 预测洞察: " + truncateString(pattern.getPredictiveInsight(), 100));
                }
            } else {
                System.out.println("  📝 经验数据不足，无法进行模式识别");
            }
            System.out.println();
            
            // 展示LLM反思统计
            Map<String, Object> llmStats = llmReflectionModule.getLLMStatistics();
            System.out.println("📊 LLM反思模块统计:");
            llmStats.forEach((key, value) -> 
                System.out.println(String.format("  %s: %s", key, value))
            );
        }
        
        System.out.println();
    }
    
    /**
     * 演示LLM增强的知识图谱
     */
    private static void demonstrateLLMKnowledgeGraph(LLMSelfEvolvingAgent agent) {
        System.out.println("🕸️  === LLM增强知识图谱演示 ===\\n");
        
        KnowledgeGraph kg = agent.getKnowledgeGraph();
        
        if (kg instanceof LLMKnowledgeGraph) {
            LLMKnowledgeGraph llmKG = (LLMKnowledgeGraph) kg;
            
            // 添加示例概念用于演示
            Map<String, Object> aiProperties = Map.of("type", "technology", "complexity", "high");
            llmKG.addConcept("人工智能", aiProperties);
            
            Map<String, Object> mlProperties = Map.of("type", "technique", "parent", "人工智能");
            llmKG.addConcept("机器学习", mlProperties);
            
            Map<String, Object> dlProperties = Map.of("type", "technique", "parent", "机器学习");
            llmKG.addConcept("深度学习", dlProperties);
            
            // 添加关系
            llmKG.addRelation("人工智能", "机器学习", "包含", 0.9);
            llmKG.addRelation("机器学习", "深度学习", "包含", 0.8);
            llmKG.addRelation("深度学习", "神经网络", "基于", 0.9);
            
            // 演示智能推理
            System.out.println("🧠 智能推理演示:");
            String reasoning = llmKG.performIntelligentReasoning(
                "人工智能与深度学习的关系", "hierarchical_analysis");
            System.out.println("  🔍 推理结果: " + truncateString(reasoning, 150));
            System.out.println();
            
            // 演示概念关系推荐
            System.out.println("💡 概念关系推荐演示:");
            List<String> recommendations = llmKG.recommendConceptRelations("机器学习");
            if (!recommendations.isEmpty()) {
                System.out.println("  🎯 为'机器学习'推荐的相关概念:");
                recommendations.forEach(rec -> System.out.println("    • " + rec));
            } else {
                System.out.println("  📝 暂无推荐结果");
            }
            System.out.println();
            
            // 演示知识图谱质量评估
            System.out.println("📊 知识图谱质量评估:");
            Map<String, Object> quality = llmKG.assessKnowledgeGraphQuality();
            quality.forEach((metric, value) -> {
                if (value instanceof Double) {
                    System.out.println(String.format("  %s: %.2f", metric, (Double) value));
                } else {
                    System.out.println(String.format("  %s: %s", metric, value));
                }
            });
            System.out.println();
            
            // 展示LLM概念信息
            System.out.println("🏷️  LLM概念增强信息:");
            LLMKnowledgeGraph.LLMConceptInfo aiInfo = llmKG.getLLMConceptInfo("人工智能");
            if (aiInfo != null) {
                System.out.println("  📝 概念: " + aiInfo.getConceptName());
                System.out.println("  📖 LLM描述: " + truncateString(aiInfo.getLlmDescription(), 100));
                System.out.println("  🏷️  类别: " + aiInfo.getLlmCategory());
                System.out.println("  ⭐ 重要度: " + String.format("%.2f", aiInfo.getConceptImportance()));
                System.out.println("  🏷️  标签: " + String.join(", ", aiInfo.getLlmTags()));
            }
        }
        
        System.out.println();
    }
    
    /**
     * 演示LLM增强的自进化过程
     */
    private static void demonstrateLLMEvolution(LLMSelfEvolvingAgent agent) {
        System.out.println("🌱 === LLM增强自进化演示 ===\\n");
        
        // 获取进化前的状态
        Map<String, Object> beforeEvolution = agent.getEnhancedPerformanceSummary();
        System.out.println("📊 进化前状态:");
        System.out.println("  策略数量: " + beforeEvolution.get("strategies_count"));
        System.out.println("  知识概念: " + beforeEvolution.get("knowledge_concepts"));
        System.out.println("  LLM辅助决策: " + beforeEvolution.get("llm_assisted_decisions"));
        System.out.println();
        
        // 执行LLM增强的自进化
        System.out.println("🚀 开始LLM增强自进化过程...");
        agent.selfEvolveWithLLM();
        System.out.println("✅ LLM增强自进化完成");
        System.out.println();
        
        // 获取进化后的状态
        Map<String, Object> afterEvolution = agent.getEnhancedPerformanceSummary();
        System.out.println("📊 进化后状态:");
        System.out.println("  策略数量: " + afterEvolution.get("strategies_count"));
        System.out.println("  知识概念: " + afterEvolution.get("knowledge_concepts"));
        System.out.println("  LLM辅助决策: " + afterEvolution.get("llm_assisted_decisions"));
        System.out.println();
        
        // 分析进化效果
        analyzeEvolutionEffect(beforeEvolution, afterEvolution);
        
        // 展示学习到的策略
        System.out.println("🎯 学习到的策略:");
        Map<String, Strategy> strategies = agent.getStrategies();
        strategies.entrySet().stream()
            .limit(5)
            .forEach(entry -> {
                Strategy strategy = entry.getValue();
                System.out.println(String.format("  📋 %s: 成功率 %.1f%%, 使用 %d 次", 
                                 entry.getKey(), 
                                 strategy.getSuccessRate() * 100, 
                                 strategy.getUsageCount()));
            });
        
        System.out.println();
    }
    
    /**
     * 演示性能对比
     */
    private static void demonstratePerformanceComparison() {
        System.out.println("⚡ === LLM增强 vs 传统智能体对比 ===\\n");
        
        // 创建传统智能体
        SelfEvolvingAgent traditionalAgent = new SelfEvolvingAgent("传统智能体");
        
        // 创建LLM增强智能体
        LLMSelfEvolvingAgent llmAgent = new LLMSelfEvolvingAgent("LLM增强智能体");
        
        // 定义对比测试任务
        List<String> testTasks = Arrays.asList(
            "复杂问题分析",
            "创新方案设计", 
            "风险评估预测",
            "策略优化建议"
        );
        
        Map<String, Object> testContext = Map.of(
            "difficulty", "high",
            "creativity_required", true,
            "analysis_depth", "deep"
        );
        
        System.out.println("🧪 执行对比测试...");
        System.out.println();
        
        // 执行对比测试
        for (String task : testTasks) {
            System.out.println("📋 任务: " + task);
            
            // 传统智能体处理
            SelfEvolvingAgent.TaskResult traditionalResult = traditionalAgent.processTask(task, new HashMap<>(testContext));
            System.out.println("  🔹 传统智能体:");
            System.out.println("    行动: " + traditionalResult.getAction());
            System.out.println("    成功: " + (traditionalResult.isSuccess() ? "✅" : "❌"));
            System.out.println("    奖励: " + String.format("%.2f", traditionalResult.getReward()));
            
            // LLM增强智能体处理
            LLMSelfEvolvingAgent.EnhancedTaskResult llmResult = llmAgent.processTaskWithLLM(task, new HashMap<>(testContext));
            System.out.println("  🔸 LLM增强智能体:");
            System.out.println("    行动: " + llmResult.getAction());
            System.out.println("    成功: " + (llmResult.isSuccess() ? "✅" : "❌"));
            System.out.println("    奖励: " + String.format("%.2f", llmResult.getReward()));
            System.out.println("    LLM洞察: " + truncateString(llmResult.getLlmReflection(), 80));
            
            System.out.println();
        }
        
        // 性能对比总结
        System.out.println("📊 性能对比总结:");
        
        Map<String, Object> traditionalPerf = traditionalAgent.getPerformanceSummary();
        Map<String, Object> llmPerf = llmAgent.getEnhancedPerformanceSummary();
        
        System.out.println("传统智能体:");
        System.out.println("  成功率: " + String.format("%.1f%%", 
                         ((Number) traditionalPerf.get("current_success_rate")).doubleValue() * 100));
        System.out.println("  策略数: " + traditionalPerf.get("strategies_count"));
        
        System.out.println("LLM增强智能体:");
        System.out.println("  成功率: " + String.format("%.1f%%", 
                         ((Number) llmPerf.get("current_success_rate")).doubleValue() * 100));
        System.out.println("  策略数: " + llmPerf.get("strategies_count"));
        System.out.println("  LLM辅助率: " + String.format("%.1f%%", 
                         ((Number) llmPerf.get("llm_assistance_rate")).doubleValue() * 100));
        
        System.out.println();
    }
    
    /**
     * 分析进化效果
     */
    private static void analyzeEvolutionEffect(Map<String, Object> before, Map<String, Object> after) {
        System.out.println("📈 进化效果分析:");
        
        // 策略数量变化
        int strategiesBefore = ((Number) before.get("strategies_count")).intValue();
        int strategiesAfter = ((Number) after.get("strategies_count")).intValue();
        int strategiesChange = strategiesAfter - strategiesBefore;
        
        System.out.println(String.format("  策略变化: %+d (从 %d 到 %d)", 
                         strategiesChange, strategiesBefore, strategiesAfter));
        
        // 知识概念变化
        int conceptsBefore = ((Number) before.get("knowledge_concepts")).intValue();
        int conceptsAfter = ((Number) after.get("knowledge_concepts")).intValue();
        int conceptsChange = conceptsAfter - conceptsBefore;
        
        System.out.println(String.format("  知识变化: %+d (从 %d 到 %d)", 
                         conceptsChange, conceptsBefore, conceptsAfter));
        
        // LLM辅助变化
        int llmBefore = ((Number) before.get("llm_assisted_decisions")).intValue();
        int llmAfter = ((Number) after.get("llm_assisted_decisions")).intValue();
        int llmChange = llmAfter - llmBefore;
        
        System.out.println(String.format("  LLM辅助增长: %+d (从 %d 到 %d)", 
                         llmChange, llmBefore, llmAfter));
        
        // 整体评估
        if (strategiesChange > 0 || conceptsChange > 0) {
            System.out.println("  🎉 进化效果: 显著提升！");
        } else if (strategiesChange == 0 && conceptsChange == 0) {
            System.out.println("  😐 进化效果: 保持稳定");
        } else {
            System.out.println("  🔄 进化效果: 优化整理");
        }
        
        System.out.println();
    }
    
    /**
     * 工具方法：截断字符串
     */
    private static String truncateString(String str, int maxLength) {
        if (str == null || str.length() <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength) + "...";
    }
    
    /**
     * 任务演示数据类
     */
    private static class TaskDemo {
        final String name;
        final Map<String, Object> context;
        
        TaskDemo(String name, Map<String, Object> context) {
            this.name = name;
            this.context = context;
        }
    }
}