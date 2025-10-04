package io.leavesfly.tinyai.agent.research;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * DeepResearch Agent 演示类
 * 展示深度研究智能体的各种功能和使用方法
 * 
 * @author 山泽
 */
public class ResearchDemo {
    
    /**
     * 主方法 - 运行所有演示
     */
    public static void main(String[] args) {
        System.out.println("🚀 DeepResearch Agent 演示开始");
        System.out.println("=" + "=".repeat(50));
        
        try {
            // 基础功能演示
            demoBasicResearch();
            
            System.out.println();
            
            // 多查询演示
            demoMultipleQueries();
            
            System.out.println();
            
            // 探索性研究演示
            demoExploratoryResearch();
            
            System.out.println();
            
            // 协作式研究演示
            demoCollaborativeResearch();
            
            System.out.println();
            
            // 性能报告演示
            demoPerformanceReport();
            
            System.out.println("\n🎊 所有演示完成！");
            
        } catch (Exception e) {
            System.err.println("❌ 演示过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 基础研究功能演示
     */
    public static void demoBasicResearch() {
        System.out.println("🧪 演示基础研究功能");
        
        // 创建研究Agent
        DeepResearchAgent agent = new DeepResearchAgent("演示研究助手", "人工智能");
        
        // 添加一些基础知识
        agent.addDomainKnowledge("人工智能是模拟人类智能的技术", "人工智能", "concept");
        agent.addDomainKnowledge("机器学习是AI的一个分支", "人工智能", "concept");
        agent.addDomainKnowledge("深度学习使用神经网络", "人工智能", "concept");
        agent.addDomainKnowledge("自然语言处理让机器理解人类语言", "人工智能", "concept");
        
        System.out.println("\n✅ Agent 创建成功");
        System.out.println("✅ 知识库初始化完成");
        
        // 执行基础研究
        System.out.println("\n🔍 执行研究测试...");
        Map<String, Object> result = agent.research(
            "什么是深度学习？",
            3,  // complexity
            3,  // depth_required
            2   // urgency
        );
        
        // 显示结果
        displayResearchResult(result);
    }
    
    /**
     * 多查询演示
     */
    public static void demoMultipleQueries() {
        System.out.println("🔬 演示多个研究查询");
        
        DeepResearchAgent agent = new DeepResearchAgent("多查询测试助手", "技术");
        
        // 添加知识
        agent.addDomainKnowledge("人工智能包括机器学习、深度学习等技术", "人工智能");
        agent.addDomainKnowledge("区块链是分布式账本技术", "区块链");
        agent.addDomainKnowledge("量子计算利用量子力学原理", "量子技术");
        agent.addDomainKnowledge("云计算提供按需的计算资源", "云计算");
        
        String[] queries = {
            "人工智能的发展历程",
            "区块链的应用场景",
            "量子计算的优势"
        };
        
        for (int i = 0; i < queries.length; i++) {
            System.out.println("\n📋 查询 " + (i + 1) + ": " + queries[i]);
            
            Map<String, Object> result = agent.research(
                queries[i],
                2,  // complexity
                2,  // depth_required
                3   // urgency
            );
            
            System.out.printf("  ✅ 完成，置信度: %.2f, 质量评分: %.3f\n", 
                result.get("totalConfidence"), result.get("qualityScore"));
        }
        
        // 显示统计
        Map<String, Object> performance = agent.getPerformanceReport();
        System.out.println("\n📈 多查询测试统计:");
        System.out.println("  📊 总查询数: " + performance.get("researchHistoryCount"));
        System.out.printf("  ⭐ 平均置信度: %.3f\n", 
            ((Map<?, ?>) performance.get("performanceMetrics")).get("avgConfidence"));
        System.out.println("  📚 知识节点数: " + performance.get("totalKnowledgeNodes"));
    }
    
    /**
     * 探索性研究演示
     */
    public static void demoExploratoryResearch() {
        System.out.println("🔎 演示探索性研究");
        
        DeepResearchAgent agent = new DeepResearchAgent("探索研究助手", "科技");
        
        // 添加相关知识
        agent.addDomainKnowledge("量子计算是基于量子力学的计算模式", "量子技术");
        agent.addDomainKnowledge("量子纠缠是量子计算的基础现象", "量子技术");
        agent.addDomainKnowledge("量子算法可以解决某些经典计算难题", "量子技术");
        
        System.out.println("\n🔍 开始探索性研究...");
        Map<String, Object> explorationResult = agent.exploreResearchTopic("量子计算", 3);
        
        System.out.println("\n📊 探索结果统计:");
        System.out.println("  🎯 主题: " + explorationResult.get("topic"));
        System.out.println("  📝 探索问题数: " + explorationResult.get("totalQuestionsExplored"));
        
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> results = (List<Map<String, Object>>) explorationResult.get("explorationResults");
        for (int i = 0; i < results.size(); i++) {
            Map<String, Object> item = results.get(i);
            System.out.println("  " + (i + 1) + ". " + item.get("question"));
        }
    }
    
    /**
     * 协作式研究演示
     */
    public static void demoCollaborativeResearch() {
        System.out.println("🤝 演示协作式研究");
        
        DeepResearchAgent agent = new DeepResearchAgent("协作研究助手", "综合");
        
        // 添加跨领域知识
        agent.addDomainKnowledge("人工智能在伦理方面存在争议", "AI伦理");
        agent.addDomainKnowledge("技术发展需要考虑社会影响", "技术社会学");
        agent.addDomainKnowledge("法律法规需要跟上技术发展", "科技法律");
        agent.addDomainKnowledge("哲学思考有助于理解技术本质", "技术哲学");
        
        List<String> perspectives = Arrays.asList("技术", "法律", "社会", "哲学");
        
        System.out.println("\n🤔 开始协作式研究...");
        Map<String, Object> collaborationResult = agent.collaborativeResearch(
            "人工智能的伦理问题", perspectives);
        
        System.out.println("\n📊 协作研究统计:");
        System.out.println("  🎯 主查询: " + collaborationResult.get("mainQuery"));
        System.out.println("  👁️ 研究视角: " + collaborationResult.get("perspectives"));
        
        @SuppressWarnings("unchecked")
        Map<String, Object> synthesis = (Map<String, Object>) collaborationResult.get("synthesis");
        System.out.println("\n🧠 综合分析结论:");
        System.out.println("  " + synthesis.get("synthesisConclusion"));
        System.out.printf("  📊 综合置信度: %.2f\n", synthesis.get("synthesisConfidence"));
    }
    
    /**
     * 性能报告演示
     */
    public static void demoPerformanceReport() {
        System.out.println("📊 演示性能报告");
        
        DeepResearchAgent agent = new DeepResearchAgent("性能测试助手", "测试");
        
        // 执行几个研究来生成性能数据
        agent.research("测试问题1", 2, 2, 2);
        agent.research("测试问题2", 3, 3, 1);
        agent.research("测试问题3", 1, 1, 4);
        
        Map<String, Object> performance = agent.getPerformanceReport();
        
        System.out.println("\n📈 Agent 性能报告:");
        System.out.println("  🤖 Agent名称: " + performance.get("agentName"));
        System.out.println("  🏷️ 主要领域: " + performance.get("primaryDomain"));
        System.out.println("  📊 研究次数: " + performance.get("researchHistoryCount"));
        System.out.println("  📚 知识节点数: " + performance.get("totalKnowledgeNodes"));
        System.out.println("  🗂️ 知识领域: " + performance.get("knowledgeDomains"));
        
        @SuppressWarnings("unchecked")
        Map<String, Object> metrics = (Map<String, Object>) performance.get("performanceMetrics");
        System.out.printf("  ⭐ 平均置信度: %.3f\n", metrics.get("avgConfidence"));
        System.out.println("  📈 推理模式使用: " + metrics.get("reasoningModeUsage"));
        
        // 显示知识图谱概览
        Map<String, Object> knowledgeOverview = agent.getKnowledgeOverview(null);
        if (!knowledgeOverview.isEmpty()) {
            System.out.println("\n🗺️ 知识图谱概览:");
            System.out.println("  📊 总节点数: " + knowledgeOverview.get("totalNodes"));
            System.out.println("  🏷️ 节点类型: " + knowledgeOverview.get("nodeTypes"));
            System.out.printf("  ⭐ 平均置信度: %.3f\n", knowledgeOverview.get("averageConfidence"));
        }
    }
    
    /**
     * 显示研究结果
     */
    private static void displayResearchResult(Map<String, Object> result) {
        System.out.println("\n🎯 研究结果:");
        System.out.printf("  ✅ 置信度: %.2f\n", result.get("totalConfidence"));
        System.out.println("  📋 研究步骤数: " + result.get("researchSteps"));
        System.out.println("  🔧 使用工具数: " + result.get("toolsUsed"));
        
        @SuppressWarnings("unchecked")
        List<String> insights = (List<String>) result.get("keyInsights");
        System.out.println("  💡 关键洞察数: " + insights.size());
        System.out.printf("  🏆 质量评分: %.3f\n", result.get("qualityScore"));
        System.out.println("  🧠 推理模式: " + result.get("reasoningMode"));
        
        System.out.println("\n📖 最终答案:");
        System.out.println("  " + result.get("finalAnswer"));
        
        if (!insights.isEmpty()) {
            System.out.println("\n💡 关键洞察:");
            for (int i = 0; i < insights.size(); i++) {
                System.out.println("  " + (i + 1) + ". " + insights.get(i));
            }
        }
    }
    
    /**
     * 简单的基础演示方法
     */
    public static void simpleDemo() {
        System.out.println("🔬 简单演示");
        
        DeepResearchAgent agent = new DeepResearchAgent();
        agent.addDomainKnowledge("Java是一种面向对象的编程语言", "编程");
        
        Map<String, Object> result = agent.research("Java的特点是什么？");
        System.out.println("研究结果: " + result.get("finalAnswer"));
        System.out.printf("置信度: %.2f\n", result.get("totalConfidence"));
    }
}
