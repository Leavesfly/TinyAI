package io.leavesfly.tinyai.agent.multi;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * 评审员Agent
 * 专门负责质量评估、代码审查和改进建议
 * 
 * @author 山泽
 */
public class CriticAgent extends BaseAgent {
    
    public CriticAgent(String agentId, MessageBus messageBus, LLMSimulator llm) {
        super(agentId, "评审员-" + agentId.substring(Math.max(0, agentId.length() - 4)), 
              "质量评审员", messageBus, llm);
    }
    
    @Override
    protected void initializeCapabilities() {
        capabilities.addAll(Arrays.asList("质量评估", "代码审查", "改进建议", "标准制定"));
    }
    
    @Override
    protected String getAgentType() {
        return "critic";
    }
    
    @Override
    protected Object performTask(AgentTask task) throws Exception {
        String description = task.getDescription().toLowerCase();
        
        if (description.contains("评审") || description.contains("评估")) {
            return performQualityReview(task);
        } else if (description.contains("代码")) {
            return performCodeReview(task);
        } else if (description.contains("标准")) {
            return developStandards(task);
        } else {
            return performGeneralCritique(task);
        }
    }
    
    /**
     * 执行质量评审
     */
    private Map<String, Object> performQualityReview(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始质量评审: %s", name, task.getTitle()));
        
        // 模拟评审过程
        Thread.sleep(2000);
        
        // 评分系统 (1-10分)
        double overallScore = 7.5 + Math.random() * 2; // 7.5-9.5之间
        
        List<String> strengths = Arrays.asList(
            "整体架构设计合理",
            "核心功能实现完整",
            "文档编写比较规范",
            "错误处理机制较好"
        );
        
        List<String> weaknesses = Arrays.asList(
            "部分细节需要完善",
            "性能优化空间较大",
            "用户体验可以改进",
            "测试覆盖率待提升"
        );
        
        List<String> recommendations = Arrays.asList(
            "增加单元测试覆盖率至80%以上",
            "优化关键算法的时间复杂度",
            "改进用户界面的响应性",
            "完善异常处理和日志记录"
        );
        
        String approvalStatus = overallScore >= 8.0 ? "通过" : 
                               overallScore >= 6.0 ? "有条件通过" : "需要修改";
        
        Map<String, Object> result = new HashMap<>();
        result.put("reviewType", "质量评审");
        result.put("overallScore", Math.round(overallScore * 10) / 10.0);
        result.put("strengths", strengths);
        result.put("weaknesses", weaknesses);
        result.put("recommendations", recommendations);
        result.put("approvalStatus", approvalStatus);
        result.put("reviewDate", System.currentTimeMillis());
        result.put("reviewer", name);
        
        return result;
    }
    
    /**
     * 执行代码审查
     */
    private Map<String, Object> performCodeReview(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始代码审查: %s", name, task.getTitle()));
        
        // 模拟代码审查过程
        Thread.sleep(2500);
        
        Map<String, Object> result = new HashMap<>();
        result.put("reviewType", "代码审查");
        result.put("codeQuality", "良好");
        result.put("codeStyleScore", 8.2);
        result.put("securityScore", 8.5);
        result.put("performanceScore", 7.8);
        
        result.put("findings", Arrays.asList(
            "代码结构清晰，易于理解",
            "变量命名规范，注释充分",
            "存在少量重复代码",
            "部分函数复杂度较高"
        ));
        
        result.put("criticalIssues", Arrays.asList(
            "SQL注入风险点需要修复",
            "密码存储方式需要加强"
        ));
        
        result.put("suggestions", Arrays.asList(
            "提取公共方法减少重复",
            "拆分复杂函数提高可读性",
            "添加输入参数验证",
            "完善单元测试用例"
        ));
        
        result.put("linesReviewed", 1250);
        result.put("filesReviewed", 15);
        
        return result;
    }
    
    /**
     * 制定标准
     */
    private Map<String, Object> developStandards(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始制定标准: %s", name, task.getTitle()));
        
        // 模拟标准制定过程
        Thread.sleep(1800);
        
        Map<String, Object> result = new HashMap<>();
        result.put("reviewType", "标准制定");
        result.put("standardType", "质量标准");
        
        result.put("codingStandards", Arrays.asList(
            "使用统一的代码格式化规则",
            "函数命名采用驼峰式命名",
            "类和接口使用PascalCase命名",
            "常量使用全大写字母命名"
        ));
        
        result.put("qualityStandards", Arrays.asList(
            "单元测试覆盖率不低于80%",
            "函数圈复杂度不超过10",
            "代码重复率不超过5%",
            "文档覆盖率不低于90%"
        ));
        
        result.put("reviewProcess", Arrays.asList(
            "开发者自测",
            "同行代码评审",
            "质量门禁检查",
            "集成测试验证"
        ));
        
        result.put("complianceLevel", "严格执行");
        
        return result;
    }
    
    /**
     * 执行通用评审
     */
    private Map<String, Object> performGeneralCritique(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始通用评审: %s", name, task.getTitle()));
        
        Thread.sleep(1000);
        
        Map<String, Object> result = new HashMap<>();
        result.put("reviewType", "通用评审");
        result.put("status", "已完成");
        result.put("summary", "评审任务已完成，整体质量符合要求。");
        result.put("score", 8.0);
        result.put("recommendation", "建议按计划继续推进");
        
        return result;
    }
    
    /**
     * 评估风险等级
     */
    private String assessRiskLevel(double score) {
        if (score >= 9.0) return "极低";
        if (score >= 8.0) return "低";
        if (score >= 7.0) return "中等";
        if (score >= 6.0) return "较高";
        return "高";
    }
    
    @Override
    protected void performPeriodicWork() throws InterruptedException {
        // 评审员的周期性工作：更新评审标准、分析质量趋势等
        if (metrics.getTasksCompleted() % 8 == 0 && metrics.getTasksCompleted() > 0) {
            System.out.println(String.format("Agent %s 执行周期性质量标准更新", name));
            
            // 模拟标准更新过程
            Thread.sleep(400);
        }
        
        super.performPeriodicWork();
    }
}