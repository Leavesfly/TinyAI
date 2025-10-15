package io.leavesfly.tinyai.agent.cursor.v1;

import java.util.List;
import java.util.Objects;

/**
 * 重构建议
 * 表示一个具体的重构建议，包含重构类型、描述、代码示例等
 * 
 * @author 山泽
 */
public class RefactorSuggestion {
    
    private final String suggestionType;    // 重构类型
    private final String description;       // 详细描述
    private final String originalCode;      // 原始代码
    private final String refactoredCode;    // 重构后代码
    private final List<String> benefits;    // 重构收益
    private final String estimatedImpact;   // 预估影响程度
    private final int priority;             // 优先级 (1-5, 5最高)
    
    /**
     * 构造函数
     * @param suggestionType 重构类型
     * @param description 详细描述
     * @param originalCode 原始代码
     * @param refactoredCode 重构后代码
     * @param benefits 重构收益
     * @param estimatedImpact 预估影响程度
     */
    public RefactorSuggestion(String suggestionType, String description, String originalCode, 
                            String refactoredCode, List<String> benefits, String estimatedImpact) {
        this.suggestionType = suggestionType;
        this.description = description;
        this.originalCode = originalCode;
        this.refactoredCode = refactoredCode;
        this.benefits = benefits;
        this.estimatedImpact = estimatedImpact;
        this.priority = calculatePriority(estimatedImpact);
    }
    
    /**
     * 根据影响程度计算优先级
     */
    private int calculatePriority(String impact) {
        if (impact == null) return 1;
        
        switch (impact.toLowerCase()) {
            case "critical":
            case "高":
                return 5;
            case "high":
            case "中等":
                return 4;
            case "medium":
            case "中":
                return 3;
            case "low":
            case "低":
                return 2;
            default:
                return 1;
        }
    }
    
    // Getter 方法
    public String getSuggestionType() {
        return suggestionType;
    }
    
    public String getDescription() {
        return description;
    }
    
    public String getOriginalCode() {
        return originalCode;
    }
    
    public String getRefactoredCode() {
        return refactoredCode;
    }
    
    public List<String> getBenefits() {
        return benefits;
    }
    
    public String getEstimatedImpact() {
        return estimatedImpact;
    }
    
    public int getPriority() {
        return priority;
    }
    
    /**
     * 获取重构收益摘要
     */
    public String getBenefitsSummary() {
        if (benefits == null || benefits.isEmpty()) {
            return "无明显收益";
        }
        return String.join(", ", benefits);
    }
    
    /**
     * 检查是否为高优先级建议
     */
    public boolean isHighPriority() {
        return priority >= 4;
    }
    
    @Override
    public String toString() {
        return String.format("RefactorSuggestion{type='%s', impact='%s', priority=%d, description='%s'}", 
                           suggestionType, estimatedImpact, priority, description);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        RefactorSuggestion that = (RefactorSuggestion) obj;
        return suggestionType.equals(that.suggestionType) &&
               description.equals(that.description) &&
               Objects.equals(originalCode, that.originalCode);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(suggestionType, description, originalCode);
    }
}