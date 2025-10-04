package io.leavesfly.tinyai.agent.research;

/**
 * 研究阶段枚举
 * 定义深度研究Agent的六个研究阶段
 * 
 * @author 山泽
 */
public enum ResearchPhase {
    /** 问题分析阶段 */
    PROBLEM_ANALYSIS("problem_analysis", "问题分析"),
    
    /** 信息收集阶段 */
    INFORMATION_GATHERING("information_gathering", "信息收集"),
    
    /** 深度分析阶段 */
    DEEP_ANALYSIS("deep_analysis", "深度分析"),
    
    /** 综合处理阶段 */
    SYNTHESIS("synthesis", "综合处理"),
    
    /** 验证检查阶段 */
    VALIDATION("validation", "验证检查"),
    
    /** 结论生成阶段 */
    CONCLUSION("conclusion", "结论生成");
    
    private final String code;
    private final String description;
    
    ResearchPhase(String code, String description) {
        this.code = code;
        this.description = description;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getDescription() {
        return description;
    }
    
    public static ResearchPhase fromCode(String code) {
        for (ResearchPhase phase : values()) {
            if (phase.code.equals(code)) {
                return phase;
            }
        }
        throw new IllegalArgumentException("未知的研究阶段代码: " + code);
    }
    
    @Override
    public String toString() {
        return String.format("%s(%s)", description, code);
    }
}