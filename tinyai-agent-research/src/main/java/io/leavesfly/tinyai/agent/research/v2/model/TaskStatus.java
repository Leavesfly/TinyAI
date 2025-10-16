package io.leavesfly.tinyai.agent.research.v2.model;

/**
 * 任务状态枚举
 * 定义研究任务的生命周期状态
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public enum TaskStatus {
    
    /**
     * 已提交 - 任务已创建，等待开始
     */
    SUBMITTED("已提交", "任务已创建，等待开始"),
    
    /**
     * 规划中 - 正在制定研究计划
     */
    PLANNING("规划中", "正在制定研究计划"),
    
    /**
     * 执行中 - 正在执行信息检索和数据收集
     */
    EXECUTING("执行中", "正在执行信息检索和数据收集"),
    
    /**
     * 分析中 - 正在进行深度分析和推理
     */
    ANALYZING("分析中", "正在进行深度分析和推理"),
    
    /**
     * 报告生成中 - 正在生成研究报告
     */
    REPORTING("报告生成中", "正在生成研究报告"),
    
    /**
     * 已完成 - 任务成功完成
     */
    COMPLETED("已完成", "任务成功完成"),
    
    /**
     * 失败 - 任务执行失败
     */
    FAILED("失败", "任务执行失败");
    
    /**
     * 状态名称
     */
    private final String name;
    
    /**
     * 状态描述
     */
    private final String description;
    
    TaskStatus(String name, String description) {
        this.name = name;
        this.description = description;
    }
    
    public String getName() {
        return name;
    }
    
    public String getDescription() {
        return description;
    }
    
    /**
     * 判断是否为终止状态
     */
    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED;
    }
    
    /**
     * 判断是否可以转换到目标状态
     */
    public boolean canTransitionTo(TaskStatus target) {
        if (this.isTerminal()) {
            return false; // 终止状态不能再转换
        }
        
        switch (this) {
            case SUBMITTED:
                return target == PLANNING || target == FAILED;
            case PLANNING:
                return target == EXECUTING || target == FAILED;
            case EXECUTING:
                return target == ANALYZING || target == FAILED;
            case ANALYZING:
                return target == REPORTING || target == FAILED;
            case REPORTING:
                return target == COMPLETED || target == FAILED;
            default:
                return false;
        }
    }
}
