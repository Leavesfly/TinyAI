package io.leavesfly.tinyai.agent.vla.model;

/**
 * 任务配置
 * 定义VLA智能体的任务参数
 * 
 * @author TinyAI
 */
public class TaskConfig {
    
    /** 任务名称 */
    private String taskName;
    
    /** 任务描述 */
    private String taskDescription;
    
    /** 最大步数 */
    private int maxSteps;
    
    /** 成功奖励 */
    private double successReward;
    
    /** 失败惩罚 */
    private double failurePenalty;
    
    /** 随机种子 */
    private long randomSeed;
    
    /** 是否渲染 */
    private boolean render;
    
    /** 步数惩罚 */
    private double stepPenalty;
    
    /** 额外参数 */
    private java.util.Map<String, Object> parameters;
    
    /**
     * 默认构造函数
     */
    public TaskConfig() {
        this.maxSteps = 100;
        this.successReward = 100.0;
        this.failurePenalty = -10.0;
        this.stepPenalty = 0.0;
        this.randomSeed = System.currentTimeMillis();
        this.render = false;
        this.parameters = new java.util.HashMap<>();
    }
    
    /**
     * 完整构造函数
     */
    public TaskConfig(String taskName, String taskDescription, int maxSteps, 
                      double successReward, double failurePenalty) {
        this.taskName = taskName;
        this.taskDescription = taskDescription;
        this.maxSteps = maxSteps;
        this.successReward = successReward;
        this.failurePenalty = failurePenalty;
        this.randomSeed = System.currentTimeMillis();
        this.render = false;
    }
    
    // Getters and Setters
    public String getTaskName() {
        return taskName;
    }
    
    public void setTaskName(String taskName) {
        this.taskName = taskName;
    }
    
    public String getTaskDescription() {
        return taskDescription;
    }
    
    public void setTaskDescription(String taskDescription) {
        this.taskDescription = taskDescription;
    }
    
    public int getMaxSteps() {
        return maxSteps;
    }
    
    public void setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
    }
    
    public double getSuccessReward() {
        return successReward;
    }
    
    public void setSuccessReward(double successReward) {
        this.successReward = successReward;
    }
    
    public double getFailurePenalty() {
        return failurePenalty;
    }
    
    public void setFailurePenalty(double failurePenalty) {
        this.failurePenalty = failurePenalty;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }
    
    public boolean isRender() {
        return render;
    }
    
    public void setRender(boolean render) {
        this.render = render;
    }
    
    public double getStepPenalty() {
        return stepPenalty;
    }
    
    public void setStepPenalty(double stepPenalty) {
        this.stepPenalty = stepPenalty;
    }
    
    /**
     * 添加自定义参数
     */
    public void addParameter(String key, Object value) {
        if (parameters == null) {
            parameters = new java.util.HashMap<>();
        }
        parameters.put(key, value);
    }
    
    /**
     * 获取自定义参数
     */
    public Object getParameter(String key) {
        return parameters != null ? parameters.get(key) : null;
    }
    
    /**
     * 获取所有参数
     */
    public java.util.Map<String, Object> getParameters() {
        return parameters;
    }
    
    @Override
    public String toString() {
        return "TaskConfig{" +
                "taskName='" + taskName + '\'' +
                ", maxSteps=" + maxSteps +
                ", successReward=" + successReward +
                ", failurePenalty=" + failurePenalty +
                ", randomSeed=" + randomSeed +
                ", render=" + render +
                '}';
    }
}
