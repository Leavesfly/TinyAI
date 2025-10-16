package io.leavesfly.tinyai.agent.multi;

import java.util.*;

import io.leavesfly.tinyai.agent.context.LLMSimulator;

/**
 * 协调员Agent
 * 专门负责任务分配、进度跟踪和团队协调
 * 
 * @author 山泽
 */
public class CoordinatorAgent extends BaseAgent {
    
    private final List<String> teamMembers; // 团队成员列表
    
    public CoordinatorAgent(String agentId, MessageBus messageBus, LLMSimulator llm) {
        super(agentId, "协调员-" + agentId.substring(Math.max(0, agentId.length() - 4)), 
              "项目协调员", messageBus, llm);
        this.teamMembers = new ArrayList<>();
    }
    
    @Override
    protected void initializeCapabilities() {
        capabilities.addAll(Arrays.asList("任务分配", "进度跟踪", "团队协调", "项目管理"));
    }
    
    @Override
    protected String getAgentType() {
        return "coordinator";
    }
    
    /**
     * 添加团队成员
     */
    public void addTeamMember(String agentId) {
        if (!teamMembers.contains(agentId)) {
            teamMembers.add(agentId);
            System.out.println(String.format("协调员 %s 添加团队成员: %s", name, agentId));
        }
    }
    
    /**
     * 移除团队成员
     */
    public void removeTeamMember(String agentId) {
        if (teamMembers.remove(agentId)) {
            System.out.println(String.format("协调员 %s 移除团队成员: %s", name, agentId));
        }
    }
    
    @Override
    protected Object performTask(AgentTask task) throws Exception {
        String description = task.getDescription().toLowerCase();
        
        if (description.contains("协调") || description.contains("分配")) {
            return coordinateTeamWork(task);
        } else if (description.contains("跟踪") || description.contains("进度")) {
            return trackProgress(task);
        } else if (description.contains("管理")) {
            return manageProject(task);
        } else {
            return performGeneralCoordination(task);
        }
    }
    
    /**
     * 协调团队工作
     */
    private Map<String, Object> coordinateTeamWork(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始协调团队工作: %s", name, task.getTitle()));
        
        // 模拟协调过程
        Thread.sleep(1500);
        
        List<String> assignments = new ArrayList<>();
        List<Map<String, Object>> subtasks = new ArrayList<>();
        
        // 为每个团队成员分配子任务
        for (int i = 0; i < teamMembers.size(); i++) {
            String memberId = teamMembers.get(i);
            String subtaskId = UUID.randomUUID().toString();
            assignments.add(subtaskId);
            
            // 创建子任务
            Map<String, Object> subtaskData = new HashMap<>();
            subtaskData.put("id", subtaskId);
            subtaskData.put("title", String.format("子任务-%d", i + 1));
            subtaskData.put("description", String.format("执行%s的第%d部分", task.getTitle(), i + 1));
            subtaskData.put("assignedTo", memberId);
            subtaskData.put("createdBy", agentId);
            
            subtasks.add(subtaskData);
            
            // 发送子任务给团队成员
            sendMessage(memberId, subtaskData, MessageType.TASK);
            
            System.out.println(String.format("协调员 %s 分配子任务给 %s: %s", name, memberId, subtaskData.get("title")));
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("coordinationType", "任务分配");
        result.put("assignedTasks", assignments);
        result.put("subtasks", subtasks);
        result.put("teamSize", teamMembers.size());
        result.put("estimatedCompletion", "2-3小时");
        result.put("coordinationStrategy", "并行处理，定期同步");
        
        return result;
    }
    
    /**
     * 跟踪进度
     */
    private Map<String, Object> trackProgress(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始跟踪进度: %s", name, task.getTitle()));
        
        // 模拟进度跟踪
        Thread.sleep(1000);
        
        // 向团队成员查询状态
        for (String memberId : teamMembers) {
            Map<String, Object> statusQuery = new HashMap<>();
            statusQuery.put("command", "status");
            statusQuery.put("requestedBy", agentId);
            
            sendMessage(memberId, statusQuery, MessageType.SYSTEM);
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("coordinationType", "进度跟踪");
        result.put("trackedAgents", teamMembers.size());
        result.put("statusRequests", teamMembers.size());
        result.put("trackingMethod", "主动查询");
        result.put("frequency", "每30分钟一次");
        
        return result;
    }
    
    /**
     * 项目管理
     */
    private Map<String, Object> manageProject(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始项目管理: %s", name, task.getTitle()));
        
        // 模拟项目管理过程
        Thread.sleep(2000);
        
        Map<String, Object> result = new HashMap<>();
        result.put("coordinationType", "项目管理");
        result.put("managementAreas", Arrays.asList(
            "资源分配优化",
            "时间进度控制",
            "质量标准确保",
            "风险识别管理"
        ));
        result.put("teamPerformance", "良好");
        result.put("projectStatus", "按计划进行");
        result.put("riskLevel", "低");
        result.put("recommendations", Arrays.asList(
            "保持当前工作节奏",
            "加强团队沟通",
            "定期评估进展",
            "及时调整资源配置"
        ));
        
        return result;
    }
    
    /**
     * 执行通用协调
     */
    private Map<String, Object> performGeneralCoordination(AgentTask task) throws InterruptedException {
        System.out.println(String.format("Agent %s 开始通用协调: %s", name, task.getTitle()));
        
        Thread.sleep(800);
        
        Map<String, Object> result = new HashMap<>();
        result.put("coordinationType", "通用协调");
        result.put("status", "已完成");
        result.put("summary", "协调任务已处理完成");
        result.put("teamStatus", "运行正常");
        
        return result;
    }
    
    @Override
    protected void performPeriodicWork() throws InterruptedException {
        // 协调员的周期性工作：定期检查团队状态
        if (teamMembers.size() > 0 && metrics.getMessagesSent() % 15 == 0) {
            System.out.println(String.format("协调员 %s 执行周期性团队状态检查", name));
            
            // 向团队成员发送状态查询
            for (String memberId : teamMembers) {
                Map<String, Object> statusQuery = new HashMap<>();
                statusQuery.put("command", "status");
                statusQuery.put("type", "periodic_check");
                
                sendMessage(memberId, statusQuery, MessageType.SYSTEM);
            }
            
            Thread.sleep(200);
        }
        
        super.performPeriodicWork();
    }
    
    // Getter 方法
    public List<String> getTeamMembers() {
        return new ArrayList<>(teamMembers);
    }
    
    public int getTeamSize() {
        return teamMembers.size();
    }
}