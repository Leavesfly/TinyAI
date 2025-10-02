package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.REINFORCEAgent;
import io.leavesfly.tinyai.rl.environment.GridWorldEnvironment;

import java.util.Map;

/**
 * GridWorld环境下使用REINFORCE算法的示例
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 这个示例展示了如何使用REINFORCE算法来解决GridWorld（网格世界）导航问题。
 * GridWorld是强化学习的经典导航问题，目标是从起始位置到达目标位置，同时避开障碍物。
 * 
 * 主要学习内容：
 * 1. REINFORCE算法的基本使用
 * 2. 策略梯度方法与价值方法的区别
 * 3. 回合制学习的特点
 * 4. 基线（baseline）的作用
 */
public class GridWorldREINFORCEExample {
    
    public static void main(String[] args) {
        System.out.println("=== GridWorld REINFORCE 训练示例 ===");
        
        // 运行两个版本的对比：带基线和不带基线
        System.out.println("\n--- 版本1：不使用基线 ---");
        runExperiment(false);
        
        System.out.println("\n--- 版本2：使用基线 ---");
        runExperiment(true);
        
        System.out.println("\n训练完成！");
    }
    
    /**
     * 运行实验
     * 
     * @param useBaseline 是否使用基线
     */
    private static void runExperiment(boolean useBaseline) {
        // 训练参数
        int numEpisodes = 2000;           // 训练回合数
        int evaluationInterval = 200;     // 评估间隔
        
        // 创建简单迷宫环境
        Environment env = GridWorldEnvironment.createSimpleMaze();
        
        // 显示环境信息
        System.out.println("环境信息:");
        System.out.println("状态维度: " + env.getStateDim());
        System.out.println("动作维度: " + env.getActionDim());
        System.out.println("初始环境状态:");
        env.reset();
        env.render();
        
        // 创建REINFORCE智能体
        REINFORCEAgent agent = createREINFORCEAgent(env, useBaseline);
        
        // 训练智能体
        trainAgent(agent, env, numEpisodes, evaluationInterval);
        
        // 最终评估
        System.out.println("\n=== 最终评估 ===");
        evaluateAgent(agent, env, 10);
        
        // 演示学习到的策略
        demonstrateLearnedPolicy(agent, env);
    }
    
    /**
     * 创建REINFORCE智能体
     * 
     * @param env 环境
     * @param useBaseline 是否使用基线
     * @return REINFORCE智能体
     */
    private static REINFORCEAgent createREINFORCEAgent(Environment env, boolean useBaseline) {
        // 网络参数
        int stateDim = env.getStateDim();           // 状态维度：2 (x, y坐标)
        int actionDim = env.getActionDim();         // 动作维度：4 (上下左右)
        int[] hiddenSizes = {64, 64};               // 隐藏层尺寸
        
        // 算法参数
        float learningRate = 0.01f;                 // 学习率
        float gamma = 0.99f;                        // 折扣因子
        
        String baselineInfo = useBaseline ? "使用基线" : "不使用基线";
        System.out.println("创建REINFORCE智能体 (" + baselineInfo + ")...");
        System.out.println("网络结构: " + stateDim + " -> " + hiddenSizes[0] + " -> " + hiddenSizes[1] + " -> " + actionDim);
        System.out.println("学习率: " + learningRate);
        
        return new REINFORCEAgent(
            "GridWorld_REINFORCE",
            stateDim, actionDim, hiddenSizes,
            learningRate, gamma, useBaseline
        );
    }
    
    /**
     * 训练智能体
     * 
     * @param agent 智能体
     * @param env 环境
     * @param numEpisodes 训练回合数
     * @param evaluationInterval 评估间隔
     */
    private static void trainAgent(REINFORCEAgent agent, Environment env, 
                                 int numEpisodes, int evaluationInterval) {
        System.out.println("\n开始训练...");
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            // 重置环境
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            // 运行一个完整回合
            while (!env.isDone() && steps < 100) { // 最多100步防止无限循环
                // 选择动作
                Variable action = agent.selectAction(state);
                
                // 执行动作
                Environment.StepResult result = env.step(action);
                Variable nextState = result.getNextState();
                float reward = result.getReward();
                boolean done = result.isDone();
                
                // 存储经验（REINFORCE在回合结束时统一学习）
                Experience experience = new Experience(state, action, reward, nextState, done, steps);
                agent.learn(experience);
                
                // 更新状态和累积奖励
                state = nextState;
                episodeReward += reward;
                steps++;
            }
            
            // 回合结束，进行学习更新
            agent.learnFromEpisode();
            
            // 打印训练进度
            if (episode % 100 == 0 || episode == numEpisodes - 1) {
                Map<String, Object> stats = agent.getTrainingStats();
                System.out.printf("Episode %d: 奖励=%.3f, 步数=%d, 平均回报=%.3f, 策略损失=%.6f\n",
                    episode, episodeReward, steps,
                    (Float) stats.get("average_return"),
                    (Float) stats.get("average_policy_loss"));
                
                if (agent.isUsingBaseline()) {
                    System.out.printf("    基线损失=%.6f\n", (Float) stats.get("average_baseline_loss"));
                }
            }
            
            // 定期评估
            if (episode > 0 && episode % evaluationInterval == 0) {
                System.out.println("\n--- 中期评估 (Episode " + episode + ") ---");
                evaluateAgent(agent, env, 5);
                System.out.println("--- 继续训练 ---\n");
            }
        }
    }
    
    /**
     * 评估智能体性能
     * 
     * @param agent 智能体
     * @param env 环境
     * @param numEvaluationEpisodes 评估回合数
     */
    private static void evaluateAgent(REINFORCEAgent agent, Environment env, int numEvaluationEpisodes) {
        // 切换到评估模式
        agent.setTraining(false);
        
        float totalReward = 0.0f;
        int totalSteps = 0;
        int successfulEpisodes = 0;
        
        for (int episode = 0; episode < numEvaluationEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!env.isDone() && steps < 100) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            totalReward += episodeReward;
            totalSteps += steps;
            
            // 检查是否成功到达目标
            GridWorldEnvironment gridEnv = (GridWorldEnvironment) env;
            Map<String, Integer> agentPos = gridEnv.getAgentPosition();
            Map<String, Integer> goalPos = gridEnv.getGoalPosition();
            boolean success = agentPos.get("x").equals(goalPos.get("x")) && 
                            agentPos.get("y").equals(goalPos.get("y"));
            
            if (success) {
                successfulEpisodes++;
            }
            
            System.out.printf("评估回合 %d: 奖励=%.3f, 步数=%d, 成功=%s\n", 
                            episode + 1, episodeReward, steps, success ? "是" : "否");
        }
        
        // 计算平均性能
        float averageReward = totalReward / numEvaluationEpisodes;
        float averageSteps = (float) totalSteps / numEvaluationEpisodes;
        float successRate = (float) successfulEpisodes / numEvaluationEpisodes * 100;
        
        System.out.println("评估结果:");
        System.out.printf("  平均奖励: %.3f\n", averageReward);
        System.out.printf("  平均步数: %.2f\n", averageSteps);
        System.out.printf("  成功率: %.1f%% (%d/%d)\n", successRate, successfulEpisodes, numEvaluationEpisodes);
        
        // 切换回训练模式
        agent.setTraining(true);
    }
    
    /**
     * 演示学习到的策略
     * 
     * @param agent 智能体
     * @param env 环境
     */
    private static void demonstrateLearnedPolicy(REINFORCEAgent agent, Environment env) {
        System.out.println("\n=== 策略演示 ===");
        
        agent.setTraining(false);
        Variable state = env.reset();
        
        System.out.println("智能体将尝试从起始位置到达目标位置:");
        env.render();
        
        String[] actionNames = {"UP", "DOWN", "LEFT", "RIGHT"};
        
        for (int step = 0; step < 20; step++) { // 最多演示20步
            Variable action = agent.selectAction(state);
            int actionIndex = (int) action.getValue().getNumber().floatValue();
            String actionName = actionNames[actionIndex];
            
            Environment.StepResult result = env.step(action);
            
            System.out.printf("步骤 %d: 动作=%s, 奖励=%.3f\n", step + 1, actionName, result.getReward());
            env.render();
            
            state = result.getNextState();
            
            if (result.isDone()) {
                GridWorldEnvironment gridEnv = (GridWorldEnvironment) env;
                Map<String, Integer> agentPos = gridEnv.getAgentPosition();
                Map<String, Integer> goalPos = gridEnv.getGoalPosition();
                boolean success = agentPos.get("x").equals(goalPos.get("x")) && 
                                agentPos.get("y").equals(goalPos.get("y"));
                
                if (success) {
                    System.out.println("成功到达目标！");
                } else {
                    System.out.println("未能到达目标。");
                }
                break;
            }
        }
        
        agent.setTraining(true);
    }
    
    /**
     * 比较不同配置的性能
     */
    public static void compareConfigurations() {
        System.out.println("\n=== 配置比较实验 ===");
        
        // 创建不同的环境
        Environment simpleEnv = GridWorldEnvironment.createSimpleMaze();
        Environment complexEnv = GridWorldEnvironment.createWithRandomObstacles(8, 8, 0.2f);
        
        System.out.println("简单环境 vs 复杂环境");
        System.out.println("不使用基线 vs 使用基线");
        
        // 这里可以扩展更多的比较实验
    }
}