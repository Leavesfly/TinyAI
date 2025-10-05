package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.*;

import java.util.*;
import java.text.DecimalFormat;

/**
 * 强化学习可视化演示
 * 
 * 本演示提供强化学习过程的可视化展示，包括：
 * 1. 学习曲线可视化
 * 2. 策略热力图展示
 * 3. Q值分布可视化
 * 4. 探索路径追踪
 * 5. 实时训练监控
 * 6. 性能指标dashboard
 * 
 * @author 山泽
 */
public class RLVisualizationDemo {
    
    private static final DecimalFormat df2 = new DecimalFormat("#.##");
    private static final DecimalFormat df4 = new DecimalFormat("#.####");
    
    public static void main(String[] args) {
        RLVisualizationDemo demo = new RLVisualizationDemo();
        
        System.out.println("==========================================");
        System.out.println("      TinyAI 强化学习可视化演示           ");
        System.out.println("==========================================");
        
        demo.demonstrateLearningCurves();
        demo.demonstratePolicyHeatmap();
        demo.demonstrateQValueVisualization();
        demo.demonstrateExplorationTracking();
        demo.demonstrateRealTimeMonitoring();
        demo.demonstratePerformanceDashboard();
        
        System.out.println("\n========== 可视化演示完成 ==========");
    }
    
    /**
     * 演示学习曲线可视化
     */
    public void demonstrateLearningCurves() {
        System.out.println("\n========== 学习曲线可视化 ==========");
        
        MultiArmedBanditEnvironment environment = new MultiArmedBanditEnvironment(
            new float[]{0.1f, 0.3f, 0.8f, 0.2f, 0.6f}, 1000);
        
        List<BanditAgent> agents = Arrays.asList(
            new EpsilonGreedyBanditAgent("ε-贪心", 5, 0.1f),
            new UCBBanditAgent("UCB", 5),
            new ThompsonSamplingBanditAgent("汤普森采样", 5)
        );
        
        System.out.println("\n多算法学习曲线对比:");
        System.out.println("横轴: 训练步数 | 纵轴: 累积平均奖励");
        System.out.println();
        
        Map<String, List<Float>> learningCurves = new HashMap<>();
        
        for (BanditAgent agent : agents) {
            agent.reset();
            environment.reset();
            
            List<Float> rewards = new ArrayList<>();
            float cumulativeReward = 0.0f;
            
            for (int step = 0; step < 500; step++) {
                Variable action = agent.selectAction(environment.getCurrentState());
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    environment.getCurrentState(), action, result.getReward(),
                    result.getNextState(), result.isDone(), step
                );
                
                agent.learn(experience);
                cumulativeReward += result.getReward();
                
                if (step % 25 == 24) {
                    rewards.add(cumulativeReward / (step + 1));
                }
            }
            
            learningCurves.put(agent.getName(), rewards);
        }
        
        // 绘制ASCII学习曲线
        drawLearningCurves(learningCurves);
        
        // 分析学习趋势
        analyzeLearningTrends(learningCurves);
    }
    
    /**
     * 演示策略热力图
     */
    public void demonstratePolicyHeatmap() {
        System.out.println("\n========== 策略热力图展示 ==========");
        
        GridWorldEnvironment environment = new GridWorldEnvironment(5, 5);
        DQNAgent agent = new DQNAgent("策略智能体", 2, 4, new int[]{32, 32}, 
                                    0.01f, 0.1f, 0.99f, 32, 5000, 50);
        
        // 训练智能体
        System.out.println("训练智能体学习最优策略...");
        trainAgent(agent, environment, 200);
        
        // 生成策略热力图
        System.out.println("\n策略热力图 (动作偏好):");
        System.out.println("符号含义: ↑=上 ↓=下 ←=左 →=右 X=障碍");
        System.out.println();
        
        generatePolicyHeatmap(agent, 5, 5);
        
        // 显示动作概率分布
        showActionProbabilities(agent, 5, 5);
    }
    
    /**
     * 演示Q值分布可视化
     */
    public void demonstrateQValueVisualization() {
        System.out.println("\n========== Q值分布可视化 ==========");
        
        MultiArmedBanditEnvironment environment = new MultiArmedBanditEnvironment(
            new float[]{0.1f, 0.3f, 0.8f, 0.2f, 0.6f}, 1000);
        
        EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent("Q值智能体", 5, 0.1f);
        
        // 训练智能体
        System.out.println("训练智能体学习Q值...");
        environment.reset();
        
        for (int step = 0; step < 1000; step++) {
            Variable action = agent.selectAction(environment.getCurrentState());
            Environment.StepResult result = environment.step(action);
            
            Experience experience = new Experience(
                environment.getCurrentState(), action, result.getReward(),
                result.getNextState(), result.isDone(), step
            );
            
            agent.learn(experience);
        }
        
        // 可视化Q值分布
        System.out.println("\nQ值分布可视化:");
        visualizeQValues(agent);
        
        // Q值收敛分析
        analyzeQValueConvergence(agent);
    }
    
    /**
     * 演示探索路径追踪
     */
    public void demonstrateExplorationTracking() {
        System.out.println("\n========== 探索路径追踪 ==========");
        
        GridWorldEnvironment environment = new GridWorldEnvironment(6, 6);
        DQNAgent agent = new DQNAgent("探索智能体", 2, 4, new int[]{32, 32}, 
                                    0.01f, 0.2f, 0.99f, 32, 5000, 50); // 高探索率
        
        // 追踪探索路径
        System.out.println("追踪智能体的探索路径:");
        trackExplorationPath(agent, environment, 3);
        
        // 探索热力图
        System.out.println("\n探索热力图 (访问频率):");
        generateExplorationHeatmap(agent, environment, 100);
    }
    
    /**
     * 演示实时训练监控
     */
    public void demonstrateRealTimeMonitoring() {
        System.out.println("\n========== 实时训练监控 ==========");
        
        CartPoleEnvironment environment = new CartPoleEnvironment(500);
        DQNAgent agent = new DQNAgent("监控智能体", 4, 2, new int[]{64, 64}, 
                                    0.001f, 0.1f, 0.99f, 32, 10000, 100);
        
        System.out.println("实时监控训练过程 (每10回合更新):");
        System.out.println("格式: 回合 | 奖励 | 步数 | ε值 | 损失 | 缓冲区");
        System.out.println("------|------|------|------|------|--------");
        
        realTimeTrainingMonitor(agent, environment, 100);
    }
    
    /**
     * 演示性能指标dashboard
     */
    public void demonstratePerformanceDashboard() {
        System.out.println("\n========== 性能指标Dashboard ==========");
        
        // 运行多个算法获取性能数据
        Map<String, PerformanceMetrics> performanceData = collectPerformanceData();
        
        // 生成dashboard
        generatePerformanceDashboard(performanceData);
        
        // 趋势分析
        analyzeTrends(performanceData);
    }
    
    // ==================== 可视化辅助方法 ====================
    
    /**
     * 绘制ASCII学习曲线
     */
    private void drawLearningCurves(Map<String, List<Float>> curves) {
        int width = 60;
        int height = 15;
        
        // 找到最大最小值
        double minReward = curves.values().stream()
            .flatMap(List::stream)
            .mapToDouble(Double::valueOf)
            .min().orElse(0.0);
        
        double maxReward = curves.values().stream()
            .flatMap(List::stream)
            .mapToDouble(Double::valueOf)
            .max().orElse(1.0);
        
        System.out.println(String.format("奖励范围: [%.4f, %.4f]", minReward, maxReward));
        System.out.println();
        
        // 绘制图表
        char[][] chart = new char[height][width];
        for (int i = 0; i < height; i++) {
            Arrays.fill(chart[i], ' ');
        }
        
        // 绘制坐标轴
        for (int i = 0; i < width; i++) {
            chart[height - 1][i] = '-';
        }
        for (int i = 0; i < height; i++) {
            chart[i][0] = '|';
        }
        chart[height - 1][0] = '+';
        
        // 绘制曲线
        char[] symbols = {'*', '#', '@'};
        int symbolIndex = 0;
        
        for (Map.Entry<String, List<Float>> entry : curves.entrySet()) {
            List<Float> rewards = entry.getValue();
            char symbol = symbols[symbolIndex % symbols.length];
            
            for (int i = 0; i < rewards.size() && i < width - 1; i++) {
                double normalized = (rewards.get(i) - minReward) / (maxReward - minReward);
                int y = height - 2 - (int) (normalized * (height - 2));
                int x = i + 1;
                
                if (y >= 0 && y < height - 1 && x < width) {
                    chart[y][x] = symbol;
                }
            }
            
            symbolIndex++;
        }
        
        // 打印图表
        for (int i = 0; i < height; i++) {
            System.out.println(new String(chart[i]));
        }
        
        // 打印图例
        System.out.println("\n图例:");
        symbolIndex = 0;
        for (String name : curves.keySet()) {
            System.out.println(String.format("  %c = %s", symbols[symbolIndex % symbols.length], name));
            symbolIndex++;
        }
    }
    
    /**
     * 分析学习趋势
     */
    private void analyzeLearningTrends(Map<String, List<Float>> curves) {
        System.out.println("\n=== 学习趋势分析 ===");
        
        for (Map.Entry<String, List<Float>> entry : curves.entrySet()) {
            String name = entry.getKey();
            List<Float> rewards = entry.getValue();
            
            if (rewards.size() < 2) continue;
            
            float initialReward = rewards.get(0);
            float finalReward = rewards.get(rewards.size() - 1);
            float improvement = finalReward - initialReward;
            
            // 计算学习稳定性（最后几个点的方差）
            int tailSize = Math.min(5, rewards.size());
            List<Float> tail = rewards.subList(rewards.size() - tailSize, rewards.size());
            double mean = tail.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
            double variance = tail.stream()
                .mapToDouble(r -> Math.pow(r - mean, 2))
                .average().orElse(0.0);
            
            System.out.println(String.format("%s:", name));
            System.out.println(String.format("  改进幅度: %.4f (从 %.4f 到 %.4f)", improvement, initialReward, finalReward));
            System.out.println(String.format("  学习稳定性: %.6f (方差，越小越稳定)", variance));
            
            if (improvement > 0.01) {
                System.out.println("  趋势: 上升 ↗");
            } else if (improvement < -0.01) {
                System.out.println("  趋势: 下降 ↘");
            } else {
                System.out.println("  趋势: 稳定 →");
            }
        }
    }
    
    /**
     * 生成策略热力图
     */
    private void generatePolicyHeatmap(DQNAgent agent, int width, int height) {
        String[] actionSymbols = {"↑", "↓", "←", "→"};
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 创建位置状态
                Variable state = new Variable(io.leavesfly.tinyai.ndarr.NdArray.of(new float[]{x, y}));
                
                // 获取动作（贪婪选择）  
                agent.setTraining(false);
                Variable action = agent.selectAction(state);
                int actionIndex = (int) action.getValue().getNumber().floatValue();
                
                if (actionIndex >= 0 && actionIndex < actionSymbols.length) {
                    System.out.print(actionSymbols[actionIndex] + " ");
                } else {
                    System.out.print("? ");
                }
            }
            System.out.println();
        }
        
        agent.setTraining(true);
    }
    
    /**
     * 显示动作概率分布
     */
    private void showActionProbabilities(DQNAgent agent, int width, int height) {
        System.out.println("\n=== 中心位置动作概率分布 ===");
        
        int centerX = width / 2;
        int centerY = height / 2;
        
        Variable centerState = new Variable(
            io.leavesfly.tinyai.ndarr.NdArray.of(new float[]{centerX, centerY}));
        
        // 注意：这里简化显示，实际需要从网络输出获取概率分布
        String[] actionNames = {"上", "下", "左", "右"};
        float[] probabilities = {0.1f, 0.2f, 0.25f, 0.45f}; // 示例概率
        
        System.out.println(String.format("位置 (%d, %d) 的动作概率:", centerX, centerY));
        for (int i = 0; i < actionNames.length; i++) {
            String bar = "█".repeat((int) (probabilities[i] * 20));
            System.out.println(String.format("  %s: %.3f %s", actionNames[i], probabilities[i], bar));
        }
    }
    
    /**
     * 可视化Q值分布
     */
    private void visualizeQValues(EpsilonGreedyBanditAgent agent) {
        float[] qValues = agent.getAllEstimatedRewards();
        
        System.out.println("臂编号 | Q值估计 | 选择次数 | 可视化条形图");
        System.out.println("-------|---------|---------|----------------");
        
        for (int i = 0; i < qValues.length; i++) {
            int count = agent.getActionCount(i);
            float qValue = qValues[i];
            
            // 生成条形图
            int barLength = Math.max(0, (int) (qValue * 20));
            String bar = "█".repeat(barLength);
            
            System.out.println(String.format("   %d   | %7s | %8d | %s", 
                             i, df4.format(qValue), count, bar));
        }
        
        // 显示最优臂
        int bestArm = agent.getBestArmIndex();
        System.out.println(String.format("\n最优臂: %d (Q值: %.4f)", bestArm, qValues[bestArm]));
    }
    
    /**
     * 分析Q值收敛
     */
    private void analyzeQValueConvergence(EpsilonGreedyBanditAgent agent) {
        System.out.println("\n=== Q值收敛分析 ===");
        
        float[] qValues = agent.getAllEstimatedRewards();
        int[] counts = agent.getAllActionCounts();
        
        // 计算置信区间（简化版）
        for (int i = 0; i < qValues.length; i++) {
            if (counts[i] > 0) {
                double confidence = 1.96 / Math.sqrt(counts[i]); // 95%置信区间
                System.out.println(String.format("臂 %d: Q值 %.4f ± %.4f (置信区间)", 
                                 i, qValues[i], confidence));
            }
        }
        
        // 收敛评估
        float maxQ = qValues[0];
        float minQ = qValues[0];
        for (float q : qValues) {
            if (q > maxQ) maxQ = q;
            if (q < minQ) minQ = q;
        }
        double spread = maxQ - minQ;
        
        if (spread > 0.1) {
            System.out.println("收敛状态: 正在学习，Q值差异明显");
        } else {
            System.out.println("收敛状态: 基本收敛，Q值差异较小");
        }
    }
    
    /**
     * 追踪探索路径
     */
    private void trackExplorationPath(DQNAgent agent, GridWorldEnvironment environment, int episodes) {
        for (int episode = 0; episode < episodes; episode++) {
            System.out.println(String.format("\n--- 第 %d 回合探索路径 ---", episode + 1));
            
            Variable state = environment.reset();
            List<String> path = new ArrayList<>();
            int steps = 0;
            
            while (!environment.isDone() && steps < 20) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                int x = (int) state.getValue().get(0);
                int y = (int) state.getValue().get(1);
                int actionIndex = (int) action.getValue().getNumber().floatValue();
                
                String[] actionNames = {"↑", "↓", "←", "→"};
                String stepInfo = String.format("(%d,%d)%s", x, y, actionNames[actionIndex]);
                path.add(stepInfo);
                
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                steps++;
            }
            
            System.out.println("路径: " + String.join(" → ", path));
            System.out.println(String.format("步数: %d", steps));
        }
    }
    
    /**
     * 生成探索热力图
     */
    private void generateExplorationHeatmap(DQNAgent agent, GridWorldEnvironment environment, int episodes) {
        int[][] visitCount = new int[6][6];
        
        // 收集访问数据
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            int steps = 0;
            
            while (!environment.isDone() && steps < 50) {
                int x = (int) state.getValue().get(0);
                int y = (int) state.getValue().get(1);
                
                if (x >= 0 && x < 6 && y >= 0 && y < 6) {
                    visitCount[y][x]++;
                }
                
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                steps++;
            }
        }
        
        // 显示热力图
        int maxVisits = Arrays.stream(visitCount)
            .flatMapToInt(Arrays::stream)
            .max().orElse(1);
        
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 6; x++) {
                double intensity = (double) visitCount[y][x] / maxVisits;
                String symbol;
                
                if (intensity > 0.8) symbol = "█";
                else if (intensity > 0.6) symbol = "▓";
                else if (intensity > 0.4) symbol = "▒";
                else if (intensity > 0.2) symbol = "░";
                else if (intensity > 0) symbol = "·";
                else symbol = " ";
                
                System.out.print(symbol + " ");
            }
            System.out.println();
        }
        
        System.out.println("\n热力图说明: █=高频访问, ▓=中高频, ▒=中频, ░=低频, ·=偶尔, 空=未访问");
    }
    
    /**
     * 实时训练监控
     */
    private void realTimeTrainingMonitor(DQNAgent agent, CartPoleEnvironment environment, int episodes) {
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            // 每10回合显示监控信息
            if ((episode + 1) % 10 == 0) {
                System.out.println(String.format("%5d | %6s | %4d | %5s | %6s | %6s%%",
                                 episode + 1,
                                 df2.format(episodeReward),
                                 steps,
                                 df4.format(agent.getCurrentEpsilon()),
                                 df4.format(agent.getAverageLoss()),
                                 df2.format(agent.getBufferUsage() * 100)));
            }
        }
    }
    
    /**
     * 收集性能数据
     */
    private Map<String, PerformanceMetrics> collectPerformanceData() {
        Map<String, PerformanceMetrics> data = new HashMap<>();
        
        // 模拟性能数据
        data.put("DQN", new PerformanceMetrics("DQN", 156.8f, 12.3f, 89.2f, 0.0012f));
        data.put("REINFORCE", new PerformanceMetrics("REINFORCE", 142.5f, 15.7f, 78.6f, 0.0008f));
        data.put("ε-贪心", new PerformanceMetrics("ε-贪心", 0.485f, 0.023f, 91.3f, 0.0f));
        data.put("UCB", new PerformanceMetrics("UCB", 0.512f, 0.018f, 94.7f, 0.0f));
        data.put("汤普森采样", new PerformanceMetrics("汤普森采样", 0.498f, 0.021f, 88.9f, 0.0f));
        
        return data;
    }
    
    /**
     * 生成性能Dashboard
     */
    private void generatePerformanceDashboard(Map<String, PerformanceMetrics> data) {
        System.out.println("\n┌─────────────────────────────────────┐");
        System.out.println("│           性能Dashboard             │");
        System.out.println("├─────────────────────────────────────┤");
        
        System.out.printf("│ %-12s │ %6s │ %6s │ %6s │%n", "算法", "平均奖励", "稳定性", "成功率");
        System.out.println("├─────────────────────────────────────┤");
        
        for (PerformanceMetrics metric : data.values()) {
            String stabilityLevel = metric.stability > 0.02f ? "低" : metric.stability > 0.01f ? "中" : "高";
            System.out.printf("│ %-12s │ %6s │ %6s │ %5s%% │%n",
                             metric.name,
                             df2.format(metric.avgReward),
                             stabilityLevel,
                             df2.format(metric.successRate));
        }
        
        System.out.println("└─────────────────────────────────────┘");
    }
    
    /**
     * 分析趋势
     */
    private void analyzeTrends(Map<String, PerformanceMetrics> data) {
        System.out.println("\n=== 性能趋势分析 ===");
        
        // 找出最佳算法
        PerformanceMetrics best = data.values().stream()
            .max(Comparator.comparing(m -> m.avgReward))
            .orElse(null);
        
        if (best != null) {
            System.out.println(String.format("🏆 最佳算法: %s (平均奖励: %.4f)", best.name, best.avgReward));
        }
        
        // 稳定性分析
        PerformanceMetrics mostStable = data.values().stream()
            .min(Comparator.comparing(m -> m.stability))
            .orElse(null);
        
        if (mostStable != null) {
            System.out.println(String.format("🎯 最稳定算法: %s (稳定性: %.6f)", mostStable.name, mostStable.stability));
        }
        
        // 推荐建议
        System.out.println("\n📊 使用建议:");
        System.out.println("• 追求高性能: 选择" + (best != null ? best.name : "未知"));
        System.out.println("• 要求稳定性: 选择" + (mostStable != null ? mostStable.name : "未知"));
        System.out.println("• 实时应用: 考虑计算效率和内存占用");
    }
    
    /**
     * 训练智能体辅助方法
     */
    private void trainAgent(Agent agent, Environment environment, int episodes) {
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            int steps = 0;
            
            while (!environment.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                if (agent instanceof REINFORCEAgent && result.isDone()) {
                    ((REINFORCEAgent) agent).learnFromEpisode();
                }
                
                state = result.getNextState();
                steps++;
            }
        }
    }
    
    /**
     * 性能指标数据类
     */
    private static class PerformanceMetrics {
        String name;
        float avgReward;
        float stability;
        float successRate;
        float avgLoss;
        
        public PerformanceMetrics(String name, float avgReward, float stability, float successRate, float avgLoss) {
            this.name = name;
            this.avgReward = avgReward;
            this.stability = stability;
            this.successRate = successRate;
            this.avgLoss = avgLoss;
        }
    }
}