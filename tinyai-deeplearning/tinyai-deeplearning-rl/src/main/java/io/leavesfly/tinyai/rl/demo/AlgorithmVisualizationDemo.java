package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Environment;

/**
 * 算法可视化演示 - 展示DQN算法的内部工作原理
 * 
 * <p>本演示通过可视化展示:
 * <ul>
 *   <li>Q值的变化过程</li>
 *   <li>经验回放缓冲区的状态</li>
 *   <li>目标网络与主网络的差异</li>
 *   <li>损失函数的变化曲线</li>
 *   <li>探索率的衰减过程</li>
 * </ul>
 * 
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmVisualizationDemo" \
 *   -pl tinyai-deeplearning-rl
 * </pre>
 * 
 * @author TinyAI Team
 */
public class AlgorithmVisualizationDemo {

    private static final int MAX_EPISODES = 100;
    private static final int VISUALIZATION_INTERVAL = 10;
    
    // 存储训练数据用于可视化
    private static float[] episodeRewards = new float[MAX_EPISODES];
    private static float[] episodeLosses = new float[MAX_EPISODES];
    private static float[] epsilonHistory = new float[MAX_EPISODES];
    private static int currentEpisode = 0;

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("      DQN算法可视化演示 - 理解内部机制    ");
        System.out.println("==========================================\n");

        explainDQNArchitecture();
        
        DQNAgent agent = createVisualizedAgent();
        CartPoleEnvironment env = new CartPoleEnvironment(500);
        
        System.out.println("\n开始训练并可视化...\n");
        
        for (int episode = 0; episode < MAX_EPISODES; episode++) {
            currentEpisode = episode;
            float reward = trainEpisode(agent, env, episode);
            episodeRewards[episode] = reward;
            epsilonHistory[episode] = agent.getCurrentEpsilon();
            
            // 定期可视化
            if (episode % VISUALIZATION_INTERVAL == 0 || episode == MAX_EPISODES - 1) {
                visualizeTrainingProgress(agent, episode);
            }
        }
        
        showFinalSummary();
    }

    /**
     * 解释DQN架构
     */
    private static void explainDQNArchitecture() {
        System.out.println("【DQN架构可视化】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println();
        System.out.println("输入层(4维状态)          隐藏层1(64)           隐藏层2(64)         输出层(Q值)");
        System.out.println("┌─────────┐            ┌─────────┐           ┌─────────┐        ┌─────────┐");
        System.out.println("│ 位置    │───────────→│         │──────────→│         │───────→│ Q(左推) │");
        System.out.println("│ 速度    │───────────→│  ReLU   │──────────→│  ReLU   │───────→│         │");
        System.out.println("│ 角度    │───────────→│  激活   │──────────→│  激活   │───────→│ Q(右推) │");
        System.out.println("│ 角速度  │───────────→│         │──────────→│         │───────→│         │");
        System.out.println("└─────────┘            └─────────┘           └─────────┘        └─────────┘");
        System.out.println();
        System.out.println("关键组件:");
        System.out.println("  1. 主网络(Q-Network):  计算当前Q值");
        System.out.println("  2. 目标网络(Target):   计算目标Q值(稳定训练)");
        System.out.println("  3. 经验回放:          存储和采样历史经验");
        System.out.println("  4. ε-贪心策略:        平衡探索与利用");
        System.out.println();
    }

    /**
     * 创建带可视化的DQN智能体
     */
    private static DQNAgent createVisualizedAgent() {
        return new DQNAgent(
            "VisualizedDQN",
            4,                      // 状态维度
            2,                      // 动作维度
            new int[]{64, 64},      // 隐藏层
            0.001f,                 // 学习率
            1.0f,                   // 初始探索率
            0.99f,                  // 折扣因子
            32,                     // 批次大小
            10000,                  // 缓冲区大小
            100                     // 目标网络更新频率
        );
    }

    /**
     * 训练一个回合
     */
    private static float trainEpisode(DQNAgent agent, CartPoleEnvironment env, int episode) {
        Variable state = env.reset();
        float totalReward = 0;
        int steps = 0;
        float episodeLoss = 0;
        int lossCount = 0;

        while (!env.isDone() && steps < 500) {
            Variable action = agent.selectAction(state);
            Environment.StepResult result = env.step(action);
            
            Experience exp = new Experience(
                state, action, result.getReward(),
                result.getNextState(), result.isDone(), steps
            );
            
            agent.learn(exp);
            
            totalReward += result.getReward();
            state = result.getNextState();
            steps++;
        }

        return totalReward;
    }

    /**
     * 可视化训练进度
     */
    private static void visualizeTrainingProgress(DQNAgent agent, int episode) {
        System.out.println("\n【训练进度 - 回合 " + (episode + 1) + "/" + MAX_EPISODES + "】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 1. 奖励曲线
        System.out.println("\n1. 奖励曲线 (最近10回合):");
        visualizeRewardCurve(episode);
        
        // 2. 探索率衰减
        System.out.println("\n2. 探索率(ε)衰减:");
        visualizeEpsilonDecay(episode);
        
        // 3. 缓冲区使用情况
        System.out.println("\n3. 经验回放缓冲区:");
        visualizeBufferUsage(agent);
        
        // 4. 训练统计
        System.out.println("\n4. 训练统计:");
        System.out.printf("   当前探索率: %.3f%n", agent.getCurrentEpsilon());
        System.out.printf("   训练步数: %d%n", agent.getTrainingStep());
        
        java.util.Map<String, Object> stats = agent.getTrainingStats();
        System.out.printf("   平均损失: %.4f%n", stats.get("average_loss"));
        System.out.printf("   缓冲区使用率: %.1f%%%n", stats.get("buffer_usage"));
    }

    /**
     * 可视化奖励曲线
     */
    private static void visualizeRewardCurve(int episode) {
        int start = Math.max(0, episode - 9);
        float maxReward = 500;
        
        for (int i = start; i <= episode; i++) {
            int barLength = (int) (episodeRewards[i] / maxReward * 30);
            String bar = "█".repeat(barLength);
            System.out.printf("   回合%3d: %s %.0f%n", i + 1, bar, episodeRewards[i]);
        }
        
        // 计算平均奖励
        float avgReward = 0;
        int count = Math.min(10, episode + 1);
        for (int i = 0; i < count; i++) {
            avgReward += episodeRewards[episode - i];
        }
        avgReward /= count;
        System.out.printf("   最近%d回合平均: %.1f%n", count, avgReward);
    }

    /**
     * 可视化探索率衰减
     */
    private static void visualizeEpsilonDecay(int episode) {
        int start = Math.max(0, episode - 9);
        
        for (int i = start; i <= episode; i++) {
            int barLength = (int) (epsilonHistory[i] * 30);
            String bar = "█".repeat(barLength);
            System.out.printf("   回合%3d: %s %.3f%n", i + 1, bar, epsilonHistory[i]);
        }
    }

    /**
     * 可视化缓冲区使用
     */
    private static void visualizeBufferUsage(DQNAgent agent) {
        java.util.Map<String, Object> stats = agent.getTrainingStats();
        float usage = ((Number) stats.get("buffer_usage")).floatValue();
        int filled = (int) (usage * 30);
        int empty = 30 - filled;
        
        String bar = "█".repeat(filled) + "░".repeat(empty);
        System.out.printf("   [%s] %.1f%%%n", bar, usage * 100);
        System.out.println("   █ = 已存储经验  ░ = 空闲空间");
    }

    /**
     * 显示最终总结
     */
    private static void showFinalSummary() {
        System.out.println("\n\n【训练完成总结】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 计算统计数据
        float totalReward = 0;
        float maxReward = 0;
        float minReward = Float.MAX_VALUE;
        
        for (float r : episodeRewards) {
            totalReward += r;
            maxReward = Math.max(maxReward, r);
            minReward = Math.min(minReward, r);
        }
        
        float avgReward = totalReward / MAX_EPISODES;
        
        // 计算前10和后10回合的平均
        float first10Avg = 0, last10Avg = 0;
        for (int i = 0; i < 10; i++) {
            first10Avg += episodeRewards[i];
            last10Avg += episodeRewards[MAX_EPISODES - 10 + i];
        }
        first10Avg /= 10;
        last10Avg /= 10;
        
        System.out.printf("总回合数: %d%n", MAX_EPISODES);
        System.out.printf("平均奖励: %.1f%n", avgReward);
        System.out.printf("最高奖励: %.0f%n", maxReward);
        System.out.printf("最低奖励: %.0f%n", minReward);
        System.out.printf("前10回合平均: %.1f%n", first10Avg);
        System.out.printf("后10回合平均: %.1f%n", last10Avg);
        System.out.printf("学习进步: %.1f%% (%+.1f)%n", 
            (last10Avg - first10Avg) / first10Avg * 100, last10Avg - first10Avg);
        
        System.out.println("\n💡 学习洞察:");
        if (last10Avg > first10Avg * 1.5) {
            System.out.println("   ✓ 智能体显著学习到了有效策略!");
        } else if (last10Avg > first10Avg) {
            System.out.println("   ✓ 智能体有一定学习效果");
        } else {
            System.out.println("   ⚠ 学习效果不明显，可能需要调整参数");
        }
        
        System.out.println("\n==========================================");
        System.out.println("         可视化演示完成!                  ");
        System.out.println("==========================================");
    }
}
