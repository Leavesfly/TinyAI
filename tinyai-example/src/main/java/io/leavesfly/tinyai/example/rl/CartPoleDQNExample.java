package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;

import java.util.Map;

/**
 * CartPole环境下使用DQN算法的示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 这个示例展示了如何使用DQN算法来解决CartPole（倒立摆）问题。
 * CartPole是强化学习的经典控制问题，目标是通过控制小车的左右移动来平衡杆子。
 * <p>
 * 主要学习内容：
 * 1. DQN算法的基本使用
 * 2. 环境与智能体的交互流程
 * 3. 经验回放机制的应用
 * 4. 训练过程的监控和评估
 */
public class CartPoleDQNExample {

    public static void main(String[] args) {
        System.out.println("=== CartPole DQN 训练示例 ===");

        // 训练参数
        int numEpisodes = 1000;          // 训练回合数
        int maxStepsPerEpisode = 500;    // 每回合最大步数
        int evaluationInterval = 100;    // 评估间隔

        // 创建环境
        Environment env = new CartPoleEnvironment(12345L); // 使用固定种子保证可重现性

        // 创建DQN智能体
        DQNAgent agent = createDQNAgent(env);

        // 训练智能体
        trainAgent(agent, env, numEpisodes, maxStepsPerEpisode, evaluationInterval);

        // 最终评估
        System.out.println("\n=== 最终评估 ===");
        evaluateAgent(agent, env, 10);

        System.out.println("训练完成！");
    }

    /**
     * 创建DQN智能体
     *
     * @param env 环境
     * @return DQN智能体
     */
    private static DQNAgent createDQNAgent(Environment env) {
        // 网络参数
        int stateDim = env.getStateDim();           // 状态维度：4
        int actionDim = env.getActionDim();         // 动作维度：2
        int[] hiddenSizes = {128, 128};             // 隐藏层尺寸

        // 算法参数
        float learningRate = 0.001f;                // 学习率
        float epsilon = 1.0f;                       // 初始探索率
        float gamma = 0.99f;                        // 折扣因子
        int batchSize = 32;                         // 批次大小
        int bufferSize = 10000;                     // 经验回放缓冲区大小
        int targetUpdateFreq = 100;                 // 目标网络更新频率

        System.out.println("创建DQN智能体...");
        System.out.println("状态维度: " + stateDim);
        System.out.println("动作维度: " + actionDim);
        System.out.println("网络结构: " + stateDim + " -> " + hiddenSizes[0] + " -> " + hiddenSizes[1] + " -> " + actionDim);
        System.out.println("学习率: " + learningRate);
        System.out.println("初始探索率: " + epsilon);

        return new DQNAgent(
                "CartPole_DQN",
                stateDim, actionDim, hiddenSizes,
                learningRate, epsilon, gamma,
                batchSize, bufferSize, targetUpdateFreq
        );
    }

    /**
     * 训练智能体
     *
     * @param agent              智能体
     * @param env                环境
     * @param numEpisodes        训练回合数
     * @param maxStepsPerEpisode 每回合最大步数
     * @param evaluationInterval 评估间隔
     */
    private static void trainAgent(DQNAgent agent, Environment env, int numEpisodes,
                                   int maxStepsPerEpisode, int evaluationInterval) {
        System.out.println("\n开始训练...");

        for (int episode = 0; episode < numEpisodes; episode++) {
            // 重置环境
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;

            for (int step = 0; step < maxStepsPerEpisode; step++) {
                // 选择动作
                Variable action = agent.selectAction(state);

                // 执行动作
                Environment.StepResult result = env.step(action);
                Variable nextState = result.getNextState();
                float reward = result.getReward();
                boolean done = result.isDone();

                // 存储经验并学习
                Experience experience = new Experience(state, action, reward, nextState, done, step);
                agent.learn(experience);

                // 更新状态和累积奖励
                state = nextState;
                episodeReward += reward;
                steps++;

                if (done) {
                    break;
                }
            }

            // 打印训练进度
            if (episode % 50 == 0 || episode == numEpisodes - 1) {
                Map<String, Object> stats = agent.getTrainingStats();
                System.out.printf("Episode %d: 奖励=%.2f, 步数=%d, Epsilon=%.3f, 损失=%.6f, 缓冲区使用率=%.2f%%\n",
                        episode, episodeReward, steps,
                        (Float) stats.get("epsilon"),
                        (Float) stats.get("average_loss"),
                        (Float) stats.get("buffer_usage") * 100);
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
     * @param agent                 智能体
     * @param env                   环境
     * @param numEvaluationEpisodes 评估回合数
     */
    private static void evaluateAgent(DQNAgent agent, Environment env, int numEvaluationEpisodes) {
        // 切换到评估模式
        agent.setTraining(false);

        float totalReward = 0.0f;
        int totalSteps = 0;
        int successfulEpisodes = 0;

        for (int episode = 0; episode < numEvaluationEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;

            for (int step = 0; step < 500; step++) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);

                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;

                if (result.isDone()) {
                    break;
                }
            }

            totalReward += episodeReward;
            totalSteps += steps;

            if (episodeReward >= 450) { // 认为450+步为成功
                successfulEpisodes++;
            }

            System.out.printf("评估回合 %d: 奖励=%.2f, 步数=%d\n", episode + 1, episodeReward, steps);
        }

        // 计算平均性能
        float averageReward = totalReward / numEvaluationEpisodes;
        float averageSteps = (float) totalSteps / numEvaluationEpisodes;
        float successRate = (float) successfulEpisodes / numEvaluationEpisodes * 100;

        System.out.println("评估结果:");
        System.out.printf("  平均奖励: %.2f\n", averageReward);
        System.out.printf("  平均步数: %.2f\n", averageSteps);
        System.out.printf("  成功率: %.1f%% (%d/%d)\n", successRate, successfulEpisodes, numEvaluationEpisodes);

        // 切换回训练模式
        agent.setTraining(true);
    }

    /**
     * 演示单回合详细过程
     *
     * @param agent 智能体
     * @param env   环境
     */
    private static void demonstrateEpisode(DQNAgent agent, Environment env) {
        System.out.println("\n=== 单回合演示 ===");

        agent.setTraining(false);
        Variable state = env.reset();

        System.out.println("初始状态:");
        env.render();

        for (int step = 0; step < 20; step++) { // 只演示前20步
            Variable action = agent.selectAction(state);
            Environment.StepResult result = env.step(action);

            String actionName = ((int) action.getValue().getNumber().floatValue() == 0) ? "LEFT" : "RIGHT";
            System.out.printf("步骤 %d: 动作=%s, 奖励=%.2f\n", step + 1, actionName, result.getReward());

            if (step % 5 == 0) { // 每5步渲染一次
                env.render();
            }

            state = result.getNextState();

            if (result.isDone()) {
                System.out.println("回合结束!");
                break;
            }
        }

        agent.setTraining(true);
    }
}