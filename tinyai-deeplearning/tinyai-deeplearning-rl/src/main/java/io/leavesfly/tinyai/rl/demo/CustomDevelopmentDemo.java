package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.rl.*;
import io.leavesfly.tinyai.rl.agent.BanditAgent;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 自定义开发完整演示
 *
 * 本演示展示如何扩展TinyAI RL框架:
 * - 自定义环境: 创建新的强化学习环境
 * - 自定义智能体: 实现新的学习算法
 * - 集成使用: 将自定义组件整合到框架中
 *
 * 场景: 简单的迷宫寻宝游戏
 * - 5x5网格迷宫
 * - 智能体需要找到宝藏
 * - 避开陷阱
 *
 * 运行方式:
 *   mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.CustomDevelopmentDemo" -pl tinyai-deeplearning-rl
 */
public class CustomDevelopmentDemo {

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("       自定义强化学习环境与智能体演示       ");
        System.out.println("==========================================\n");

        demonstrateCustomEnvironment();
        demonstrateCustomAgent();
        demonstrateIntegration();

        System.out.println("\n==========================================");
        System.out.println("          自定义开发演示完成!             ");
        System.out.println("==========================================");
    }

    /**
     * 演示自定义环境
     */
    private static void demonstrateCustomEnvironment() {
        System.out.println("【步骤1: 创建自定义环境】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("场景: 5x5迷宫寻宝\n");

        TreasureMazeEnvironment env = new TreasureMazeEnvironment();
        
        System.out.println("迷宫布局:");
        env.render();
        
        System.out.println("\n环境特性:");
        System.out.println("  状态维度: " + env.getStateDim() + " (x坐标 + y坐标)");
        System.out.println("  动作维度: " + env.getActionDim() + " (上下左右)");
        
        System.out.println("\n奖励设计:");
        System.out.println("  • 找到宝藏: +10");
        System.out.println("  • 踩到陷阱: -5");
        System.out.println("  • 每步移动: -0.1");
        System.out.println("  • 到达边界: -1\n");
    }

    /**
     * 演示自定义智能体
     */
    private static void demonstrateCustomAgent() {
        System.out.println("【步骤2: 创建自定义智能体】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("算法: 简单Q-Learning (表格型)\n");

        QLearningAgent agent = new QLearningAgent(
            "Q-Learning探险家",
            2,  // 状态维度
            4,  // 动作维度  
            0.1f,  // 学习率
            0.1f,  // 探索率
            0.9f   // 折扣因子
        );

        System.out.println("智能体特性:");
        System.out.println("  算法类型: Q-Learning");
        System.out.println("  Q表结构: 25个状态 × 4个动作");
        System.out.println("  学习方式: 时序差分学习");
        System.out.println("  策略: ε-贪心\n");
    }

    /**
     * 演示集成使用
     */
    private static void demonstrateIntegration() {
        System.out.println("【步骤3: 训练自定义智能体】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        TreasureMazeEnvironment env = new TreasureMazeEnvironment();
        QLearningAgent agent = new QLearningAgent("探险家", 2, 4, 0.1f, 0.1f, 0.9f);

        System.out.println("训练进度:");
        System.out.println("回合 | 步数 | 奖励 | 结果");
        System.out.println("-----|------|------|----------");

        int maxEpisodes = 100;
        int successCount = 0;

        for (int episode = 0; episode < maxEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0;
            int steps = 0;

            while (!env.isDone() && steps < 50) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);

                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                agent.learn(experience);

                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }

            if (episodeReward > 5) {
                successCount++;
            }

            if (episode < 10 || (episode + 1) % 20 == 0) {
                String result = episodeReward > 5 ? "找到宝藏✓" : "未找到";
                System.out.printf(" %3d | %3d  | %5.1f | %s\n",
                    episode + 1, steps, episodeReward, result);
            }
        }

        System.out.println("\n训练结果:");
        System.out.println("  成功次数: " + successCount + "/" + maxEpisodes);
        System.out.println("  成功率: " + String.format("%.1f%%", (float) successCount / maxEpisodes * 100));

        System.out.println("\n学到的策略(最后一次):");
        env.reset();
        env.render();

        System.out.println("\n【关键代码模板】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("\n1. 自定义环境模板:");
        System.out.println("```java");
        System.out.println("public class CustomEnvironment extends Environment {");
        System.out.println("    public CustomEnvironment() {");
        System.out.println("        super(stateDim, actionDim, maxSteps);");
        System.out.println("    }");
        System.out.println("    ");
        System.out.println("    @Override");
        System.out.println("    public Variable reset() {");
        System.out.println("        // 重置环境到初始状态");
        System.out.println("        return initialState;");
        System.out.println("    }");
        System.out.println("    ");
        System.out.println("    @Override");
        System.out.println("    public StepResult step(Variable action) {");
        System.out.println("        // 执行动作,计算下一状态和奖励");
        System.out.println("        return new StepResult(nextState, reward, done, info);");
        System.out.println("    }");
        System.out.println("}");
        System.out.println("```");

        System.out.println("\n2. 自定义智能体模板:");
        System.out.println("```java");
        System.out.println("public class CustomAgent implements Agent {");
        System.out.println("    @Override");
        System.out.println("    public Variable selectAction(Variable state) {");
        System.out.println("        // 实现动作选择逻辑");
        System.out.println("        return selectedAction;");
        System.out.println("    }");
        System.out.println("    ");
        System.out.println("    @Override");
        System.out.println("    public void learn(Experience experience) {");
        System.out.println("        // 实现学习更新逻辑");
        System.out.println("    }");
        System.out.println("}");
        System.out.println("```");

        System.out.println("\n💡 扩展建议:");
        System.out.println("  • 参考现有环境实现: CartPoleEnvironment, GridWorldEnvironment");
        System.out.println("  • 参考现有智能体实现: DQNAgent, REINFORCEAgent");
        System.out.println("  • 遵循接口规范,确保兼容性");
        System.out.println("  • 添加详细注释,便于维护");
    }

    /**
     * 自定义环境: 寻宝迷宫
     */
    private static class TreasureMazeEnvironment extends Environment {
        private static final int SIZE = 5;
        private int[] agentPos;
        private int[] treasurePos;
        private int[] trapPos;
        private Random random;

        public TreasureMazeEnvironment() {
            super(2, 4, 50);
            this.random = new Random();
            this.treasurePos = new int[]{4, 4};  // 宝藏在右下角
            this.trapPos = new int[]{2, 2};      // 陷阱在中间
        }

        @Override
        public Variable reset() {
            this.agentPos = new int[]{0, 0};  // 起点在左上角
            this.done = false;
            this.currentStep = 0;
            return new Variable(NdArray.of(new float[]{agentPos[0], agentPos[1]}));
        }

        @Override
        public StepResult step(Variable action) {
            int actionValue = (int) action.getValue().getNumber().floatValue();
            
            // 移动: 0=上, 1=下, 2=左, 3=右
            int[] newPos = agentPos.clone();
            switch (actionValue) {
                case 0: newPos[1] = Math.max(0, newPos[1] - 1); break;  // 上
                case 1: newPos[1] = Math.min(SIZE - 1, newPos[1] + 1); break;  // 下
                case 2: newPos[0] = Math.max(0, newPos[0] - 1); break;  // 左
                case 3: newPos[0] = Math.min(SIZE - 1, newPos[0] + 1); break;  // 右
            }

            float reward = -0.1f;  // 移动惩罚
            boolean reachedGoal = false;

            // 检查边界惩罚
            if (newPos[0] == agentPos[0] && newPos[1] == agentPos[1]) {
                reward = -1.0f;  // 撞墙
            }

            agentPos = newPos;

            // 检查宝藏
            if (agentPos[0] == treasurePos[0] && agentPos[1] == treasurePos[1]) {
                reward = 10.0f;
                done = true;
                reachedGoal = true;
            }

            // 检查陷阱
            if (agentPos[0] == trapPos[0] && agentPos[1] == trapPos[1]) {
                reward = -5.0f;
                done = true;
            }

            currentStep++;
            if (currentStep >= maxSteps) {
                done = true;
            }

            Variable nextState = new Variable(NdArray.of(new float[]{agentPos[0], agentPos[1]}));
            
            Map<String, Object> info = new HashMap<>();
            info.put("reachedGoal", reachedGoal);

            return new StepResult(nextState, reward, done, info);
        }

        @Override
        public Variable sampleAction() {
            return new Variable(NdArray.of(random.nextInt(4)));
        }

        @Override
        public boolean isValidAction(Variable action) {
            int actionValue = (int) action.getValue().getNumber().floatValue();
            return actionValue >= 0 && actionValue < 4;
        }

        @Override
        public void render() {
            System.out.println("  图例: A=智能体, T=宝藏, X=陷阱, .=空地");
            for (int y = 0; y < SIZE; y++) {
                System.out.print("  ");
                for (int x = 0; x < SIZE; x++) {
                    if (x == agentPos[0] && y == agentPos[1]) {
                        System.out.print("A ");
                    } else if (x == treasurePos[0] && y == treasurePos[1]) {
                        System.out.print("T ");
                    } else if (x == trapPos[0] && y == trapPos[1]) {
                        System.out.print("X ");
                    } else {
                        System.out.print(". ");
                    }
                }
                System.out.println();
            }
        }
    }

    /**
     * 自定义智能体: Q-Learning
     */
    private static class QLearningAgent implements Agent {
        private final String name;
        private final int stateDim;
        private final int actionDim;
        private final float learningRate;
        private float epsilon;
        private final float gamma;
        private int trainingStep;
        private boolean training;
        private Map<String, float[]> qTable;
        private Random random;

        public QLearningAgent(String name, int stateDim, int actionDim,
                            float learningRate, float epsilon, float gamma) {
            this.name = name;
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.learningRate = learningRate;
            this.epsilon = epsilon;
            this.gamma = gamma;
            this.trainingStep = 0;
            this.training = true;
            this.qTable = new HashMap<>();
            this.random = new Random();
        }

        @Override
        public Variable selectAction(Variable state) {
            String stateKey = getStateKey(state);
            
            if (!qTable.containsKey(stateKey)) {
                qTable.put(stateKey, new float[actionDim]);
            }

            // ε-贪心策略
            if (training && random.nextFloat() < epsilon) {
                return new Variable(NdArray.of(random.nextInt(actionDim)));
            } else {
                float[] qValues = qTable.get(stateKey);
                int bestAction = 0;
                for (int i = 1; i < actionDim; i++) {
                    if (qValues[i] > qValues[bestAction]) {
                        bestAction = i;
                    }
                }
                return new Variable(NdArray.of(bestAction));
            }
        }

        @Override
        public void learn(Experience experience) {
            String stateKey = getStateKey(experience.getState());
            String nextStateKey = getStateKey(experience.getNextState());
            
            if (!qTable.containsKey(stateKey)) {
                qTable.put(stateKey, new float[actionDim]);
            }
            if (!qTable.containsKey(nextStateKey)) {
                qTable.put(nextStateKey, new float[actionDim]);
            }

            int action = (int) experience.getAction().getValue().getNumber().floatValue();
            float reward = experience.getReward();
            
            // Q-Learning更新: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
            float[] qValues = qTable.get(stateKey);
            float[] nextQValues = qTable.get(nextStateKey);
            
            float maxNextQ = experience.isDone() ? 0 : getMaxQValue(nextQValues);
            float tdTarget = reward + gamma * maxNextQ;
            float tdError = tdTarget - qValues[action];
            
            qValues[action] += learningRate * tdError;
            trainingStep++;
        }

        @Override
        public void storeExperience(Experience experience) {
            // Q-Learning不需要经验回放
        }

        @Override
        public void learnBatch(Experience[] experiences) {
            for (Experience exp : experiences) {
                learn(exp);
            }
        }

        @Override
        public void loadModel(String modelPath) {
            // Q-Learning不支持模型加载
            throw new UnsupportedOperationException("Q-Learning不支持模型加载");
        }

        @Override
        public void saveModel(String modelPath) {
            // Q-Learning不支持模型保存
            throw new UnsupportedOperationException("Q-Learning不支持模型保存");
        }

        @Override
        public void setTraining(boolean training) {
            this.training = training;
        }

        @Override
        public boolean isTraining() {
            return training;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public int getStateDim() {
            return stateDim;
        }

        @Override
        public int getActionDim() {
            return actionDim;
        }

        @Override
        public int getTrainingStep() {
            return trainingStep;
        }

        @Override
        public void reset() {
            this.qTable.clear();
            this.trainingStep = 0;
        }

        private String getStateKey(Variable state) {
            float[] data = state.getValue().getArray();
            return String.format("%d,%d", (int) data[0], (int) data[1]);
        }

        private float getMaxQValue(float[] qValues) {
            float max = qValues[0];
            for (int i = 1; i < qValues.length; i++) {
                if (qValues[i] > max) {
                    max = qValues[i];
                }
            }
            return max;
        }
    }
}
