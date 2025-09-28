package io.leavesfly.tinyai.rl.environment;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * GridWorld（网格世界）环境实现
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * GridWorldEnvironment实现了经典的网格世界导航问题。
 * 智能体在网格中移动，目标是从起始位置到达目标位置，同时避开障碍物。
 * 这是强化学习领域的经典导航问题，适合测试路径规划和策略学习算法。
 * 
 * 状态空间：智能体的位置坐标 [x, y]
 * 动作空间：4个方向 [上, 下, 左, 右]
 * 奖励：到达目标+10，碰撞障碍物-1，每步-0.01（鼓励快速到达）
 */
public class GridWorldEnvironment extends Environment {
    
    // 环境配置
    private final int gridWidth;     // 网格宽度
    private final int gridHeight;    // 网格高度
    private final boolean[] obstacles; // 障碍物位置
    private final int startX, startY; // 起始位置
    private final int goalX, goalY;   // 目标位置
    
    // 当前位置
    private int agentX;
    private int agentY;
    
    // 动作定义：0=上, 1=下, 2=左, 3=右
    private static final int ACTION_UP = 0;
    private static final int ACTION_DOWN = 1;
    private static final int ACTION_LEFT = 2;
    private static final int ACTION_RIGHT = 3;
    
    // 奖励设置
    private static final float REWARD_GOAL = 10.0f;      // 到达目标的奖励
    private static final float REWARD_OBSTACLE = -1.0f;  // 碰撞障碍物的惩罚
    private static final float REWARD_STEP = -0.01f;     // 每步的小惩罚
    private static final float REWARD_OUT_OF_BOUNDS = -1.0f; // 越界惩罚
    
    private final Random random;
    
    /**
     * 构造函数 - 创建简单的网格世界
     * 
     * @param width 网格宽度
     * @param height 网格高度
     */
    public GridWorldEnvironment(int width, int height) {
        this(width, height, 0, 0, width - 1, height - 1, new boolean[width * height]);
    }
    
    /**
     * 完整构造函数
     * 
     * @param width 网格宽度
     * @param height 网格高度
     * @param startX 起始X坐标
     * @param startY 起始Y坐标
     * @param goalX 目标X坐标
     * @param goalY 目标Y坐标
     * @param obstacles 障碍物数组（一维，按行优先顺序）
     */
    public GridWorldEnvironment(int width, int height, int startX, int startY, 
                               int goalX, int goalY, boolean[] obstacles) {
        super(2, 4, 1000); // 2维状态（位置），4个动作，最大1000步
        
        this.gridWidth = width;
        this.gridHeight = height;
        this.startX = startX;
        this.startY = startY;
        this.goalX = goalX;
        this.goalY = goalY;
        this.obstacles = obstacles.clone();
        this.random = new Random();
        
        validateConfiguration();
    }
    
    /**
     * 验证环境配置的有效性
     */
    private void validateConfiguration() {
        if (gridWidth <= 0 || gridHeight <= 0) {
            throw new IllegalArgumentException("网格尺寸必须为正数");
        }
        
        if (obstacles.length != gridWidth * gridHeight) {
            throw new IllegalArgumentException("障碍物数组大小不匹配");
        }
        
        if (!isValidPosition(startX, startY) || !isValidPosition(goalX, goalY)) {
            throw new IllegalArgumentException("起始或目标位置超出网格范围");
        }
        
        if (isObstacle(startX, startY) || isObstacle(goalX, goalY)) {
            throw new IllegalArgumentException("起始或目标位置不能是障碍物");
        }
    }
    
    @Override
    public Variable reset() {
        agentX = startX;
        agentY = startY;
        currentStep = 0;
        done = false;
        
        currentState = getStateVariable();
        return currentState;
    }
    
    @Override
    public StepResult step(Variable action) {
        if (done) {
            throw new IllegalStateException("环境已结束，请先调用reset()");
        }
        
        int actionValue = (int) action.getValue().getNumber().floatValue();
        if (!isValidAction(action)) {
            throw new IllegalArgumentException("无效动作：" + actionValue + "，必须是0-3");
        }
        
        // 计算新位置
        int newX = agentX;
        int newY = agentY;
        
        switch (actionValue) {
            case ACTION_UP:
                newY = agentY - 1;
                break;
            case ACTION_DOWN:
                newY = agentY + 1;
                break;
            case ACTION_LEFT:
                newX = agentX - 1;
                break;
            case ACTION_RIGHT:
                newX = agentX + 1;
                break;
        }
        
        // 计算奖励并更新位置
        float reward = REWARD_STEP; // 基础步骤惩罚
        
        if (!isValidPosition(newX, newY)) {
            // 越界，保持原位置
            reward = REWARD_OUT_OF_BOUNDS;
        } else if (isObstacle(newX, newY)) {
            // 碰撞障碍物，保持原位置
            reward = REWARD_OBSTACLE;
        } else {
            // 有效移动
            agentX = newX;
            agentY = newY;
            
            // 检查是否到达目标
            if (agentX == goalX && agentY == goalY) {
                reward = REWARD_GOAL;
                done = true;
            }
        }
        
        currentStep++;
        
        // 检查是否超过最大步数
        if (currentStep >= maxSteps) {
            done = true;
        }
        
        // 更新当前状态
        currentState = getStateVariable();
        
        // 创建附加信息
        Map<String, Object> info = new HashMap<>();
        info.put("agent_x", agentX);
        info.put("agent_y", agentY);
        info.put("goal_reached", (agentX == goalX && agentY == goalY));
        info.put("action_name", getActionName(actionValue));
        
        return new StepResult(currentState, reward, done, info);
    }
    
    /**
     * 检查位置是否有效
     * 
     * @param x X坐标
     * @param y Y坐标
     * @return 是否有效
     */
    private boolean isValidPosition(int x, int y) {
        return x >= 0 && x < gridWidth && y >= 0 && y < gridHeight;
    }
    
    /**
     * 检查位置是否是障碍物
     * 
     * @param x X坐标
     * @param y Y坐标
     * @return 是否是障碍物
     */
    private boolean isObstacle(int x, int y) {
        if (!isValidPosition(x, y)) {
            return true; // 越界视为障碍物
        }
        return obstacles[y * gridWidth + x];
    }
    
    /**
     * 将当前位置转换为状态Variable
     * 
     * @return 状态Variable
     */
    private Variable getStateVariable() {
        float[] stateArray = {(float) agentX, (float) agentY};
        return new Variable(NdArray.of(stateArray, Shape.of(1, 2)));
    }

    @Override
    public Variable sampleAction() {
        int randomAction = random.nextInt(4); // 0, 1, 2, 3
        return new Variable(NdArray.of(randomAction));
    }

    @Override
    public boolean isValidAction(Variable action) {
        float actionValue = action.getValue().getNumber().floatValue();
        return actionValue >= 0.0f && actionValue <= 3.0f && actionValue == (int) actionValue;
    }

    @Override
    public void render() {
        System.out.println("=== GridWorld Step " + currentStep + " ===");

        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                if (x == agentX && y == agentY) {
                    System.out.print(" A "); // Agent
                } else if (x == goalX && y == goalY) {
                    System.out.print(" G "); // Goal
                } else if (isObstacle(x, y)) {
                    System.out.print(" # "); // Obstacle
                } else {
                    System.out.print(" . "); // Empty
                }
            }
            System.out.println();
        }

        System.out.println("位置: (" + agentX + ", " + agentY + ")");
        System.out.println("目标: (" + goalX + ", " + goalY + ")");
        System.out.println("步数: " + currentStep + "/" + maxSteps);
        System.out.println("完成: " + done);
        System.out.println();
    }

    /**
     * 获取动作名称
     *
     * @param action 动作值
     * @return 动作名称
     */
    private String getActionName(int action) {
        switch (action) {
            case ACTION_UP: return "UP";
            case ACTION_DOWN: return "DOWN";
            case ACTION_LEFT: return "LEFT";
            case ACTION_RIGHT: return "RIGHT";
            default: return "UNKNOWN";
        }
    }

    /**
     * 获取状态的独热编码表示
     *
     * @return 独热编码状态
     */
    public Variable getOneHotState() {
        int totalStates = gridWidth * gridHeight;
        float[] oneHot = new float[totalStates];
        int stateIndex = agentY * gridWidth + agentX;
        oneHot[stateIndex] = 1.0f;

        return new Variable(NdArray.of(oneHot, Shape.of(1, totalStates)));
    }
    
    /**
     * 创建带障碍物的网格世界
     * 
     * @param width 宽度
     * @param height 高度
     * @param obstacleRatio 障碍物比例
     * @return 网格世界环境
     */
    public static GridWorldEnvironment createWithRandomObstacles(int width, int height, float obstacleRatio) {
        boolean[] obstacles = new boolean[width * height];
        Random random = new Random();
        
        int startIndex = 0; // 左上角
        int goalIndex = width * height - 1; // 右下角
        
        for (int i = 0; i < obstacles.length; i++) {
            if (i != startIndex && i != goalIndex && random.nextFloat() < obstacleRatio) {
                obstacles[i] = true;
            }
        }
        
        return new GridWorldEnvironment(width, height, 0, 0, width - 1, height - 1, obstacles);
    }
    
    /**
     * 创建简单的迷宫环境
     * 
     * @return 迷宫环境
     */
    public static GridWorldEnvironment createSimpleMaze() {
        int width = 5;
        int height = 5;
        boolean[] obstacles = new boolean[width * height];
        
        // 创建一个简单的迷宫布局
        // # 表示障碍物，. 表示空地
        // . . # . G
        // . # # . .
        // . . . . #
        // # . # . .
        // A . . . .
        
        obstacles[2] = true;  // (2,0)
        obstacles[6] = true;  // (1,1)
        obstacles[7] = true;  // (2,1)
        obstacles[19] = true; // (4,2)
        obstacles[15] = true; // (0,3)
        obstacles[17] = true; // (2,3)
        
        return new GridWorldEnvironment(width, height, 0, 4, 4, 0, obstacles);
    }
    
    /**
     * 获取网格尺寸信息
     * 
     * @return 包含宽度和高度的映射
     */
    public Map<String, Integer> getGridSize() {
        Map<String, Integer> size = new HashMap<>();
        size.put("width", gridWidth);
        size.put("height", gridHeight);
        return size;
    }
    
    /**
     * 获取智能体当前位置
     * 
     * @return 包含x和y坐标的映射
     */
    public Map<String, Integer> getAgentPosition() {
        Map<String, Integer> position = new HashMap<>();
        position.put("x", agentX);
        position.put("y", agentY);
        return position;
    }
    
    /**
     * 获取目标位置
     * 
     * @return 包含x和y坐标的映射
     */
    public Map<String, Integer> getGoalPosition() {
        Map<String, Integer> position = new HashMap<>();
        position.put("x", goalX);
        position.put("y", goalY);
        return position;
    }
    
    /**
     * 计算曼哈顿距离到目标
     * 
     * @return 曼哈顿距离
     */
    public int getManhattanDistanceToGoal() {
        return Math.abs(agentX - goalX) + Math.abs(agentY - goalY);
    }
}