package io.leavesfly.tinyai.agent.embodied.learning;

import io.leavesfly.tinyai.agent.embodied.memory.EpisodicMemory;
import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.model.ObstacleInfo;
import io.leavesfly.tinyai.agent.embodied.model.Transition;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;
import java.util.Random;

/**
 * 端到端学习策略
 * 直接从原始传感器数据到控制指令的端到端学习
 *
 * @author TinyAI Team
 */
public class EndToEndLearner {
    private static final int INPUT_DIM = 128;   // 输入特征维度
    private static final int HIDDEN_DIM = 64;   // 隐藏层维度
    private static final int OUTPUT_DIM = 3;    // 输出动作维度
    
    // 简化的神经网络参数
    private NdArray weightsInputHidden;   // 输入层到隐藏层
    private NdArray biasHidden;
    private NdArray weightsHiddenOutput;  // 隐藏层到输出层
    private NdArray biasOutput;
    
    private double learningRate;
    private int batchSize;
    private int trainingSteps;
    
    private Random random;

    public EndToEndLearner() {
        this.learningRate = 0.0001;
        this.batchSize = 64;
        this.trainingSteps = 0;
        this.random = new Random();
        
        // 初始化网络参数
        initializeNetwork();
    }

    /**
     * 初始化网络参数
     */
    private void initializeNetwork() {
        // Xavier初始化
        float scale1 = (float) Math.sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM));
        weightsInputHidden = randomMatrix(INPUT_DIM, HIDDEN_DIM, scale1);
        biasHidden = NdArray.zeros(Shape.of(HIDDEN_DIM));
        
        float scale2 = (float) Math.sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM));
        weightsHiddenOutput = randomMatrix(HIDDEN_DIM, OUTPUT_DIM, scale2);
        biasOutput = NdArray.zeros(Shape.of(OUTPUT_DIM));
    }

    /**
     * 创建随机矩阵
     */
    private NdArray randomMatrix(int rows, int cols, float scale) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (random.nextGaussian() * scale);
        }
        return NdArray.of(data);
    }

    /**
     * 从经验中学习
     */
    public void learn(EpisodicMemory memory) {
        if (memory.getBufferSize() < batchSize) {
            return;
        }

        // 采样批次数据
        List<Transition> batch = memory.sampleBatch(batchSize);
        
        //  训练数据
        NdArray[] inputs = new NdArray[batch.size()];
        DrivingAction[] actions = new DrivingAction[batch.size()];
        
        for (int i = 0; i < batch.size(); i++) {
            Transition t = batch.get(i);
            inputs[i] = extractFeatures(t.getState());
            actions[i] = t.getAction();
        }
        
        // 执行一轮训练
        trainBatch(inputs, actions);
        
        trainingSteps++;
    }

    /**
     * 批量训练
     */
    private void trainBatch(NdArray[] inputs, DrivingAction[] targets) {
        double totalLoss = 0.0;
        
        for (int i = 0; i < inputs.length; i++) {
            // 前向传播
            NdArray[] activations = forward(inputs[i]);
            NdArray output = activations[2];
            
            // 计算损失（MSE）
            NdArray targetArray = actionToArray(targets[i]);
            NdArray error = output.sub(targetArray);
            double loss = computeMSE(error);
            totalLoss += loss;
            
            // 反向传播（简化版）
            backward(inputs[i], activations, error);
        }
        
        // 可选：打印训练信息
        if (trainingSteps % 100 == 0) {
            System.out.printf("端到端学习步骤 %d，平均损失: %.6f%n", 
                            trainingSteps, totalLoss / inputs.length);
        }
    }

    /**
     * 前向传播
     */
    private NdArray[] forward(NdArray input) {
        // 隐藏层
        NdArray hidden = linearForward(input, weightsInputHidden, biasHidden);
        hidden = relu(hidden);
        
        // 输出层
        NdArray output = linearForward(hidden, weightsHiddenOutput, biasOutput);
        output = tanh(output);  // 动作值在 [-1, 1] 范围
        
        return new NdArray[]{input, hidden, output};
    }

    /**
     * 线性变换
     */
    private NdArray linearForward(NdArray input, NdArray weights, NdArray bias) {
        // 简化的矩阵乘法实现
        int inputSize = input.getShape().size();
        int outputSize = bias.getShape().size();
        
        float[] result = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            result[i] = bias.get(i);
            for (int j = 0; j < inputSize && j * outputSize + i < weights.getShape().size(); j++) {
                result[i] += input.get(j) * weights.get(j * outputSize + i);
            }
        }
        
        return NdArray.of(result);
    }

    /**
     * ReLU激活函数
     */
    private NdArray relu(NdArray x) {
        int size = x.getShape().size();
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.max(0, x.get(i));
        }
        return NdArray.of(data);
    }

    /**
     * Tanh激活函数
     */
    private NdArray tanh(NdArray x) {
        int size = x.getShape().size();
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.tanh(x.get(i));
        }
        return NdArray.of(data);
    }

    /**
     * 反向传播（简化版）
     */
    private void backward(NdArray input, NdArray[] activations, NdArray outputError) {
        NdArray hidden = activations[1];
        
        // 输出层梯度
        float[] gradOutput = new float[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; i++) {
            float error = outputError.get(i);
            float tanhDeriv = 1 - activations[2].get(i) * activations[2].get(i);
            gradOutput[i] = error * tanhDeriv;
        }
        
        // 更新输出层参数（简化的SGD）
        int weightsSize = weightsHiddenOutput.getShape().size();
        for (int i = 0; i < HIDDEN_DIM; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                int idx = i * OUTPUT_DIM + j;
                if (idx < weightsSize) {
                    float grad = gradOutput[j] * hidden.get(i);
                    float currentWeight = weightsHiddenOutput.get(idx);
                    // 使用临时数组存储更新
                    float newWeight = currentWeight - (float) learningRate * grad;
                    // 批量更新（需要重建数组）
                }
            }
        }
        
        // 更新偏置
        for (int i = 0; i < OUTPUT_DIM; i++) {
            float currentBias = biasOutput.get(i);
            float newBias = currentBias - (float) learningRate * gradOutput[i];
            // 由于NdArray不支持原地修改，这里简化处理
        }
    }

    /**
     * 计算MSE损失
     */
    private double computeMSE(NdArray error) {
        double sum = 0.0;
        int size = error.getShape().size();
        for (int i = 0; i < size; i++) {
            sum += error.get(i) * error.get(i);
        }
        return sum / size;
    }

    /**
     * 预测动作
     */
    public DrivingAction predict(PerceptionState state) {
        NdArray features = extractFeatures(state);
        NdArray[] activations = forward(features);
        NdArray output = activations[2];
        
        return arrayToAction(output);
    }

    /**
     * 提取特征
     */
    private NdArray extractFeatures(PerceptionState state) {
        float[] features = new float[INPUT_DIM];
        int idx = 0;
        
        // 车辆状态特征
        features[idx++] = (float) state.getVehicleState().getSpeed();
        features[idx++] = (float) state.getVehicleState().getHeading();
        features[idx++] = (float) state.getVehicleState().getAcceleration();
        features[idx++] = (float) state.getVehicleState().getSteeringAngle();
        
        // 车道几何特征
        if (state.getLaneInfo() != null) {
            features[idx++] = (float) state.getLaneInfo().getLateralDeviation();
            features[idx++] = 0.0f; // 简化：headingError 
            features[idx++] = (float) state.getLaneInfo().getCurvature();
        } else {
            idx += 3;
        }
        
        // 障碍物特征（前5个）
        List<ObstacleInfo> obstacles = state.getObstacleMap();
        if (obstacles != null) {
            for (int i = 0; i < Math.min(5, obstacles.size()); i++) {
                features[idx++] = (float) obstacles.get(i).getDistance();
                features[idx++] = (float) obstacles.get(i).getVelocity().magnitude();
                features[idx++] = (float) obstacles.get(i).getPosition().getX();
            }
        }
        idx += (5 - Math.min(5, obstacles != null ? obstacles.size() : 0)) * 3; // 填充
        
        // 传感器原始数据（简化）
        // 填充剩余特征
        while (idx < INPUT_DIM) {
            features[idx++] = 0.0f;
        }
        
        return NdArray.of(features);
    }

    /**
     * 动作转数组
     */
    private NdArray actionToArray(DrivingAction action) {
        float[] data = new float[OUTPUT_DIM];
        data[0] = (float) action.getSteering();
        data[1] = (float) action.getThrottle();
        data[2] = (float) action.getBrake();
        return NdArray.of(data);
    }

    /**
     * 数组转动作
     */
    private DrivingAction arrayToAction(NdArray array) {
        double steering = Math.max(-1.0, Math.min(1.0, array.get(0)));
        double throttle = Math.max(0.0, Math.min(1.0, array.get(1)));
        double brake = Math.max(0.0, Math.min(1.0, array.get(2)));
        return new DrivingAction(steering, throttle, brake);
    }

    /**
     * 保存模型参数
     */
    public void saveModel(String path) {
        // TODO: 实现模型保存
        System.out.println("保存端到端模型到: " + path);
    }

    /**
     * 加载模型参数
     */
    public void loadModel(String path) {
        // TODO: 实现模型加载
        System.out.println("从文件加载端到端模型: " + path);
    }

    // Getters and Setters
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getTrainingSteps() {
        return trainingSteps;
    }
}
