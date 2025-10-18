package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.ProprioceptionInput;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * 本体感知编码器
 * 使用MLP将机器人关节状态编码为向量表示
 * 
 * @author TinyAI
 */
public class ProprioceptionEncoder extends Block {
    
    private final int inputDim;
    private final int hiddenDim;
    
    // MLP层
    private LinearLayer fc1;
    private LinearLayer fc2;
    private LinearLayer fc3;
    
    // 层归一化
    private LayerNorm norm1;
    private LayerNorm norm2;
    
    /**
     * 构造函数
     * 
     * @param inputDim 输入维度（关节数 * 2，包括位置和速度）
     * @param hiddenDim 隐藏层维度（与其他模态对齐）
     */
    public ProprioceptionEncoder(int inputDim, int hiddenDim) {
        super("ProprioceptionEncoder", null);
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        
        // 三层MLP：inputDim -> 256 -> 512 -> hiddenDim
        this.fc1 = new LinearLayer("fc1", inputDim, 256, true);
        this.norm1 = new LayerNorm("norm1", 256);
        
        this.fc2 = new LinearLayer("fc2", 256, 512, true);
        this.norm2 = new LayerNorm("norm2", 512);
        
        this.fc3 = new LinearLayer("fc3", 512, hiddenDim, true);
    }
    
    @Override
    public void init() {
        // 初始化已在构造函数中完成
    }
    
    /**
     * 编码本体感知输入
     * 
     * @param proprioInput 本体感知输入
     * @return 本体感知嵌入向量，维度 [1, hiddenDim]
     */
    public NdArray encode(ProprioceptionInput proprioInput) {
        // 拼接关节位置和速度
        NdArray jointPositions = proprioInput.getJointPositions();
        NdArray jointVelocities = proprioInput.getJointVelocities();
        
        int numJoints = jointPositions.getShape().getDimension(0);
        double[] stateVector = new double[numJoints * 2 + 1];
        
        // 复制关节位置
        for (int i = 0; i < numJoints; i++) {
            stateVector[i] = jointPositions.get(i);
        }
        
        // 复制关节速度
        for (int i = 0; i < numJoints; i++) {
            stateVector[numJoints + i] = jointVelocities.get(i);
        }
        
        // 添加夹爪状态
        stateVector[numJoints * 2] = proprioInput.getGripperState();
        
        NdArray inputVector = NdArray.of(stateVector).reshape(io.leavesfly.tinyai.ndarr.Shape.of(1, stateVector.length));
        
        // 前向传播
        Variable input = new Variable(inputVector);
        
        // Layer 1
        Variable fc1Out = fc1.layerForward(input);
        Variable norm1Out = norm1.layerForward(fc1Out);
        Variable relu1Out = norm1Out.relu();
        
        // Layer 2
        Variable fc2Out = fc2.layerForward(relu1Out);
        Variable norm2Out = norm2.layerForward(fc2Out);
        Variable relu2Out = norm2Out.relu();
        
        // Layer 3
        Variable fc3Out = fc3.layerForward(relu2Out);
        
        return fc3Out.getValue();
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        // 简化的前向传播接口
        // Layer 1
        Variable fc1Out = fc1.layerForward(input);
        Variable norm1Out = norm1.layerForward(fc1Out);
        Variable relu1Out = norm1Out.relu();
        
        // Layer 2
        Variable fc2Out = fc2.layerForward(relu1Out);
        Variable norm2Out = norm2.layerForward(fc2Out);
        Variable relu2Out = norm2Out.relu();
        
        // Layer 3
        Variable fc3Out = fc3.layerForward(relu2Out);
        
        return fc3Out;
    }
}
