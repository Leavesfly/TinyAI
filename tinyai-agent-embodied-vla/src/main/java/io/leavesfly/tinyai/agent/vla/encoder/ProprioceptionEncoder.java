package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.ProprioceptionInput;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.block.Block;
import io.leavesfly.tinyai.nnet.layer.Linear;
import io.leavesfly.tinyai.nnet.layer.LayerNorm;

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
    private Linear fc1;
    private Linear fc2;
    private Linear fc3;
    
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
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        
        // 三层MLP：inputDim -> 256 -> 512 -> hiddenDim
        this.fc1 = new Linear(inputDim, 256, true);
        this.norm1 = new LayerNorm(256);
        
        this.fc2 = new Linear(256, 512, true);
        this.norm2 = new LayerNorm(512);
        
        this.fc3 = new Linear(512, hiddenDim, true);
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
        
        int numJoints = jointPositions.getShape()[0];
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
        
        NdArray inputVector = new NdArray(stateVector).reshape(1, stateVector.length);
        
        // 前向传播
        Variable input = new Variable(inputVector, true);
        
        // Layer 1
        Variable fc1Out = fc1.forward(input);
        Variable norm1Out = norm1.forward(fc1Out);
        Variable relu1Out = norm1Out.relu();
        
        // Layer 2
        Variable fc2Out = fc2.forward(relu1Out);
        Variable norm2Out = norm2.forward(fc2Out);
        Variable relu2Out = norm2Out.relu();
        
        // Layer 3
        Variable fc3Out = fc3.forward(relu2Out);
        
        return fc3Out.getData();
    }
    
    @Override
    public Variable forward(Variable input) {
        // 简化的前向传播接口
        // Layer 1
        Variable fc1Out = fc1.forward(input);
        Variable norm1Out = norm1.forward(fc1Out);
        Variable relu1Out = norm1Out.relu();
        
        // Layer 2
        Variable fc2Out = fc2.forward(relu1Out);
        Variable norm2Out = norm2.forward(fc2Out);
        Variable relu2Out = norm2Out.relu();
        
        // Layer 3
        Variable fc3Out = fc3.forward(relu2Out);
        
        return fc3Out;
    }
    
    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.addAll(fc1.parameters());
        params.addAll(norm1.parameters());
        params.addAll(fc2.parameters());
        params.addAll(norm2.parameters());
        params.addAll(fc3.parameters());
        return params;
    }
}
