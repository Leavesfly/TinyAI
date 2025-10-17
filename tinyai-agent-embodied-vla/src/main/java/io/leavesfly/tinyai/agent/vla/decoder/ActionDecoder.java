package io.leavesfly.tinyai.agent.vla.decoder;

import io.leavesfly.tinyai.agent.vla.model.ActionType;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.Linear;

import java.util.ArrayList;
import java.util.List;

/**
 * 动作解码器
 * 将融合特征解码为连续动作和离散动作
 * 
 * @author TinyAI
 */
public class ActionDecoder extends Block {
    
    private final int hiddenDim;
    private final int continuousActionDim;
    private final int discreteActionNum;
    
    // 连续动作头
    private Linear continuousHead1;
    private Linear continuousHead2;
    private Linear continuousHead3;
    
    // 离散动作头
    private Linear discreteHead1;
    private Linear discreteHead2;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏维度
     * @param continuousActionDim 连续动作维度（如末端执行器7自由度）
     * @param discreteActionNum 离散动作数量
     */
    public ActionDecoder(int hiddenDim, int continuousActionDim, int discreteActionNum) {
        this.hiddenDim = hiddenDim;
        this.continuousActionDim = continuousActionDim;
        this.discreteActionNum = discreteActionNum;
        
        // 连续动作头：hiddenDim -> 512 -> 256 -> actionDim
        this.continuousHead1 = new Linear(hiddenDim, 512, true);
        this.continuousHead2 = new Linear(512, 256, true);
        this.continuousHead3 = new Linear(256, continuousActionDim, true);
        
        // 离散动作头：hiddenDim -> 256 -> discreteActionNum
        this.discreteHead1 = new Linear(hiddenDim, 256, true);
        this.discreteHead2 = new Linear(256, discreteActionNum, true);
    }
    
    /**
     * 解码动作
     * 
     * @param fusedFeatures 融合后的特征 [seq_len, hiddenDim]
     * @return VLA动作
     */
    public VLAAction decode(NdArray fusedFeatures) {
        // 取最后一个token或做平均池化
        int seqLen = fusedFeatures.getShape()[0];
        double[] lastToken = new double[hiddenDim];
        
        for (int i = 0; i < hiddenDim; i++) {
            lastToken[i] = fusedFeatures.get((seqLen - 1) * hiddenDim + i);
        }
        
        NdArray aggregated = new NdArray(lastToken).reshape(1, hiddenDim);
        Variable input = new Variable(aggregated, false);
        
        // 解码连续动作
        Variable cont1 = continuousHead1.forward(input);
        Variable contRelu1 = cont1.relu();
        Variable cont2 = continuousHead2.forward(contRelu1);
        Variable contRelu2 = cont2.relu();
        Variable cont3 = continuousHead3.forward(contRelu2);
        
        // Tanh激活，归一化到[-1, 1]
        NdArray continuousAction = tanh(cont3.getData());
        
        // 解码离散动作
        Variable disc1 = discreteHead1.forward(input);
        Variable discRelu = disc1.relu();
        Variable disc2 = discreteHead2.forward(discRelu);
        
        // Softmax得到概率分布
        NdArray discreteProbs = softmax(disc2.getData());
        int discreteAction = argmax(discreteProbs);
        
        // 计算置信度
        double confidence = discreteProbs.get(discreteAction);
        
        // 映射到ActionType
        ActionType actionType = mapToActionType(discreteAction);
        
        return new VLAAction(continuousAction, discreteAction, actionType, confidence, null);
    }
    
    /**
     * Tanh激活函数
     */
    private NdArray tanh(NdArray input) {
        double[] data = input.toDoubleArray();
        double[] result = new double[data.length];
        
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.tanh(data[i]);
        }
        
        return new NdArray(result).reshape(input.getShape());
    }
    
    /**
     * Softmax函数
     */
    private NdArray softmax(NdArray input) {
        double[] data = input.toDoubleArray();
        double max = data[0];
        for (double v : data) {
            max = Math.max(max, v);
        }
        
        double[] exp = new double[data.length];
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            exp[i] = Math.exp(data[i] - max);
            sum += exp[i];
        }
        
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = exp[i] / sum;
        }
        
        return new NdArray(result);
    }
    
    /**
     * Argmax函数
     */
    private int argmax(NdArray input) {
        double[] data = input.toDoubleArray();
        int maxIdx = 0;
        double maxVal = data[0];
        
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }
    
    /**
     * 映射离散动作索引到ActionType
     */
    private ActionType mapToActionType(int discreteAction) {
        ActionType[] types = ActionType.values();
        if (discreteAction >= 0 && discreteAction < types.length) {
            return types[discreteAction];
        }
        return ActionType.MOVE_END_EFFECTOR;
    }
    
    @Override
    public Variable forward(Variable input) {
        // 简化接口
        VLAAction action = decode(input.getData());
        return new Variable(action.getContinuousAction(), false);
    }
    
    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.addAll(continuousHead1.parameters());
        params.addAll(continuousHead2.parameters());
        params.addAll(continuousHead3.parameters());
        params.addAll(discreteHead1.parameters());
        params.addAll(discreteHead2.parameters());
        return params;
    }
}
