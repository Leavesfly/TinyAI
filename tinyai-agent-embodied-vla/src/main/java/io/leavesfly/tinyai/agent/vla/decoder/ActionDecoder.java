package io.leavesfly.tinyai.agent.vla.decoder;

import io.leavesfly.tinyai.agent.vla.model.ActionType;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

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
    private LinearLayer continuousHead1;
    private LinearLayer continuousHead2;
    private LinearLayer continuousHead3;
    
    // 离散动作头
    private LinearLayer discreteHead1;
    private LinearLayer discreteHead2;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏维度
     * @param continuousActionDim 连续动作维度（如末端执行器7自由度）
     * @param discreteActionNum 离散动作数量
     */
    public ActionDecoder(int hiddenDim, int continuousActionDim, int discreteActionNum) {
        super("ActionDecoder", null);
        this.hiddenDim = hiddenDim;
        this.continuousActionDim = continuousActionDim;
        this.discreteActionNum = discreteActionNum;
        
        // 连续动作头：hiddenDim -> 512 -> 256 -> actionDim
        this.continuousHead1 = new LinearLayer("continuous1", hiddenDim, 512, true);
        this.continuousHead2 = new LinearLayer("continuous2", 512, 256, true);
        this.continuousHead3 = new LinearLayer("continuous3", 256, continuousActionDim, true);
        
        // 离散动作头：hiddenDim -> 256 -> discreteActionNum
        this.discreteHead1 = new LinearLayer("discrete1", hiddenDim, 256, true);
        this.discreteHead2 = new LinearLayer("discrete2", 256, discreteActionNum, true);
    }
    
    /**
     * 解码动作
     * 
     * @param fusedFeatures 融合后的特征 [seq_len, hiddenDim]
     * @return VLA动作
     */
    public VLAAction decode(NdArray fusedFeatures) {
        // 取最后一个token或做平均池化
        int seqLen = fusedFeatures.getShape().getDimension(0);
        double[] lastToken = new double[hiddenDim];
        
        for (int i = 0; i < hiddenDim; i++) {
            lastToken[i] = fusedFeatures.get((seqLen - 1) * hiddenDim + i);
        }
        
        NdArray aggregated = NdArray.of(lastToken).reshape(io.leavesfly.tinyai.ndarr.Shape.of(1, hiddenDim));
        Variable input = new Variable(aggregated);
        
        // 解码连续动作
        Variable cont1 = continuousHead1.layerForward(input);
        Variable contRelu1 = cont1.relu();
        Variable cont2 = continuousHead2.layerForward(contRelu1);
        Variable contRelu2 = cont2.relu();
        Variable cont3 = continuousHead3.layerForward(contRelu2);
        
        // Tanh激活，归一化到[-1, 1]
        NdArray continuousAction = tanh(cont3.getValue());
        
        // 解码离散动作
        Variable disc1 = discreteHead1.layerForward(input);
        Variable discRelu = disc1.relu();
        Variable disc2 = discreteHead2.layerForward(discRelu);
        
        // Softmax得到概率分布
        NdArray discreteProbs = softmax(disc2.getValue());
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
        io.leavesfly.tinyai.ndarr.Shape shape = input.getShape();
        float[][] matrix = input.getMatrix();
        int rows = shape.getDimension(0);
        int cols = shape.getDimNum() > 1 ? shape.getDimension(1) : 1;
        
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.tanh(matrix[i][j]);
            }
        }
        
        return NdArray.of(result).reshape(input.getShape());
    }
    
    /**
     * Softmax函数
     */
    private NdArray softmax(NdArray input) {
        float[][] matrix = input.getMatrix();
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        double max = matrix[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, matrix[i][j]);
            }
        }
        
        double[] exp = new double[cols];
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            exp[j] = Math.exp(matrix[0][j] - max);
            sum += exp[j];
        }
        
        float[] result = new float[cols];
        for (int j = 0; j < cols; j++) {
            result[j] = (float)(exp[j] / sum);
        }
        
        return NdArray.of(result);
    }
    
    /**
     * Argmax函数
     */
    private int argmax(NdArray input) {
        float[][] matrix = input.getMatrix();
        int maxIdx = 0;
        float maxVal = matrix[0][0];
        
        int cols = matrix[0].length;
        for (int i = 0; i < cols; i++) {
            if (matrix[0][i] > maxVal) {
                maxVal = matrix[0][i];
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
    public Variable layerForward(Variable... inputs) {
        // 简化接口
        VLAAction action = decode(inputs[0].getValue());
        return new Variable(action.getContinuousAction());
    }
    
    @Override
    public void init() {
        // 初始化已在构造函数中完成
    }
}
