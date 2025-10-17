package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.decoder.ActionDecoder;
import io.leavesfly.tinyai.agent.vla.decoder.LanguageFeedbackGenerator;
import io.leavesfly.tinyai.agent.vla.encoder.LanguageEncoder;
import io.leavesfly.tinyai.agent.vla.encoder.ProprioceptionEncoder;
import io.leavesfly.tinyai.agent.vla.encoder.VisionEncoder;
import io.leavesfly.tinyai.agent.vla.fusion.CrossModalAttention;
import io.leavesfly.tinyai.agent.vla.fusion.VLATransformerCore;
import io.leavesfly.tinyai.agent.vla.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;

/**
 * VLA智能体核心
 * 集成视觉、语言、动作三种模态的端到端具身智能系统
 * 
 * @author TinyAI
 */
public class VLAAgent {
    
    // 配置参数
    private final int hiddenDim;
    private final int numHeads;
    private final int numLayers;
    private final int actionDim;
    
    // 编码器
    private final VisionEncoder visionEncoder;
    private final LanguageEncoder languageEncoder;
    private final ProprioceptionEncoder proprioceptionEncoder;
    
    // 融合层
    private final CrossModalAttention crossModalAttention;
    private final VLATransformerCore transformerCore;
    
    // 解码器
    private final ActionDecoder actionDecoder;
    private final LanguageFeedbackGenerator feedbackGenerator;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏层维度
     * @param numHeads 注意力头数
     * @param numLayers Transformer层数
     * @param actionDim 动作维度
     */
    public VLAAgent(int hiddenDim, int numHeads, int numLayers, int actionDim) {
        this.hiddenDim = hiddenDim;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.actionDim = actionDim;
        
        // 初始化编码器
        this.visionEncoder = new VisionEncoder(3, hiddenDim, 64, 8);
        this.languageEncoder = new LanguageEncoder(10000, hiddenDim, 128, 6);
        this.proprioceptionEncoder = new ProprioceptionEncoder(15, hiddenDim);
        
        // 初始化融合层
        this.crossModalAttention = new CrossModalAttention(hiddenDim, numHeads);
        this.transformerCore = new VLATransformerCore(hiddenDim, numLayers, numHeads);
        
        // 初始化解码器
        this.actionDecoder = new ActionDecoder(hiddenDim, actionDim, 7);
        this.feedbackGenerator = new LanguageFeedbackGenerator();
        
        System.out.println("VLAAgent initialized with:");
        System.out.println("  Hidden Dim: " + hiddenDim);
        System.out.println("  Num Heads: " + numHeads);
        System.out.println("  Num Layers: " + numLayers);
        System.out.println("  Action Dim: " + actionDim);
    }
    
    /**
     * 预测动作
     * 
     * @param state VLA状态
     * @return VLA动作
     */
    public VLAAction predict(VLAState state) {
        // 1. 编码各模态输入
        NdArray visionFeatures = visionEncoder.encode(state.getVisionInput());
        NdArray languageFeatures = languageEncoder.encode(state.getLanguageInput());
        
        NdArray proprioFeatures = null;
        if (state.getProprioceptionInput() != null) {
            proprioFeatures = proprioceptionEncoder.encode(state.getProprioceptionInput());
        }
        
        // 2. 拼接多模态特征
        NdArray concatenatedFeatures = concatenateFeatures(
            visionFeatures, 
            languageFeatures, 
            proprioFeatures
        );
        
        // 3. 跨模态融合
        Variable fusedVar = transformerCore.fuse(new Variable(concatenatedFeatures, false));
        NdArray fusedFeatures = fusedVar.getData();
        
        // 保存融合特征到状态
        state.setFusedFeatures(fusedFeatures);
        
        // 4. 解码动作
        VLAAction action = actionDecoder.decode(fusedFeatures);
        
        // 5. 生成语言反馈
        String feedback = feedbackGenerator.generateFeedback(
            action.getActionType(), 
            action.getConfidence()
        );
        action.setLanguageFeedback(feedback);
        
        return action;
    }
    
    /**
     * 拼接多模态特征
     */
    private NdArray concatenateFeatures(NdArray vision, NdArray language, NdArray proprio) {
        int visionLen = vision.getShape()[0];
        int langLen = language.getShape()[0];
        int totalLen = visionLen + langLen;
        
        if (proprio != null) {
            totalLen += proprio.getShape()[0];
        }
        
        double[][] concatenated = new double[totalLen][hiddenDim];
        
        // 复制视觉特征
        for (int i = 0; i < visionLen; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                concatenated[i][j] = vision.get(i * hiddenDim + j);
            }
        }
        
        // 复制语言特征
        for (int i = 0; i < langLen; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                concatenated[visionLen + i][j] = language.get(i * hiddenDim + j);
            }
        }
        
        // 复制本体感知特征
        if (proprio != null) {
            int proprioLen = proprio.getShape()[0];
            for (int i = 0; i < proprioLen; i++) {
                for (int j = 0; j < hiddenDim; j++) {
                    concatenated[visionLen + langLen + i][j] = proprio.get(i * hiddenDim + j);
                }
            }
        }
        
        return new NdArray(concatenated);
    }
    
    /**
     * 获取模型参数数量
     */
    public int getParameterCount() {
        int count = 0;
        // 简化计算
        count += hiddenDim * hiddenDim * numLayers * 4; // Transformer参数
        count += hiddenDim * actionDim; // 解码器参数
        return count;
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("\n========== VLA Agent Model Info ==========");
        System.out.println("Architecture: Vision-Language-Action Transformer");
        System.out.println("Hidden Dimension: " + hiddenDim);
        System.out.println("Attention Heads: " + numHeads);
        System.out.println("Transformer Layers: " + numLayers);
        System.out.println("Action Dimension: " + actionDim);
        System.out.println("Estimated Parameters: " + getParameterCount());
        System.out.println("==========================================\n");
    }
}
