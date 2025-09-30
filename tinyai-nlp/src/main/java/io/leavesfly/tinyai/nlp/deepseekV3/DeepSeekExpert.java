package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;

/**
 * DeepSeek专家网络实现
 * 
 * 基于SwiGLU激活函数的前馈神经网络专家，
 * 这是DeepSeek-V3中使用的标准专家架构。
 * 
 * 网络结构：
 * Input → Linear1 → SwiGLU → Linear2 → Output
 * 
 * SwiGLU = Swish(Linear_gate(x)) ⊙ Linear_up(x)
 * 其中 Swish(x) = x * sigmoid(x)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekExpert extends Layer {
    
    private LinearLayer gateProjection;    // 门控投影层
    private LinearLayer upProjection;      // 上投影层
    private LinearLayer downProjection;    // 下投影层
    
    private int expertId;                  // 专家ID
    private int dModel;                    // 输入/输出维度
    private int dExpert;                   // 专家隐藏层维度
    private String activationType;         // 激活函数类型
    
    /**
     * 构造专家网络
     * 
     * @param name 专家网络名称
     * @param expertId 专家ID
     * @param dModel 输入和输出维度
     * @param dExpert 隐藏层维度
     * @param activationType 激活函数类型
     */
    public DeepSeekExpert(String name, int expertId, int dModel, int dExpert, String activationType) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || dExpert <= 0) {
            throw new IllegalArgumentException("维度参数必须大于0");
        }
        
        this.expertId = expertId;
        this.dModel = dModel;
        this.dExpert = dExpert;
        this.activationType = activationType != null ? activationType : "SwiGLU";
        
        init();
    }
    
    /**
     * 简化构造函数，使用默认SwiGLU激活
     */
    public DeepSeekExpert(String name, int expertId, int dModel, int dExpert) {
        this(name, expertId, dModel, dExpert, "SwiGLU");
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            if ("SwiGLU".equals(activationType)) {
                // SwiGLU需要两个投影层：gate和up
                gateProjection = new LinearLayer(
                    name + "_gate_proj", 
                    dModel, 
                    dExpert, 
                    false  // 不使用偏置
                );
                
                upProjection = new LinearLayer(
                    name + "_up_proj", 
                    dModel, 
                    dExpert, 
                    false
                );
                
                downProjection = new LinearLayer(
                    name + "_down_proj", 
                    dExpert, 
                    dModel, 
                    false
                );
            } else {
                // 传统ReLU激活只需要两层
                gateProjection = new LinearLayer(
                    name + "_first_linear", 
                    dModel, 
                    dExpert, 
                    false
                );
                
                downProjection = new LinearLayer(
                    name + "_second_linear", 
                    dExpert, 
                    dModel, 
                    false
                );
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        // 验证输入维度
        if (input.getValue().getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                String.format("专家网络输入维度不匹配。期望%d，实际%d", 
                             dModel, input.getValue().getShape().getDimension(2))
            );
        }
        
        if ("SwiGLU".equals(activationType)) {
            return forwardSwiGLU(input);
        } else {
            return forwardReLU(input);
        }
    }
    
    /**
     * SwiGLU前向传播
     */
    private Variable forwardSwiGLU(Variable input) {
        // 计算门控和上投影
        Variable gate = gateProjection.layerForward(input);    // [batch, seq, dExpert]
        Variable up = upProjection.layerForward(input);        // [batch, seq, dExpert]
        
        // 应用SwiGLU：Swish(gate) ⊙ up
        Variable gateSwish = applySwiish(gate);
        Variable gated = elementWiseMultiply(gateSwish, up);
        
        // 下投影回到原始维度
        Variable output = downProjection.layerForward(gated);  // [batch, seq, dModel]
        
        return output;
    }
    
    /**
     * ReLU前向传播
     */
    private Variable forwardReLU(Variable input) {
        // 第一层 + ReLU
        Variable hidden = gateProjection.layerForward(input);
        Variable activated = applyReLU(hidden);
        
        // 第二层
        Variable output = downProjection.layerForward(activated);
        
        return output;
    }
    
    /**
     * 应用Swish激活函数：x * sigmoid(x)
     */
    private Variable applySwiish(Variable input) {
        // Swish(x) = x * sigmoid(x)
        Variable sigmoid = new Variable(input.getValue().sigmoid());
        return elementWiseMultiply(input, sigmoid);
    }
    
    /**
     * 应用ReLU激活函数
     */
    private Variable applyReLU(Variable input) {
        // ReLU(x) = max(0, x)
        return new Variable(input.getValue().maximum(0.0f));
    }
    
    /**
     * 逐元素乘法
     */
    private Variable elementWiseMultiply(Variable a, Variable b) {
        return new Variable(a.getValue().mul(b.getValue()));
    }
    
    /**
     * 获取专家参数数量
     */
    public long getParameterCount() {
        if ("SwiGLU".equals(activationType)) {
            // gate_proj + up_proj + down_proj
            return (long) dModel * dExpert * 2 + (long) dExpert * dModel;
        } else {
            // first_linear + second_linear
            return (long) dModel * dExpert + (long) dExpert * dModel;
        }
    }
    
    /**
     * 获取专家配置信息
     */
    public String getExpertConfig() {
        return String.format(
            "Expert-%d Config:\n" +
            "  - Input/Output Dim: %d\n" +
            "  - Hidden Dim: %d\n" +
            "  - Activation: %s\n" +
            "  - Parameters: %,d",
            expertId, dModel, dExpert, activationType, getParameterCount()
        );
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // Getter方法
    public int getExpertId() { return expertId; }
    public int getDModel() { return dModel; }
    public int getDExpert() { return dExpert; }
    public String getActivationType() { return activationType; }
    
    public LinearLayer getGateProjection() { return gateProjection; }
    public LinearLayer getUpProjection() { return upProjection; }
    public LinearLayer getDownProjection() { return downProjection; }
}