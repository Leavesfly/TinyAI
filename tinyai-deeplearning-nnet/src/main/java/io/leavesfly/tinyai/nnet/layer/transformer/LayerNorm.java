package io.leavesfly.tinyai.nnet.layer.transformer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * 层归一化（Layer Normalization）实现
 * <p>
 * 层归一化对每个样本的特征维度进行归一化，公式如下：
 * LayerNorm(x) = γ * (x - μ) / σ + β
 * <p>
 * 其中：
 * - μ 是均值
 * - σ 是标准差
 * - γ 是学习的缩放参数
 * - β 是学习的偏移参数
 */
public class LayerNorm extends Layer {

    private Parameter gamma;      // 缩放参数
    private Parameter beta;       // 偏移参数
    private int normalizedShape;  // 归一化维度大小
    private double epsilon;       // 防止除零的小常数

    public LayerNorm(String _name, Shape _inputShape) {
        super(_name, _inputShape, _inputShape);
        if (_inputShape != null && _inputShape.getDimNum() > 0) {
            this.normalizedShape = _inputShape.getDimension(_inputShape.getDimNum() - 1);
        } else {
            this.normalizedShape = 1;
        }
        this.epsilon = 1e-6;
        init();
    }

    public LayerNorm(String _name, Shape _inputShape, Shape _outputShape) {
        super(_name, _inputShape, _outputShape);
        if (_inputShape != null && _inputShape.getDimNum() > 0) {
            this.normalizedShape = _inputShape.getDimension(_inputShape.getDimNum() - 1);
        } else {
            this.normalizedShape = 1;
        }
        this.epsilon = 1e-6;
        init();
    }

    /**
     * 构造层归一化层
     *
     * @param name            层名称
     * @param normalizedShape 归一化的特征维度大小
     * @param epsilon         防止除零的小常数
     */
    public LayerNorm(String name, int normalizedShape, double epsilon) {
        super(name, Shape.of(-1, -1, normalizedShape), Shape.of(-1, -1, normalizedShape));
        this.normalizedShape = normalizedShape;
        this.epsilon = epsilon;
        init();
    }

    public LayerNorm(String name, int normalizedShape) {
        this(name, normalizedShape, 1e-6);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化缩放参数γ为1
            gamma = new Parameter(NdArray.ones(Shape.of(normalizedShape)));
            gamma.setName(name + "_gamma");
            addParam("gamma", gamma);
            
            // 初始化偏移参数β为0
            beta = new Parameter(NdArray.zeros(Shape.of(normalizedShape)));
            beta.setName(name + "_beta");
            addParam("beta", beta);
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();
        Shape inputShape = inputData.getShape();
        
        // 验证输入形状的最后一维与归一化维度匹配
        int inputLastDim = inputShape.getDimension(inputShape.getDimNum() - 1);
        if (inputLastDim != normalizedShape) {
            throw new IllegalArgumentException(
                String.format("输入的最后一维 (%d) 与归一化维度 (%d) 不匹配", 
                             inputLastDim, normalizedShape)
            );
        }
        
        // 计算最后一个维度的均值和方差
        Variable mean = calculateLastDimMean(x);
        Variable variance = calculateLastDimVariance(x, mean);
        
        // 归一化：(x - μ) / √(σ² + ε)
        Variable normalized = x.sub(mean).div(
            variance.add(new Variable(NdArray.of(epsilon))).pow(0.5f)
        );
        
        // 应用缩放和偏移：γ * normalized + β
        // 创建与输入形状兼容的gamma和beta
        Variable gammaBroadcasted = broadcastParameterToInput(gamma.getValue(), inputShape);
        Variable betaBroadcasted = broadcastParameterToInput(beta.getValue(), inputShape);
        Variable output = normalized.mul(gammaBroadcasted).add(betaBroadcasted);
        
        return output;
    }
    
    /**
     * 计算最后一个维度的均值
     */
    private Variable calculateLastDimMean(Variable x) {
        NdArray data = x.getValue();
        Shape shape = data.getShape();
        
        // 对最后一个维度计算均值
        if (shape.getDimNum() >= 1) {
            // 计算最后一个维度的轴索引
            int lastAxis = shape.getDimNum() - 1;
            NdArray meanData = data.mean(lastAxis);
            
            // 创建一个新的形状，将最后一个维度设为1，以便能够广播回原始形状
            int[] newDims = new int[shape.getDimNum()];
            for (int i = 0; i < shape.getDimNum() - 1; i++) {
                newDims[i] = shape.getDimension(i);
            }
            newDims[shape.getDimNum() - 1] = 1;
            Shape broadcastShape = Shape.of(newDims);
            
            // 将均值广播到新形状，然后再广播回原始形状
            return new Variable(meanData.reshape(broadcastShape).broadcastTo(shape));
        }
        
        return x;
    }
    
    /**
     * 计算最后一个维度的方差
     */
    private Variable calculateLastDimVariance(Variable x, Variable mean) {
        Variable diff = x.sub(mean);
        Variable squaredDiff = diff.mul(diff);
        
        NdArray data = squaredDiff.getValue();
        Shape shape = data.getShape();
        
        if (shape.getDimNum() >= 1) {
            // 计算最后一个维度的轴索引
            int lastAxis = shape.getDimNum() - 1;
            NdArray varData = data.mean(lastAxis);
            
            // 创建一个新的形状，将最后一个维度设为1，以便能够广播回原始形状
            int[] newDims = new int[shape.getDimNum()];
            for (int i = 0; i < shape.getDimNum() - 1; i++) {
                newDims[i] = shape.getDimension(i);
            }
            newDims[shape.getDimNum() - 1] = 1;
            Shape broadcastShape = Shape.of(newDims);
            
            // 将方差广播到新形状，然后再广播回原始形状
            return new Variable(varData.reshape(broadcastShape).broadcastTo(shape));
        }
        
        return squaredDiff;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 层归一化的反向传播比较复杂，这里提供简化版本
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    /**
     * 将参数广播到输入形状
     */
    private Variable broadcastParameterToInput(NdArray param, Shape targetShape) {
        // 参数形状: (normalizedShape,)
        // 目标形状: (..., normalizedShape)
        
        int targetDims = targetShape.getDimNum();
        int[] broadcastShape = new int[targetDims];
        
        // 前面的维度都设为1，最后一个维度保持不变
        for (int i = 0; i < targetDims - 1; i++) {
            broadcastShape[i] = 1;
        }
        broadcastShape[targetDims - 1] = normalizedShape;
        
        // 重塑参数并广播到目标形状
        NdArray reshaped = param.reshape(Shape.of(broadcastShape));
        return new Variable(reshaped.broadcastTo(targetShape));
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
}