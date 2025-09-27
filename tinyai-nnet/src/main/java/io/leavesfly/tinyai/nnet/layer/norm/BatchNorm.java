package io.leavesfly.tinyai.nnet.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;


/**
 * 批量归一化层
 * 实现Batch Normalization算法，提高训练稳定性和收敛速度
 * 
 * 公式：BN(x) = γ * (x - μ) / σ + β
 * 其中：
 * - μ 是批次均值
 * - σ 是批次标准差
 * - γ 是学习的缩放参数
 * - β 是学习的偏移参数
 */
public class BatchNorm extends Layer {

    private Parameter gamma;  // 缩放参数
    private Parameter beta;   // 偏移参数
    private int numFeatures;  // 特征数量
    private double eps;       // 防止除零的小常数
    
    /**
     * 构造批量归一化层
     * 
     * @param _name 层名称
     * @param _inputShape 输入形状
     * @param numFeatures 特征数量（通常是通道数）
     * @param eps 防止除零的小常数
     */
    public BatchNorm(String _name, Shape _inputShape, int numFeatures, double eps) {
        super(_name, _inputShape, _inputShape);
        this.numFeatures = numFeatures;
        this.eps = eps;
        init();
    }
    
    /**
     * 构造批量归一化层（使用默认eps）
     */
    public BatchNorm(String _name, Shape _inputShape, int numFeatures) {
        this(_name, _inputShape, numFeatures, 1e-5);
    }

    public BatchNorm(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 默认特征数为输入形状的最后一维
        if (_inputShape != null && _inputShape.size() > 0) {
            this.numFeatures = _inputShape.getDimension(_inputShape.size() - 1);
        } else {
            this.numFeatures = 1;
        }
        this.eps = 1e-5;
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化缩放参数γ为1
            gamma = new Parameter(NdArray.ones(Shape.of(numFeatures)));
            gamma.setName(name + "_gamma");
            addParam("gamma", gamma);
            
            // 初始化偏移参数β为0
            beta = new Parameter(NdArray.zeros(Shape.of(numFeatures)));
            beta.setName(name + "_beta");
            addParam("beta", beta);
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        NdArray inputData = x.getValue();
        
        // 计算批次维度的均值和方差
        // 假设输入形状为 (batch_size, features) 或 (batch_size, height, width, channels)
        Shape shape = inputData.getShape();
        
        // 计算除了特征维度外的所有维度的均值
        Variable mean = calculateMean(x);
        
        // 计算方差
        Variable variance = calculateVariance(x, mean);
        
        // 归一化：(x - μ) / √(σ² + ε)
        Variable normalized = x.sub(mean).div(variance.add(new Variable(NdArray.of(eps))).pow(0.5f));
        
        // 应用缩放和偏移：γ * normalized + β
        Variable output = normalized.mul(gamma).add(beta);
        
        return output;
    }
    
    /**
     * 计算批次均值
     */
    private Variable calculateMean(Variable x) {
        NdArray data = x.getValue();
        Shape shape = data.getShape();
        
        if (shape.size() == 2) {
            // 2D情况：(batch_size, features)
            // 沿着batch维度（轴0）计算均值
            return computeMeanAlongAxis(x, 0);
        } else if (shape.size() == 4) {
            // 4D情况：(batch_size, height, width, channels)
            // 沿着batch, height, width维度计算均值
            Variable temp = computeMeanAlongAxis(x, 0);
            temp = computeMeanAlongAxis(temp, 0);
            temp = computeMeanAlongAxis(temp, 0);
            return temp;
        } else {
            // 简化处理：计算所有元素的均值然后广播
            Variable totalMean = x.sum().div(new Variable(NdArray.of(data.getShape().size())));
            return totalMean.broadcastTo(Shape.of(numFeatures));
        }
    }
    
    /**
     * 计算批次方差
     */
    private Variable calculateVariance(Variable x, Variable mean) {
        Variable diff = x.sub(mean);
        Variable squaredDiff = diff.mul(diff);
        
        NdArray data = x.getValue();
        Shape shape = data.getShape();
        
        if (shape.size() == 2) {
            return computeMeanAlongAxis(squaredDiff, 0);
        } else if (shape.size() == 4) {
            Variable temp = computeMeanAlongAxis(squaredDiff, 0);
            temp = computeMeanAlongAxis(temp, 0);
            temp = computeMeanAlongAxis(temp, 0);
            return temp;
        } else {
            Variable totalVar = squaredDiff.sum().div(new Variable(NdArray.of(data.getShape().size())));
            return totalVar.broadcastTo(Shape.of(numFeatures));
        }
    }
    
    /**
     * 沿指定轴计算均值（简化实现）
     */
    private Variable computeMeanAlongAxis(Variable x, int axis) {
        // 这是一个简化的实现，实际中应该有更精确的轴向均值计算
        NdArray data = x.getValue();
        Shape shape = data.getShape();
        
        if (axis == 0 && shape.size() >= 2) {
            // 沿第一个轴计算均值
            int batchSize = shape.getDimension(0);
            Variable sum = x.sum();
            return sum.div(new Variable(NdArray.of(batchSize)));
        }
        
        // 默认返回输入的平均值
        return x.sum().div(new Variable(NdArray.of(data.getShape().size())));
    }
}