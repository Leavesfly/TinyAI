package io.leavesfly.tinyai.nnet.layer.norm;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.Collections;
import java.util.List;

/**
 * 将矩阵打平层
 * 将多维输入重塑为二维输出，通常用在卷积层和全连接层之间
 * 
 * 例如：(batch_size, height, width, channels) -> (batch_size, height*width*channels)
 */
public class Flatten extends Layer {

    private Shape originalShape;  // 保存原始形状用于反向传播
    
    /**
     * 构造函数
     * 
     * @param _name 层名称
     * @param _inputShape 输入形状
     */
    public Flatten(String _name, Shape _inputShape) {
        super(_name, _inputShape, calculateOutputShape(_inputShape));
    }
    
    /**
     * 获取输出形状
     * 
     * @return 输出形状
     */
    public Shape getOutputShape() {
        return outputShape;
    }
    
    /**
     * 计算输出形状
     */
    private static Shape calculateOutputShape(Shape inputShape) {
        // 处理null或空形状
        if (inputShape == null || inputShape.getDimNum() == 0) {
            return Shape.of(-1, 1);
        }
        
        // 如果输入是一维的
        if (inputShape.getDimNum() == 1) {
            // 将一维输入转换为(batch_size, 1)的形式
            // 这里假设一维输入的维度就是batch_size
            int batchSize = inputShape.getDimension(0);
            return Shape.of(batchSize, 1);
        }
        
        // 如果输入是二维或更高维度
        int batchSize = inputShape.getDimension(0);
        int flattenedSize = 1;
        
        // 计算除了batch维度外的所有维度的乘积
        for (int i = 1; i < inputShape.getDimNum(); i++) {
            int dim = inputShape.getDimension(i);
            if (dim > 0) {
                flattenedSize *= dim;
            }
        }
        
        return Shape.of(batchSize, flattenedSize);
    }

    @Override
    public void init() {
        // Flatten层没有可训练的参数，无需初始化
        alreadyInit = true;
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        originalShape = x.getValue().getShape();
        
        // 计算输出形状
        Shape outputShape = calculateOutputShape(originalShape);
        
        // 重塑为二维
        return x.reshape(outputShape);
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        originalShape = inputs[0].getShape();
        
        // 计算输出形状
        Shape outputShape = calculateOutputShape(originalShape);
        
        return inputs[0].reshape(outputShape);
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 反向传播时将梯度重塑回原始形状
        return Collections.singletonList(yGrad.reshape(originalShape));
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
}