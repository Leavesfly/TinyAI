package io.leavesfly.tinyai.nl.core;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 上下文流（ContextFlow）
 * 管理嵌套优化层级之间的信息流动
 * 
 * <p>在嵌入学习范式中，不同层级的优化问题通过上下文流交换信息。
 * 上下文流负责在层级间传播、压缩和合并上下文数据。</p>
 * 
 * @author TinyAI Team
 */
public class ContextFlow {
    
    /**
     * 当前上下文数据
     */
    private Variable contextData;
    
    /**
     * 流动方向
     */
    private FlowDirection flowDirection;
    
    /**
     * 上下文压缩率（0-1之间，1表示不压缩）
     */
    private float compressionRate;
    
    /**
     * 构造函数
     * 
     * @param contextData 初始上下文数据
     * @param flowDirection 流动方向
     * @param compressionRate 压缩率
     */
    public ContextFlow(Variable contextData, FlowDirection flowDirection, float compressionRate) {
        this.contextData = contextData;
        this.flowDirection = flowDirection;
        this.compressionRate = Math.max(0.0f, Math.min(1.0f, compressionRate));
    }
    
    /**
     * 简化构造函数，默认双向流动，不压缩
     * 
     * @param contextData 初始上下文数据
     */
    public ContextFlow(Variable contextData) {
        this(contextData, FlowDirection.BIDIRECTIONAL, 1.0f);
    }
    
    /**
     * 执行上下文流动
     * 根据流动方向和压缩率处理输入上下文
     * 
     * @param inputContext 输入上下文
     * @return 处理后的上下文
     */
    public Variable flow(Variable inputContext) {
        if (inputContext == null) {
            return this.contextData;
        }
        
        Variable processedContext = inputContext;
        
        // 根据流动方向决定处理方式
        switch (flowDirection) {
            case UPWARD:
                // 向上流动时压缩信息（从子层级到父层级，信息需要抽象化）
                if (compressionRate < 1.0f) {
                    processedContext = compress(inputContext, compressionRate);
                }
                break;
            case DOWNWARD:
                // 向下流动时直接传递（从父层级到子层级，保留完整信息）
                break;
            case BIDIRECTIONAL:
                // 双向流动时应用压缩
                if (compressionRate < 1.0f) {
                    processedContext = compress(inputContext, compressionRate);
                }
                break;
        }
        
        // 如果已有上下文数据，与新数据融合
        if (this.contextData != null) {
            processedContext = fuseContext(this.contextData, processedContext);
        }
        
        // 更新当前上下文
        this.contextData = processedContext;
        
        return processedContext;
    }
    
    /**
     * 融合旧上下文和新上下文
     * 使用指数移动平均实现平滑过渡
     * 
     * @param oldContext 旧上下文
     * @param newContext 新上下文
     * @return 融合后的上下文
     */
    private Variable fuseContext(Variable oldContext, Variable newContext) {
        if (oldContext == null) {
            return newContext;
        }
        if (newContext == null) {
            return oldContext;
        }
        
        // 检查形状是否兼容
        int[] oldShape = oldContext.getValue().getShape().getShapeDims();
        int[] newShape = newContext.getValue().getShape().getShapeDims();
        if (!java.util.Arrays.equals(oldShape, newShape)) {
            // 形状不兼容时直接使用新上下文
            return newContext;
        }
        
        // 指数移动平均：result = 0.7 * newContext + 0.3 * oldContext
        float newWeight = 0.7f;
        float oldWeight = 1.0f - newWeight;
        return newContext.mul(new Variable(newWeight)).add(oldContext.mul(new Variable(oldWeight)));
    }
    
    /**
     * 压缩上下文信息
     * 使用随机投影矩阵降低维度，保留主要特征
     * 
     * @param context 原始上下文
     * @param rate 压缩率（0-1，越小压缩越多）
     * @return 压缩后的上下文
     */
    public Variable compress(Variable context, float rate) {
        if (context == null || rate >= 1.0f) {
            return context;
        }
        
        int[] shape = context.getValue().getShape().getShapeDims();
        if (shape.length == 2) {
            int originalDim = shape[1];
            int compressedDim = Math.max(1, (int) (originalDim * rate));
            
            if (compressedDim < originalDim) {
                // 使用均值池化进行压缩：将相邻特征分组取平均
                int groupSize = originalDim / compressedDim;
                int batchSize = shape[0];
                float[] compressedData = new float[batchSize * compressedDim];
                
                NdArray contextData = context.getValue();
                for (int b = 0; b < batchSize; b++) {
                    for (int j = 0; j < compressedDim; j++) {
                        float sum = 0.0f;
                        int count = 0;
                        int startIdx = j * groupSize;
                        int endIdx = (j == compressedDim - 1) ? originalDim : startIdx + groupSize;
                        for (int k = startIdx; k < endIdx; k++) {
                            sum += contextData.get(new int[]{b, k});
                            count++;
                        }
                        compressedData[b * compressedDim + j] = sum / count;
                    }
                }
                
                NdArray compressedArray = NdArray.of(compressedData, Shape.of(batchSize, compressedDim));
                return new Variable(compressedArray);
            }
        }
        
        return context;
    }
    
    /**
     * 合并多个上下文流
     * 将另一个上下文流的信息合并到当前流中
     * 
     * @param otherContext 其他上下文流
     * @return 合并后的新上下文流
     */
    public ContextFlow merge(ContextFlow otherContext) {
        if (otherContext == null) {
            return this;
        }
        
        Variable mergedData = this.contextData;
        if (otherContext.contextData != null) {
            if (this.contextData != null) {
                // 检查形状兼容性
                int[] thisShape = this.contextData.getValue().getShape().getShapeDims();
                int[] otherShape = otherContext.contextData.getValue().getShape().getShapeDims();
                
                if (java.util.Arrays.equals(thisShape, otherShape)) {
                    // 形状相同时取加权平均
                    mergedData = this.contextData.add(otherContext.contextData).mul(new Variable(0.5f));
                } else {
                    // 形状不同时保留当前上下文
                    mergedData = this.contextData;
                }
            } else {
                mergedData = otherContext.contextData;
            }
        }
        
        return new ContextFlow(mergedData, this.flowDirection, this.compressionRate);
    }
    
    // Getters and Setters
    
    public Variable getContextData() {
        return contextData;
    }
    
    public void setContextData(Variable contextData) {
        this.contextData = contextData;
    }
    
    public FlowDirection getFlowDirection() {
        return flowDirection;
    }
    
    public void setFlowDirection(FlowDirection flowDirection) {
        this.flowDirection = flowDirection;
    }
    
    public float getCompressionRate() {
        return compressionRate;
    }
    
    public void setCompressionRate(float compressionRate) {
        this.compressionRate = Math.max(0.0f, Math.min(1.0f, compressionRate));
    }
}
