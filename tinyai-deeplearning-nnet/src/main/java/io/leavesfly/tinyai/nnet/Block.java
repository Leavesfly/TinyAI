package io.leavesfly.tinyai.nnet;


import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 表示由层组合起来的更大的神经网络的块
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Block是神经网络中用于组合多个Layer的容器类，可以包含其他Layer或Block，
 * 是构建复杂神经网络结构的基础组件。
 */
public abstract class Block extends LayerAble {

    /**
     * 存储该Block中包含的所有Layer
     */
    protected List<LayerAble> layers;

    /**
     * 构造函数，初始化Block的基本属性
     * 
     * @param _name Block的名称
     * @param _inputShape 输入数据的形状
     */
    public Block(String _name, Shape _inputShape) {
        name = _name;
        this.params = new HashMap<>();
        layers = new ArrayList<>();
        inputShape = _inputShape;
    }

    /**
     * 构造函数，初始化Block的基本属性（包含输出形状）
     * 
     * @param _name Block的名称
     * @param _inputShape 输入数据的形状
     * @param _outputShape 输出数据的形状
     */
    public Block(String _name, Shape _inputShape, Shape _outputShape) {
        name = _name;
        this.params = new HashMap<>();
        layers = new ArrayList<>();
        inputShape = _inputShape;
        outputShape = _outputShape;
    }


    @Override
    public void clearGrads() {
        for (Parameter parameter : params.values()) {
            parameter.clearGrad();
        }
        for (LayerAble layer : layers) {
            layer.clearGrads();
        }
    }

    /**
     * 向Block中添加一个Layer
     * 
     * @param layerAble 要添加的Layer实例
     */
    public void addLayer(LayerAble layerAble) {
        layerAble.init();
        layers.add(layerAble);
    }


    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];
        Variable y = layers.get(0).layerForward(x);
        for (int i = 1; i < layers.size(); i++) {
            y = layers.get(i).layerForward(y);
        }
        return y;
    }

    /**
     * 获取Block中所有的参数
     * 
     * @return 包含所有参数的Map
     */
    public Map<String, Parameter> getAllParams() {
        Map<String, Parameter> allParams = new HashMap<>();
        putAll(allParams);
        return allParams;
    }

    /**
     * 递归收集所有参数
     * 
     * @param allParams 用于存储所有参数的Map
     */
    private void putAll(Map<String, Parameter> allParams) {
        allParams.putAll(params);
        for (LayerAble layer : layers) {
            // 对于Block类型的层，使用getAllParams()递归收集参数
            // 对于普通Layer，使用getParams()获取直接参数
            if (layer instanceof Block) {
                allParams.putAll(((Block) layer).getAllParams());
            } else {
                allParams.putAll(layer.getParams());
            }
        }
    }

    /**
     * 重置Block中所有RNN层的状态
     */
    public void resetState() {
        for (LayerAble layerAble : layers) {
            if (layerAble instanceof RnnLayer) {
                ((RnnLayer) layerAble).resetState();
            } else if (layerAble instanceof Block) {
                ((Block) layerAble).resetState();
            }
        }
    }


//    public List<LayerAble> getLayers() {
//        return layers;
//    }


    @Override
    public NdArray forward(NdArray... inputs) {
        return null;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        return null;
    }


}