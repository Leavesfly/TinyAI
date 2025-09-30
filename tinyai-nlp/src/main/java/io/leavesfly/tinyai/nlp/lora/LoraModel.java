package io.leavesfly.tinyai.nlp.lora;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * LoRA模型 - 展示完整的LoRA微调流程
 * 
 * 该模型演示了如何构建一个包含多个LoRA层的深度神经网络，
 * 并展示LoRA微调的核心优势：
 * 1. 大幅减少可训练参数
 * 2. 保持模型性能
 * 3. 快速适应新任务
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraModel extends Block {
    
    /**
     * LoRA配置
     */
    private final LoraConfig config;
    
    /**
     * LoRA层列表
     */
    private final List<LoraLinearLayer> loraLayers;
    
    /**
     * 激活层列表
     */
    private final List<ReLuLayer> activationLayers;
    
    /**
     * 模型架构配置
     */
    private final int[] layerSizes;
    
    /**
     * 是否包含输出层的激活函数
     */
    private final boolean useOutputActivation;
    
    /**
     * 构造函数 - 创建多层LoRA模型
     * 
     * @param _name 模型名称
     * @param layerSizes 各层大小配置（包括输入层）
     * @param config LoRA配置
     * @param useOutputActivation 是否在输出层使用激活函数
     */
    public LoraModel(String _name, int[] layerSizes, LoraConfig config, boolean useOutputActivation) {
        super(_name, Shape.of(-1, layerSizes[0]), Shape.of(-1, layerSizes[layerSizes.length - 1]));
        
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("至少需要输入层和输出层");
        }
        
        this.config = config;
        this.layerSizes = layerSizes.clone();
        this.useOutputActivation = useOutputActivation;
        this.loraLayers = new ArrayList<>();
        this.activationLayers = new ArrayList<>();
        
        // 构建网络层
        buildNetwork();
    }
    
    /**
     * 构建网络结构
     */
    private void buildNetwork() {
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int inputDim = layerSizes[i];
            int outputDim = layerSizes[i + 1];
            
            // 创建LoRA线性层
            String layerName = String.format("lora_layer_%d", i);
            LoraLinearLayer loraLayer = new LoraLinearLayer(
                layerName, inputDim, outputDim, config, true);
            loraLayers.add(loraLayer);
            addLayer(loraLayer);
            
            // 为除最后一层外的所有层添加ReLU激活函数
            // 或者如果指定了输出层也使用激活函数
            if (i < layerSizes.length - 2 || useOutputActivation) {
                String activationName = String.format("relu_%d", i);
                ReLuLayer activation = new ReLuLayer(activationName, Shape.of(-1, outputDim));
                activationLayers.add(activation);
                addLayer(activation);
            }
        }
    }
    
    @Override
    public void init() {
        // 层已在构造函数中初始化
    }
    
    /**
     * 从预训练模型创建LoRA模型
     * 
     * @param _name 模型名称
     * @param pretrainedWeights 预训练权重列表
     * @param pretrainedBiases 预训练偏置列表（可包含null）
     * @param config LoRA配置
     * @param useOutputActivation 是否在输出层使用激活函数
     * @return LoRA模型
     */
    public static LoraModel fromPretrained(String _name, List<NdArray> pretrainedWeights, 
                                         List<NdArray> pretrainedBiases, LoraConfig config,
                                         boolean useOutputActivation) {
        if (pretrainedWeights.isEmpty()) {
            throw new IllegalArgumentException("预训练权重不能为空");
        }
        
        // 推断层大小
        int[] layerSizes = new int[pretrainedWeights.size() + 1];
        layerSizes[0] = pretrainedWeights.get(0).getShape().getDimension(0); // 输入维度
        for (int i = 0; i < pretrainedWeights.size(); i++) {
            layerSizes[i + 1] = pretrainedWeights.get(i).getShape().getDimension(1); // 输出维度
        }
        
        // 创建基础模型结构
        LoraModel model = new LoraModel(_name, layerSizes, config, useOutputActivation);
        
        // 替换权重为预训练权重
        for (int i = 0; i < pretrainedWeights.size(); i++) {
            NdArray weight = pretrainedWeights.get(i);
            NdArray bias = (pretrainedBiases != null && i < pretrainedBiases.size()) ? 
                          pretrainedBiases.get(i) : null;
            
            // 更新对应层的权重
            LoraLinearLayer layer = model.loraLayers.get(i);
            layer.getFrozenWeight().setValue(weight);
            if (bias != null && layer.getBias() != null) {
                layer.getBias().setValue(bias);
            }
        }
        
        return model;
    }
    
    /**
     * 启用所有LoRA适配器
     */
    public void enableAllLora() {
        for (LoraLinearLayer layer : loraLayers) {
            layer.enableLora();
        }
    }
    
    /**
     * 禁用所有LoRA适配器
     */
    public void disableAllLora() {
        for (LoraLinearLayer layer : loraLayers) {
            layer.disableLora();
        }
    }
    
    /**
     * 冻结所有原始权重
     */
    public void freezeAllOriginalWeights() {
        for (LoraLinearLayer layer : loraLayers) {
            layer.freezeOriginalWeights();
        }
    }
    
    /**
     * 解冻所有原始权重
     */
    public void unfreezeAllOriginalWeights() {
        for (LoraLinearLayer layer : loraLayers) {
            layer.unfreezeOriginalWeights();
        }
    }
    
    /**
     * 获取模型的可训练参数数量
     * 
     * @return 可训练参数数量
     */
    public int getTrainableParameterCount() {
        return loraLayers.stream()
                .mapToInt(LoraLinearLayer::getTrainableParameterCount)
                .sum();
    }
    
    /**
     * 获取模型的总参数数量
     * 
     * @return 总参数数量
     */
    public int getTotalParameterCount() {
        return loraLayers.stream()
                .mapToInt(LoraLinearLayer::getTotalParameterCount)
                .sum();
    }
    
    /**
     * 获取参数减少比例
     * 
     * @return 参数减少比例
     */
    public double getParameterReduction() {
        int totalParams = getTotalParameterCount();
        int trainableParams = getTrainableParameterCount();
        return 1.0 - (double) trainableParams / totalParams;
    }
    
    /**
     * 获取所有LoRA参数
     * 
     * @return LoRA参数映射
     */
    public Map<String, Parameter> getAllLoraParameters() {
        Map<String, Parameter> allLoraParams = new HashMap<>();
        for (LoraLinearLayer layer : loraLayers) {
            allLoraParams.putAll(layer.getLoraParameters());
        }
        return allLoraParams;
    }
    
    /**
     * 合并所有LoRA权重到原始权重中
     * 这个操作会将模型转换为等效的标准神经网络
     * 
     * @return 合并后的权重列表
     */
    public List<NdArray> mergeAllLoraWeights() {
        List<NdArray> mergedWeights = new ArrayList<>();
        for (LoraLinearLayer layer : loraLayers) {
            mergedWeights.add(layer.mergeLoraWeights());
        }
        return mergedWeights;
    }
    
    /**
     * 保存LoRA参数到映射中
     * 
     * @return LoRA参数状态
     */
    public Map<String, NdArray> saveLoraState() {
        Map<String, NdArray> state = new HashMap<>();
        for (int i = 0; i < loraLayers.size(); i++) {
            LoraLinearLayer layer = loraLayers.get(i);
            String prefix = String.format("layer_%d", i);
            
            state.put(prefix + ".lora_A", layer.getLoraAdapter().getMatrixA().getValue());
            state.put(prefix + ".lora_B", layer.getLoraAdapter().getMatrixB().getValue());
            
            if (layer.getBias() != null) {
                state.put(prefix + ".bias", layer.getBias().getValue());
            }
        }
        return state;
    }
    
    /**
     * 从映射中加载LoRA参数
     * 
     * @param state LoRA参数状态
     */
    public void loadLoraState(Map<String, NdArray> state) {
        for (int i = 0; i < loraLayers.size(); i++) {
            LoraLinearLayer layer = loraLayers.get(i);
            String prefix = String.format("layer_%d", i);
            
            NdArray loraA = state.get(prefix + ".lora_A");
            NdArray loraB = state.get(prefix + ".lora_B");
            NdArray bias = state.get(prefix + ".bias");
            
            if (loraA != null) {
                layer.getLoraAdapter().getMatrixA().setValue(loraA);
            }
            if (loraB != null) {
                layer.getLoraAdapter().getMatrixB().setValue(loraB);
            }
            if (bias != null && layer.getBias() != null) {
                layer.getBias().setValue(bias);
            }
        }
    }
    
    /**
     * 获取模型层信息
     * 
     * @return 层信息字符串
     */
    public String getModelInfo() {
        StringBuilder info = new StringBuilder();
        info.append(String.format("LoRA模型: %s\n", getName()));
        info.append(String.format("配置: %s\n", config.toString()));
        info.append(String.format("架构: %s\n", java.util.Arrays.toString(layerSizes)));
        info.append(String.format("总参数: %,d\n", getTotalParameterCount()));
        info.append(String.format("可训练参数: %,d\n", getTrainableParameterCount()));
        info.append(String.format("参数减少: %.2f%%\n", getParameterReduction() * 100));
        
        info.append("\n层详情:\n");
        for (int i = 0; i < loraLayers.size(); i++) {
            LoraLinearLayer layer = loraLayers.get(i);
            info.append(String.format("  %d. %s\n", i + 1, layer.toString()));
        }
        
        return info.toString();
    }
    
    /**
     * 验证模型配置的合理性
     */
    public void validateConfiguration() {
        for (LoraLinearLayer layer : loraLayers) {
            int inputDim = layer.getInputShape().getDimension(1);
            int outputDim = layer.getOutputShape().getDimension(1);
            config.validate(inputDim, outputDim);
        }
    }
    
    /**
     * 获取LoRA层列表
     * 
     * @return LoRA层列表
     */
    public List<LoraLinearLayer> getLoraLayers() {
        return new ArrayList<>(loraLayers);
    }
    
    /**
     * 获取指定索引的LoRA层
     * 
     * @param index 层索引
     * @return LoRA层
     */
    public LoraLinearLayer getLoraLayer(int index) {
        if (index < 0 || index >= loraLayers.size()) {
            throw new IndexOutOfBoundsException("Layer index out of bounds: " + index);
        }
        return loraLayers.get(index);
    }
    
    /**
     * 获取LoRA配置
     * 
     * @return LoRA配置
     */
    public LoraConfig getLoraConfig() {
        return config;
    }
    
    /**
     * 获取模型架构
     * 
     * @return 层大小数组
     */
    public int[] getLayerSizes() {
        return layerSizes.clone();
    }
    
    @Override
    public String toString() {
        return String.format("LoraModel{name='%s', layers=%d, trainableParams=%,d/%,d (%.1f%% reduction)}", 
                           getName(), loraLayers.size(), getTrainableParameterCount(), 
                           getTotalParameterCount(), getParameterReduction() * 100);
    }
}