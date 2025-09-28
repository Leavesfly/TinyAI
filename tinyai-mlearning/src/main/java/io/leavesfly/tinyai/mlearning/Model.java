package io.leavesfly.tinyai.mlearning;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.mlearning.inference.Predictor;
import io.leavesfly.tinyai.mlearning.inference.Translator;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.Parameter;
// import io.leavesfly.tinyai.nnet.Uml;

import java.io.*;
import java.util.Map;

/**
 * 机器学习模型类
 * <p>
 * 该类是TinyDL框架中模型的核心表示，提供了模型的完整生命周期管理功能，
 * 包括模型的创建、训练、保存、加载、推理等操作。
 * <p>
 * 主要功能：
 * 1. 模型结构管理：封装神经网络的架构（Block）
 * 2. 模型序列化：支持完整模型、参数、检查点等多种保存方式
 * 3. 模型推理：提供预测接口
 * 4. 模型信息管理：维护模型的元数据信息
 *
 * @author TinyDL
 * @version 1.0
 */
public class Model implements Serializable {

    private static final long serialVersionUID = 1L;

    private String name;

    private Block block;

    // 模型元数据信息
    private ModelInfo modelInfo;

    public transient Variable tmpPredict;

    /**
     * 构造函数
     *
     * @param _name  模型名称
     * @param _block 模型的神经网络结构
     */
    public Model(String _name, Block _block) {
        name = _name;
        block = _block;
        modelInfo = new ModelInfo(_name);
        initializeModelInfo();
    }

    /**
     * 初始化模型信息
     * 包括输入输出形状、参数数量、架构类型等基本信息
     */
    private void initializeModelInfo() {
        if (block != null) {
            modelInfo.setInputShape(block.getInputShape());
            modelInfo.setOutputShape(block.getOutputShape());

            // 统计参数数量
            Map<String, Parameter> params = block.getAllParams();
            long totalParams = 0;
            for (Parameter param : params.values()) {
                totalParams += param.getValue().getShape().size();
            }
            modelInfo.setTotalParameters(totalParams);

            // 设置架构类型（根据Block类型推断）
            String blockClassName = block.getClass().getSimpleName();
            modelInfo.setArchitectureType(blockClassName);
        }
    }

    /**
     * 绘制模型计算图
     * 通过可视化方式展示模型的前向传播计算过程
     */
    public void plot() {
        Shape xInputShape = block.getInputShape();
        if (xInputShape != null) {
            Shape shape = block.getInputShape();
            tmpPredict = block.layerForward(new Variable(NdArray.ones(shape)));
        }
        // System.out.println(Uml.getDotGraph(tmpPredict)); // 暂时注释，因为Uml类不可用
    }

    /**
     * 保存模型到文件（传统方式）
     *
     * @param modelFile 模型文件
     */
    public void save(File modelFile) {
        try (FileOutputStream fileOut = new FileOutputStream(modelFile); ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(this);
        } catch (Exception e) {
            throw new RuntimeException("Model save error!");
        }
    }

    /**
     * 使用ModelSerializer保存模型（推荐使用）
     *
     * @param filePath 保存路径
     */
    public void saveModel(String filePath) {
        ModelSerializer.saveModel(this, filePath);
    }

    /**
     * 保存压缩模型
     *
     * @param filePath 保存路径
     */
    public void saveModelCompressed(String filePath) {
        ModelSerializer.saveModel(this, filePath, true);
    }

    /**
     * 仅保存模型参数
     *
     * @param filePath 保存路径
     */
    public void saveParameters(String filePath) {
        ModelSerializer.saveParameters(this, filePath);
    }

    /**
     * 保存训练检查点
     *
     * @param filePath 保存路径
     * @param epoch    当前训练轮次
     * @param loss     当前损失
     */
    public void saveCheckpoint(String filePath, int epoch, double loss) {
        // 更新模型信息
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epoch);
            modelInfo.setFinalLoss(loss);
            if (loss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(loss);
            }
        }
        ModelSerializer.saveCheckpoint(this, epoch, loss, filePath);
    }

    /**
     * 从文件加载模型（传统方式）
     *
     * @param modelFile 模型文件
     * @return 加载的模型
     */
    public static Model load(File modelFile) {
        try (FileInputStream fileIn = new FileInputStream(modelFile); ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Model) in.readObject();
        } catch (Exception e) {
            throw new RuntimeException("model load error!");
        }
    }

    /**
     * 使用ModelSerializer加载模型（推荐使用）
     *
     * @param filePath 模型文件路径
     * @return 加载的模型
     */
    public static Model loadModel(String filePath) {
        return ModelSerializer.loadModel(filePath);
    }

    /**
     * 加载参数到当前模型
     *
     * @param filePath 参数文件路径
     */
    public void loadParameters(String filePath) {
        ModelSerializer.loadParameters(this, filePath);
    }

    /**
     * 从检查点恢复模型
     *
     * @param filePath 检查点文件路径
     * @return 恢复的模型
     */
    public static Model resumeFromCheckpoint(String filePath) {
        return ModelSerializer.resumeFromCheckpoint(filePath);
    }

    /**
     * 重置模型状态
     * 主要用于RNN等有状态的模型，清除历史状态信息
     */
    public void resetState() {
        block.resetState();
    }

    /**
     * 模型前向传播
     *
     * @param inputs 输入变量
     * @return 输出变量
     */
    public Variable forward(Variable... inputs) {
        return block.layerForward(inputs);
    }

    /**
     * 清除梯度
     * 在每次反向传播前调用，清除历史梯度信息
     */
    public void clearGrads() {
        block.clearGrads();
    }

    /**
     * 获取所有参数
     *
     * @return 参数映射
     */
    public Map<String, Parameter> getAllParams() {
        return block.getAllParams();
    }

    /**
     * 获取预测器
     *
     * @param translator 翻译器
     * @return 预测器
     */
    public <I, O> Predictor<I, O> getPredictor(Translator<I, O> translator) {
        return new Predictor<>(translator, this);
    }

    /**
     * 获取模型名称
     *
     * @return 模型名称
     */
    public String getName() {
        return name;
    }

    /**
     * 获取模型结构
     *
     * @return 模型结构块
     */
    public Block getBlock() {
        return block;
    }

    // =========== 模型信息相关方法 ===========

    /**
     * 获取模型信息
     *
     * @return 模型信息
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * 设置模型信息
     *
     * @param modelInfo 模型信息
     */
    public void setModelInfo(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    /**
     * 更新训练信息
     *
     * @param epochs       训练轮次
     * @param finalLoss    最终损失
     * @param optimizer    优化器名称
     * @param learningRate 学习率
     */
    public void updateTrainingInfo(int epochs, double finalLoss, String optimizer, double learningRate) {
        if (modelInfo != null) {
            modelInfo.setTrainedEpochs(epochs);
            modelInfo.setFinalLoss(finalLoss);
            modelInfo.setOptimizerType(optimizer);
            modelInfo.setLearningRate(learningRate);

            if (finalLoss < modelInfo.getBestLoss() || modelInfo.getBestLoss() == 0) {
                modelInfo.setBestLoss(finalLoss);
            }
        }
    }

    /**
     * 添加性能指标
     *
     * @param metricName 指标名称
     * @param value      指标值
     */
    public void addMetric(String metricName, double value) {
        if (modelInfo != null) {
            modelInfo.addMetric(metricName, value);
        }
    }

    /**
     * 设置模型描述
     *
     * @param description 描述
     */
    public void setDescription(String description) {
        if (modelInfo != null) {
            modelInfo.setDescription(description);
        }
    }

    /**
     * 获取模型简要信息
     *
     * @return 简要信息
     */
    public String getModelSummary() {
        if (modelInfo != null) {
            return modelInfo.getSummary();
        }
        return "Model: " + name + "\nNo detailed info available.";
    }

    /**
     * 获取模型详细信息
     *
     * @return 详细信息
     */
    public String getModelDetailedInfo() {
        if (modelInfo != null) {
            return modelInfo.getDetailedInfo();
        }
        return getModelSummary();
    }

    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println(getModelDetailedInfo());
    }

    /**
     * 验证模型文件
     *
     * @param filePath 文件路径
     * @return 是否有效
     */
    public static boolean validateModel(String filePath) {
        return ModelSerializer.validateModelFile(filePath);
    }

}