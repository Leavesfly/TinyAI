package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.activate.SigmoidLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.HashMap;
import java.util.Map;

/**
 * 代码生成专门模块
 * 
 * 专门针对代码生成任务优化的模块，包含以下核心功能：
 * 1. 代码语言识别 - 自动识别目标编程语言
 * 2. 代码结构分析 - 分析代码的结构特征
 * 3. 语法验证 - 验证生成代码的语法正确性
 * 4. 代码质量评估 - 评估代码的质量和可维护性
 * 5. 编程范式适配 - 适配不同的编程范式
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CodeGenerationBlock extends Block {
    
    /**
     * 模型维度
     */
    private final int dModel;
    
    /**
     * 支持的编程语言数量
     */
    private final int numProgrammingLanguages;
    
    /**
     * 代码语言识别器
     */
    private LinearLayer languageClassifierLayer1;
    private ReLuLayer languageClassifierActivation;
    private LinearLayer languageClassifierLayer2;
    private SigmoidLayer languageClassifierSoftmax;
    
    /**
     * 代码结构分析器
     */
    private LinearLayer structureAnalyzerLayer1;
    private ReLuLayer structureAnalyzerActivation1;
    private LinearLayer structureAnalyzerLayer2;
    private ReLuLayer structureAnalyzerActivation2;
    private LinearLayer structureAnalyzerLayer3;
    
    /**
     * 语法验证器
     */
    private LinearLayer syntaxValidatorLayer1;
    private ReLuLayer syntaxValidatorActivation1;
    private LinearLayer syntaxValidatorLayer2;
    private ReLuLayer syntaxValidatorActivation2;
    private LinearLayer syntaxValidatorLayer3;
    private SigmoidLayer syntaxValidatorSigmoid;
    
    /**
     * 代码质量评估器
     */
    private LinearLayer qualityAssessorLayer1;
    private ReLuLayer qualityAssessorActivation1;
    private LinearLayer qualityAssessorLayer2;
    private ReLuLayer qualityAssessorActivation2;
    private LinearLayer qualityAssessorLayer3;
    private SigmoidLayer qualityAssessorSigmoid;
    
    /**
     * 编程范式适配器
     */
    private LinearLayer paradigmAdapterLayer1;
    private ReLuLayer paradigmAdapterActivation;
    private LinearLayer paradigmAdapterLayer2;
    
    /**
     * 编程语言映射
     */
    private final Map<Integer, String> languageMapping;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param dModel 模型维度
     * @param numProgrammingLanguages 支持的编程语言数量
     */
    public CodeGenerationBlock(String name, int dModel, int numProgrammingLanguages) {
        super(name, Shape.of(-1, dModel), Shape.of(-1, dModel));
        
        this.dModel = dModel;
        this.numProgrammingLanguages = numProgrammingLanguages;
        this.languageMapping = initializeLanguageMapping();
        
        init();
    }
    
    /**
     * 默认构造函数 - 支持10种编程语言
     */
    public CodeGenerationBlock(String name, int dModel) {
        this(name, dModel, 10);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            initLanguageClassifier();
            initStructureAnalyzer();
            initSyntaxValidator();
            initQualityAssessor();
            initParadigmAdapter();
            
            alreadyInit = true;
        }
    }
    
    /**
     * 初始化编程语言映射
     */
    private Map<Integer, String> initializeLanguageMapping() {
        Map<Integer, String> mapping = new HashMap<>();
        mapping.put(0, "Java");
        mapping.put(1, "Python");
        mapping.put(2, "JavaScript");
        mapping.put(3, "C++");
        mapping.put(4, "C#");
        mapping.put(5, "Go");
        mapping.put(6, "Rust");
        mapping.put(7, "TypeScript");
        mapping.put(8, "Kotlin");
        mapping.put(9, "Swift");
        return mapping;
    }
    
    /**
     * 初始化代码语言识别器
     */
    private void initLanguageClassifier() {
        // 语言分类器：dModel -> 128 -> numProgrammingLanguages
        languageClassifierLayer1 = new LinearLayer(name + "_lang_classifier1", dModel, 128, true);
        addLayer(languageClassifierLayer1);
        
        languageClassifierActivation = new ReLuLayer(name + "_lang_classifier_relu", Shape.of(-1, 128));
        addLayer(languageClassifierActivation);
        
        languageClassifierLayer2 = new LinearLayer(name + "_lang_classifier2", 128, numProgrammingLanguages, true);
        addLayer(languageClassifierLayer2);
        
        languageClassifierSoftmax = new SigmoidLayer(name + "_lang_classifier_softmax");
        addLayer(languageClassifierSoftmax);
    }
    
    /**
     * 初始化代码结构分析器
     */
    private void initStructureAnalyzer() {
        // 结构分析器：dModel -> dModel*2 -> dModel*2 -> dModel
        structureAnalyzerLayer1 = new LinearLayer(name + "_structure1", dModel, dModel * 2, true);
        addLayer(structureAnalyzerLayer1);
        
        structureAnalyzerActivation1 = new ReLuLayer(name + "_structure_relu1", Shape.of(-1, dModel * 2));
        addLayer(structureAnalyzerActivation1);
        
        structureAnalyzerLayer2 = new LinearLayer(name + "_structure2", dModel * 2, dModel * 2, true);
        addLayer(structureAnalyzerLayer2);
        
        structureAnalyzerActivation2 = new ReLuLayer(name + "_structure_relu2", Shape.of(-1, dModel * 2));
        addLayer(structureAnalyzerActivation2);
        
        structureAnalyzerLayer3 = new LinearLayer(name + "_structure3", dModel * 2, dModel, true);
        addLayer(structureAnalyzerLayer3);
    }
    
    /**
     * 初始化语法验证器
     */
    private void initSyntaxValidator() {
        // 语法验证器：dModel -> 256 -> 128 -> 1
        syntaxValidatorLayer1 = new LinearLayer(name + "_syntax1", dModel, 256, true);
        addLayer(syntaxValidatorLayer1);
        
        syntaxValidatorActivation1 = new ReLuLayer(name + "_syntax_relu1", Shape.of(-1, 256));
        addLayer(syntaxValidatorActivation1);
        
        syntaxValidatorLayer2 = new LinearLayer(name + "_syntax2", 256, 128, true);
        addLayer(syntaxValidatorLayer2);
        
        syntaxValidatorActivation2 = new ReLuLayer(name + "_syntax_relu2", Shape.of(-1, 128));
        addLayer(syntaxValidatorActivation2);
        
        syntaxValidatorLayer3 = new LinearLayer(name + "_syntax3", 128, 1, true);
        addLayer(syntaxValidatorLayer3);
        
        syntaxValidatorSigmoid = new SigmoidLayer(name + "_syntax_sigmoid");
        addLayer(syntaxValidatorSigmoid);
    }
    
    /**
     * 初始化代码质量评估器
     */
    private void initQualityAssessor() {
        // 质量评估器：dModel -> 512 -> 256 -> 1
        qualityAssessorLayer1 = new LinearLayer(name + "_quality1", dModel, 512, true);
        addLayer(qualityAssessorLayer1);
        
        qualityAssessorActivation1 = new ReLuLayer(name + "_quality_relu1", Shape.of(-1, 512));
        addLayer(qualityAssessorActivation1);
        
        qualityAssessorLayer2 = new LinearLayer(name + "_quality2", 512, 256, true);
        addLayer(qualityAssessorLayer2);
        
        qualityAssessorActivation2 = new ReLuLayer(name + "_quality_relu2", Shape.of(-1, 256));
        addLayer(qualityAssessorActivation2);
        
        qualityAssessorLayer3 = new LinearLayer(name + "_quality3", 256, 1, true);
        addLayer(qualityAssessorLayer3);
        
        qualityAssessorSigmoid = new SigmoidLayer(name + "_quality_sigmoid");
        addLayer(qualityAssessorSigmoid);
    }
    
    /**
     * 初始化编程范式适配器
     */
    private void initParadigmAdapter() {
        // 范式适配器：dModel -> dModel*3 -> dModel
        paradigmAdapterLayer1 = new LinearLayer(name + "_paradigm1", dModel, dModel * 3, true);
        addLayer(paradigmAdapterLayer1);
        
        paradigmAdapterActivation = new ReLuLayer(name + "_paradigm_relu", Shape.of(-1, dModel * 3));
        addLayer(paradigmAdapterActivation);
        
        paradigmAdapterLayer2 = new LinearLayer(name + "_paradigm2", dModel * 3, dModel, true);
        addLayer(paradigmAdapterLayer2);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable reasoningOutput = inputs[0];
        
        // 执行代码生成分析
        CodeGenerationResult result = performCodeGenerationAnalysis(reasoningOutput);
        
        // 返回增强的推理输出
        return result.enhancedOutput;
    }
    
    /**
     * 执行代码生成分析
     * 
     * @param reasoningOutput 推理模块的输出
     * @return 代码生成分析结果
     */
    public CodeGenerationResult performCodeGenerationAnalysis(Variable reasoningOutput) {
        // 代码语言识别
        LanguageDetectionResult languageResult = detectProgrammingLanguage(reasoningOutput);
        
        // 代码结构分析
        Variable structureFeatures = analyzeCodeStructure(reasoningOutput);
        
        // 语法验证
        float syntaxScore = validateSyntax(structureFeatures);
        
        // 代码质量评估
        float qualityScore = assessCodeQuality(structureFeatures);
        
        // 编程范式适配
        Variable adaptedOutput = adaptProgrammingParadigm(reasoningOutput);
        
        // 计算综合代码置信度
        float codeConfidence = computeCodeConfidence(syntaxScore, qualityScore, languageResult.confidence);
        
        // 创建代码信息
        Map<String, Object> codeInfo = new HashMap<>();
        codeInfo.put("language_distribution", languageResult.languageDistribution);
        codeInfo.put("detected_language", languageResult.detectedLanguage);
        codeInfo.put("language_confidence", languageResult.confidence);
        codeInfo.put("structure_quality", computeStructureQuality(structureFeatures));
        codeInfo.put("syntax_score", syntaxScore);
        codeInfo.put("quality_score", qualityScore);
        codeInfo.put("code_confidence", codeConfidence);
        codeInfo.put("paradigm_adaptation", "applied");
        
        return new CodeGenerationResult(adaptedOutput, codeInfo);
    }
    
    /**
     * 检测编程语言
     */
    private LanguageDetectionResult detectProgrammingLanguage(Variable input) {
        // 语言分类
        Variable langFeatures = languageClassifierLayer1.layerForward(input);
        langFeatures = languageClassifierActivation.layerForward(langFeatures);
        Variable langLogits = languageClassifierLayer2.layerForward(langFeatures);
        Variable langProbs = languageClassifierSoftmax.layerForward(langLogits);
        
        NdArray probsData = langProbs.getValue();
        float[] languageDistribution = new float[numProgrammingLanguages];
        
        int maxIndex = 0;
        float maxProb = 0.0f;
        
        for (int i = 0; i < numProgrammingLanguages && i < probsData.getShape().getDimension(1); i++) {
            float prob = probsData.get(0, i);
            languageDistribution[i] = prob;
            
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = i;
            }
        }
        
        String detectedLanguage = languageMapping.getOrDefault(maxIndex, "Unknown");
        
        return new LanguageDetectionResult(detectedLanguage, maxProb, languageDistribution);
    }
    
    /**
     * 分析代码结构
     */
    private Variable analyzeCodeStructure(Variable input) {
        Variable features1 = structureAnalyzerLayer1.layerForward(input);
        features1 = structureAnalyzerActivation1.layerForward(features1);
        
        Variable features2 = structureAnalyzerLayer2.layerForward(features1);
        features2 = structureAnalyzerActivation2.layerForward(features2);
        
        return structureAnalyzerLayer3.layerForward(features2);
    }
    
    /**
     * 验证语法
     */
    private float validateSyntax(Variable structureFeatures) {
        Variable hidden1 = syntaxValidatorLayer1.layerForward(structureFeatures);
        hidden1 = syntaxValidatorActivation1.layerForward(hidden1);
        
        Variable hidden2 = syntaxValidatorLayer2.layerForward(hidden1);
        hidden2 = syntaxValidatorActivation2.layerForward(hidden2);
        
        Variable syntaxScore = syntaxValidatorLayer3.layerForward(hidden2);
        syntaxScore = syntaxValidatorSigmoid.layerForward(syntaxScore);
        
        return syntaxScore.getValue().get(0, 0);
    }
    
    /**
     * 评估代码质量
     */
    private float assessCodeQuality(Variable structureFeatures) {
        Variable hidden1 = qualityAssessorLayer1.layerForward(structureFeatures);
        hidden1 = qualityAssessorActivation1.layerForward(hidden1);
        
        Variable hidden2 = qualityAssessorLayer2.layerForward(hidden1);
        hidden2 = qualityAssessorActivation2.layerForward(hidden2);
        
        Variable qualityScore = qualityAssessorLayer3.layerForward(hidden2);
        qualityScore = qualityAssessorSigmoid.layerForward(qualityScore);
        
        return qualityScore.getValue().get(0, 0);
    }
    
    /**
     * 适配编程范式
     */
    private Variable adaptProgrammingParadigm(Variable input) {
        Variable paradigmFeatures = paradigmAdapterLayer1.layerForward(input);
        paradigmFeatures = paradigmAdapterActivation.layerForward(paradigmFeatures);
        return paradigmAdapterLayer2.layerForward(paradigmFeatures);
    }
    
    /**
     * 计算结构质量
     */
    private float computeStructureQuality(Variable structureFeatures) {
        NdArray featuresData = structureFeatures.getValue();
        
        // 计算特征向量的L2范数作为结构质量指标
        float sum = 0.0f;
        int totalElements = featuresData.getShape().size();
        
        for (int i = 0; i < totalElements; i++) {
            float value = featuresData.getByIndex(i);
            sum += value * value;
        }
        
        return (float) Math.sqrt(sum / totalElements);
    }
    
    /**
     * 计算综合代码置信度
     */
    private float computeCodeConfidence(float syntaxScore, float qualityScore, float languageConfidence) {
        // 加权平均计算综合置信度
        float syntaxWeight = 0.4f;
        float qualityWeight = 0.4f;
        float languageWeight = 0.2f;
        
        return syntaxWeight * syntaxScore + 
               qualityWeight * qualityScore + 
               languageWeight * languageConfidence;
    }
    
    /**
     * 语言检测结果
     */
    private static class LanguageDetectionResult {
        final String detectedLanguage;
        final float confidence;
        final float[] languageDistribution;
        
        LanguageDetectionResult(String detectedLanguage, float confidence, float[] languageDistribution) {
            this.detectedLanguage = detectedLanguage;
            this.confidence = confidence;
            this.languageDistribution = languageDistribution;
        }
    }
    
    /**
     * 代码生成结果包装类
     */
    public static class CodeGenerationResult {
        public final Variable enhancedOutput;
        public final Map<String, Object> codeInfo;
        
        public CodeGenerationResult(Variable enhancedOutput, Map<String, Object> codeInfo) {
            this.enhancedOutput = enhancedOutput;
            this.codeInfo = codeInfo;
        }
        
        /**
         * 获取代码置信度
         */
        public float getCodeConfidence() {
            Object confidence = codeInfo.get("code_confidence");
            return confidence instanceof Float ? (Float) confidence : 0.0f;
        }
        
        /**
         * 获取检测到的编程语言
         */
        public String getDetectedLanguage() {
            Object language = codeInfo.get("detected_language");
            return language instanceof String ? (String) language : "Unknown";
        }
        
        /**
         * 获取语法得分
         */
        public float getSyntaxScore() {
            Object score = codeInfo.get("syntax_score");
            return score instanceof Float ? (Float) score : 0.0f;
        }
        
        /**
         * 获取质量得分
         */
        public float getQualityScore() {
            Object score = codeInfo.get("quality_score");
            return score instanceof Float ? (Float) score : 0.0f;
        }
    }
    
    // Getters
    public int getDModel() {
        return dModel;
    }
    
    public int getNumProgrammingLanguages() {
        return numProgrammingLanguages;
    }
    
    public Map<Integer, String> getLanguageMapping() {
        return languageMapping;
    }
    
    @Override
    public String toString() {
        return String.format("CodeGenerationBlock{name='%s', dModel=%d, numLanguages=%d}", 
                           name, dModel, numProgrammingLanguages);
    }
}