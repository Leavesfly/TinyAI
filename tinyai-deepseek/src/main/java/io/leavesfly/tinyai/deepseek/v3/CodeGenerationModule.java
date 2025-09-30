package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.func.math.ReLu;

import java.util.*;

/**
 * 代码生成专门模块
 * 
 * 专门针对代码生成任务优化的模块，提供：
 * 1. 编程语言识别和分类
 * 2. 代码结构分析
 * 3. 语法验证
 * 4. 代码质量评估
 * 5. 多语言支持
 * 6. 代码风格检查
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CodeGenerationModule extends LayerAble {
    
    // ========== 配置参数 ==========
    private int dModel;                          // 模型维度
    private int numSupportedLanguages;           // 支持的编程语言数量
    private double syntaxThreshold;              // 语法验证阈值
    
    // ========== 网络组件 ==========
    private LinearLayer languageClassifier;           // 编程语言分类器
    private LayerAble structureAnalyzer;         // 代码结构分析器
    private LinearLayer syntaxValidator;              // 语法验证器
    private LayerAble qualityAssessor;           // 质量评估器
    private LayerAble styleChecker;              // 代码风格检查器
    
    // ========== 语言支持 ==========
    private List<String> supportedLanguages;     // 支持的编程语言列表
    private Map<String, Double> languageWeights; // 语言权重映射
    
    // ========== 运行时状态 ==========
    private CodeGenerationResult lastResult;     // 最近的生成结果
    private Map<String, Integer> languageUsageCount; // 语言使用统计
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param dModel 模型维度
     */
    public CodeGenerationModule(String name, int dModel) {
        this.name = name;
        this.dModel = dModel;
        this.syntaxThreshold = 0.7;
        
        // 初始化支持的编程语言
        initializeSupportedLanguages();
        this.numSupportedLanguages = supportedLanguages.size();
        
        // 设置输入输出形状
        this.inputShape = Shape.of(-1, dModel);  // [batch, dModel]
        this.outputShape = Shape.of(-1, 1);      // [batch, 1] - 代码质量分数
        
        // 初始化统计数据
        this.languageUsageCount = new HashMap<>();
        for (String lang : supportedLanguages) {
            languageUsageCount.put(lang, 0);
        }
        
        init();
    }
    
    /**
     * 初始化支持的编程语言
     */
    private void initializeSupportedLanguages() {
        supportedLanguages = Arrays.asList(
            "Java", "Python", "JavaScript", "C++", "C#",
            "Go", "Rust", "TypeScript", "Kotlin", "Swift"
        );
        
        // 设置语言权重（基于流行度和复杂度）
        languageWeights = new HashMap<>();
        languageWeights.put("Java", 1.0);
        languageWeights.put("Python", 0.9);
        languageWeights.put("JavaScript", 0.8);
        languageWeights.put("C++", 1.1);
        languageWeights.put("C#", 0.95);
        languageWeights.put("Go", 0.85);
        languageWeights.put("Rust", 1.05);
        languageWeights.put("TypeScript", 0.9);
        languageWeights.put("Kotlin", 0.9);
        languageWeights.put("Swift", 0.85);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化编程语言分类器
            languageClassifier = new Linear(name + "_lang_classifier", dModel, numSupportedLanguages, false);
            languageClassifier.init();
            
            // 2. 初始化代码结构分析器
            structureAnalyzer = new StructureAnalyzer(name + "_structure", dModel);
            structureAnalyzer.init();
            
            // 3. 初始化语法验证器
            syntaxValidator = new Linear(name + "_syntax", dModel, 1, false);
            syntaxValidator.init();
            
            // 4. 初始化质量评估器
            qualityAssessor = new QualityAssessor(name + "_quality", dModel);
            qualityAssessor.init();
            
            // 5. 初始化代码风格检查器
            styleChecker = new StyleChecker(name + "_style", dModel);
            styleChecker.init();
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable reasoningOutput = inputs[0]; // 来自推理模块的输出
        
        // 执行代码生成分析
        CodeGenerationResult result = analyzeCodeGeneration(reasoningOutput);
        this.lastResult = result;
        
        // 返回代码质量分数
        return new Variable(NdArray.of(result.getCodeConfidence()).reshape(Shape.of(1, 1)));
    }
    
    /**
     * 分析代码生成
     * 
     * @param reasoningOutput 推理模块输出
     * @return 代码生成分析结果
     */
    public CodeGenerationResult analyzeCodeGeneration(Variable reasoningOutput) {
        // 1. 编程语言识别
        LanguageRecognitionResult langResult = recognizeLanguage(reasoningOutput);
        
        // 2. 代码结构分析
        StructureAnalysisResult structureResult = analyzeStructure(reasoningOutput);
        
        // 3. 语法验证
        double syntaxScore = validateSyntax(reasoningOutput);
        
        // 4. 质量评估
        QualityAssessmentResult qualityResult = assessQuality(reasoningOutput);
        
        // 5. 代码风格检查
        StyleCheckResult styleResult = checkStyle(reasoningOutput);
        
        // 6. 综合计算代码置信度
        double codeConfidence = calculateCodeConfidence(
            langResult, structureResult, syntaxScore, qualityResult, styleResult);
        
        return new CodeGenerationResult(
            langResult,
            structureResult,
            syntaxScore,
            qualityResult,
            styleResult,
            codeConfidence
        );
    }
    
    /**
     * 识别编程语言
     * 
     * @param input 输入张量
     * @return 语言识别结果
     */
    private LanguageRecognitionResult recognizeLanguage(Variable input) {
        // 1. 语言分类
        Variable languageProbs = languageClassifier.layerForward(input);
        languageProbs = applySoftmax(languageProbs);
        
        // 2. 解析概率分布
        NdArray probsData = languageProbs.getValue();
        double[] probsArray = probsData.toDoubleArray();
        
        // 3. 找到概率最高的语言
        int maxIdx = 0;
        double maxProb = probsArray[0];
        for (int i = 1; i < probsArray.length && i < supportedLanguages.size(); i++) {
            if (probsArray[i] > maxProb) {
                maxProb = probsArray[i];
                maxIdx = i;
            }
        }
        
        String detectedLanguage = supportedLanguages.get(maxIdx);
        
        // 4. 构建语言分布
        Map<String, Double> languageDistribution = new HashMap<>();
        for (int i = 0; i < supportedLanguages.size() && i < probsArray.length; i++) {
            languageDistribution.put(supportedLanguages.get(i), (double) probsArray[i]);
        }
        
        // 5. 更新使用统计
        languageUsageCount.put(detectedLanguage, languageUsageCount.get(detectedLanguage) + 1);
        
        return new LanguageRecognitionResult(detectedLanguage, maxProb, languageDistribution);
    }
    
    /**
     * 分析代码结构
     * 
     * @param input 输入张量
     * @return 结构分析结果
     */
    private StructureAnalysisResult analyzeStructure(Variable input) {
        Variable structureFeatures = structureAnalyzer.layerForward(input);
        
        // 计算结构质量分数
        NdArray featuresData = structureFeatures.getValue();
        double structureQuality = calculateStructureQuality(featuresData);
        
        // 分析结构特征
        Map<String, Double> structureMetrics = extractStructureMetrics(featuresData);
        
        return new StructureAnalysisResult(structureQuality, structureMetrics);
    }
    
    /**
     * 验证语法
     * 
     * @param input 输入张量
     * @return 语法分数
     */
    private double validateSyntax(Variable input) {
        Variable syntaxOutput = syntaxValidator.layerForward(input);
        NdArray syntaxData = syntaxOutput.getValue();
        double[] syntaxArray = syntaxData.toDoubleArray();
        
        // 使用Sigmoid激活得到语法分数
        double syntaxScore = 0.0;
        for (double value : syntaxArray) {
            syntaxScore += 1.0 / (1.0 + Math.exp(-value));
        }
        return syntaxScore / syntaxArray.length;
    }
    
    /**
     * 评估代码质量
     * 
     * @param input 输入张量
     * @return 质量评估结果
     */
    private QualityAssessmentResult assessQuality(Variable input) {
        Variable qualityOutput = qualityAssessor.layerForward(input);
        
        // 解析质量指标
        NdArray qualityData = qualityOutput.getValue();
        Map<String, Double> qualityMetrics = extractQualityMetrics(qualityData);
        
        // 计算总体质量分数
        double overallQuality = qualityMetrics.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
        
        return new QualityAssessmentResult(overallQuality, qualityMetrics);
    }
    
    /**
     * 检查代码风格
     * 
     * @param input 输入张量
     * @return 风格检查结果
     */
    private StyleCheckResult checkStyle(Variable input) {
        Variable styleOutput = styleChecker.layerForward(input);
        
        // 解析风格指标
        NdArray styleData = styleOutput.getValue();
        Map<String, Double> styleMetrics = extractStyleMetrics(styleData);
        
        // 计算风格分数
        double styleScore = styleMetrics.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
        
        return new StyleCheckResult(styleScore, styleMetrics);
    }
    
    /**
     * 计算代码置信度
     * 
     * @param langResult 语言识别结果
     * @param structureResult 结构分析结果
     * @param syntaxScore 语法分数
     * @param qualityResult 质量评估结果
     * @param styleResult 风格检查结果
     * @return 代码置信度
     */
    private double calculateCodeConfidence(LanguageRecognitionResult langResult,
                                         StructureAnalysisResult structureResult,
                                         double syntaxScore,
                                         QualityAssessmentResult qualityResult,
                                         StyleCheckResult styleResult) {
        
        // 权重配置
        double langWeight = 0.2;      // 语言识别权重
        double syntaxWeight = 0.3;    // 语法验证权重
        double structureWeight = 0.2; // 结构分析权重
        double qualityWeight = 0.2;   // 质量评估权重
        double styleWeight = 0.1;     // 风格检查权重
        
        // 加权计算置信度
        double confidence = langWeight * langResult.getConfidence() +
                           syntaxWeight * syntaxScore +
                           structureWeight * structureResult.getStructureQuality() +
                           qualityWeight * qualityResult.getOverallQuality() +
                           styleWeight * styleResult.getStyleScore();
        
        // 语言特定调整
        String detectedLang = langResult.getDetectedLanguage();
        if (languageWeights.containsKey(detectedLang)) {
            confidence *= languageWeights.get(detectedLang);
        }
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * 应用Softmax激活
     */
    private Variable applySoftmax(Variable x) {
        return x.softMax();
    }
    
    /**
     * 计算结构质量
     */
    private double calculateStructureQuality(NdArray features) {
        double[] featuresArray = features.toDoubleArray();
        
        // 简化计算：使用特征向量的模长作为质量指标
        double norm = 0.0;
        for (double value : featuresArray) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        
        // 归一化到[0,1]区间
        return 1.0 / (1.0 + Math.exp(-norm / featuresArray.length));
    }
    
    /**
     * 提取结构特征
     */
    private Map<String, Double> extractStructureMetrics(NdArray features) {
        Map<String, Double> metrics = new HashMap<>();
        double[] featuresArray = features.toDoubleArray();
        
        // 模拟结构特征（实际实现中会有更复杂的分析）
        metrics.put("complexity", calculateComplexity(featuresArray));
        metrics.put("modularity", calculateModularity(featuresArray));
        metrics.put("cohesion", calculateCohesion(featuresArray));
        metrics.put("coupling", calculateCoupling(featuresArray));
        
        return metrics;
    }
    
    /**
     * 提取质量特征
     */
    private Map<String, Double> extractQualityMetrics(NdArray quality) {
        Map<String, Double> metrics = new HashMap<>();
        double[] qualityArray = quality.toDoubleArray();
        
        // 模拟质量特征
        metrics.put("readability", Math.abs(qualityArray[0 % qualityArray.length]));
        metrics.put("maintainability", Math.abs(qualityArray[1 % qualityArray.length]));
        metrics.put("efficiency", Math.abs(qualityArray[2 % qualityArray.length]));
        metrics.put("reliability", Math.abs(qualityArray[3 % qualityArray.length]));
        
        // 归一化到[0,1]
        metrics.replaceAll((k, v) -> 1.0 / (1.0 + Math.exp(-v)));
        
        return metrics;
    }
    
    /**
     * 提取风格特征
     */
    private Map<String, Double> extractStyleMetrics(NdArray style) {
        Map<String, Double> metrics = new HashMap<>();
        double[] styleArray = style.toDoubleArray();
        
        // 模拟风格特征
        metrics.put("naming_convention", Math.abs(styleArray[0 % styleArray.length]));
        metrics.put("indentation", Math.abs(styleArray[1 % styleArray.length]));
        metrics.put("commenting", Math.abs(styleArray[2 % styleArray.length]));
        metrics.put("line_length", Math.abs(styleArray[3 % styleArray.length]));
        
        // 归一化到[0,1]
        metrics.replaceAll((k, v) -> 1.0 / (1.0 + Math.exp(-v)));
        
        return metrics;
    }
    
    // 简化的结构特征计算方法
    private double calculateComplexity(double[] features) {
        return Math.abs(features[0]) / (1.0 + Math.abs(features[0]));
    }
    
    private double calculateModularity(double[] features) {
        return Math.abs(features[1 % features.length]) / (1.0 + Math.abs(features[1 % features.length]));
    }
    
    private double calculateCohesion(double[] features) {
        return Math.abs(features[2 % features.length]) / (1.0 + Math.abs(features[2 % features.length]));
    }
    
    private double calculateCoupling(double[] features) {
        return Math.abs(features[3 % features.length]) / (1.0 + Math.abs(features[3 % features.length]));
    }
    
    /**
     * 获取代码生成统计报告
     * 
     * @return 统计报告
     */
    public String getCodeGenerationReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 代码生成模块报告 ===\n");
        sb.append(String.format("支持语言数: %d\n", numSupportedLanguages));
        sb.append(String.format("语法验证阈值: %.2f\n", syntaxThreshold));
        
        if (lastResult != null) {
            sb.append("最近分析结果:\n");
            sb.append(lastResult.getSummary());
        }
        
        sb.append("语言使用统计:\n");
        languageUsageCount.forEach((lang, count) ->
            sb.append(String.format("  - %s: %d次\n", lang, count)));
        
        sb.append("====================");
        return sb.toString();
    }
    
    @Override
    public void clearGrads() {
        languageClassifier.clearGrads();
        structureAnalyzer.clearGrads();
        syntaxValidator.clearGrads();
        qualityAssessor.clearGrads();
        styleChecker.clearGrads();
    }
    
    // ========== Inner Classes ==========
    
    /**
     * 结构分析器
     */
    private static class StructureAnalyzer extends LayerAble {
        private LinearLayer layer1, layer2;
        
        public StructureAnalyzer(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel);
            this.outputShape = Shape.of(-1, dModel);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), inputShape.getDimension(-1) * 2, false);
                layer2 = new LinearLayer(name + "_l2", inputShape.getDimension(-1) * 2, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // 使用ReLU激活
            x = layer2.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
        }
    }
    
    /**
     * 质量评估器
     */
    private static class QualityAssessor extends LayerAble {
        private LinearLayer layer1, layer2, layer3;
        
        public QualityAssessor(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel);
            this.outputShape = Shape.of(-1, 64);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_l1", inputShape.getDimension(-1), 128, false);
                layer2 = new LinearLayer(name + "_l2", 128, 64, false);
                layer3 = new LinearLayer(name + "_l3", 64, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                layer3.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new ReLu(name + "_relu1").layerForward(x);
            x = layer2.layerForward(x);
            x = new ReLu(name + "_relu2").layerForward(x);
            x = layer3.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
            layer3.clearGrads();
        }
    }
    
    /**
     * 风格检查器
     */
    private static class StyleChecker extends LayerAble {
        private LinearLayer layer1, layer2;
        
        public StyleChecker(String name, int dModel) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel);
            this.outputShape = Shape.of(-1, 32);
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new Linear(name + "_l1", inputShape.getDimension(-1), 64, false);
                layer2 = new Linear(name + "_l2", 64, outputShape.getDimension(-1), false);
                layer1.init();
                layer2.init();
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new ReLu().forward(x.getValue())); // 使用ReLU激活
            x = layer2.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
        }
    }
    
    // ========== Result Classes ==========
    
    /**
     * 语言识别结果
     */
    public static class LanguageRecognitionResult {
        private String detectedLanguage;
        private double confidence;
        private Map<String, Double> languageDistribution;
        
        public LanguageRecognitionResult(String detectedLanguage, double confidence, 
                                       Map<String, Double> languageDistribution) {
            this.detectedLanguage = detectedLanguage;
            this.confidence = confidence;
            this.languageDistribution = languageDistribution;
        }
        
        public String getDetectedLanguage() { return detectedLanguage; }
        public double getConfidence() { return confidence; }
        public Map<String, Double> getLanguageDistribution() { return languageDistribution; }
    }
    
    /**
     * 结构分析结果
     */
    public static class StructureAnalysisResult {
        private double structureQuality;
        private Map<String, Double> structureMetrics;
        
        public StructureAnalysisResult(double structureQuality, Map<String, Double> structureMetrics) {
            this.structureQuality = structureQuality;
            this.structureMetrics = structureMetrics;
        }
        
        public double getStructureQuality() { return structureQuality; }
        public Map<String, Double> getStructureMetrics() { return structureMetrics; }
    }
    
    /**
     * 质量评估结果
     */
    public static class QualityAssessmentResult {
        private double overallQuality;
        private Map<String, Double> qualityMetrics;
        
        public QualityAssessmentResult(double overallQuality, Map<String, Double> qualityMetrics) {
            this.overallQuality = overallQuality;
            this.qualityMetrics = qualityMetrics;
        }
        
        public double getOverallQuality() { return overallQuality; }
        public Map<String, Double> getQualityMetrics() { return qualityMetrics; }
    }
    
    /**
     * 风格检查结果
     */
    public static class StyleCheckResult {
        private double styleScore;
        private Map<String, Double> styleMetrics;
        
        public StyleCheckResult(double styleScore, Map<String, Double> styleMetrics) {
            this.styleScore = styleScore;
            this.styleMetrics = styleMetrics;
        }
        
        public double getStyleScore() { return styleScore; }
        public Map<String, Double> getStyleMetrics() { return styleMetrics; }
    }
    
    /**
     * 代码生成结果
     */
    public static class CodeGenerationResult {
        private LanguageRecognitionResult languageResult;
        private StructureAnalysisResult structureResult;
        private double syntaxScore;
        private QualityAssessmentResult qualityResult;
        private StyleCheckResult styleResult;
        private double codeConfidence;
        
        public CodeGenerationResult(LanguageRecognitionResult languageResult,
                                  StructureAnalysisResult structureResult,
                                  double syntaxScore,
                                  QualityAssessmentResult qualityResult,
                                  StyleCheckResult styleResult,
                                  double codeConfidence) {
            this.languageResult = languageResult;
            this.structureResult = structureResult;
            this.syntaxScore = syntaxScore;
            this.qualityResult = qualityResult;
            this.styleResult = styleResult;
            this.codeConfidence = codeConfidence;
        }
        
        public String getSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("  检测语言: %s (置信度: %.3f)\n", 
                languageResult.getDetectedLanguage(), languageResult.getConfidence()));
            sb.append(String.format("  语法分数: %.3f\n", syntaxScore));
            sb.append(String.format("  结构质量: %.3f\n", structureResult.getStructureQuality()));
            sb.append(String.format("  整体质量: %.3f\n", qualityResult.getOverallQuality()));
            sb.append(String.format("  风格分数: %.3f\n", styleResult.getStyleScore()));
            sb.append(String.format("  代码置信度: %.3f\n", codeConfidence));
            return sb.toString();
        }
        
        // Getters
        public LanguageRecognitionResult getLanguageResult() { return languageResult; }
        public StructureAnalysisResult getStructureResult() { return structureResult; }
        public double getSyntaxScore() { return syntaxScore; }
        public QualityAssessmentResult getQualityResult() { return qualityResult; }
        public StyleCheckResult getStyleResult() { return styleResult; }
        public double getCodeConfidence() { return codeConfidence; }
    }
    
    // ========== Getter Methods ==========
    
    public List<String> getSupportedLanguages() { return supportedLanguages; }
    public CodeGenerationResult getLastResult() { return lastResult; }
    public Map<String, Integer> getLanguageUsageCount() { return languageUsageCount; }
    public double getSyntaxThreshold() { return syntaxThreshold; }
}