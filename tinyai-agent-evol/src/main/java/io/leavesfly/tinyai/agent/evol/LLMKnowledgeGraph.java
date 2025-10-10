package io.leavesfly.tinyai.agent.evol;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * LLM增强的知识图谱
 * 利用大语言模型的语义理解能力，构建更智能的知识关联和推理系统
 *
 * @author 山泽
 */
public class LLMKnowledgeGraph extends KnowledgeGraph {

    /**
     * LLM增强的概念节点信息
     */
    public static class LLMConceptInfo {
        private String conceptName;
        private String llmDescription;              // LLM生成的描述
        private Map<String, Double> semanticFeatures; // 语义特征向量
        private List<String> llmTags;               // LLM标注的标签
        private double conceptImportance;           // 概念重要度
        private String llmCategory;                 // LLM分类
        private Map<String, Object> llmMetadata;    // LLM元数据

        public LLMConceptInfo(String conceptName) {
            this.conceptName = conceptName;
            this.llmDescription = "";
            this.semanticFeatures = new HashMap<>();
            this.llmTags = new ArrayList<>();
            this.conceptImportance = 0.5;
            this.llmCategory = "unknown";
            this.llmMetadata = new HashMap<>();
        }

        // Getters and Setters
        public String getConceptName() { return conceptName; }
        public String getLlmDescription() { return llmDescription; }
        public void setLlmDescription(String description) { this.llmDescription = description; }

        public Map<String, Double> getSemanticFeatures() { return semanticFeatures; }
        public void setSemanticFeatures(Map<String, Double> features) { this.semanticFeatures = features; }

        public List<String> getLlmTags() { return llmTags; }
        public void addLlmTag(String tag) { llmTags.add(tag); }

        public double getConceptImportance() { return conceptImportance; }
        public void setConceptImportance(double importance) { this.conceptImportance = importance; }

        public String getLlmCategory() { return llmCategory; }
        public void setLlmCategory(String category) { this.llmCategory = category; }

        public Map<String, Object> getLlmMetadata() { return llmMetadata; }
        public void setLlmMetadata(String key, Object value) { llmMetadata.put(key, value); }
    }

    /**
     * LLM增强的关系信息
     */
    public static class LLMRelationInfo {
        private String fromConcept;
        private String toConcept;
        private String relationType;
        private String llmExplanation;              // LLM对关系的解释
        private double semanticStrength;            // 语义强度
        private String inferenceType;               // 推理类型
        private double confidenceScore;             // 置信度评分
        private List<String> supportingEvidence;    // 支撑证据
        private String llmJustification;            // LLM证明

        public LLMRelationInfo(String fromConcept, String toConcept, String relationType) {
            this.fromConcept = fromConcept;
            this.toConcept = toConcept;
            this.relationType = relationType;
            this.llmExplanation = "";
            this.semanticStrength = 0.5;
            this.inferenceType = "direct";
            this.confidenceScore = 0.5;
            this.supportingEvidence = new ArrayList<>();
            this.llmJustification = "";
        }

        // Getters and Setters
        public String getFromConcept() { return fromConcept; }
        public String getToConcept() { return toConcept; }
        public String getRelationType() { return relationType; }

        public String getLlmExplanation() { return llmExplanation; }
        public void setLlmExplanation(String explanation) { this.llmExplanation = explanation; }

        public double getSemanticStrength() { return semanticStrength; }
        public void setSemanticStrength(double strength) { this.semanticStrength = strength; }

        public String getInferenceType() { return inferenceType; }
        public void setInferenceType(String type) { this.inferenceType = type; }

        public double getConfidenceScore() { return confidenceScore; }
        public void setConfidenceScore(double score) { this.confidenceScore = score; }

        public List<String> getSupportingEvidence() { return supportingEvidence; }
        public void addSupportingEvidence(String evidence) { supportingEvidence.add(evidence); }

        public String getLlmJustification() { return llmJustification; }
        public void setLlmJustification(String justification) { this.llmJustification = justification; }
    }

    /** LLM模拟器 */
    private final EvolLLMSimulator llmSimulator;

    /** LLM增强的概念信息 */
    private final Map<String, LLMConceptInfo> llmConceptInfos;

    /** LLM增强的关系信息 */
    private final Map<String, LLMRelationInfo> llmRelationInfos;

    /** 语义分析缓存 */
    private final Map<String, String> semanticAnalysisCache;

    /** 推理结果缓存 */
    private final Map<String, String> inferenceCache;

    /** 启用异步处理 */
    private boolean enableAsyncProcessing;

    /** LLM置信度阈值 */
    private double llmConfidenceThreshold;

    public LLMKnowledgeGraph() {
        this(new EvolLLMSimulator(), true, 0.6);
    }

    public LLMKnowledgeGraph(EvolLLMSimulator llmSimulator, boolean enableAsync, double confidenceThreshold) {
        super();
        this.llmSimulator = llmSimulator;
        this.llmConceptInfos = new HashMap<>();
        this.llmRelationInfos = new HashMap<>();
        this.semanticAnalysisCache = new HashMap<>();
        this.inferenceCache = new HashMap<>();
        this.enableAsyncProcessing = enableAsync;
        this.llmConfidenceThreshold = confidenceThreshold;
    }

    /**
     * LLM增强的概念添加
     */
    @Override
    public void addConcept(String concept, Map<String, Object> properties) {
        // 调用父类方法
        super.addConcept(concept, properties);

        // 创建LLM增强的概念信息
        LLMConceptInfo llmInfo = new LLMConceptInfo(concept);

        // LLM增强分析
        if (llmSimulator != null) {
            try {
                // 生成概念描述
                String llmDescription = generateConceptDescription(concept, properties);
                llmInfo.setLlmDescription(llmDescription);

                // 提取语义特征
                Map<String, Double> semanticFeatures = extractSemanticFeatures(concept, properties);
                llmInfo.setSemanticFeatures(semanticFeatures);

                // 生成标签
                List<String> llmTags = generateConceptTags(concept, properties);
                llmTags.forEach(llmInfo::addLlmTag);

                // 计算重要度
                double importance = calculateConceptImportance(concept, properties);
                llmInfo.setConceptImportance(importance);

                // 分类
                String category = categorizeConceptWithLLM(concept, properties);
                llmInfo.setLlmCategory(category);

                // 添加LLM元数据
                llmInfo.setLlmMetadata("creation_time", System.currentTimeMillis());
                llmInfo.setLlmMetadata("llm_processed", true);

            } catch (Exception e) {
                System.err.println("LLM概念分析过程中出现异常: " + e.getMessage());
                llmInfo.setLlmMetadata("llm_processed", false);
                llmInfo.setLlmMetadata("error", e.getMessage());
            }
        }

        llmConceptInfos.put(concept, llmInfo);
    }

    /**
     * LLM增强的关系添加
     */
    @Override
    public void addRelation(String from, String to, String relationType, double weight) {
        // 调用父类方法
        super.addRelation(from, to, relationType, weight);

        // 创建LLM增强的关系信息
        String relationKey = from + "->" + to + ":" + relationType;
        LLMRelationInfo llmInfo = new LLMRelationInfo(from, to, relationType);

        // LLM增强分析
        if (llmSimulator != null) {
            try {
                // 生成关系解释
                String explanation = generateRelationExplanation(from, to, relationType);
                llmInfo.setLlmExplanation(explanation);

                // 计算语义强度
                double semanticStrength = calculateSemanticStrength(from, to, relationType);
                llmInfo.setSemanticStrength(semanticStrength);

                // 确定推理类型
                String inferenceType = determineInferenceType(from, to, relationType);
                llmInfo.setInferenceType(inferenceType);

                // 计算置信度
                double confidence = calculateRelationConfidence(from, to, relationType, weight);
                llmInfo.setConfidenceScore(confidence);

                // 收集支撑证据
                List<String> evidence = collectSupportingEvidence(from, to, relationType);
                evidence.forEach(llmInfo::addSupportingEvidence);

                // 生成LLM证明
                String justification = generateLLMJustification(from, to, relationType, weight);
                llmInfo.setLlmJustification(justification);

            } catch (Exception e) {
                System.err.println("LLM关系分析过程中出现异常: " + e.getMessage());
                llmInfo.setLlmExplanation("LLM分析失败: " + e.getMessage());
            }
        }

        llmRelationInfos.put(relationKey, llmInfo);
    }

    /**
     * LLM增强的概念相似度计算
     */
    @Override
    public double getConceptSimilarity(String concept1, String concept2) {
        // 基础相似度计算
        double basicSimilarity = super.getConceptSimilarity(concept1, concept2);

        // LLM增强相似度计算
        if (llmSimulator != null && llmConceptInfos.containsKey(concept1) && llmConceptInfos.containsKey(concept2)) {
            try {
                // 语义特征相似度
                double semanticSimilarity = calculateSemanticSimilarity(concept1, concept2);

                // LLM推理相似度
                double llmSimilarity = calculateLLMSimilarity(concept1, concept2);

                // 综合相似度计算
                return combineSimularities(basicSimilarity, semanticSimilarity, llmSimilarity);

            } catch (Exception e) {
                System.err.println("LLM相似度计算过程中出现异常: " + e.getMessage());
            }
        }

        return basicSimilarity;
    }

    /**
     * LLM增强的相关概念查找
     */
    @Override
    public List<String> findRelatedConcepts(String concept, int maxDistance) {
        // 基础相关概念
        List<String> basicRelated = super.findRelatedConcepts(concept, maxDistance);

        // LLM增强的相关概念
        if (llmSimulator != null) {
            try {
                List<String> llmRelated = findLLMRelatedConcepts(concept, maxDistance);

                // 合并和去重
                Set<String> combined = new HashSet<>(basicRelated);
                combined.addAll(llmRelated);

                // 按相似度排序
                return combined.stream()
                        .sorted((c1, c2) -> Double.compare(
                                getConceptSimilarity(concept, c2),
                                getConceptSimilarity(concept, c1)
                        ))
                        .collect(Collectors.toList());

            } catch (Exception e) {
                System.err.println("LLM相关概念查找过程中出现异常: " + e.getMessage());
            }
        }

        return basicRelated;
    }

    /**
     * 智能推理 - 基于LLM进行知识推理
     */
    public String performIntelligentReasoning(String query, String reasoningType) {
        if (llmSimulator == null) {
            return "智能推理需要LLM支持";
        }

        // 检查缓存
        String cacheKey = query + ":" + reasoningType;
        if (inferenceCache.containsKey(cacheKey)) {
            return inferenceCache.get(cacheKey);
        }

        try {
            // 构建推理上下文
            String reasoningContext = buildReasoningContext(query);

            // 执行LLM推理
            String reasoning = llmSimulator.generateEvolResponse(
                    query,
                    reasoningContext,
                    "knowledge_inference"
            );

            // 缓存结果
            inferenceCache.put(cacheKey, reasoning);

            return reasoning;

        } catch (Exception e) {
            return "智能推理过程中出现异常: " + e.getMessage();
        }
    }

    /**
     * 概念关系推荐 - 基于LLM推荐潜在的概念关系
     */
    public List<String> recommendConceptRelations(String concept) {
        List<String> recommendations = new ArrayList<>();

        if (llmSimulator == null || !llmConceptInfos.containsKey(concept)) {
            return recommendations;
        }

        try {
            LLMConceptInfo targetInfo = llmConceptInfos.get(concept);

            // 基于语义特征推荐
            recommendations.addAll(recommendBasedOnSemanticFeatures(targetInfo));

            // 基于LLM分析推荐
            recommendations.addAll(recommendBasedOnLLMAnalysis(concept));

            // 基于现有关系模式推荐
            recommendations.addAll(recommendBasedOnRelationPatterns(concept));

            return recommendations.stream()
                    .distinct()
                    .limit(10)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            System.err.println("概念关系推荐过程中出现异常: " + e.getMessage());
            return recommendations;
        }
    }

    /**
     * 知识图谱质量评估
     */
    public Map<String, Object> assessKnowledgeGraphQuality() {
        Map<String, Object> quality = new HashMap<>();

        // 基础统计
        quality.put("total_concepts", llmConceptInfos.size());
        quality.put("total_relations", llmRelationInfos.size());

        // LLM增强质量指标
        if (llmSimulator != null) {
            // 概念描述完整性
            long conceptsWithDescription = llmConceptInfos.values().stream()
                    .filter(info -> !info.getLlmDescription().isEmpty())
                    .count();
            quality.put("concept_description_rate",
                    llmConceptInfos.isEmpty() ? 0.0 : (double) conceptsWithDescription / llmConceptInfos.size());

            // 平均置信度
            double avgConfidence = llmRelationInfos.values().stream()
                    .mapToDouble(LLMRelationInfo::getConfidenceScore)
                    .average().orElse(0.0);
            quality.put("avg_relation_confidence", avgConfidence);

            // 语义覆盖度
            long conceptsWithSemantics = llmConceptInfos.values().stream()
                    .filter(info -> !info.getSemanticFeatures().isEmpty())
                    .count();
            quality.put("semantic_coverage",
                    llmConceptInfos.isEmpty() ? 0.0 : (double) conceptsWithSemantics / llmConceptInfos.size());

            // 分类覆盖度
            long categorizedConcepts = llmConceptInfos.values().stream()
                    .filter(info -> !"unknown".equals(info.getLlmCategory()))
                    .count();
            quality.put("categorization_rate",
                    llmConceptInfos.isEmpty() ? 0.0 : (double) categorizedConcepts / llmConceptInfos.size());
        }

        return quality;
    }

    /**
     * 异步概念分析
     */
    public CompletableFuture<String> analyzeConceptAsync(String concept) {
        if (!enableAsyncProcessing || llmSimulator == null) {
            return CompletableFuture.completedFuture("同步模式或LLM不可用");
        }

        return CompletableFuture.supplyAsync(() -> {
            try {
                LLMConceptInfo info = llmConceptInfos.get(concept);
                if (info != null) {
                    StringBuilder analysis = new StringBuilder();
                    analysis.append("概念分析: ").append(concept).append("\\n");
                    analysis.append("描述: ").append(info.getLlmDescription()).append("\\n");
                    analysis.append("类别: ").append(info.getLlmCategory()).append("\\n");
                    analysis.append("重要度: ").append(info.getConceptImportance()).append("\\n");
                    analysis.append("标签: ").append(String.join(", ", info.getLlmTags()));
                    return analysis.toString();
                }
                return "概念未找到";
            } catch (Exception e) {
                return "异步概念分析失败: " + e.getMessage();
            }
        });
    }

    // ================================
    // LLM增强的具体实现方法
    // ================================

    private String generateConceptDescription(String concept, Map<String, Object> properties) {
        String cacheKey = "desc:" + concept;
        if (semanticAnalysisCache.containsKey(cacheKey)) {
            return semanticAnalysisCache.get(cacheKey);
        }

        StringBuilder prompt = new StringBuilder();
        prompt.append("为概念'").append(concept).append("'生成简洁描述，");
        prompt.append("属性包括: ").append(properties.toString());

        String description = llmSimulator.generateEvolResponse(
                prompt.toString(),
                "concept_analysis",
                "knowledge_inference"
        );

        semanticAnalysisCache.put(cacheKey, description);
        return description;
    }

    private Map<String, Double> extractSemanticFeatures(String concept, Map<String, Object> properties) {
        Map<String, Double> features = new HashMap<>();

        // 基于概念名称提取特征
        if (concept.contains("task")) {
            features.put("task_related", 0.8);
        }
        if (concept.contains("action")) {
            features.put("action_related", 0.8);
        }
        if (concept.contains("success")) {
            features.put("positive_outcome", 0.9);
        }
        if (concept.contains("fail")) {
            features.put("negative_outcome", 0.9);
        }

        // 基于属性提取特征
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            if ("type".equals(key)) {
                features.put("type_" + value.toString(), 0.7);
            }
            if ("difficulty".equals(key)) {
                features.put("difficulty_" + value.toString(), 0.6);
            }
        }

        // 添加随机特征模拟LLM嵌入
        for (int i = 0; i < 5; i++) {
            features.put("feature_" + i, Math.random());
        }

        return features;
    }

    private List<String> generateConceptTags(String concept, Map<String, Object> properties) {
        List<String> tags = new ArrayList<>();

        // 基于概念名称生成标签
        if (concept.startsWith("task:")) {
            tags.add("任务类型");
        }
        if (concept.startsWith("action:")) {
            tags.add("动作类型");
        }

        // 基于属性生成标签
        if (properties.containsKey("type")) {
            tags.add(properties.get("type").toString());
        }

        // LLM生成的标签
        try {
            String prompt = "为概念'" + concept + "'生成3-5个标签";
            String llmTags = llmSimulator.generateEvolResponse(prompt, properties.toString(), "pattern_analysis");
            // 简化解析
            if (llmTags.contains("标签")) {
                tags.add("LLM标签1");
                tags.add("LLM标签2");
            }
        } catch (Exception e) {
            // 静默处理
        }

        return tags;
    }

    private double calculateConceptImportance(String concept, Map<String, Object> properties) {
        double importance = 0.5; // 基础重要度

        // 基于概念类型
        if (concept.contains("success")) {
            importance += 0.2;
        }
        if (concept.contains("critical") || concept.contains("important")) {
            importance += 0.3;
        }

        // 基于属性
        if (properties.containsKey("priority")) {
            Object priority = properties.get("priority");
            if ("high".equals(priority)) {
                importance += 0.2;
            }
        }

        return Math.min(1.0, importance);
    }

    private String categorizeConceptWithLLM(String concept, Map<String, Object> properties) {
        try {
            String prompt = "将概念'" + concept + "'分类到以下类别之一：任务、动作、结果、工具、策略、其他";
            String category = llmSimulator.generateEvolResponse(prompt, properties.toString(), "strategic_reasoning");

            // 简化分类提取
            if (concept.contains("task")) return "任务";
            if (concept.contains("action")) return "动作";
            if (concept.contains("result")) return "结果";
            if (concept.contains("tool")) return "工具";
            if (concept.contains("strategy")) return "策略";

            return "其他";
        } catch (Exception e) {
            return "未分类";
        }
    }

    private String generateRelationExplanation(String from, String to, String relationType) {
        try {
            String prompt = String.format("解释概念'%s'与'%s'之间的'%s'关系", from, to, relationType);
            return llmSimulator.generateEvolResponse(prompt, "", "knowledge_inference");
        } catch (Exception e) {
            return "关系解释生成失败";
        }
    }

    private double calculateSemanticStrength(String from, String to, String relationType) {
        double strength = 0.5;

        // 基于关系类型
        if ("succeeds_with".equals(relationType)) {
            strength = 0.8;
        } else if ("fails_with".equals(relationType)) {
            strength = 0.6;
        } else if ("similar_to".equals(relationType)) {
            strength = 0.7;
        }

        // 基于语义特征相似度
        if (llmConceptInfos.containsKey(from) && llmConceptInfos.containsKey(to)) {
            Map<String, Double> features1 = llmConceptInfos.get(from).getSemanticFeatures();
            Map<String, Double> features2 = llmConceptInfos.get(to).getSemanticFeatures();

            double featureSimilarity = calculateFeatureSimilarity(features1, features2);
            strength = (strength + featureSimilarity) / 2;
        }

        return strength;
    }

    private String determineInferenceType(String from, String to, String relationType) {
        if ("succeeds_with".equals(relationType) || "fails_with".equals(relationType)) {
            return "empirical"; // 基于经验的推理
        } else if ("similar_to".equals(relationType)) {
            return "analogical"; // 类比推理
        } else if ("llm_inferred".equals(relationType)) {
            return "semantic"; // 语义推理
        }
        return "direct"; // 直接关系
    }

    private double calculateRelationConfidence(String from, String to, String relationType, double weight) {
        double confidence = Math.abs(weight); // 基于权重

        // 基于推理类型调整
        String inferenceType = determineInferenceType(from, to, relationType);
        switch (inferenceType) {
            case "empirical":
                confidence += 0.2; // 经验推理置信度更高
                break;
            case "semantic":
                confidence += 0.1; // 语义推理置信度中等
                break;
            case "analogical":
                confidence -= 0.1; // 类比推理置信度较低
                break;
        }

        return Math.min(1.0, Math.max(0.0, confidence));
    }

    private List<String> collectSupportingEvidence(String from, String to, String relationType) {
        List<String> evidence = new ArrayList<>();

        // 收集经验证据
        if ("succeeds_with".equals(relationType)) {
            evidence.add("基于成功经验的实证支持");
        } else if ("fails_with".equals(relationType)) {
            evidence.add("基于失败经验的反面证据");
        }

        // 收集语义证据
        if (llmConceptInfos.containsKey(from) && llmConceptInfos.containsKey(to)) {
            LLMConceptInfo info1 = llmConceptInfos.get(from);
            LLMConceptInfo info2 = llmConceptInfos.get(to);

            if (info1.getLlmCategory().equals(info2.getLlmCategory())) {
                evidence.add("相同类别的概念支持");
            }

            // 检查共同标签
            Set<String> commonTags = new HashSet<>(info1.getLlmTags());
            commonTags.retainAll(info2.getLlmTags());
            if (!commonTags.isEmpty()) {
                evidence.add("共同标签证据: " + String.join(", ", commonTags));
            }
        }

        return evidence;
    }

    private String generateLLMJustification(String from, String to, String relationType, double weight) {
        try {
            String prompt = String.format(
                    "为关系'%s %s %s'(权重:%.2f)提供推理证明",
                    from, relationType, to, weight
            );
            return llmSimulator.generateEvolResponse(prompt, "", "knowledge_inference");
        } catch (Exception e) {
            return "LLM证明生成失败";
        }
    }

    private double calculateSemanticSimilarity(String concept1, String concept2) {
        LLMConceptInfo info1 = llmConceptInfos.get(concept1);
        LLMConceptInfo info2 = llmConceptInfos.get(concept2);

        if (info1 == null || info2 == null) {
            return 0.0;
        }

        return calculateFeatureSimilarity(info1.getSemanticFeatures(), info2.getSemanticFeatures());
    }

    private double calculateFeatureSimilarity(Map<String, Double> features1, Map<String, Double> features2) {
        if (features1.isEmpty() || features2.isEmpty()) {
            return 0.0;
        }

        Set<String> commonFeatures = new HashSet<>(features1.keySet());
        commonFeatures.retainAll(features2.keySet());

        if (commonFeatures.isEmpty()) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (String feature : commonFeatures) {
            double val1 = features1.get(feature);
            double val2 = features2.get(feature);

            dotProduct += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }

        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private double calculateLLMSimilarity(String concept1, String concept2) {
        try {
            String prompt = String.format("评估概念'%s'与'%s'的语义相似度(0-1)", concept1, concept2);
            String similarity = llmSimulator.generateEvolResponse(prompt, "", "knowledge_inference");

            // 简化解析：寻找数字
            for (String part : similarity.split("\\\\s+")) {
                try {
                    double value = Double.parseDouble(part);
                    if (value >= 0 && value <= 1) {
                        return value;
                    }
                } catch (NumberFormatException e) {
                    // 继续尝试
                }
            }

            return 0.5; // 默认中等相似度
        } catch (Exception e) {
            return 0.0;
        }
    }

    private double combineSimularities(double basic, double semantic, double llm) {
        // 加权组合相似度
        return 0.4 * basic + 0.3 * semantic + 0.3 * llm;
    }

    private List<String> findLLMRelatedConcepts(String concept, int maxDistance) {
        List<String> related = new ArrayList<>();

        try {
            String prompt = String.format("找出与概念'%s'相关的概念，限制在距离%d以内", concept, maxDistance);
            String llmRelated = llmSimulator.generateEvolResponse(prompt, "", "knowledge_inference");

            // 简化解析：提取概念名称
            String[] parts = llmRelated.split("[,，;；\\\\s]+");
            for (String part : parts) {
                String cleaned = part.trim().replaceAll("['\\\"]", "");
                if (cleaned.length() > 2 && llmConceptInfos.containsKey(cleaned)) {
                    related.add(cleaned);
                }
            }

        } catch (Exception e) {
            System.err.println("LLM相关概念查找失败: " + e.getMessage());
        }

        return related;
    }

    private String buildReasoningContext(String query) {
        StringBuilder context = new StringBuilder();
        context.append("知识图谱统计:\\n");
        context.append("概念数量: ").append(llmConceptInfos.size()).append("\\n");
        context.append("关系数量: ").append(llmRelationInfos.size()).append("\\n");

        // 添加相关概念信息
        for (Map.Entry<String, LLMConceptInfo> entry : llmConceptInfos.entrySet().stream().limit(5).collect(Collectors.toList())) {
            LLMConceptInfo info = entry.getValue();
            context.append("概念: ").append(entry.getKey())
                    .append(", 类别: ").append(info.getLlmCategory())
                    .append("\\n");
        }

        return context.toString();
    }

    private List<String> recommendBasedOnSemanticFeatures(LLMConceptInfo targetInfo) {
        List<String> recommendations = new ArrayList<>();

        Map<String, Double> targetFeatures = targetInfo.getSemanticFeatures();

        for (Map.Entry<String, LLMConceptInfo> entry : llmConceptInfos.entrySet()) {
            String conceptName = entry.getKey();
            LLMConceptInfo info = entry.getValue();

            if (!conceptName.equals(targetInfo.getConceptName())) {
                double similarity = calculateFeatureSimilarity(targetFeatures, info.getSemanticFeatures());
                if (similarity > 0.7) {
                    recommendations.add(conceptName);
                }
            }
        }

        return recommendations;
    }

    private List<String> recommendBasedOnLLMAnalysis(String concept) {
        List<String> recommendations = new ArrayList<>();

        try {
            String prompt = "推荐与概念'" + concept + "'可能相关的其他概念";
            String llmRecommendations = llmSimulator.generateEvolResponse(prompt, "", "strategic_reasoning");

            // 简化解析
            String[] parts = llmRecommendations.split("[,，;；]");
            for (String part : parts) {
                String cleaned = part.trim();
                if (cleaned.length() > 2 && llmConceptInfos.containsKey(cleaned)) {
                    recommendations.add(cleaned);
                }
            }

        } catch (Exception e) {
            System.err.println("LLM推荐分析失败: " + e.getMessage());
        }

        return recommendations;
    }

    private List<String> recommendBasedOnRelationPatterns(String concept) {
        List<String> recommendations = new ArrayList<>();

        // 基于关系模式推荐（简化实现）
        for (LLMRelationInfo relationInfo : llmRelationInfos.values()) {
            if (relationInfo.getFromConcept().equals(concept)) {
                String relatedConcept = relationInfo.getToConcept();
                // 基于传递性推荐
                for (LLMRelationInfo transitive : llmRelationInfos.values()) {
                    if (transitive.getFromConcept().equals(relatedConcept) &&
                            !transitive.getToConcept().equals(concept)) {
                        recommendations.add(transitive.getToConcept());
                    }
                }
            }
        }

        return recommendations;
    }

    // ================================
    // LLM增强的Getter方法
    // ================================

    /**
     * 获取LLM增强的概念信息
     */
    public Map<String, LLMConceptInfo> getLLMConceptInfos() {
        return new HashMap<>(llmConceptInfos);
    }

    /**
     * 获取LLM增强的关系信息
     */
    public Map<String, LLMRelationInfo> getLLMRelationInfos() {
        return new HashMap<>(llmRelationInfos);
    }

    /**
     * 获取特定概念的LLM信息
     */
    public LLMConceptInfo getLLMConceptInfo(String concept) {
        return llmConceptInfos.get(concept);
    }

    /**
     * 获取特定关系的LLM信息
     */
    public LLMRelationInfo getLLMRelationInfo(String from, String to, String relationType) {
        String relationKey = from + "->" + to + ":" + relationType;
        return llmRelationInfos.get(relationKey);
    }

    /**
     * 获取LLM统计信息
     */
    public Map<String, Object> getLLMStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("llm_enabled", llmSimulator != null);
        stats.put("llm_concepts_count", llmConceptInfos.size());
        stats.put("llm_relations_count", llmRelationInfos.size());
        stats.put("semantic_analysis_cache_size", semanticAnalysisCache.size());
        stats.put("inference_cache_size", inferenceCache.size());
        stats.put("async_processing_enabled", enableAsyncProcessing);
        stats.put("confidence_threshold", llmConfidenceThreshold);

        return stats;
    }

    // Getters and Setters
    public boolean isAsyncProcessingEnabled() { return enableAsyncProcessing; }
    public void setAsyncProcessingEnabled(boolean enabled) { this.enableAsyncProcessing = enabled; }

    public double getLlmConfidenceThreshold() { return llmConfidenceThreshold; }
    public void setLlmConfidenceThreshold(double threshold) {
        this.llmConfidenceThreshold = Math.max(0.0, Math.min(1.0, threshold));
    }

    public EvolLLMSimulator getLlmSimulator() { return llmSimulator; }
}