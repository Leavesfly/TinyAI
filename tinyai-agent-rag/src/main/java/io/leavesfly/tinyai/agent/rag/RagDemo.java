package io.leavesfly.tinyai.agent.rag;

import java.util.*;

/**
 * RAG系统演示类
 * 演示RAG检索增强生成系统的完整功能
 */
public class RagDemo {

    /**
     * 创建示例文档数据
     * @return 示例文档列表
     */
    private static List<Map<String, Object>> createSampleDocuments() {
        List<Map<String, Object>> documents = new ArrayList<>();
        
        // 文档1：Python编程
        Map<String, Object> doc1 = new HashMap<>();
        doc1.put("id", "python_intro");
        doc1.put("content", "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它具有简洁的语法和强大的功能，广泛用于Web开发、数据科学、人工智能等领域。Python的设计哲学强调代码的可读性和简洁性。");
        Map<String, Object> metadata1 = new HashMap<>();
        metadata1.put("category", "编程语言");
        metadata1.put("difficulty", "入门");
        doc1.put("metadata", metadata1);
        documents.add(doc1);
        
        // 文档2：机器学习
        Map<String, Object> doc2 = new HashMap<>();
        doc2.put("id", "machine_learning_basics");
        doc2.put("content", "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下学习。机器学习算法构建数学模型，基于训练数据进行预测或决策。常见的机器学习类型包括监督学习、无监督学习和强化学习。");
        Map<String, Object> metadata2 = new HashMap<>();
        metadata2.put("category", "人工智能");
        metadata2.put("difficulty", "中级");
        doc2.put("metadata", metadata2);
        documents.add(doc2);
        
        // 文档3：深度学习
        Map<String, Object> doc3 = new HashMap<>();
        doc3.put("id", "deep_learning_intro");
        doc3.put("content", "深度学习是机器学习的一个子集，它基于人工神经网络，特别是深度神经网络。深度学习在图像识别、语音识别、自然语言处理等任务中取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是常用的深度学习架构。");
        Map<String, Object> metadata3 = new HashMap<>();
        metadata3.put("category", "人工智能");
        metadata3.put("difficulty", "高级");
        doc3.put("metadata", metadata3);
        documents.add(doc3);
        
        // 文档4：数据科学
        Map<String, Object> doc4 = new HashMap<>();
        doc4.put("id", "data_science_overview");
        doc4.put("content", "数据科学是一个跨学科领域，结合了统计学、计算机科学和领域知识来从数据中提取洞察。数据科学家使用各种工具和技术，包括数据挖掘、机器学习、可视化等，来分析复杂的数据集并解决业务问题。");
        Map<String, Object> metadata4 = new HashMap<>();
        metadata4.put("category", "数据科学");
        metadata4.put("difficulty", "中级");
        doc4.put("metadata", metadata4);
        documents.add(doc4);
        
        // 文档5：Web开发
        Map<String, Object> doc5 = new HashMap<>();
        doc5.put("id", "web_development_python");
        doc5.put("content", "Python在Web开发中非常流行，有许多强大的框架可供选择。Django是一个高级Web框架，提供了完整的解决方案。Flask是一个轻量级框架，更适合小型项目。FastAPI是一个现代框架，专为构建API而设计，支持异步编程。");
        Map<String, Object> metadata5 = new HashMap<>();
        metadata5.put("category", "Web开发");
        metadata5.put("difficulty", "中级");
        doc5.put("metadata", metadata5);
        documents.add(doc5);
        
        // 文档6：自然语言处理
        Map<String, Object> doc6 = new HashMap<>();
        doc6.put("id", "natural_language_processing");
        doc6.put("content", "自然语言处理（NLP）是人工智能的一个重要分支，专注于使计算机理解和生成人类语言。NLP技术包括文本分类、情感分析、机器翻译、问答系统等。近年来，基于Transformer的大语言模型如GPT、BERT等在NLP任务中表现出色。");
        Map<String, Object> metadata6 = new HashMap<>();
        metadata6.put("category", "自然语言处理");
        metadata6.put("difficulty", "高级");
        doc6.put("metadata", metadata6);
        documents.add(doc6);
        
        // 文档7：数据库
        Map<String, Object> doc7 = new HashMap<>();
        doc7.put("id", "database_fundamentals");
        doc7.put("content", "数据库是存储和管理数据的系统。关系型数据库使用SQL语言进行查询，如MySQL、PostgreSQL等。NoSQL数据库适合处理非结构化数据，如MongoDB、Redis等。数据库设计需要考虑数据模型、索引、事务处理等方面。");
        Map<String, Object> metadata7 = new HashMap<>();
        metadata7.put("category", "数据库");
        metadata7.put("difficulty", "中级");
        doc7.put("metadata", metadata7);
        documents.add(doc7);
        
        // 文档8：云计算
        Map<String, Object> doc8 = new HashMap<>();
        doc8.put("id", "cloud_computing_intro");
        doc8.put("content", "云计算是通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件等。主要的云服务模型有IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。AWS、Azure、Google Cloud是主要的云服务提供商。");
        Map<String, Object> metadata8 = new HashMap<>();
        metadata8.put("category", "云计算");
        metadata8.put("difficulty", "中级");
        doc8.put("metadata", metadata8);
        documents.add(doc8);
        
        // 文档9：软件工程
        Map<String, Object> doc9 = new HashMap<>();
        doc9.put("id", "software_engineering_practices");
        doc9.put("content", "软件工程是一门关于如何系统化、规范化、可量化地开发软件的学科。良好的软件工程实践包括版本控制、代码审查、单元测试、持续集成、敏捷开发等。这些实践有助于提高软件质量、降低维护成本、提升团队协作效率。");
        Map<String, Object> metadata9 = new HashMap<>();
        metadata9.put("category", "软件工程");
        metadata9.put("difficulty", "中级");
        doc9.put("metadata", metadata9);
        documents.add(doc9);
        
        // 文档10：网络安全
        Map<String, Object> doc10 = new HashMap<>();
        doc10.put("id", "cybersecurity_basics");
        doc10.put("content", "网络安全是保护计算机系统和网络免受数字攻击的实践。常见的安全威胁包括恶意软件、钓鱼攻击、数据泄露等。网络安全措施包括防火墙、加密、身份验证、访问控制等。安全开发生命周期（SDLC）将安全考虑融入软件开发的各个阶段。");
        Map<String, Object> metadata10 = new HashMap<>();
        metadata10.put("category", "网络安全");
        metadata10.put("difficulty", "高级");
        doc10.put("metadata", metadata10);
        documents.add(doc10);
        
        return documents;
    }

    /**
     * 演示RAG系统基本功能
     */
    public static void demoRAGSystem() {
        System.out.println("=".repeat(60));
        System.out.println("🔍 RAG检索增强生成系统演示");
        System.out.println("=".repeat(60));
        
        // 创建RAG系统
        RAGSystem rag = new RAGSystem(256, 0.05);
        
        // 创建示例文档
        System.out.println("\n📚 准备示例文档...");
        List<Map<String, Object>> documents = createSampleDocuments();
        
        // 添加文档到RAG系统
        rag.addDocuments(documents);
        
        // 显示系统统计信息
        Map<String, Object> stats = rag.getStatistics();
        System.out.println("\n📊 系统统计信息:");
        stats.forEach((key, value) -> System.out.println("  " + key + ": " + value));
        
        System.out.println("\n" + "=".repeat(40));
        System.out.println("🔍 开始检索演示");
        System.out.println("=".repeat(40));
        
        // 测试查询
        String[] testQueries = {
            "Python编程语言的特点",
            "机器学习算法",
            "深度学习神经网络",
            "Web开发框架",
            "数据库管理系统",
            "云计算服务模型",
            "网络安全防护"
        };
        
        for (String query : testQueries) {
            System.out.println("\n🔎 查询: '" + query + "'");
            System.out.println("-".repeat(50));
            
            // 执行检索
            List<QueryResult> results = rag.search(query, 3);
            
            if (!results.isEmpty()) {
                for (QueryResult result : results) {
                    Document doc = result.getDocument();
                    System.out.println("📄 文档ID: " + doc.getId());
                    System.out.println("📊 相似度: " + String.format("%.4f", result.getSimilarity()));
                    System.out.println("📝 内容: " + (doc.getContent().length() > 100 ? 
                        doc.getContent().substring(0, 100) + "..." : doc.getContent()));
                    System.out.println("🏷️  类别: " + doc.getMetadata().getOrDefault("category", "N/A"));
                    System.out.println();
                }
            } else {
                System.out.println("❌ 未找到相关文档");
            }
            
            // 生成上下文
            String context = rag.generateContext(query, 300);
            System.out.println("📋 生成的上下文:");
            System.out.println(context.length() > 200 ? context.substring(0, 200) + "..." : context);
            System.out.println();
        }
        
        // 关闭系统
        rag.close();
    }

    /**
     * 演示向量操作
     */
    public static void demoVectorOperations() {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("🧮 向量操作演示");
        System.out.println("=".repeat(50));
        
        // 创建示例文本
        List<String> texts = Arrays.asList(
            "人工智能是一门计算机科学",
            "机器学习是人工智能的分支",
            "深度学习使用神经网络",
            "Python是流行的编程语言",
            "数据科学分析大量数据"
        );
        
        // 创建向量化器
        TFIDFVectorizer vectorizer = new TFIDFVectorizer(20);
        
        // 训练并转换
        System.out.println("\n📊 训练TF-IDF向量化器...");
        List<List<Double>> vectors = vectorizer.fitTransform(texts);
        
        System.out.println("词汇表大小: " + vectorizer.getTokenizer().getVocabSize());
        System.out.println("向量维度: " + (vectors.isEmpty() ? 0 : vectors.get(0).size()));
        
        // 显示特征词汇
        List<String> featureNames = vectorizer.getFeatureNames();
        System.out.println("\n🔤 特征词汇: " + featureNames.subList(0, Math.min(10, featureNames.size())) + "...");
        
        // 显示向量信息
        System.out.println("\n📈 文本向量:");
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            List<Double> vector = vectors.get(i);
            
            // 找出非零特征
            List<String> nonZeroFeatures = new ArrayList<>(); 
            for (int j = 0; j < Math.min(vector.size(), featureNames.size()); j++) {
                if (vector.get(j) > 0) {
                    nonZeroFeatures.add(featureNames.get(j) + ":" + String.format("%.3f", vector.get(j)));
                    if (nonZeroFeatures.size() >= 3) break;
                }
            }
            
            System.out.println((i+1) + ". '" + text + "'");
            System.out.println("   非零特征: " + nonZeroFeatures + "...");
        }
        
        // 计算相似度矩阵
        System.out.println("\n📏 余弦相似度矩阵:");
        System.out.print("    ");
        for (int i = 0; i < texts.size(); i++) {
            System.out.printf("%6d", i+1);
        }
        System.out.println();
        
        for (int i = 0; i < vectors.size(); i++) {
            System.out.printf("%2d: ", i+1);
            for (int j = 0; j < vectors.size(); j++) {
                double similarity = VectorSimilarity.cosineSimilarity(vectors.get(i), vectors.get(j));
                System.out.printf("%5.3f ", similarity);
            }
            System.out.println();
        }
        
        // 查询相似度
        System.out.println("\n🔍 查询相似度测试:");
        String queryText = "机器学习算法";
        List<Double> queryVector = vectorizer.transform(queryText);
        
        System.out.println("查询: '" + queryText + "'");
        List<QuerySimilarity> similarities = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            List<Double> vector = vectors.get(i);
            double similarity = VectorSimilarity.cosineSimilarity(queryVector, vector);
            similarities.add(new QuerySimilarity(i, text, similarity));
        }
        
        // 按相似度排序
        similarities.sort((a, b) -> Double.compare(b.similarity, a.similarity));
        
        System.out.println("相似度排序结果:");
        for (int rank = 0; rank < similarities.size(); rank++) {
            QuerySimilarity qs = similarities.get(rank);
            System.out.println((rank+1) + ". " + String.format("%.4f", qs.similarity) + " - '" + qs.text + "'");
        }
    }

    /**
     * 交互式查询演示
     */
    public static void demoInteractiveQuery() {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("💬 交互式查询模式");
        System.out.println("=".repeat(40));
        System.out.println("输入查询内容（输入 'quit' 退出）:");
        
        // 创建RAG系统并添加文档
        RAGSystem rag = new RAGSystem(256, 0.05);
        rag.addDocuments(createSampleDocuments());
        
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            try {
                System.out.print("\n🔍 请输入查询: ");
                String userQuery = scanner.nextLine().trim();
                
                if (userQuery.toLowerCase().matches("(quit|exit|退出|q)")) {
                    System.out.println("👋 退出查询模式");
                    break;
                }
                
                if (userQuery.isEmpty()) {
                    continue;
                }
                
                // 执行检索
                List<QueryResult> results = rag.search(userQuery, 3);
                
                if (!results.isEmpty()) {
                    System.out.println("\n📋 找到 " + results.size() + " 个相关文档:");
                    for (int i = 0; i < results.size(); i++) {
                        QueryResult result = results.get(i);
                        Document doc = result.getDocument();
                        System.out.println("\n" + (i+1) + ". 📄 " + doc.getId());
                        System.out.println("   📊 相似度: " + String.format("%.4f", result.getSimilarity()));
                        System.out.println("   🏷️  类别: " + doc.getMetadata().getOrDefault("category", "N/A"));
                        System.out.println("   📝 内容: " + doc.getContent());
                    }
                    
                    // 生成上下文用于回答
                    String context = rag.generateContext(userQuery);
                    System.out.println("\n📋 基于检索结果的上下文:");
                    System.out.println(context);
                } else {
                    System.out.println("❌ 未找到相关文档，请尝试其他关键词");
                }
                
            } catch (Exception e) {
                System.out.println("❌ 发生错误: " + e.getMessage());
            }
        }
        
        scanner.close();
        rag.close();
    }

    /**
     * 主函数
     */
    public static void main(String[] args) {
        System.out.println("🚀 RAG与向量数据实现演示");
        System.out.println("选择演示模式:");
        System.out.println("1. RAG检索增强生成系统演示");
        System.out.println("2. 向量操作基础演示");
        System.out.println("3. 交互式查询演示");
        System.out.println("4. 完整演示（包含所有部分）");
        
        Scanner scanner = new Scanner(System.in);
        
        try {
            System.out.print("\n请选择 (1-4): ");
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    demoRAGSystem();
                    break;
                case "2":
                    demoVectorOperations();
                    break;
                case "3":
                    demoInteractiveQuery();
                    break;
                case "4":
                    demoVectorOperations();
                    demoRAGSystem();
                    demoInteractiveQuery();
                    break;
                default:
                    System.out.println("无效选择，运行基础演示");
                    demoRAGSystem();
                    break;
            }
            
        } catch (Exception e) {
            System.out.println("❌ 程序执行出错: " + e.getMessage());
            e.printStackTrace();
        } finally {
            scanner.close();
        }
        
        System.out.println("\n👋 程序结束，感谢使用！");
    }

    /**
     * 查询相似度内部类
     */
    private static class QuerySimilarity {
        final int index;
        final String text;
        final double similarity;
        
        QuerySimilarity(int index, String text, double similarity) {
            this.index = index;
            this.text = text;
            this.similarity = similarity;
        }
    }
}
