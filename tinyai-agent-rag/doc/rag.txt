# -*- coding: utf-8 -*-
"""
RAG（检索增强生成）与向量数据简单实现
演示向量化、相似度计算、检索和生成的完整流程
"""

import json
import math
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime
import sqlite3


@dataclass
class Document:
    """文档结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class QueryResult:
    """查询结果结构"""
    document: Document
    similarity: float
    rank: int


class SimpleTokenizer:
    """简单的分词器"""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的基于正则表达式的分词
        text = text.lower()
        # 移除标点符号，保留中英文字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        tokens = text.split()

        # 对中文进行字符级分词
        result = []
        for token in tokens:
            if re.search(r'[\u4e00-\u9fff]', token):
                # 中文字符，按字符分词
                result.extend(list(token))
            else:
                # 英文单词
                result.append(token)

        return [t for t in result if t.strip()]

    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        word_counts = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        # 按频率排序，构建词汇表
        self.vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
        self.vocab_size = len(self.vocab)

        print(f"构建词汇表完成，共 {self.vocab_size} 个词汇")

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将词汇转换为ID"""
        return [self.vocab.get(token, 0) for token in tokens]


class TFIDFVectorizer:
    """TF-IDF向量化器"""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tokenizer = SimpleTokenizer()
        self.idf_scores = {}
        self.feature_names = []

    def fit(self, documents: List[str]):
        """训练TF-IDF模型"""
        print("开始训练TF-IDF模型...")

        # 构建词汇表
        self.tokenizer.build_vocab(documents)

        # 计算文档频率
        doc_frequencies = defaultdict(int)
        total_docs = len(documents)

        for doc in documents:
            tokens = set(self.tokenizer.tokenize(doc))  # 使用set去重
            for token in tokens:
                doc_frequencies[token] += 1

        # 计算IDF分数
        for token, df in doc_frequencies.items():
            self.idf_scores[token] = math.log(total_docs / df)

        # 选择前max_features个最重要的特征
        sorted_features = sorted(self.idf_scores.items(), key=lambda x: x[1], reverse=True)
        self.feature_names = [token for token, _ in sorted_features[:self.max_features]]

        print(f"TF-IDF模型训练完成，特征维度: {len(self.feature_names)}")

    def transform(self, text: str) -> List[float]:
        """将文本转换为TF-IDF向量"""
        if not self.feature_names:
            return [0.0] * self.max_features

        tokens = self.tokenizer.tokenize(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # 计算TF-IDF向量
        vector = []
        for feature in self.feature_names:
            tf = token_counts.get(feature, 0) / total_tokens if total_tokens > 0 else 0
            idf = self.idf_scores.get(feature, 0)
            tfidf = tf * idf
            vector.append(tfidf)

        return vector

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """训练并转换文档"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


class VectorSimilarity:
    """向量相似度计算"""

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """欧氏距离"""
        if len(vec1) != len(vec2):
            return float('inf')

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    @staticmethod
    def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
        """曼哈顿距离"""
        if len(vec1) != len(vec2):
            return float('inf')

        return sum(abs(a - b) for a, b in zip(vec1, vec2))


class VectorDatabase:
    """向量数据库"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding TEXT,
                created_at REAL
            )
        """)
        self.conn.commit()

    def add_document(self, document: Document):
        """添加文档"""
        self.conn.execute("""
            INSERT OR REPLACE INTO documents 
            (id, content, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            document.id,
            document.content,
            json.dumps(document.metadata),
            json.dumps(document.embedding) if document.embedding else None,
            (document.created_at or datetime.now()).timestamp()
        ))
        self.conn.commit()

    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()

        if row:
            return Document(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=json.loads(row[3]) if row[3] else None,
                created_at=datetime.fromtimestamp(row[4])
            )
        return None

    def get_all_documents(self) -> List[Document]:
        """获取所有文档"""
        cursor = self.conn.execute("SELECT * FROM documents")
        documents = []

        for row in cursor.fetchall():
            documents.append(Document(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=json.loads(row[3]) if row[3] else None,
                created_at=datetime.fromtimestamp(row[4])
            ))

        return documents

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        cursor = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_documents(self) -> int:
        """文档总数"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]


class RAGSystem:
    """RAG检索增强生成系统"""

    def __init__(self, vector_dim: int = 512, similarity_threshold: float = 0.1):
        self.vector_dim = vector_dim
        self.similarity_threshold = similarity_threshold

        # 核心组件
        self.vectorizer = TFIDFVectorizer(max_features=vector_dim)
        self.vector_db = VectorDatabase()
        self.similarity_calculator = VectorSimilarity()

        # 状态
        self.is_trained = False
        self.documents_count = 0

    def add_documents(self, documents: List[Dict[str, Any]]):
        """批量添加文档"""
        print(f"正在添加 {len(documents)} 个文档...")

        # 准备文档内容用于训练向量化器
        contents = [doc['content'] for doc in documents]

        # 训练向量化器（如果还未训练）
        if not self.is_trained:
            self.vectorizer.fit(contents)
            self.is_trained = True

        # 向量化文档并存储
        for i, doc_data in enumerate(documents):
            doc_id = doc_data.get('id', f"doc_{int(time.time())}_{i}")
            content = doc_data['content']
            metadata = doc_data.get('metadata', {})

            # 计算文档向量
            embedding = self.vectorizer.transform(content)

            # 创建文档对象
            document = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )

            # 存储到向量数据库
            self.vector_db.add_document(document)

            if (i + 1) % 10 == 0:  # 每10个文档显示一次进度
                print(f"已处理 {i + 1}/{len(documents)} 个文档")

        self.documents_count = self.vector_db.count_documents()
        print(f"文档添加完成！当前共有 {self.documents_count} 个文档")

    def search(self, query: str, top_k: int = 5, similarity_method: str = 'cosine') -> List[QueryResult]:
        """检索相关文档"""
        if not self.is_trained:
            print("RAG系统尚未训练，请先添加文档")
            return []

        print(f"检索查询: '{query}'")

        # 向量化查询
        query_embedding = self.vectorizer.transform(query)

        # 获取所有文档
        all_documents = self.vector_db.get_all_documents()

        # 计算相似度
        similarities = []
        for doc in all_documents:
            if doc.embedding:
                if similarity_method == 'cosine':
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, doc.embedding
                    )
                elif similarity_method == 'euclidean':
                    distance = self.similarity_calculator.euclidean_distance(
                        query_embedding, doc.embedding
                    )
                    similarity = 1 / (1 + distance)  # 转换为相似度
                else:
                    similarity = self.similarity_calculator.cosine_similarity(
                        query_embedding, doc.embedding
                    )

                if similarity >= self.similarity_threshold:
                    similarities.append((doc, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for rank, (doc, similarity) in enumerate(similarities[:top_k]):
            results.append(QueryResult(
                document=doc,
                similarity=similarity,
                rank=rank + 1
            ))

        print(f"找到 {len(results)} 个相关文档")
        return results

    def generate_context(self, query: str, max_context_length: int = 1000) -> str:
        """为查询生成上下文"""
        search_results = self.search(query, top_k=5)

        if not search_results:
            return "未找到相关内容。"

        context_parts = []
        current_length = 0

        for result in search_results:
            doc = result.document
            content = doc.content

            # 添加文档信息头
            doc_header = f"[文档 {doc.id}, 相似度: {result.similarity:.3f}]\n"

            if current_length + len(doc_header) + len(content) <= max_context_length:
                context_parts.append(doc_header + content)
                current_length += len(doc_header) + len(content)
            else:
                # 截断内容以适应长度限制
                remaining_space = max_context_length - current_length - len(doc_header)
                if remaining_space > 50:  # 确保有足够空间显示有意义的内容
                    truncated_content = content[:remaining_space - 3] + "..."
                    context_parts.append(doc_header + truncated_content)
                break

        return "\n\n".join(context_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "documents_count": self.documents_count,
            "vector_dimension": self.vector_dim,
            "is_trained": self.is_trained,
            "vocabulary_size": self.vectorizer.tokenizer.vocab_size if self.is_trained else 0,
            "similarity_threshold": self.similarity_threshold
        }


def create_sample_documents() -> List[Dict[str, Any]]:
    """创建示例文档"""
    return [
        {
            "id": "python_intro",
            "content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它具有简洁的语法和强大的功能，广泛用于Web开发、数据科学、人工智能等领域。Python的设计哲学强调代码的可读性和简洁性。",
            "metadata": {"category": "编程语言", "difficulty": "入门"}
        },
        {
            "id": "machine_learning_basics",
            "content": "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下学习。机器学习算法构建数学模型，基于训练数据进行预测或决策。常见的机器学习类型包括监督学习、无监督学习和强化学习。",
            "metadata": {"category": "人工智能", "difficulty": "中级"}
        },
        {
            "id": "deep_learning_intro",
            "content": "深度学习是机器学习的一个子集，它基于人工神经网络，特别是深度神经网络。深度学习在图像识别、语音识别、自然语言处理等任务中取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是常用的深度学习架构。",
            "metadata": {"category": "人工智能", "difficulty": "高级"}
        },
        {
            "id": "data_science_overview",
            "content": "数据科学是一个跨学科领域，结合了统计学、计算机科学和领域知识来从数据中提取洞察。数据科学家使用各种工具和技术，包括数据挖掘、机器学习、可视化等，来分析复杂的数据集并解决业务问题。",
            "metadata": {"category": "数据科学", "difficulty": "中级"}
        },
        {
            "id": "web_development_python",
            "content": "Python在Web开发中非常流行，有许多强大的框架可供选择。Django是一个高级Web框架，提供了完整的解决方案。Flask是一个轻量级框架，更适合小型项目。FastAPI是一个现代框架，专为构建API而设计，支持异步编程。",
            "metadata": {"category": "Web开发", "difficulty": "中级"}
        },
        {
            "id": "natural_language_processing",
            "content": "自然语言处理（NLP）是人工智能的一个重要分支，专注于使计算机理解和生成人类语言。NLP技术包括文本分类、情感分析、机器翻译、问答系统等。近年来，基于Transformer的大语言模型如GPT、BERT等在NLP任务中表现出色。",
            "metadata": {"category": "自然语言处理", "difficulty": "高级"}
        },
        {
            "id": "database_fundamentals",
            "content": "数据库是存储和管理数据的系统。关系型数据库使用SQL语言进行查询，如MySQL、PostgreSQL等。NoSQL数据库适合处理非结构化数据，如MongoDB、Redis等。数据库设计需要考虑数据模型、索引、事务处理等方面。",
            "metadata": {"category": "数据库", "difficulty": "中级"}
        },
        {
            "id": "cloud_computing_intro",
            "content": "云计算是通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件等。主要的云服务模型有IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。AWS、Azure、Google Cloud是主要的云服务提供商。",
            "metadata": {"category": "云计算", "difficulty": "中级"}
        },
        {
            "id": "software_engineering_practices",
            "content": "软件工程是一门关于如何系统化、规范化、可量化地开发软件的学科。良好的软件工程实践包括版本控制、代码审查、单元测试、持续集成、敏捷开发等。这些实践有助于提高软件质量、降低维护成本、提升团队协作效率。",
            "metadata": {"category": "软件工程", "difficulty": "中级"}
        },
        {
            "id": "cybersecurity_basics",
            "content": "网络安全是保护计算机系统和网络免受数字攻击的实践。常见的安全威胁包括恶意软件、钓鱼攻击、数据泄露等。网络安全措施包括防火墙、加密、身份验证、访问控制等。安全开发生命周期（SDLC）将安全考虑融入软件开发的各个阶段。",
            "metadata": {"category": "网络安全", "difficulty": "高级"}
        }
    ]


def demo_rag_system():
    """演示RAG系统"""
    print("=" * 60)
    print("🔍 RAG检索增强生成系统演示")
    print("=" * 60)

    # 创建RAG系统
    rag = RAGSystem(vector_dim=256, similarity_threshold=0.05)

    # 创建示例文档
    print("\n📚 准备示例文档...")
    documents = create_sample_documents()

    # 添加文档到RAG系统
    rag.add_documents(documents)

    # 显示系统统计信息
    stats = rag.get_statistics()
    print(f"\n📊 系统统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 40)
    print("🔍 开始检索演示")
    print("=" * 40)

    # 测试查询
    test_queries = [
        "Python编程语言的特点",
        "机器学习算法",
        "深度学习神经网络",
        "Web开发框架",
        "数据库管理系统",
        "云计算服务模型",
        "网络安全防护"
    ]

    for query in test_queries:
        print(f"\n🔎 查询: '{query}'")
        print("-" * 50)

        # 执行检索
        results = rag.search(query, top_k=3)

        if results:
            for result in results:
                doc = result.document
                print(f"📄 文档ID: {doc.id}")
                print(f"📊 相似度: {result.similarity:.4f}")
                print(f"📝 内容: {doc.content[:100]}...")
                print(f"🏷️  类别: {doc.metadata.get('category', 'N/A')}")
                print()
        else:
            print("❌ 未找到相关文档")

        # 生成上下文
        context = rag.generate_context(query, max_context_length=300)
        print(f"📋 生成的上下文:\n{context[:200]}...\n")

    # 交互式查询
    print("\n" + "=" * 40)
    print("💬 交互式查询模式")
    print("=" * 40)
    print("输入查询内容（输入 'quit' 退出）:")

    while True:
        try:
            user_query = input("\n🔍 请输入查询: ").strip()

            if user_query.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 退出查询模式")
                break

            if not user_query:
                continue

            # 执行检索
            results = rag.search(user_query, top_k=3)

            if results:
                print(f"\n📋 找到 {len(results)} 个相关文档:")
                for i, result in enumerate(results, 1):
                    doc = result.document
                    print(f"\n{i}. 📄 {doc.id}")
                    print(f"   📊 相似度: {result.similarity:.4f}")
                    print(f"   🏷️  类别: {doc.metadata.get('category', 'N/A')}")
                    print(f"   📝 内容: {doc.content}")

                # 生成上下文用于回答
                context = rag.generate_context(user_query)
                print(f"\n📋 基于检索结果的上下文:")
                print(context)
            else:
                print("❌ 未找到相关文档，请尝试其他关键词")

        except KeyboardInterrupt:
            print("\n👋 程序被中断，退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def demo_vector_operations():
    """演示向量操作"""
    print("\n" + "=" * 50)
    print("🧮 向量操作演示")
    print("=" * 50)

    # 创建示例文本
    texts = [
        "人工智能是一门计算机科学",
        "机器学习是人工智能的分支",
        "深度学习使用神经网络",
        "Python是流行的编程语言",
        "数据科学分析大量数据"
    ]

    # 创建向量化器
    vectorizer = TFIDFVectorizer(max_features=20)

    # 训练并转换
    print("\n📊 训练TF-IDF向量化器...")
    vectors = vectorizer.fit_transform(texts)

    print(f"词汇表大小: {len(vectorizer.feature_names)}")
    print(f"向量维度: {len(vectors[0])}")

    # 显示特征词汇
    print(f"\n🔤 特征词汇: {vectorizer.feature_names[:10]}...")

    # 显示向量
    print(f"\n📈 文本向量:")
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        non_zero_features = [(vectorizer.feature_names[j], vector[j])
                             for j in range(len(vector)) if vector[j] > 0]
        print(f"{i+1}. '{text}'")
        print(f"   非零特征: {non_zero_features[:3]}...")

    # 计算相似度矩阵
    print(f"\n📏 余弦相似度矩阵:")
    print("    ", end="")
    for i in range(len(texts)):
        print(f"{i+1:6}", end="")
    print()

    for i in range(len(vectors)):
        print(f"{i+1:2}: ", end="")
        for j in range(len(vectors)):
            similarity = VectorSimilarity.cosine_similarity(vectors[i], vectors[j])
            print(f"{similarity:5.3f} ", end="")
        print()

    # 查询相似度
    print(f"\n🔍 查询相似度测试:")
    query_text = "机器学习算法"
    query_vector = vectorizer.transform(query_text)

    print(f"查询: '{query_text}'")
    similarities = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        similarity = VectorSimilarity.cosine_similarity(query_vector, vector)
        similarities.append((i, text, similarity))

    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)

    print("相似度排序结果:")
    for rank, (idx, text, sim) in enumerate(similarities, 1):
        print(f"{rank}. {sim:.4f} - '{text}'")


def main():
    """主函数"""
    print("🚀 RAG与向量数据实现演示")
    print("选择演示模式:")
    print("1. RAG检索增强生成系统演示")
    print("2. 向量操作基础演示")
    print("3. 完整演示（包含两个部分）")

    try:
        choice = input("\n请选择 (1-3): ").strip()

        if choice == '1':
            demo_rag_system()
        elif choice == '2':
            demo_vector_operations()
        elif choice == '3':
            demo_vector_operations()
            demo_rag_system()
        else:
            print("无效选择，运行完整演示")
            demo_vector_operations()
            demo_rag_system()

    except KeyboardInterrupt:
        print("\n👋 程序被中断，再见！")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")


if __name__ == "__main__":
    main()