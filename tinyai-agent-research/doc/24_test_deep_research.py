#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepResearch Agent 测试脚本
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入模块
try:
    from importlib import import_module
    deep_research_module = import_module('24_deep_research_agent')
    DeepResearchAgent = deep_research_module.DeepResearchAgent
    demo_basic_research = deep_research_module.demo_basic_research
except ImportError:
    print("❌ 无法导入 DeepResearch Agent 模块")
    sys.exit(1)

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 测试 DeepResearch Agent 基础功能")
    
    try:
        # 创建研究Agent
        agent = DeepResearchAgent(name="测试研究助手", domain="人工智能")
        
        # 添加一些基础知识
        agent.add_domain_knowledge("人工智能是模拟人类智能的技术", "人工智能", "concept")
        agent.add_domain_knowledge("机器学习是AI的一个分支", "人工智能", "concept")
        agent.add_domain_knowledge("深度学习使用神经网络", "人工智能", "concept")
        
        print("\n✅ Agent 创建成功")
        print("✅ 知识库初始化完成")
        
        # 测试基础研究功能
        print("\n🔍 执行研究测试...")
        result = agent.research(
            query="什么是深度学习？",
            complexity=3,
            depth_required=3,
            urgency=2
        )
        
        print(f"\n🎯 研究结果:")
        print(f"  ✅ 置信度: {result['total_confidence']:.2f}")
        print(f"  📋 研究步骤数: {result['research_steps']}")
        print(f"  🔧 使用工具数: {result['tools_used']}")
        print(f"  💡 关键洞察数: {len(result['key_insights'])}")
        print(f"  🏆 质量评分: {result['quality_score']}")
        print(f"  🧠 推理模式: {result['reasoning_mode']}")
        
        print(f"\n📖 最终答案:")
        print(result['final_answer'])
        
        if result['key_insights']:
            print(f"\n💡 关键洞察:")
            for i, insight in enumerate(result['key_insights'], 1):
                print(f"  {i}. {insight}")
        
        # 测试性能报告
        print(f"\n📊 性能报告:")
        performance = agent.get_performance_report()
        print(f"  🤖 Agent名称: {performance['agent_name']}")
        print(f"  🔢 研究次数: {performance['performance_metrics']['total_research_count']}")
        print(f"  ⭐ 平均置信度: {performance['performance_metrics']['avg_confidence']:.3f}")
        print(f"  📚 知识节点数: {performance['total_knowledge_nodes']}")
        
        print(f"\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_queries():
    """测试多个查询"""
    print("\n🔬 测试多个研究查询")
    
    agent = DeepResearchAgent(name="多查询测试助手", domain="技术")
    
    # 添加知识
    knowledge_items = [
        ("人工智能包括机器学习、深度学习等技术", "人工智能", "concept"),
        ("区块链是分布式账本技术", "区块链", "concept"),
        ("量子计算利用量子力学原理", "量子技术", "concept")
    ]
    
    for content, domain, node_type in knowledge_items:
        agent.add_domain_knowledge(content, domain, node_type)
    
    queries = [
        "人工智能的发展历程",
        "区块链的应用场景",
        "量子计算的优势"
    ]
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n📋 查询 {i}: {query}")
        
        result = agent.research(
            query=query,
            complexity=2,
            depth_required=2,
            urgency=3
        )
        
        results.append(result)
        print(f"  ✅ 完成，置信度: {result['total_confidence']:.2f}")
    
    # 统计
    avg_confidence = sum(r['total_confidence'] for r in results) / len(results)
    avg_steps = sum(r['research_steps'] for r in results) / len(results)
    
    print(f"\n📈 多查询测试统计:")
    print(f"  📊 总查询数: {len(queries)}")
    print(f"  ⭐ 平均置信度: {avg_confidence:.2f}")
    print(f"  📋 平均步骤数: {avg_steps:.1f}")
    
    return results

if __name__ == "__main__":
    print("🚀 DeepResearch Agent 测试开始")
    print("="*50)
    
    # 基础功能测试
    success = test_basic_functionality()
    
    if success:
        # 多查询测试
        test_multiple_queries()
        
        print(f"\n🎊 所有测试完成！")
    else:
        print(f"\n💥 测试失败！")