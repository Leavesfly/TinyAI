# -*- coding: utf-8 -*-
"""
OpenManus Agent系统演示程序
=====================================

完整演示OpenManus Agent系统的四大核心特征：
1. 双执行机制演示
2. 分层架构展示
3. 计划驱动任务分解演示
4. 动态工具调用演示

Author: 山泽
Date: 2025-10-03
"""

# 为了简化演示，直接导入核心模块
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import uuid
import time
import re
import logging

# 导入核心系统
exec(open('22_openmanus_agent_system.py').read())
import time


def demo_direct_agent_mode():
    """演示直接Agent模式"""
    print("\n" + "=" * 60)
    print("🤖 OpenManus - 直接Agent模式演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Direct")
    manus.set_execution_mode(ExecutionMode.DIRECT_AGENT)
    manus.planning_enabled = False  # 关闭计划模式，使用基础ReAct

    test_queries = [
        "计算 25 * 8 + 15",
        "现在几点了？",
        "分析这个文本: 'OpenManus是一个强大的Agent系统'"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试 {i}: {query}")
        print("-" * 40)

        message = Message(content=query, role="user")
        response = manus.process_message(message)

        print(f"🎯 回答: {response.content}")
        print(f"📊 状态: {manus.get_status()['state']}")
        time.sleep(1)  # 演示间隔


def demo_planning_driven_mode():
    """演示计划驱动模式"""
    print("\n" + "=" * 60)
    print("📋 OpenManus - 计划驱动模式演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Planning")
    manus.set_execution_mode(ExecutionMode.DIRECT_AGENT)
    manus.planning_enabled = True  # 启用计划模式

    complex_queries = [
        "详细分析计算 100 * 25 的结果",
        "研究当前时间并进行深入分析",
        "制定一个完整的学习计划"
    ]

    for i, query in enumerate(complex_queries, 1):
        print(f"\n📝 复杂查询 {i}: {query}")
        print("-" * 50)

        message = Message(content=query, role="user")
        response = manus.process_message(message)

        print(f"🎯 计划执行结果:")
        print(response.content)
        print(f"📊 执行信息: {response.metadata}")
        time.sleep(1)


def demo_flow_orchestration():
    """演示Flow编排模式"""
    print("\n" + "=" * 60)
    print("🔄 OpenManus - Flow编排模式演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Flow")

    # 注册示例Flow定义
    calculation_flow = {
        "name": "计算流程",
        "description": "专门处理数学计算的流程",
        "nodes": [
            {"type": "agent", "name": "问题分析", "description": "分析数学问题"},
            {"type": "tool", "name": "calculator", "description": "执行计算"},
            {"type": "agent", "name": "结果整理", "description": "整理计算结果"}
        ]
    }

    time_flow = {
        "name": "时间查询流程",
        "description": "处理时间相关查询的流程",
        "nodes": [
            {"type": "tool", "name": "get_time", "description": "获取时间"},
            {"type": "agent", "name": "时间格式化", "description": "格式化时间信息"}
        ]
    }

    analysis_flow = {
        "name": "分析流程",
        "description": "处理分析任务的流程",
        "nodes": [
            {"type": "tool", "name": "text_analyzer", "description": "文本分析"},
            {"type": "agent", "name": "结果解释", "description": "解释分析结果"}
        ]
    }

    # 注册Flow
    manus.register_flow("calculation_flow", calculation_flow)
    manus.register_flow("time_flow", time_flow)
    manus.register_flow("analysis_flow", analysis_flow)
    manus.set_execution_mode(ExecutionMode.FLOW_ORCHESTRATION)

    flow_queries = [
        "计算 15 + 25 * 3",
        "查询当前时间",
        "分析文本内容",
        "这是一个通用查询"  # 测试回退机制
    ]

    for i, query in enumerate(flow_queries, 1):
        print(f"\n📝 Flow查询 {i}: {query}")
        print("-" * 40)

        message = Message(content=query, role="user")
        response = manus.process_message(message)

        print(f"🎯 Flow执行结果: {response.content}")
        if response.metadata:
            print(f"📊 Flow信息: {response.metadata}")
        time.sleep(1)


def demo_layered_architecture():
    """演示分层架构"""
    print("\n" + "=" * 60)
    print("🏗️ OpenManus - 分层架构演示")
    print("=" * 60)

    print("创建各层Agent实例...")

    # 创建各层实例
    base_agent = BaseAgent(name="基础层Agent")
    react_agent = ReActAgent(name="ReAct层Agent")
    toolcall_agent = ToolCallAgent(name="工具调用层Agent")
    manus_agent = Manus(name="Manus核心层Agent")

    agents = [base_agent, react_agent, toolcall_agent, manus_agent]

    # 展示各层状态
    print("\n各层Agent状态信息:")
    for agent in agents:
        if hasattr(agent, 'get_status'):
            status = agent.get_status()
            print(f"📊 {agent.name}: {status}")

    # 测试消息处理 (只有非抽象类可以处理)
    test_message = Message(content="计算 10 + 20", role="user")

    print(f"\n测试消息: {test_message.content}")
    print("-" * 40)

    for agent in [react_agent, toolcall_agent, manus_agent]:
        print(f"\n{agent.name} 处理结果:")
        try:
            response = agent.process_message(test_message)
            print(f"回答: {response.content[:100]}...")
        except Exception as e:
            print(f"处理失败: {e}")


def demo_tool_management():
    """演示工具管理系统"""
    print("\n" + "=" * 60)
    print("🔧 OpenManus - 工具管理系统演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Tools")

    # 展示内置工具
    print("内置工具列表:")
    tools = manus.tool_registry.list_tools()
    for tool in tools:
        print(f"  🛠️ {tool['name']}: {tool['description']}")

    # 注册自定义工具
    def weather_tool(city: str) -> Dict[str, Any]:
        """天气查询工具"""
        weather_data = {
            "北京": "晴天 25°C",
            "上海": "多云 22°C",
            "广州": "雨天 28°C"
        }
        return {"city": city, "weather": weather_data.get(city, "暂无数据")}

    def translator_tool(text: str, target_lang: str = "en") -> Dict[str, Any]:
        """翻译工具（模拟）"""
        translations = {
            "你好": "Hello",
            "谢谢": "Thank you",
            "再见": "Goodbye"
        }
        return {"original": text, "translated": translations.get(text, f"[{target_lang}] {text}")}

    print("\n注册自定义工具...")
    manus.register_custom_tool("weather", weather_tool, "天气查询工具")
    manus.register_custom_tool("translator", translator_tool, "文本翻译工具")

    # 展示更新后的工具列表
    print("\n更新后的工具列表:")
    tools = manus.tool_registry.list_tools()
    for tool in tools:
        print(f"  🛠️ {tool['name']}: {tool['description']}")

    # 测试工具调用
    print("\n测试工具调用:")
    test_calls = [
        ToolCall(name="weather", arguments={"city": "北京"}),
        ToolCall(name="translator", arguments={"text": "你好", "target_lang": "en"}),
        ToolCall(name="calculator", arguments={"expression": "50 * 2"})
    ]

    for tool_call in test_calls:
        result = manus.tool_registry.call_tool(tool_call)
        print(f"🔧 {tool_call.name}: {result.result if result.result else result.error}")

    # 展示工具统计
    manus.tool_call_history.extend(test_calls)
    print(f"\n📊 工具使用统计: {manus.get_tool_stats()}")


def demo_system_monitoring():
    """演示系统监控"""
    print("\n" + "=" * 60)
    print("📊 OpenManus - 系统监控演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Monitor")

    # 配置系统
    manus.set_execution_mode(ExecutionMode.DIRECT_AGENT)
    manus.planning_enabled = True

    # 注册Flow
    sample_flow = {
        "name": "示例流程",
        "nodes": [{"type": "agent", "name": "处理节点"}]
    }
    manus.register_flow("sample_flow", sample_flow)

    # 处理一些消息来生成数据
    test_messages = [
        "计算 100 + 200",
        "查询时间",
        "详细分析系统状态"
    ]

    print("处理测试消息...")
    for msg_content in test_messages:
        message = Message(content=msg_content, role="user")
        manus.process_message(message)
        time.sleep(0.5)

    # 展示系统状态
    print("\n系统状态监控:")
    status = manus.get_system_status()

    for key, value in status.items():
        print(f"📈 {key}: {value}")

    # 展示消息历史
    print(f"\n消息历史 (共 {len(manus.messages)} 条):")
    for i, msg in enumerate(manus.messages[-6:], 1):  # 显示最近6条
        print(f"  {i}. [{msg.role}] {msg.content[:50]}...")


def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("💬 OpenManus - 交互式演示")
    print("=" * 60)

    manus = Manus(name="OpenManus-Interactive")

    # 配置系统
    manus.set_execution_mode(ExecutionMode.DIRECT_AGENT)
    manus.planning_enabled = True

    # 注册Flow
    flows = {
        "math_flow": {
            "name": "数学计算流程",
            "nodes": [{"type": "tool", "name": "calculator"}]
        },
        "time_flow": {
            "name": "时间查询流程",
            "nodes": [{"type": "tool", "name": "get_time"}]
        }
    }

    for flow_id, flow_def in flows.items():
        manus.register_flow(flow_id, flow_def)

    print("🎮 OpenManus Agent系统已启动！")
    print("\n可用功能:")
    print("- 数学计算: '计算 10 + 20'")
    print("- 时间查询: '现在几点?'")
    print("- 文本分析: '分析这段文本'")
    print("- 复杂任务: '详细研究某个主题'")
    print("- 系统控制: 'mode:flow' (切换到Flow模式), 'status' (查看状态)")
    print("\n输入 'quit' 退出演示")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n👋 感谢使用OpenManus Agent系统演示！")
                break

            if not user_input:
                continue

            # 处理系统命令
            if user_input.startswith('mode:'):
                mode_name = user_input.split(':', 1)[1].strip()
                if mode_name == 'flow':
                    manus.set_execution_mode(ExecutionMode.FLOW_ORCHESTRATION)
                    print("🔄 已切换到Flow编排模式")
                elif mode_name == 'direct':
                    manus.set_execution_mode(ExecutionMode.DIRECT_AGENT)
                    print("🤖 已切换到直接Agent模式")
                else:
                    print("❌ 不支持的模式")
                continue

            if user_input == 'status':
                status = manus.get_system_status()
                print("\n📊 系统状态:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                continue

            # 处理用户消息
            message = Message(content=user_input, role="user")
            response = manus.process_message(message)

            print(f"\n🤖 Manus: {response.content}")

            if response.metadata:
                print(f"💡 执行信息: {response.metadata}")

        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def comprehensive_demo():
    """综合演示"""
    print("\n" + "🌟" * 30)
    print("OpenManus Agent系统完整演示")
    print("🌟" * 30)

    print("""
🎯 演示内容：
1. 直接Agent模式 - 基础的推理与行动
2. 计划驱动模式 - 复杂任务的分解与执行  
3. Flow编排模式 - 工作流程的灵活编排
4. 分层架构展示 - 四层架构的协同工作
5. 工具管理系统 - 动态工具注册与调用
6. 系统监控 - 实时状态监控与统计
7. 交互式体验 - 与系统的实时互动

OpenManus的四大核心特征：
✅ 双执行机制（直接Agent模式 & Flow编排模式）
✅ 分层架构（BaseAgent → ReActAgent → ToolCallAgent → Manus）
✅ 计划驱动任务分解
✅ 动态工具调用
    """)

    demo_options = {
        "1": ("直接Agent模式演示", demo_direct_agent_mode),
        "2": ("计划驱动模式演示", demo_planning_driven_mode),
        "3": ("Flow编排模式演示", demo_flow_orchestration),
        "4": ("分层架构演示", demo_layered_architecture),
        "5": ("工具管理演示", demo_tool_management),
        "6": ("系统监控演示", demo_system_monitoring),
        "7": ("交互式演示", interactive_demo),
        "8": ("全部演示", None)
    }

    while True:
        print("\n" + "=" * 50)
        print("选择演示内容:")
        for key, (name, _) in demo_options.items():
            print(f"{key}. {name}")
        print("0. 退出")

        choice = input("\n请选择 (0-8): ").strip()

        if choice == "0":
            print("\n👋 感谢使用OpenManus Agent系统演示！")
            break
        elif choice == "8":
            # 全部演示
            for i in range(1, 8):
                print(f"\n🎬 开始第{i}个演示...")
                demo_options[str(i)][1]()
                time.sleep(2)
            print("\n🎉 全部演示完成！")
        elif choice in demo_options and choice != "8":
            demo_options[choice][1]()
        else:
            print("❌ 无效选择，请输入 0-8")


if __name__ == "__main__":
    # 设置日志级别为WARNING，减少演示中的日志输出
    logging.basicConfig(level=logging.WARNING)

    try:
        comprehensive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示被中断，再见！")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()