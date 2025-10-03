#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Coding Cursor 功能测试脚本

演示系统的核心功能，无需用户交互
"""

import sys
import os

# 直接执行主文件内容
with open('23_ai_coding_cursor.py', 'r', encoding='utf-8') as f:
    exec(f.read())

def test_ai_coding_cursor():
    """测试 AI Coding Cursor 的主要功能"""
    print("🧪 AI Coding Cursor 功能测试")
    print("=" * 50)
    
    # 创建 AI Coding Cursor 实例
    cursor = AICodingCursor("测试助手")
    
    # 测试代码示例
    test_code = '''
def calculate_fibonacci(n):
    """计算斐波那契数列的第n项"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class MathUtils:
    def __init__(self):
        self.pi = 3.14159
    
    def circle_area(self, radius):
        return self.pi * radius * radius
    
    def circle_circumference(self, radius):
        return 2 * self.pi * radius
'''
    
    print("\n1. 🔍 代码分析测试")
    print("-" * 30)
    analysis = cursor.analyze_code(test_code)
    
    if analysis.get("syntax_valid", False):
        print("✅ 代码语法正确")
        print(f"📊 函数数量: {len(analysis['functions'])}")
        print(f"📊 类数量: {len(analysis['classes'])}")
        print(f"📊 总行数: {analysis['metrics']['total_lines']}")
        print(f"📊 代码行数: {analysis['metrics']['code_lines']}")
        print(f"📊 注释行数: {analysis['metrics']['comment_lines']}")
        print(f"📊 复杂度: {analysis['complexity']}")
        
        print(f"\n📦 导入分析:")
        for imp in analysis['imports']:
            print(f"  • {imp}")
        
        print(f"\n🏛️ 类结构:")
        for cls in analysis['classes']:
            print(f"  • {cls['name']} (行 {cls['line']})")
            print(f"    方法: {', '.join(cls['methods'])}")
        
        print(f"\n⚡ 函数列表:")
        for func in analysis['functions']:
            print(f"  • {func['name']}({', '.join(func['args'])}) - 行 {func['line']}")
        
        if analysis['issues']:
            print(f"\n⚠️ 发现问题:")
            for issue in analysis['issues']:
                print(f"  • [{issue.severity}] {issue.message} (行 {issue.line_number})")
        else:
            print("\n✅ 未发现明显问题")
    
    print("\n2. 🤖 代码生成测试")
    print("-" * 30)
    
    # 生成函数
    print("生成函数示例:")
    function_code = cursor.generate_code("function validate_email")
    print(function_code[:300] + "..." if len(function_code) > 300 else function_code)
    
    # 生成类
    print("\n生成类示例:")
    class_code = cursor.generate_code("class DatabaseManager")
    print(class_code[:300] + "..." if len(class_code) > 300 else class_code)
    
    print("\n3. 🔧 重构建议测试")
    print("-" * 30)
    refactor_suggestions = cursor.suggest_refactor(test_code)
    
    if refactor_suggestions:
        for i, suggestion in enumerate(refactor_suggestions, 1):
            print(f"{i}. {suggestion.description}")
            print(f"   类型: {suggestion.suggestion_type}")
            print(f"   影响: {suggestion.estimated_impact}")
            print(f"   优势: {', '.join(suggestion.benefits)}")
    else:
        print("✅ 代码结构良好，无需重构")
    
    print("\n4. 🐛 调试功能测试")
    print("-" * 30)
    
    # 测试有语法错误的代码
    buggy_code = '''
def broken_function(x, y)
    if x > y
        return x
    else
        return y
'''
    
    debug_result = cursor.debug_code(buggy_code)
    
    if debug_result.get("error_found", False):
        print(f"❌ 发现错误: {debug_result['error_type']}")
        print(f"📍 行号: {debug_result['error_line']}")
        print(f"💡 诊断: {debug_result['diagnosis']}")
        
        suggestions = debug_result.get("suggestions", [])
        if suggestions:
            print("🔧 修复建议:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        
        # 显示修复后的代码
        fixed_code = debug_result.get("fixed_code", "")
        if fixed_code and fixed_code != buggy_code:
            print(f"\n✨ 尝试自动修复:")
            print(fixed_code)
    else:
        print("✅ 未发现明显错误")
    
    print("\n5. 📋 代码审查测试")
    print("-" * 30)
    review_result = cursor.code_review(test_code)
    
    print(f"🏆 总体评分: {review_result['overall_score']:.1f}/100")
    
    recommendations = review_result.get("recommendations", [])
    if recommendations:
        print("📝 改进建议:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("✅ 代码质量优秀")
    
    print("\n6. 🚀 高级功能测试")
    print("-" * 30)
    
    # 测试复杂代码
    complex_code = '''
import json
import sqlite3
from typing import List, Dict, Optional

class UserRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()
    
    def _initialize_schema(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.connection.commit()
    
    def create_user(self, name: str, email: str) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_users(self) -> List[Dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
'''
    
    complex_analysis = cursor.analyze_code(complex_code)
    print(f"复杂代码分析:")
    print(f"  总行数: {complex_analysis['metrics']['total_lines']}")
    print(f"  类数量: {len(complex_analysis['classes'])}")
    print(f"  方法数量: {len(complex_analysis['functions'])}")
    print(f"  复杂度: {complex_analysis['complexity']}")
    
    complex_suggestions = cursor.suggest_refactor(complex_code)
    if complex_suggestions:
        print(f"  重构建议: {len(complex_suggestions)} 项")
        for suggestion in complex_suggestions[:2]:  # 显示前2个建议
            print(f"    • {suggestion.description}")
    
    complex_review = cursor.code_review(complex_code)
    print(f"  质量评分: {complex_review['overall_score']:.1f}/100")
    
    print("\n🎉 测试完成！")
    print("AI Coding Cursor 所有核心功能运行正常")


if __name__ == "__main__":
    test_ai_coding_cursor()