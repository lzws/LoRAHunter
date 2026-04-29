"""
模块1: Concept Extractor (概念提取器)

功能: 基于大语言模型(LLM)将用户prompt分解为多个语义不重叠的concepts

论文参考:
- Section 4.1: Concept Extractor
- Supplementary Material (SM) §8: Complete prompt for concept extraction
"""

import json
from typing import List, Optional
import openai


class ConceptExtractor:
    """
    概念提取器
    
    使用LLM将用户输入的prompt分解为明确的、不重叠的概念chunks。
    例如: "a British shorthair cat playing in a cherry blossom garden"
    输出: ["British shorthair cat", "cherry blossom garden"]
    
    数学表示 (论文公式1):
        C(s) = {t_i | t_i ∈ T(s), t_i ⊆ s}
    其中:
        s: 输入prompt
        T(s): 从s中提取的概念集合
        t_i: 第i个概念
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 200):
        """
        初始化概念提取器
        
        Args:
            model: 使用的LLM模型，论文使用gpt-4o-mini
            api_key: OpenAI API密钥
            temperature: 采样温度，低温度确保输出一致性
            max_tokens: 最大输出token数
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 设置API密钥
        if api_key:
            openai.api_key = api_key
        
        # =========================================================================
        # 系统Prompt设计 (基于论文Supplementary Material §8)
        # 指导LLM如何从prompt中提取概念
        # =========================================================================
        self.system_prompt = """You are a concept extraction assistant for image generation prompts.

Your task is to analyze the given prompt and extract distinct, non-overlapping conceptual chunks. Each concept should represent a unified entity including its descriptive adjectives or qualifiers.

Instructions:
- Extract keywords that include any relevant descriptive adjectives or qualifiers
- If multiple keywords describe a single entity, treat them as one unified concept
- Treat multiple keywords describing a single entity as one unified concept
- Return concepts as a JSON list of strings
- Each concept should be self-contained and semantically distinct
- Extract 1-3 concepts depending on the complexity of the prompt

Examples:

Input: "a British shorthair cat playing in a cherry blossom garden"
Output: ["British shorthair cat", "cherry blossom garden"]

Input: "A market square in a medieval village"
Output: ["market square", "medieval village"]

Input: "An impressionist painting of the geyser Old Faithful"
Output: ["impressionist painting", "geyser Old Faithful"]

Input: "A modern racecar"
Output: ["modern racecar"]

Respond ONLY with a JSON array of strings, nothing else."""
    
    def extract_concepts(self, prompt: str) -> List[str]:
        """
        从用户prompt中提取concepts集合 T(s)
        
        这是论文公式1的实现:
            C(s) = {t_i | t_i ∈ T(s), t_i ⊆ s}
        
        Args:
            prompt: 用户输入的文本提示 (s)
            
        Returns:
            List[str]: 提取的概念列表 [t₁, t₂, ..., tₙ]
                      每个概念t_i都是原prompt s的子集
        """
        try:
            # 调用OpenAI API进行概念提取
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Extract concepts from: \"{prompt}\""}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # 解析JSON响应
            concepts = self._parse_response(content)
            
            # 后处理：确保每个concept都是有效的
            concepts = [c.strip() for c in concepts if c.strip()]
            
            # 如果没有提取到概念，使用fallback方法
            if not concepts:
                concepts = self._fallback_extract(prompt)
            
            return concepts
            
        except Exception as e:
            print(f"Error in concept extraction: {e}")
            return self._fallback_extract(prompt)
    
    def _parse_response(self, content: str) -> List[str]:
        """
        解析LLM的JSON响应
        
        Args:
            content: LLM返回的原始文本
            
        Returns:
            List[str]: 解析后的概念列表
        """
        try:
            # 尝试直接解析JSON
            concepts = json.loads(content)
            if isinstance(concepts, list):
                return [str(c) for c in concepts]
        except json.JSONDecodeError:
            pass
        
        # 如果直接解析失败，尝试提取方括号中的内容
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                concepts = json.loads(content[start:end+1])
                if isinstance(concepts, list):
                    return [str(c) for c in concepts]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 如果还是失败，尝试按行分割
        lines = [line.strip().strip('"\'') for line in content.split('\n') 
                 if line.strip() and not line.startswith('```')]
        return lines
    
    def _fallback_extract(self, prompt: str) -> List[str]:
        """
        Fallback概念提取方法
        
        当LLM调用失败或解析失败时使用简单的启发式规则提取概念。
        实际应用中应该实现更智能的NLP方法。
        
        Args:
            prompt: 用户输入的文本提示
            
        Returns:
            List[str]: 提取的概念列表
        """
        import re
        
        # 按常见标点符号分割
        parts = re.split(r'[,;:]', prompt)
        concepts = [p.strip() for p in parts if len(p.strip()) > 3]
        
        # 如果概念太多，合并为2-3个
        if len(concepts) > 3:
            mid = len(concepts) // 2
            return [
                ' '.join(concepts[:mid]).strip(),
                ' '.join(concepts[mid:]).strip()
            ]
        
        return concepts if concepts else [prompt]
    
    def extract_concepts_batch(self, prompts: List[str]) -> List[List[str]]:
        """
        批量提取概念
        
        Args:
            prompts: 用户输入的文本提示列表
            
        Returns:
            List[List[str]]: 每个prompt对应的概念列表
        """
        return [self.extract_concepts(p) for p in prompts]


# =========================================================================
# 单机测试代码
# =========================================================================
if __name__ == "__main__":
    import os
    
    # 从环境变量读取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 创建实例
    extractor = ConceptExtractor(api_key=api_key)
    
    # 测试用例 (来自论文)
    test_prompts = [
        "a British shorthair cat playing in a cherry blossom garden",
        "A market square in a medieval village",
        "An impressionist painting of the geyser Old Faithful",
        "A fantastical witch in a forest in vibrant style"
    ]
    
    print("Testing Concept Extractor:")
    print("=" * 50)
    
    for prompt in test_prompts:
        concepts = extractor.extract_concepts(prompt)
        print(f"\nInput: {prompt}")
        print(f"Output: {concepts}")
