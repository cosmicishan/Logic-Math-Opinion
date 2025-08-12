#!/usr/bin/env python3
"""
Question Classifier with Gemini API Support
Classifies user questions as 'factual', 'opinion', or 'math' and provides responses.
"""

import re
import math
import operator
import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    response: str


class QuestionClassifier:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key
        self.model = None
        
        # Math operators for safe evaluation
        self.math_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '^': operator.pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'exp': math.exp,
        }
        
        # Initialize Gemini if API key provided
        if self.gemini_api_key:
            self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini AI initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini AI: {e}")
            self.model = None
    
    def _classify_with_patterns(self, question: str) -> ClassificationResult:
        """Classify question using pattern matching (fallback method)."""
        question_lower = question.lower().strip()
        
        # Math patterns
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
            r'what\s+is\s+\d+.*[\+\-\*\/].*\d+',  # "what is 5 + 3"
            r'calculate|compute|solve',  # Math keywords
            r'square\s+root|sqrt|logarithm|sin|cos|tan',  # Math functions
            r'\d+\s*\%',  # Percentages
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, question_lower):
                response = self._handle_math_question(question)
                return ClassificationResult("math", 0.8, response)
        
        # Opinion patterns
        opinion_patterns = [
            r'\b(think|believe|opinion|feel|prefer|like|love|hate)\b',
            r'\b(should|ought|better|worse|best|worst)\b',
            r'\b(good|bad|beautiful|ugly|amazing|terrible)\b',
            r'what.*do.*you.*think',
            r'which.*better',
            r'your.*opinion',
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, question_lower):
                response = "That's a great question that depends on personal perspective and individual experiences. Different people might have varying viewpoints based on their background, values, and circumstances."
                return ClassificationResult("opinion", 0.7, response)
        
        # Default to factual
        response = "That's an interesting factual question. I'd need to research reliable sources to provide you with accurate information."
        return ClassificationResult("factual", 0.6, response)
    
    def _classify_with_gemini(self, question: str) -> ClassificationResult:
        """Classify question using Gemini API."""
        if not self.model:
            return self._classify_with_patterns(question)
        
        prompt = f"""
        Classify the following question into exactly one of these categories:
        - "factual": Questions about facts, data, definitions, or objective information
        - "opinion": Questions asking for subjective views, preferences, or judgments  
        - "math": Questions involving calculations, mathematical problems, or numerical operations
        
        Question: "{question}"
        
        Respond with a JSON object containing:
        - "category": one of ["factual", "opinion", "math"]
        - "confidence": a number between 0 and 1
        - "reasoning": brief explanation of classification
        
        Example: {{"category": "math", "confidence": 0.95, "reasoning": "Contains arithmetic calculation"}}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith('```'):
                result_text = result_text[3:-3].strip()
            
            result = json.loads(result_text)
            category = result.get('category', 'factual')
            confidence = float(result.get('confidence', 0.5))
            
            # Generate appropriate response based on category
            if category == "math":
                response_text = self._handle_math_question(question)
            elif category == "opinion":
                response_text = self._generate_opinion_response(question)
            else:  # factual
                response_text = self._generate_factual_response(question)
            
            return ClassificationResult(category, confidence, response_text)
            
        except Exception as e:
            print(f"⚠️ Gemini classification failed: {e}")
            return self._classify_with_patterns(question)
    
    def _handle_math_question(self, question: str) -> str:
        """Handle mathematical questions and calculations."""
        # Extract mathematical expressions
        math_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)'
        match = re.search(math_pattern, question)
        
        if match:
            try:
                num1 = float(match.group(1))
                operation = match.group(2)
                num2 = float(match.group(3))
                
                if operation == '+':
                    result = num1 + num2
                elif operation == '-':
                    result = num1 - num2
                elif operation == '*':
                    result = num1 * num2
                elif operation == '/':
                    if num2 == 0:
                        return "Error: Division by zero is undefined."
                    result = num1 / num2
                elif operation in ['^', '**']:
                    result = num1 ** num2
                else:
                    return "I can help with basic arithmetic operations (+, -, *, /, ^)."
                
                # Format result nicely
                if result.is_integer():
                    return f"The answer is {int(result)}."
                else:
                    return f"The answer is {result:.2f}."
                    
            except (ValueError, ZeroDivisionError) as e:
                return f"I encountered an error calculating that: {e}"
        
        # Handle percentage calculations
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)'
        percent_match = re.search(percent_pattern, question)
        if percent_match:
            try:
                percentage = float(percent_match.group(1))
                number = float(percent_match.group(2))
                result = (percentage / 100) * number
                return f"{percentage}% of {number} is {result:.2f}."
            except ValueError:
                return "I had trouble parsing that percentage calculation."
        
        return "I can help with basic math problems. Try asking something like 'What is 15 + 27?' or '30% of 150'."
    
    def _generate_opinion_response(self, question: str) -> str:
        """Generate response for opinion-based questions."""
        responses = [
            "That's a thoughtful question that really depends on personal values and individual perspectives.",
            "Different people might have varying opinions on this based on their experiences and beliefs.",
            "This is subjective and could have multiple valid viewpoints depending on one's background and preferences.",
            "That's an interesting question where reasonable people might disagree based on their personal experiences."
        ]
        
        # Simple hash-based selection for consistency
        index = hash(question) % len(responses)
        return responses[index]
    
    def _generate_factual_response(self, question: str) -> str:
        """Generate response for factual questions."""
        return "That's an interesting factual question. For accurate information, I'd recommend checking reliable sources or databases related to your specific topic."
    
    def classify_question(self, question: str) -> ClassificationResult:
        """Main method to classify a question and generate response."""
        if not question or not question.strip():
            return ClassificationResult("factual", 0.0, "Please ask a question!")
        
        # Use Gemini if available, otherwise fall back to patterns
        if self.model:
            return self._classify_with_gemini(question)
        else:
            return self._classify_with_patterns(question)
    
    def interactive_mode(self):
        """Run interactive question-answering session."""
        print("🤖 Question Classifier Ready!")
        print("Ask me anything, and I'll classify it as factual, opinion, or math.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    print("⚠️ Please enter a question.\n")
                    continue
                
                result = self.classify_question(question)
                
                print(f"\n📂 Category: {result.category.upper()}")
                print(f"🎯 Confidence: {result.confidence:.2f}")
                print(f"💬 Response: {result.response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the question classifier."""
    print("Question Classifier with Gemini AI Support")
    print("=" * 40)
    
    # Try to get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("💡 Tip: Set GEMINI_API_KEY environment variable for AI-powered classification")
        print("Running in pattern-matching mode...\n")
    
    # Initialize classifier
    classifier = QuestionClassifier(api_key)
    
    # Example usage
    test_questions = [
        "What is 25 + 17?",
        "What do you think about climate change?",
        "What is the capital of France?",
        "Which programming language is better?",
        "Calculate 15% of 200",
        "How tall is the Eiffel Tower?"
    ]
    
    print("🧪 Testing with example questions:\n")
    for question in test_questions:
        result = classifier.classify_question(question)
        print(f"Q: {question}")
        print(f"Category: {result.category} (confidence: {result.confidence:.2f})")
        print(f"Response: {result.response}")
        print("-" * 50)
    
    print("\n🚀 Starting interactive mode...\n")
    classifier.interactive_mode()


if __name__ == "__main__":
    main()