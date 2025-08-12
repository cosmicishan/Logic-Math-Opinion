# Logic-Math-Opinion

# Question Classifier with Gemini API

A Python script that classifies user questions as `factual`, `opinion`, or `math` and provides appropriate responses. Features both pattern-matching fallback and AI-powered classification using Google's Gemini API.

## Features

- **Question Classification**: Automatically categorizes questions into factual, opinion, or math types
- **Math Problem Solving**: Handles basic arithmetic, percentages, and mathematical operations
- **AI-Powered Analysis**: Uses Gemini API for more accurate classification (optional)
- **Fallback System**: Works with pattern matching when API is unavailable
- **Interactive Mode**: Command-line interface for real-time question answering

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
```

2. Install dependencies:
```bash
uv sync
```

3. (Optional) Set up Gemini API:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Usage

### Command Line
```bash
python main.py
```

### As a Module
```python
from question_classifier import QuestionClassifier

# Initialize with API key (optional)
classifier = QuestionClassifier(gemini_api_key="your-key")

# Classify a question
result = classifier.classify_question("What is 15 + 25?")
print(f"Category: {result.category}")
print(f"Response: {result.response}")
```

## API Integration Details

### Current Implementation (Gemini API)
- Uses `google-generativeai` library
- Fallback to pattern matching if API fails
- JSON-structured responses for consistent parsing
- Error handling and retry logic

### Adding Other APIs

To add support for other LLM APIs (OpenAI, Anthropic, etc.), extend the classifier:

```python
def _classify_with_openai(self, question: str) -> ClassificationResult:
    """Example OpenAI integration"""
    import openai
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Classify this question as factual/opinion/math: {question}"
        }]
    )
    # Process response...
```

## Example Classifications

- **Math**: "What is 25 + 17?" → Calculates and returns "42"
- **Opinion**: "What do you think about AI?" → Acknowledges subjectivity
- **Factual**: "What is the capital of France?" → Suggests reliable sources

##Demo 

<img width="1479" height="656" alt="Screenshot from 2025-08-13 04-07-23" src="https://github.com/user-attachments/assets/aa6a2405-89f2-4ae4-b032-6a188429cf62" />

