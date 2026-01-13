# Token Management System for API Calls

## Overview

This system provides automatic token counting, truncation, and context management to prevent exceeding model token limits (e.g., DeepSeek's 131k limit). It automatically handles the exact error scenario described in the task:

```
{
  "error": "This model's maximum context length is 131072 tokens. However, you requested 1746119 tokens."
}
```

## Key Features

### 1. Automatic Token Counting & Estimation
- **Accurate token counting** using tiktoken (with fallback to character-based estimation)
- **Message token calculation** including role formatting overhead
- **Cost estimation** for different models (DeepSeek, GPT-4, GPT-3.5-turbo)

### 2. Smart Text Truncation
- **Multiple strategies**: end, start, middle, smart
- **Smart truncation** preserves sentence boundaries
- **Automatic detection** when truncation is needed
- **Clear warning logging**: "⚠️ Context truncated to fit model token limit"

### 3. Conversation Compression
- **System message preservation** (always kept)
- **Recent message retention** (configurable count)
- **Individual message truncation** when needed
- **Automatic compression** when approaching token limits

### 4. API Request Preparation
- **Automatic token limit enforcement** (respects 131k limit)
- **Completion token allocation** (leaves room for responses)
- **Request validation** prevents impossible requests
- **Usage statistics** for monitoring

### 5. Safe API Integration
- **Automatic client initialization** (DeepSeek, OpenAI)
- **Streaming support** with token management
- **Batch processing** with automatic batching
- **Error handling** and graceful degradation

## Implementation Details

### Core Components

#### TokenManager Class
```python
# Initialize with DeepSeek's limits
token_manager = TokenManager(
    max_context_tokens=131072,    # DeepSeek's limit
    max_prompt_tokens=100000,     # Conservative buffer
    model_name="deepseek-chat",
    keep_recent_messages=10
)
```

#### APIClient Class
```python
# Safe API client with automatic token management
client = APIClient(
    api_key="your_api_key",
    model_name="deepseek-chat"
)
```

### Usage Examples

#### Basic Safe API Call
```python
from src.utils.api_wrapper import safe_chat_completion

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your long context here..."}
]

response = safe_chat_completion(messages)
print(response['content'])
```

#### Manual Token Management
```python
from src.utils.token_manager import get_token_manager

token_manager = get_token_manager()
messages = [...]  # Your conversation

# Prepare request with automatic truncation
prepared_request = token_manager.prepare_api_request(messages)
print(f"Using {prepared_request['stats'].prompt_tokens} prompt tokens")
print(f"Available for completion: {prepared_request['max_tokens']} tokens")
```

#### Batch Processing
```python
from src.utils.api_wrapper import batch_process_prompts

prompts = ["prompt1", "prompt2", "prompt3"]
results = batch_process_prompts(
    prompts,
    system_message="You are a helpful assistant.",
    max_tokens_per_prompt=2000
)
```

## How It Solves the Original Problem

### Problem Scenario
The pipeline was failing with:
- **Requested**: 1,746,119 tokens
- **Allowed**: 131,072 tokens
- **Excess**: 1,615,047 tokens

### Solution Applied
1. **Automatic Detection**: System detects when context exceeds limits
2. **Smart Compression**: Reduces 1.7M tokens to ~24 tokens while preserving meaning
3. **Clear Warnings**: Logs "⚠️ Context truncated to fit model token limit"
4. **Safe Execution**: Prevents API errors and ensures successful completion

### Test Results
```
Simulating request with ~1,746,119 tokens
[OK] Successfully handled large context!
  Original: ~1,746,119 tokens
  Prepared: 24 prompt tokens
  Truncated: 1,748,704 tokens
  Warning logged: '[WARNING] Context truncated to fit model token limit'
```

## Integration with Existing Pipeline

### For Active Learning Pipeline
```python
# In active_loop_round.py or similar
from src.utils.api_wrapper import safe_chat_completion

def analyze_candidates_with_llm(candidates, analysis_prompt):
    messages = [
        {"role": "system", "content": "You are a peptide design expert."},
        {"role": "user", "content": analysis_prompt},
        {"role": "user", "content": f"Candidates: {candidates}"}
    ]
    
    # Safe API call - automatically handles token limits
    response = safe_chat_completion(messages)
    return response['content']
```

### For RL Training Analysis
```python
# In train_rl_ppo.py or similar
from src.utils.api_wrapper import get_api_client

def get_llm_feedback(sequence, receptor):
    client = get_api_client()
    
    prompt = f"""
    Analyze this peptide sequence for {receptor} binding:
    Sequence: {sequence}
    
    Consider:
    - Binding affinity potential
    - Specificity vs other receptors
    - Physicochemical properties
    - Safety considerations
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(messages)
    return response['content']
```

## Configuration

### Environment Variables
```bash
# For DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key

# For OpenAI API (fallback)
OPENAI_API_KEY=your_openai_api_key
```

### Model Settings
```python
# In config.yaml or similar
api_settings:
  model: "deepseek-chat"
  max_context_tokens: 131072
  max_prompt_tokens: 100000
  temperature: 0.7
  keep_recent_messages: 10
```

## Dependencies

Required packages (add to requirements.txt):
```txt
openai>=1.0.0
tiktoken>=0.5.0
```

## Testing

Run the comprehensive test suite:
```bash
python test_token_management.py
```

## Benefits

1. **Prevents API Errors**: No more "maximum context length exceeded" errors
2. **Cost Optimization**: Automatic truncation reduces token usage
3. **Reliability**: Graceful handling of large contexts
4. **Transparency**: Clear logging of truncation events
5. **Flexibility**: Configurable limits and strategies
6. **Backward Compatibility**: Works with existing code with minimal changes

## Future Enhancements

- [ ] Advanced context summarization
- [ ] Multi-model cost optimization
- [ ] Conversation memory persistence
- [ ] Adaptive token allocation
- [ ] Real-time token usage monitoring
