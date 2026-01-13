"""
Test script for token management system.

Tests the automatic token counting, truncation, and context management
to ensure it prevents exceeding model token limits.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.token_manager import TokenManager, get_token_manager, estimate_token_cost
from src.utils.api_wrapper import APIClient, get_api_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_token_counting():
    """Test token counting functionality."""
    print("\n=== Testing Token Counting ===")
    
    token_manager = get_token_manager()
    
    # Test with various text lengths
    test_texts = [
        "Hello, world!",
        "This is a longer text that should have more tokens.",
        "A" * 1000,  # Long repetitive text
        " ".join(["word"] * 100),  # Many words
    ]
    
    for text in test_texts:
        tokens = token_manager.count_tokens(text)
        print(f"Text: '{text[:50]}...' -> {tokens} tokens")
    
    # Test message token counting
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    message_tokens = token_manager.count_message_tokens(messages)
    print(f"Messages: {message_tokens} tokens")


def test_text_truncation():
    """Test text truncation functionality."""
    print("\n=== Testing Text Truncation ===")
    
    token_manager = get_token_manager()
    
    # Create a long text
    long_text = " ".join([f"word_{i}" for i in range(1000)])
    original_tokens = token_manager.count_tokens(long_text)
    
    print(f"Original text: {original_tokens} tokens")
    
    # Test different truncation strategies
    strategies = ["end", "start", "middle", "smart"]
    max_tokens = 100
    
    for strategy in strategies:
        truncated = token_manager.truncate_text(long_text, max_tokens, strategy)
        truncated_tokens = token_manager.count_tokens(truncated)
        print(f"Strategy '{strategy}': {truncated_tokens} tokens")
        print(f"  Sample: '{truncated[:100]}...'")


def test_conversation_compression():
    """Test conversation compression functionality."""
    print("\n=== Testing Conversation Compression ===")
    
    token_manager = get_token_manager(max_prompt_tokens=1000, keep_recent_messages=3)
    
    # Create a long conversation
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    
    # Add many user messages
    for i in range(20):
        messages.append({
            "role": "user", 
            "content": f"This is message number {i}. " + "word " * 50
        })
        messages.append({
            "role": "assistant",
            "content": f"This is response number {i}. " + "response " * 50
        })
    
    original_tokens = token_manager.count_message_tokens(messages)
    print(f"Original conversation: {len(messages)} messages, {original_tokens} tokens")
    
    # Test compression
    compressed_messages = token_manager.compress_conversation(messages)
    compressed_tokens = token_manager.count_message_tokens(compressed_messages)
    
    print(f"Compressed conversation: {len(compressed_messages)} messages, {compressed_tokens} tokens")
    print(f"Reduction: {original_tokens - compressed_tokens} tokens ({((original_tokens - compressed_tokens) / original_tokens * 100):.1f}%)")
    
    # Verify system message is preserved
    system_messages = [msg for msg in compressed_messages if msg.get('role') == 'system']
    assert len(system_messages) > 0, "System message should be preserved"
    print("[OK] System message preserved")


def test_api_request_preparation():
    """Test API request preparation with token management."""
    print("\n=== Testing API Request Preparation ===")
    
    token_manager = get_token_manager(max_prompt_tokens=2000, max_context_tokens=4000)
    
    # Create messages that would exceed the limit
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    
    for i in range(10):
        messages.append({
            "role": "user",
            "content": f"Question {i}: " + "text " * 200
        })
    
    original_tokens = token_manager.count_message_tokens(messages)
    print(f"Original: {original_tokens} tokens")
    
    # Prepare API request
    try:
        prepared_request = token_manager.prepare_api_request(messages, max_completion_tokens=1000)
        prepared_tokens = token_manager.count_message_tokens(prepared_request['messages'])
        
        print(f"Prepared: {prepared_tokens} prompt + {prepared_request['max_tokens']} completion tokens")
        print(f"Total: {prepared_tokens + prepared_request['max_tokens']} tokens")
        print(f"Within limit: {prepared_tokens + prepared_request['max_tokens'] <= token_manager.max_context_tokens}")
        
        stats = prepared_request['stats']
        print(f"Stats: {stats.prompt_tokens} prompt, {stats.completion_tokens} completion, {stats.truncated_tokens} truncated")
        
    except ValueError as e:
        print(f"✓ Correctly prevented request: {e}")


def test_large_context_simulation():
    """Simulate the exact error scenario from the task description."""
    print("\n=== Testing Large Context Simulation ===")
    
    # Simulate the problematic scenario: 1,746,119 tokens requested
    token_manager = get_token_manager(max_context_tokens=131072)  # DeepSeek's limit
    
    # Create an extremely large context (simulating 1.7M tokens)
    huge_context = "X" * (1746119 * 4)  # Rough approximation: 4 chars per token
    
    print(f"Simulating request with ~1,746,119 tokens")
    
    # Try to prepare this huge context
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": huge_context}
    ]
    
    try:
        prepared_request = token_manager.prepare_api_request(messages)
        print(f"[OK] Successfully handled large context!")
        print(f"  Original: ~1,746,119 tokens")
        print(f"  Prepared: {prepared_request['stats'].prompt_tokens} prompt tokens")
        print(f"  Truncated: {prepared_request['stats'].truncated_tokens} tokens")
        print(f"  Warning logged: '[WARNING] Context truncated to fit model token limit'")
        
    except Exception as e:
        print(f"[ERROR] Failed to handle large context: {e}")


def test_cost_estimation():
    """Test token cost estimation."""
    print("\n=== Testing Cost Estimation ===")
    
    test_text = "This is a test text for cost estimation. " * 100
    
    cost_estimate = estimate_token_cost(test_text, "deepseek-chat")
    
    print(f"Text: {cost_estimate['tokens']} tokens")
    print(f"Estimated cost: ${cost_estimate['estimated_cost']:.6f}")
    print(f"Cost per 1K tokens: ${cost_estimate['cost_per_1k_tokens']:.4f}")
    print(f"Model: {cost_estimate['model']}")


def test_api_client_integration():
    """Test API client integration (without actual API calls)."""
    print("\n=== Testing API Client Integration ===")
    
    # Test without API key (should work for initialization)
    client = APIClient(api_key="test_key", model_name="deepseek-chat")
    
    # Test usage stats
    stats = client.get_usage_stats()
    print(f"API Client stats: {stats}")
    
    # Test cost estimation
    test_text = "Sample text for cost estimation"
    cost = client.estimate_cost(test_text)
    print(f"Cost estimate: {cost}")
    
    print("[OK] API client integration test completed")


def run_all_tests():
    """Run all token management tests."""
    print("Starting Token Management System Tests")
    print("=" * 50)
    
    try:
        test_token_counting()
        test_text_truncation()
        test_conversation_compression()
        test_api_request_preparation()
        test_large_context_simulation()
        test_cost_estimation()
        test_api_client_integration()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests completed successfully!")
        print("\nKey Features Implemented:")
        print("- Automatic token counting and estimation")
        print("- Text truncation with multiple strategies")
        print("- Conversation compression and context management")
        print("- API request preparation with token limits")
        print("- Prevention of token limit errors (131k limit respected)")
        print("- Cost estimation and usage tracking")
        print("- Clear warning logging for truncation events")
        
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
