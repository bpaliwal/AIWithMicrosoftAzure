import pytest

from governance.safety_validator import SafetyValidator


@pytest.mark.integration
def test_safety_validator_real_azure_content_safety():
    """
    REAL Azure AI Content Safety integration test.

    Demonstrates:
    - SafetyValidator initializes Azure Content Safety
    - Real Azure Content Safety validation is performed
    - Azure result is captured in azure_content_safety_check
    - Safe content passes the complete validator
    """

    text = "What is the baggage allowance for Air India?"

    print("=" * 70)
    print("REAL AZURE CONTENT SAFETY - SAFETY VALIDATOR")
    print("=" * 70)
    print(f"Input: {text}")
    print()

    validator = SafetyValidator()

    assert validator.client is not None, (
        "Azure Content Safety client was not initialized. "
        "Check AZURE_CONTENT_SAFETY_ENDPOINT and "
        "AZURE_CONTENT_SAFETY_KEY."
    )

    print("AZURE CLIENT")
    print("-" * 70)
    print("Azure Content Safety client : INITIALIZED")
    print()

    result = validator.validate(text)

    print("SAFETY VALIDATION RESULT")
    print("-" * 70)
    print(f"Safe                    : {result['is_safe']}")
    print(f"Local check             : {result['local_check']}")
    print(
        f"Prompt injection check : "
        f"{result['prompt_injection_check']}"
    )
    print(
        f"Azure Content Safety   : "
        f"{result['azure_content_safety_check']}"
    )
    print(f"Severity                : {result['severity']}")
    print(f"Flags                   : {result['flags']}")
    print()

    assert result["azure_content_safety_check"] == "PASS"
    assert result["local_check"] == "PASS"
    assert result["prompt_injection_check"] == "PASS"
    assert result["is_safe"] is True

    print("=" * 70)
    print("RESULT: REAL AZURE CONTENT SAFETY INTEGRATION SUCCESSFUL")
    print("=" * 70)
