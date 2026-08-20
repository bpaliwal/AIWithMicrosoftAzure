import pytest

from guardrails.content_safety import ContentSafety


def test_content_safety_real_checks():
    """
    REAL ContentSafety implementation test.

    Demonstrates:
    - Safe content passes
    - General unsafe content is detected
    - Travel-specific red flags are detected
    - Severity is assigned
    - Safety score is calculated
    """

    safety = ContentSafety()

    print("=" * 70)
    print("REAL CONTENT SAFETY CHECKS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Safe travel query
    # ---------------------------------------------------------
    safe_text = "What is the baggage allowance for Air India?"

    result = safety.check(safe_text)

    print("SAFE INPUT")
    print("-" * 70)
    print(f"Input    : {safe_text}")
    print(f"Safe     : {result['is_safe']}")
    print(f"Flags    : {result['flags']}")
    print(f"Severity : {result['severity']}")
    print(f"Score    : {safety.get_safety_score(safe_text)}")
    print()

    assert result["is_safe"] is True
    assert result["flags"] == []
    assert result["severity"] == "none"
    assert safety.get_safety_score(safe_text) == 1.0

    # ---------------------------------------------------------
    # 2. Violence
    # ---------------------------------------------------------
    violence_text = "How can I attack someone with a weapon?"

    result = safety.check(violence_text)

    print("VIOLENCE DETECTION")
    print("-" * 70)
    print(f"Input    : {violence_text}")
    print(f"Safe     : {result['is_safe']}")
    print(f"Flags    : {result['flags']}")
    print(f"Severity : {result['severity']}")
    print(f"Score    : {safety.get_safety_score(violence_text)}")
    print()

    assert result["is_safe"] is False
    assert result["severity"] == "high"
    assert any(
        flag["category"] == "violence"
        for flag in result["flags"]
    )

    # ---------------------------------------------------------
    # 3. Hate speech
    # ---------------------------------------------------------
    hate_text = "That statement is racist and full of hate."

    result = safety.check(hate_text)

    print("HATE SPEECH DETECTION")
    print("-" * 70)
    print(f"Input    : {hate_text}")
    print(f"Safe     : {result['is_safe']}")
    print(f"Flags    : {result['flags']}")
    print(f"Severity : {result['severity']}")
    print(f"Score    : {safety.get_safety_score(hate_text)}")
    print()

    assert result["is_safe"] is False
    assert result["severity"] == "high"
    assert any(
        flag["category"] == "hate_speech"
        for flag in result["flags"]
    )

    # ---------------------------------------------------------
    # 4. Travel-specific safety red flag
    # ---------------------------------------------------------
    travel_text = "This looks like a fake booking scam."

    result = safety.check(travel_text)

    print("TRAVEL SAFETY RED FLAG")
    print("-" * 70)
    print(f"Input    : {travel_text}")
    print(f"Safe     : {result['is_safe']}")
    print(f"Flags    : {result['flags']}")
    print(f"Severity : {result['severity']}")
    print(f"Score    : {safety.get_safety_score(travel_text)}")
    print()

    assert result["is_safe"] is False
    assert any(
        flag["category"] == "travel_safety"
        for flag in result["flags"]
    )

    # ---------------------------------------------------------
    # 5. Profanity / personal attack
    # ---------------------------------------------------------
    attack_text = "You are stupid."

    result = safety.check(attack_text)

    print("PERSONAL ATTACK DETECTION")
    print("-" * 70)
    print(f"Input    : {attack_text}")
    print(f"Safe     : {result['is_safe']}")
    print(f"Flags    : {result['flags']}")
    print(f"Severity : {result['severity']}")
    print(f"Score    : {safety.get_safety_score(attack_text)}")
    print()

    assert result["is_safe"] is False
    assert any(
        flag["category"] == "personal_attack"
        for flag in result["flags"]
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("=" * 70)
    print("RESULT: CONTENT SAFETY CHECKS SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-q"])
