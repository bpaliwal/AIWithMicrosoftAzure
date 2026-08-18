"""
Safety Validator

RUBRIC: Governance & Guardrails
- Safety validator with Azure Content Safety (3 marks)

TASK:
Implement safety validation using:
1. Local content-safety checks
2. Prompt-injection detection
3. Azure AI Content Safety
"""

import re
from typing import Dict, Any

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from src.config import Config
from guardrails.content_safety import ContentSafety


class SafetyValidator:
    """
    Validates content safety and detects adversarial attacks.

    Safety checks:
        1. Local keyword/content-safety guardrail
        2. Prompt injection detection
        3. Azure AI Content Safety

    The validator is used for both:
        - User input
        - LLM output
    """

    # Azure AI Content Safety severity levels:
    #
    # 0 = Safe
    # 2 = Low
    # 4 = Medium
    # 6 = High
    #
    # These values are defined by Azure Content Safety.
    SEVERITY_LEVELS = {
        "low": 2,
        "medium": 4,
        "high": 6,
    }

    def __init__(self):

        # ========================================================
        # Local Content Safety
        # ========================================================

        self.content_safety = ContentSafety()

        # ========================================================
        # Prompt Injection Detection
        # ========================================================

        self.injection_patterns = [
            r"ignore previous instructions",
            r"ignore all previous instructions",
            r"ignore prior instructions",
            r"disregard previous instructions",
            r"disregard all previous instructions",
            r"bypass safety",
            r"bypass security",
            r"override guidelines",
            r"override safety",
            r"override system instructions",
            r"you are now in developer mode",
            r"you are now in admin mode",
            r"developer mode",
            r"delete all data",
            r"reveal system prompt",
            r"show system prompt",
            r"print system prompt",
            r"what is your system prompt",
            r"system prompt",
        ]

        # ========================================================
        # Azure AI Content Safety
        # ========================================================

        self.client = None

        endpoint = (
            Config.AZURE_CONTENT_SAFETY_ENDPOINT
        )

        key = (
            Config.AZURE_CONTENT_SAFETY_KEY
        )

        if endpoint and key:

            try:

                self.client = ContentSafetyClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(key),
                )

                print(
                    "DEBUG: Azure Content Safety "
                    "client initialized successfully."
                )

            except Exception as e:

                print(
                    "WARNING: Failed to initialize "
                    f"Azure Content Safety: {e}"
                )

                self.client = None

        else:

            print(
                "WARNING: Azure Content Safety "
                "credentials are not configured."
            )

    # ============================================================
    # Main Validation
    # ============================================================

    def validate(
        self,
        text: str,
        severity_threshold: str = "high",
    ) -> Dict[str, Any]:
        """
        Validate text using local checks and Azure Content Safety.

        Parameters
        ----------
        text:
            Text to validate.

        severity_threshold:
            Azure severity level at which content is blocked.

            Supported values:
                low
                medium
                high

            Default:
                high

        Returns
        -------
        Dict containing:

            is_safe
            flags
            severity
            local_check
            prompt_injection_check
            azure_content_safety_check
        """

        # ========================================================
        # Validate Input Parameters
        # ========================================================

        if not isinstance(text, str):
            return {
                "is_safe": False,
                "flags": [
                    "Invalid input: text must be a string."
                ],
                "severity": "high",
                "local_check": "FAIL",
                "prompt_injection_check": "FAIL",
                "azure_content_safety_check": "NOT_RUN",
            }

        if not text.strip():
            return {
                "is_safe": True,
                "flags": [],
                "severity": "low",
                "local_check": "PASS",
                "prompt_injection_check": "PASS",
                "azure_content_safety_check": "NOT_RUN",
            }

        severity_threshold = severity_threshold.lower()

        if severity_threshold not in self.SEVERITY_LEVELS:

            raise ValueError(
                "severity_threshold must be one of: "
                "low, medium, high"
            )

        azure_threshold = (
            self.SEVERITY_LEVELS[
                severity_threshold
            ]
        )

        flags = []
        is_safe = True

        local_check = "PASS"
        injection_check = "PASS"
        azure_check = "NOT_RUN"

        # ========================================================
        # 1. Local Content Safety Check
        # ========================================================

        try:

            local_result = self.content_safety.check(
                text
            )

            if not local_result.get(
                "is_safe",
                True,
            ):

                is_safe = False
                local_check = "FAIL"

                for flag in local_result.get(
                    "flags",
                    [],
                ):

                    category = flag.get(
                        "category",
                        "Unknown",
                    )

                    keyword = flag.get(
                        "keyword",
                        "Unknown",
                    )

                    flags.append(
                        "Unsafe Keyword "
                        f"({category}): {keyword}"
                    )

        except Exception as e:

            # A local safety-check failure should not silently
            # allow content through.
            is_safe = False
            local_check = "ERROR"

            flags.append(
                "Local Content Safety check failed: "
                f"{type(e).__name__}"
            )

            print(
                "WARNING: Local Content Safety "
                f"check failed: {e}"
            )

        # ========================================================
        # 2. Prompt Injection Detection
        # ========================================================

        for pattern in self.injection_patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                is_safe = False
                injection_check = "FAIL"

                flags.append(
                    "Prompt Injection Detected: "
                    f"{pattern}"
                )

        # ========================================================
        # 3. Azure AI Content Safety
        # ========================================================

        if self.client:

            try:

                request = AnalyzeTextOptions(
                    text=text
                )

                response = self.client.analyze_text(
                    request
                )

                azure_check = "PASS"

                categories_analysis = (
                    response.categories_analysis
                    or []
                )

                for analysis in categories_analysis:

                    category = getattr(
                        analysis,
                        "category",
                        "Unknown",
                    )

                    severity = getattr(
                        analysis,
                        "severity",
                        0,
                    )

                    # Azure uses:
                    # 0 = Safe
                    # 2 = Low
                    # 4 = Medium
                    # 6 = High
                    #
                    # Block when the configured threshold
                    # is reached or exceeded.

                    if severity >= azure_threshold:

                        is_safe = False
                        azure_check = "FAIL"

                        flags.append(
                            "Azure Content Safety "
                            f"Violation: {category} "
                            f"(severity={severity})"
                        )

            except HttpResponseError as e:

                # Azure is configured, therefore an Azure
                # validation failure should not silently result
                # in content being treated as safe.
                is_safe = False
                azure_check = "ERROR"

                flags.append(
                    "Azure Content Safety "
                    "validation failed."
                )

                print(
                    "WARNING: Azure Content Safety "
                    f"request failed: {e}"
                )

            except Exception as e:

                is_safe = False
                azure_check = "ERROR"

                flags.append(
                    "Azure Content Safety "
                    "validation failed."
                )

                print(
                    "WARNING: Azure Content Safety "
                    f"unexpected error: {e}"
                )

        else:

            # Azure credentials are not available.
            #
            # We do not claim that Azure validation passed.
            # Local checks still operate, but the result records
            # that Azure was unavailable.

            azure_check = "NOT_CONFIGURED"

        # ========================================================
        # Determine Overall Severity
        # ========================================================

        if not is_safe:

            severity = "high"

        elif azure_check == "NOT_CONFIGURED":

            severity = "low"

        else:

            severity = "low"

        # ========================================================
        # Final Result
        # ========================================================

        return {
            "is_safe": is_safe,
            "flags": flags,
            "severity": severity,
            "local_check": local_check,
            "prompt_injection_check": injection_check,
            "azure_content_safety_check": azure_check,
        }
