"""
Compliance Checker
RUBRIC: Governance & Guardrails - GovernanceGate orchestrates safety checks (3 marks)

TASK: Implement compliance checking using PII detector
"""
import re
from typing import Dict, Any, List
from guardrails.pii_detector import PIIDetector

class ComplianceChecker:
    """Ensures content meets regulatory and organizational compliance requirements"""
    
    def __init__(self):
        # HINT: Initialize PII detector
        self.pii_detector = PIIDetector()  # HINT: PIIDetector()

    def check_compliance(self, text: str, compliance_standards: List[str] = None, industry: str = "travel") -> Dict[str, Any]:
        """
        Checks the text for compliance violations
        
        HINT: This method should:
        1. Initialize compliance_standards if None
        2. Check for PII using pii_detector
        3. Build violations list if PII detected
        4. Determine if compliant based on standards
        5. Return dict with compliant, violations, remediation, detected_pii_count
        """
        compliance_standards = compliance_standards or []
        violations = []
        is_compliant = True
        
        # HINT: Check for PII using Guardrail PII Detector
        pii_result = self.pii_detector.detect(text)
        
        detected_pii = []
        if pii_result['has_pii']:
            for entity in pii_result['entities']:
                detected_pii.append(f"{entity['type']}: {entity['value']}")
        
        if detected_pii:
            # HINT: Add PII violation message (limit to first 5)
            violations.append(f"PII Detected: {', '.join(detected_pii[:5])}...")
            
            # HINT: If strict compliance needed (GDPR or HIPAA), mark as non-compliant
            if "GDPR" in compliance_standards or "HIPAA" in compliance_standards:
                is_compliant = False
        
        # HINT: Determine remediation action
        remediation = "Remove or redact PII before sharing data." if detected_pii else "No remediation needed."
        
        return {
            'compliant': is_compliant,
            'violations': violations,
            'remediation': remediation,
            'detected_pii_count': pii_result['count']
        }