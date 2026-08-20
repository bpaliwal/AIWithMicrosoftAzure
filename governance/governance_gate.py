"""
Governance Gate
RUBRIC: Governance & Guardrails - GovernanceGate orchestrates safety checks (3 marks)

TASK: Implement main governance orchestrator with audit logging
"""
from typing import Dict, Any
import datetime
from governance.safety_validator import SafetyValidator
from governance.compliance_checker import ComplianceChecker

class GovernanceGate:
    """The main orchestrator that coordinates all governance checks"""
    
    def __init__(self):
        # HINT: Initialize safety validator and compliance checker
        self.safety_validator = SafetyValidator()
        self.compliance_checker = ComplianceChecker()
        self.audit_log = []

    def validate_input(self, text: str) -> Dict[str, Any]:
        """
        Validates user input before processing
        
        HINT: This method should:
        1. Run safety validation
        2. Run compliance check with GDPR standard
        3. Combine results (passed if both safe and compliant)
        4. Log audit entry
        5. Return result dict
        """
        # HINT: 1. Safety Check
        safety_result = self.safety_validator.validate(text)
        
        # HINT: 2. Compliance Check (ensure no PII in query if strict)
        compliance_result = self.compliance_checker.check_compliance(
            text,
            compliance_standards=["GDPR"]
        )
        
        # HINT: Combine results - passed only if both checks pass
        passed = safety_result['is_safe'] and compliance_result['compliant']
        violations = safety_result.get('flags', []) + compliance_result.get('violations', [])
        
        result = {
            'passed': passed,
            'violations': violations,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # HINT: Log audit entry
        self._log_audit("validate_input", result)
        return result

    def validate_output(self, text: str) -> Dict[str, Any]:
        """
        Validates LLM output before returning to user
        
        HINT: Similar to validate_input - run both safety and compliance checks
        """
        # HINT: Similar checks for output
        safety_result = self.safety_validator.validate(text)
        compliance_result = self.compliance_checker.check_compliance(
            text,
            compliance_standards=["GDPR"]
        )
        
        passed = safety_result['is_safe'] and compliance_result['compliant']
        violations = safety_result.get('flags', []) + compliance_result.get('violations', [])
        
        result = {
            'passed': passed,
            'violations': violations,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self._log_audit("validate_output", result)
        return result

    def get_audit_log(self):
        """Return the audit log"""
        return self.audit_log

    def _log_audit(self, action: str, result: Dict[str, Any]):
        """
        Log audit entry
        
        HINT: Create entry dict with action, result status, details, timestamp
        """
        entry = {
            'action': action,
            'result': 'PASS' if result['passed'] else 'FAIL',
            'details': result,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.audit_log.append(entry)  # HINT: append