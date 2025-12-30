"""border_guard/rules/parameterized_rules_engine.py - Flexible rule evaluation"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RuleOperator(str, Enum):
    """Parameterized rule operators"""
    EQ = "eq"              # equals
    NEQ = "neq"            # not equals
    GT = "gt"              # greater than
    GTE = "gte"            # greater than or equal
    LT = "lt"              # less than
    LTE = "lte"            # less than or equal
    IN = "in"              # in list
    NOT_IN = "not_in"      # not in list
    CONTAINS = "contains"  # string contains
    NOT_CONTAINS = "not_contains"  # string not contains
    MATCHES = "matches"    # regex match
    NOT_MATCHES = "not_matches"  # regex not match
    IS_NULL = "is_null"    # is null/None
    NOT_NULL = "not_null"  # not null/None
    BETWEEN = "between"    # between values
    NOT_BETWEEN = "not_between"  # not between values


class AggregateOperator(str, Enum):
    """Aggregate operators for distributions"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STDDEV = "stddev"
    VARIANCE = "variance"
    PERCENTILE = "percentile"


@dataclass
class RuleCondition:
    """Individual rule condition"""
    field: str
    operator: RuleOperator
    value: Any
    severity: str = "warning"  # warning, critical
    message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class ParameterizedRule:
    """Flexible, parameterized rule"""
    name: str
    description: str
    enabled: bool = True
    rule_type: str = "custom"
    conditions: List[RuleCondition] = field(default_factory=list)
    logic: str = "AND"  # AND, OR, XOR, NAND
    timeout_seconds: int = 30
    enabled_for_datasets: List[str] = field(default_factory=list)  # Empty = all
    disabled_for_datasets: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_applicable_to_dataset(self, dataset_name: str) -> bool:
        """Check if rule applies to dataset"""
        if self.disabled_for_datasets and dataset_name in self.disabled_for_datasets:
            return False
        if self.enabled_for_datasets and dataset_name not in self.enabled_for_datasets:
            return False
        return self.enabled
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "rule_type": self.rule_type,
            "conditions": [c.to_dict() for c in self.conditions],
            "logic": self.logic,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class RuleEvaluator(ABC):
    """Base class for rule evaluation"""
    
    @abstractmethod
    def evaluate(self, data: Any, rule: ParameterizedRule) -> tuple[bool, List[str]]:
        """
        Evaluate rule against data
        Returns: (is_valid, violations)
        """
        pass


class FieldLevelEvaluator(RuleEvaluator):
    """Evaluate rules at field level"""
    
    def __init__(self):
        self.operators = {
            RuleOperator.EQ: lambda v, c: v == c,
            RuleOperator.NEQ: lambda v, c: v != c,
            RuleOperator.GT: lambda v, c: v > c,
            RuleOperator.GTE: lambda v, c: v >= c,
            RuleOperator.LT: lambda v, c: v < c,
            RuleOperator.LTE: lambda v, c: v <= c,
            RuleOperator.IN: lambda v, c: v in c,
            RuleOperator.NOT_IN: lambda v, c: v not in c,
            RuleOperator.CONTAINS: lambda v, c: c in str(v),
            RuleOperator.NOT_CONTAINS: lambda v, c: c not in str(v),
            RuleOperator.MATCHES: lambda v, c: re.match(c, str(v)) is not None,
            RuleOperator.NOT_MATCHES: lambda v, c: re.match(c, str(v)) is None,
            RuleOperator.IS_NULL: lambda v, c: v is None,
            RuleOperator.NOT_NULL: lambda v, c: v is not None,
            RuleOperator.BETWEEN: lambda v, c: c[0] <= v <= c[1],
            RuleOperator.NOT_BETWEEN: lambda v, c: not (c[0] <= v <= c[1]),
        }
    
    def evaluate(self, data: List[Dict], rule: ParameterizedRule) -> tuple[bool, List[str]]:
        """Evaluate field-level conditions"""
        violations = []
        results = []
        
        for condition in rule.conditions:
            condition_results = []
            
            for record in data:
                field_value = record.get(condition.field)
                
                try:
                    operator_func = self.operators[condition.operator]
                    passed = operator_func(field_value, condition.value)
                    condition_results.append(passed)
                    
                    if not passed:
                        msg = condition.message or \
                              f"Field '{condition.field}' {condition.operator.value} {condition.value}"
                        violations.append(msg)
                
                except Exception as e:
                    logger.error(f"Error evaluating {condition.field}: {e}")
                    violations.append(f"Error evaluating {condition.field}: {str(e)}")
            
            results.append(all(condition_results) if condition_results else False)
        
        # Apply logic (AND, OR, XOR, NAND)
        if rule.logic == "AND":
            is_valid = all(results) if results else False
        elif rule.logic == "OR":
            is_valid = any(results) if results else False
        elif rule.logic == "XOR":
            is_valid = sum(results) % 2 == 1 if results else False
        elif rule.logic == "NAND":
            is_valid = not (all(results) if results else True)
        else:
            is_valid = False
        
        return is_valid, violations


class DistributionEvaluator(RuleEvaluator):
    """Evaluate statistical distributions"""
    
    def __init__(self):
        pass
    
    def evaluate(self, data: List[Dict], rule: ParameterizedRule) -> tuple[bool, List[str]]:
        """Evaluate distribution-based rules"""
        violations = []
        
        for condition in rule.conditions:
            field_values = [r.get(condition.field) for r in data if r.get(condition.field) is not None]
            
            if not field_values:
                violations.append(f"No valid values for field '{condition.field}'")
                continue
            
            try:
                if condition.operator == RuleOperator.EQ:
                    # Check if average equals value
                    avg = sum(field_values) / len(field_values)
                    if abs(avg - condition.value) > 0.01:
                        violations.append(
                            f"Average of '{condition.field}' is {avg}, expected {condition.value}"
                        )
                
                elif condition.operator == RuleOperator.GT:
                    avg = sum(field_values) / len(field_values)
                    if avg <= condition.value:
                        violations.append(
                            f"Average of '{condition.field}' ({avg}) not > {condition.value}"
                        )
                
                elif condition.operator == RuleOperator.BETWEEN:
                    avg = sum(field_values) / len(field_values)
                    if not (condition.value[0] <= avg <= condition.value[1]):
                        violations.append(
                            f"Average of '{condition.field}' ({avg}) not between {condition.value}"
                        )
            
            except Exception as e:
                logger.error(f"Distribution evaluation error: {e}")
                violations.append(f"Error in distribution evaluation: {str(e)}")
        
        is_valid = len(violations) == 0
        return is_valid, violations


class CustomRuleEvaluator(RuleEvaluator):
    """Evaluate custom Python functions"""
    
    def __init__(self, custom_functions: Optional[Dict[str, Callable]] = None):
        self.custom_functions = custom_functions or {}
    
    def register_function(self, name: str, func: Callable):
        """Register custom evaluation function"""
        self.custom_functions[name] = func
    
    def evaluate(self, data: List[Dict], rule: ParameterizedRule) -> tuple[bool, List[str]]:
        """Evaluate using custom function"""
        violations = []
        
        func_name = rule.metadata.get("function_name")
        if not func_name or func_name not in self.custom_functions:
            return False, [f"Custom function '{func_name}' not found"]
        
        try:
            func = self.custom_functions[func_name]
            is_valid, msgs = func(data, rule)
            violations.extend(msgs or [])
            return is_valid, violations
        except Exception as e:
            logger.error(f"Custom rule evaluation error: {e}")
            return False, [f"Error in custom rule: {str(e)}"]


@dataclass
class ParameterizedRulesEngine:
    """Main rules engine with parametrization"""
    
    rules: List[ParameterizedRule] = field(default_factory=list)
    field_evaluator: FieldLevelEvaluator = field(default_factory=FieldLevelEvaluator)
    distribution_evaluator: DistributionEvaluator = field(default_factory=DistributionEvaluator)
    custom_evaluator: CustomRuleEvaluator = field(default_factory=CustomRuleEvaluator)
    
    def add_rule(self, rule: ParameterizedRule):
        """Add rule to engine"""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove rule by name"""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def get_rule(self, rule_name: str) -> Optional[ParameterizedRule]:
        """Get rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def update_rule(self, rule_name: str, updates: Dict):
        """Update rule parameters"""
        rule = self.get_rule(rule_name)
        if rule:
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            rule.updated_at = datetime.utcnow()
            logger.info(f"Updated rule: {rule_name}")
    
    def evaluate_dataset(self, data: List[Dict], dataset_name: str, 
                        tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate all applicable rules for dataset
        
        Returns:
            {
                "dataset": dataset_name,
                "valid": bool,
                "violations": [],
                "applied_rules": [],
                "skipped_rules": [],
            }
        """
        result = {
            "dataset": dataset_name,
            "valid": True,
            "violations": [],
            "applied_rules": [],
            "skipped_rules": [],
        }
        
        for rule in self.rules:
            # Check applicability
            if not rule.is_applicable_to_dataset(dataset_name):
                result["skipped_rules"].append(rule.name)
                continue
            
            # Check tags
            if tags and not any(tag in rule.tags for tag in tags):
                result["skipped_rules"].append(rule.name)
                continue
            
            # Evaluate
            try:
                if rule.rule_type == "field":
                    is_valid, violations = self.field_evaluator.evaluate(data, rule)
                elif rule.rule_type == "distribution":
                    is_valid, violations = self.distribution_evaluator.evaluate(data, rule)
                elif rule.rule_type == "custom":
                    is_valid, violations = self.custom_evaluator.evaluate(data, rule)
                else:
                    is_valid, violations = self.field_evaluator.evaluate(data, rule)
                
                result["applied_rules"].append(rule.name)
                
                if not is_valid:
                    result["valid"] = False
                    result["violations"].extend([
                        {
                            "rule": rule.name,
                            "severity": "critical",
                            "message": msg,
                        }
                        for msg in violations
                    ])
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                result["valid"] = False
                result["violations"].append({
                    "rule": rule.name,
                    "severity": "critical",
                    "message": f"Error: {str(e)}",
                })
        
        return result
    
    def get_rules_by_tag(self, tag: str) -> List[ParameterizedRule]:
        """Get all rules with specific tag"""
        return [r for r in self.rules if tag in r.tags]
    
    def enable_rule(self, rule_name: str):
        """Enable specific rule"""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
    
    def disable_rule(self, rule_name: str):
        """Disable specific rule"""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
