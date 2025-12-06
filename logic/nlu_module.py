#!/usr/bin/env python3
"""
NLU Module for AI Analyst Integration
======================================

This module provides a production-ready NLU component that can be integrated
into the AI Analyst application. It uses a trained BERT model for fast,
reliable intent classification and slot extraction.

Architecture:
-------------
    User Input
         │
         ▼
    ┌─────────────┐
    │ BERT NLU    │ ◄── Fast inference (~50ms)
    │ (Local)     │
    └─────────────┘
         │
         ├── High confidence (>0.8) ──► Direct action
         │
         └── Low confidence ──► LLM fallback

Benefits over LLM-only approach:
- 10-100x faster inference
- Works offline
- Consistent, deterministic outputs
- Lower resource usage
- Multi-turn dialogue support

Usage:
------
    from integration.nlu_module import NLUModule
    
    nlu = NLUModule()
    result = nlu.understand("show me a histogram of age")
    # Returns: {'intent': 'plot', 'slots': {'plot_type': 'histogram', 'feature': 'age'}, ...}

Author: AI Analyst Project
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re

# Check for torch availability
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. NLU module will use rule-based fallback.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Intent(Enum):
    """Supported intents in AI Analyst."""
    CHAT = "chat"
    PLOT = "plot"
    ML_MODEL = "ml_model"


@dataclass
class NLUConfig:
    """Configuration for NLU module."""
    model_name: str = "prajjwal1/bert-tiny"
    model_path: Optional[str] = None  # Path to trained model weights
    confidence_threshold: float = 0.7
    max_context_turns: int = 3
    max_length: int = 128
    device: str = "cpu"


@dataclass
class DialogueState:
    """Tracks the current dialogue state for multi-turn conversations."""
    intent: Optional[str] = None
    slots: Dict[str, str] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    is_complete: bool = False
    context: List[Dict[str, str]] = field(default_factory=list)
    
    def add_turn(self, speaker: str, text: str):
        """Add a turn to the dialogue context."""
        self.context.append({"speaker": speaker, "text": text})
        # Keep only last N turns
        if len(self.context) > 6:
            self.context = self.context[-6:]
    
    def get_context_string(self) -> str:
        """Get context as a string for the model."""
        return " [SEP] ".join([t["text"] for t in self.context[-3:]])
    
    def reset(self):
        """Reset the dialogue state."""
        self.intent = None
        self.slots = {}
        self.missing_slots = []
        self.is_complete = False
        self.context = []


@dataclass  
class NLUResult:
    """Result from NLU processing."""
    intent: str
    confidence: float
    slots: Dict[str, str]
    is_complete: bool
    missing_slots: List[str]
    clarification_question: Optional[str]
    raw_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.intent,
            "confidence": self.confidence,
            "slots": self.slots,
            "plot_type": self.slots.get("plot_type"),
            "features": [self.slots.get("feature")] if self.slots.get("feature") else [],
            "model_type": self.slots.get("model_type"),
            "target": self.slots.get("target"),
            "is_complete": self.is_complete,
            "missing_slots": self.missing_slots,
            "clarification_question": self.clarification_question,
            "explanation": f"Intent: {self.intent} (conf: {self.confidence:.2f})"
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# LABEL DEFINITIONS (must match training)
# =============================================================================

INTENT_LABELS = ['chat', 'plot', 'ml_model']
INTENT2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2INTENT = {i: label for label, i in INTENT2ID.items()}

SLOT_LABELS = [
    'O', 
    'B-plot_type', 'I-plot_type', 
    'B-model_type', 'I-model_type',
    'B-feature', 'I-feature', 
    'B-target', 'I-target',
    'B-aggregation', 'I-aggregation'
]
SLOT2ID = {label: i for i, label in enumerate(SLOT_LABELS)}
ID2SLOT = {i: label for label, i in SLOT2ID.items()}

# Required slots for each intent
# Note: scatter plots need 2 features, which is handled dynamically
REQUIRED_SLOTS = {
    "plot": ["plot_type"],  # features checked dynamically based on plot_type
    "ml_model": ["model_type", "target"],
    "chat": []
}

# Slots that require 2 values (e.g., scatter needs x and y features)
MULTI_FEATURE_PLOTS = ["scatter", "scatter plot", "correlation"]

# Clarification questions
CLARIFICATION_QUESTIONS = {
    "plot_type": "What type of chart would you like? (histogram, scatter, timeseries, frequency_domain)",
    "feature": "Which column/feature would you like to visualize?",
    "model_type": "What type of model would you like to train? (logistic_regression, decision_tree, random_forest)",
    "target": "Which column should be the target variable for prediction?",
}


# =============================================================================
# BERT NLU MODEL
# =============================================================================

if TORCH_AVAILABLE:
    class JointNLUModel(nn.Module):
        """
        Joint BERT model for intent classification and slot filling.
        Matches the architecture from train_multiturn.py and comparative_analysis.py
        """
        
        def __init__(self, model_name: str, num_intents: int, num_slots: int, dropout: float = 0.1):
            super().__init__()
            self.model_name = model_name
            self.config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            hidden_size = self.config.hidden_size
            
            self.intent_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_intents)
            )
            
            self.slot_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_slots)
            )
            
            # State classifier for dialogue management (INCOMPLETE vs COMPLETE)
            self.state_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )
        
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
            
            intent_logits = self.intent_classifier(pooled_output)
            slot_logits = self.slot_classifier(sequence_output)
            state_logits = self.state_classifier(pooled_output)
            
            return {
                'intent_logits': intent_logits,
                'slot_logits': slot_logits,
                'state_logits': state_logits
            }


# =============================================================================
# RULE-BASED FALLBACK
# =============================================================================

class RuleBasedNLU:
    """
    Rule-based NLU as fallback when BERT model is not available.
    Uses pattern matching for intent and slot extraction.
    
    Supports dynamic feature extraction from dataset columns.
    """
    
    PLOT_PATTERNS = [
        r'\b(show|display|create|make|generate|plot|draw|visualize)\b.*\b(chart|plot|graph|diagram|histogram|scatter|timeseries|visualization)\b',
        r'\b(histogram|scatter|timeseries|bar chart|line chart|frequency|fft|spectrum)\b',
        r'\b(need|want|give me)\b.*\b(scatter|plot|chart|graph)\b',
        r'\b(time\s*frequency|timefrequen|freq\s*domain|frequency_domain)\b',  # Typo-tolerant patterns
        r'\bfreq\w*\s+(for|of|with)\b',  # "freqXXX for a1" pattern
        r'\b(histogram|scatter|fft|spectrum)\s+(for|of|with)\b',  # "plottype for feature" pattern
    ]
    
    ML_PATTERNS = [
        r'\b(train|build|create|fit)\b.*\b(model|classifier|regressor)\b',
        r'\b(logistic regression|decision tree|random forest|neural network|svm)\b',
        r'\b(predict|classify|regression)\b',
    ]
    
    PLOT_TYPES = {
        'histogram': ['histogram', 'hist', 'distribution', 'frequency distribution', 'bar chart', 'barchart'],
        'scatter': ['scatter', 'scatter plot', 'scatterplot', 'correlation', 'xy plot', 'xyplot'],
        'timeseries': ['timeseries', 'time series', 'time plot', 'trend', 'over time', 'temporal', 'time-series'],
        'frequency_domain': [
            'frequency domain', 'frequency_domain', 'fft', 'spectrum', 'spectral', 
            'fourier', 'freq domain', 'freqdomain', 'frequency analysis',
            # Common typos and variations
            'timefrequency', 'time frequency', 'time-frequency', 'timefreq',
            'frequenct', 'frequncy', 'frequancy', 'freqency',  # typos
            'timefrequenct', 'timefrequncy',  # User's typo
        ],
    }
    
    MODEL_TYPES = {
        'logistic_regression': ['logistic regression', 'logistic', 'log reg', 'logreg'],
        'decision_tree': ['decision tree', 'tree', 'dt', 'decisiontree'],
        'random_forest': ['random forest', 'rf', 'forest', 'randomforest'],
    }
    
    # Pattern to extract potential column names (alphanumeric with optional underscores)
    COLUMN_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    
    @staticmethod
    def _fuzzy_match(text: str, target: str, threshold: float = 0.7) -> bool:
        """
        Simple fuzzy matching to handle typos.
        Returns True if text contains something similar to target.
        """
        text = text.lower()
        target = target.lower()
        
        # Exact match
        if target in text:
            return True
        
        # Check each word in text
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            # Skip very short words
            if len(word) < 4 or len(target) < 4:
                continue
            
            # Check if word starts similarly (prefix match)
            if len(word) >= 4 and len(target) >= 4:
                if word[:4] == target[:4]:
                    return True
            
            # Simple edit distance check (Levenshtein-like)
            if abs(len(word) - len(target)) <= 2:
                matches = sum(c1 == c2 for c1, c2 in zip(word, target))
                similarity = matches / max(len(word), len(target))
                if similarity >= threshold:
                    return True
        
        return False
    
    def understand(self, text: str, available_features: List[str] = None) -> NLUResult:
        """Extract intent and slots using rules.
        
        Args:
            text: User input text
            available_features: List of available column names from dataset
        
        Returns:
            NLUResult with extracted intent and slots
        """
        text_lower = text.lower()
        available_features = available_features or []
        available_features_lower = [f.lower() for f in available_features]
        
        # Detect intent
        intent = "chat"
        confidence = 0.5
        
        for pattern in self.PLOT_PATTERNS:
            if re.search(pattern, text_lower):
                intent = "plot"
                confidence = 0.85
                break
        
        # Fallback: check if text contains something that looks like a plot type (with typos)
        if intent == "chat":
            plot_type_hints = [
                ('frequency_domain', ['freq', 'fft', 'spectrum', 'fourier', 'spectral']),
                ('timeseries', ['time', 'series', 'temporal', 'trend']),
                ('histogram', ['hist', 'distribution']),
                ('scatter', ['scatter', 'correlation']),
            ]
            for ptype, hints in plot_type_hints:
                for hint in hints:
                    if hint in text_lower and len(text_lower) < 100:  # Short query with plot hint
                        # Check if combined with features
                        if available_features:
                            for feat in available_features:
                                if feat.lower() in text_lower:
                                    intent = "plot"
                                    confidence = 0.75
                                    break
                        if intent == "plot":
                            break
                if intent == "plot":
                    break
        
        if intent == "chat":
            for pattern in self.ML_PATTERNS:
                if re.search(pattern, text_lower):
                    intent = "ml_model"
                    confidence = 0.85
                    break
        
        # Extract slots
        slots = {}
        
        if intent == "plot":
            # Find plot type - with fuzzy matching for typos
            for ptype, keywords in self.PLOT_TYPES.items():
                for kw in keywords:
                    if kw in text_lower:
                        slots["plot_type"] = ptype
                        break
                if "plot_type" in slots:
                    break
            
            # If no exact match, try fuzzy matching
            if "plot_type" not in slots:
                fuzzy_targets = [
                    ('frequency_domain', ['frequency', 'freqdomain', 'timefrequency']),
                    ('timeseries', ['timeseries', 'temporal']),
                    ('histogram', ['histogram', 'distribution']),
                    ('scatter', ['scatter', 'correlation']),
                ]
                for ptype, targets in fuzzy_targets:
                    for target in targets:
                        if self._fuzzy_match(text_lower, target):
                            slots["plot_type"] = ptype
                            break
                    if "plot_type" in slots:
                        break
            
            # Find features - supports multiple features for scatter plots
            found_features = self._extract_features(text, available_features)
            if found_features:
                slots["features"] = found_features
                # Keep "feature" for backward compatibility (first feature)
                slots["feature"] = found_features[0]
        
        elif intent == "ml_model":
            # Find model type
            for mtype, keywords in self.MODEL_TYPES.items():
                for kw in keywords:
                    if kw in text_lower:
                        slots["model_type"] = mtype
                        break
                if "model_type" in slots:
                    break
            
            # Find target column - multiple strategies
            target = self._extract_target(text, available_features)
            if target:
                slots["target"] = target
        
        # Check completeness - special handling for scatter plots needing 2 features
        required = REQUIRED_SLOTS.get(intent, []).copy()
        missing = [s for s in required if s not in slots]
        
        # Special check: scatter plots need exactly 2 features
        if intent == "plot" and slots.get("plot_type") == "scatter":
            features = slots.get("features", [])
            if len(features) < 2:
                if "features" not in missing:
                    missing.append("features")
        
        is_complete = len(missing) == 0
        
        clarification = None
        if not is_complete and missing:
            if "features" in missing and slots.get("plot_type") == "scatter":
                clarification = "I need 2 features for a scatter plot. Which columns would you like to compare? (e.g., 'a1 and a2')"
            else:
                clarification = CLARIFICATION_QUESTIONS.get(missing[0])
        
        return NLUResult(
            intent=intent,
            confidence=confidence,
            slots=slots,
            is_complete=is_complete,
            missing_slots=missing,
            clarification_question=clarification,
            raw_text=text
        )
    
    def _extract_features(self, text: str, available_features: List[str]) -> List[str]:
        """
        Extract feature/column names from text.
        
        Uses two strategies:
        1. Match against known column names (if available_features provided)
        2. Extract potential column names using patterns (for dynamic detection)
        
        Args:
            text: User input text
            available_features: List of known column names from dataset
        
        Returns:
            List of extracted feature names
        """
        text_lower = text.lower()
        found_features = []
        
        # Strategy 1: Match against known features (case-insensitive)
        if available_features:
            for feature in available_features:
                # Check if feature name appears in text
                if feature.lower() in text_lower:
                    if feature not in found_features:
                        found_features.append(feature)
        
        # Strategy 2: If we have available_features, check for abbreviated references
        # e.g., "a1 and a2" when columns are "a1", "a2", "a3"
        if available_features:
            words = re.findall(r'\b\w+\b', text_lower)
            for word in words:
                for feature in available_features:
                    if word == feature.lower():
                        if feature not in found_features:
                            found_features.append(feature)
        
        # Strategy 3: Extract potential column names even without known features
        # This helps when dataset hasn't been loaded yet
        if not found_features:
            # Look for patterns like "a1 and a2", "column1, column2"
            # Match alphanumeric identifiers that look like column names
            potential_cols = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', text)
            
            # Filter out common words that aren't column names
            stopwords = {
                'a', 'an', 'the', 'and', 'or', 'for', 'to', 'of', 'in', 'on',
                'show', 'me', 'i', 'want', 'need', 'create', 'make', 'plot',
                'scatter', 'histogram', 'chart', 'graph', 'diagram', 'with',
                'between', 'using', 'vs', 'versus', 'compare'
            }
            potential_cols = [c for c in potential_cols if c.lower() not in stopwords]
            
            # If we found exactly 2 or more potential columns, use them
            if len(potential_cols) >= 2:
                found_features = potential_cols
        
        return found_features
    
    def _extract_target(self, text: str, available_features: List[str]) -> Optional[str]:
        """
        Extract target column name from text for ML model training.
        
        Uses multiple strategies:
        1. Pattern matching: "predict X", "target X", "for X"
        2. Direct column name matching against available features
        3. Handle quoted column names: "label", 'target_col'
        
        Args:
            text: User input text
            available_features: List of known column names from dataset
        
        Returns:
            Target column name if found, None otherwise
        """
        text_lower = text.lower().strip()
        
        # Strategy 1: Pattern-based extraction
        target_patterns = [
            r'predict\s+["\']?(\w+)["\']?',
            r'target\s+["\']?(\w+)["\']?',
            r'classify\s+["\']?(\w+)["\']?',
            r'for\s+["\']?(\w+)["\']?\s*(column)?',
            r'(\w+)\s+column',
            r'["\'](\w+)["\']',  # Quoted text like "label" or 'label'
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, text_lower)
            if match:
                candidate = match.group(1)
                # Check if candidate is an available feature
                if available_features:
                    for feature in available_features:
                        if feature.lower() == candidate:
                            return feature
                # If no available features, check if it looks like a column name
                if not available_features and candidate not in {'column', 'the', 'a', 'an'}:
                    return candidate
        
        # Strategy 2: Direct match against available features
        if available_features:
            for feature in available_features:
                if feature.lower() in text_lower:
                    return feature
        
        # Strategy 3: For very short inputs (likely follow-up answers),
        # check if the whole input is a column name
        words = text_lower.split()
        if len(words) <= 3 and available_features:
            for word in words:
                # Remove quotes if present
                word_clean = word.strip('\'"')
                for feature in available_features:
                    if feature.lower() == word_clean:
                        return feature
        
        return None


# =============================================================================
# MAIN NLU MODULE
# =============================================================================

class NLUModule:
    """
    Main NLU module for AI Analyst integration.
    
    Provides:
    - Fast BERT-based intent classification and slot filling
    - Multi-turn dialogue support with state tracking
    - Clarification question generation for incomplete requests
    - Rule-based fallback when BERT is unavailable
    """
    
    def __init__(self, config: NLUConfig = None):
        self.config = config or NLUConfig()
        self.dialogue_states: Dict[str, DialogueState] = {}
        self.model = None
        self.tokenizer = None
        self.rule_based = RuleBasedNLU()
        self.available_features: Dict[str, List[str]] = {}  # room -> features
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the trained BERT model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = JointNLUModel(
                model_name=self.config.model_name,
                num_intents=len(INTENT_LABELS),
                num_slots=len(SLOT_LABELS)
            )
            
            # Load trained weights if available
            if self.config.model_path and os.path.exists(self.config.model_path):
                state_dict = torch.load(
                    self.config.model_path, 
                    map_location=self.config.device
                )
                self.model.load_state_dict(state_dict)
                print(f"Loaded NLU model from {self.config.model_path}")
            
            self.model.to(self.config.device)
            self.model.eval()
            print(f"NLU model initialized: {self.config.model_name}")
            
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            print("Using rule-based fallback.")
            self.model = None
    
    def set_available_features(self, room_name: str, features: List[str]):
        """
        Set available features/columns for a room (from loaded dataset).
        
        This enables dynamic slot extraction:
        - Feature names like 'a1', 'a2', 'label' will be recognized
        - Target column names will be matched
        - Scatter plot can find both x and y columns
        
        Args:
            room_name: The room/session identifier
            features: List of column names from the loaded dataset
        """
        self.available_features[room_name] = features
        print(f"[NLU] Injected {len(features)} dataset columns for '{room_name}': {features[:5]}{'...' if len(features) > 5 else ''}")
    
    def get_dialogue_state(self, room_name: str) -> DialogueState:
        """Get or create dialogue state for a room."""
        if room_name not in self.dialogue_states:
            self.dialogue_states[room_name] = DialogueState()
        return self.dialogue_states[room_name]
    
    def reset_dialogue(self, room_name: str):
        """Reset dialogue state for a room."""
        if room_name in self.dialogue_states:
            self.dialogue_states[room_name].reset()
    
    def understand(
        self, 
        text: str, 
        room_name: str = "default",
        use_context: bool = True
    ) -> NLUResult:
        """
        Understand user input and extract intent/slots.
        
        HYBRID APPROACH:
        ================
        1. BERT BIO Schema:
           - Intent classification (chat, plot, ml_model)
           - Known slot extraction (plot_type, model_type from training vocabulary)
        
        2. Rule-Based with Injected Headers:
           - Dynamic feature/column extraction (matches dataset headers)
           - Target column extraction for ML models
           - Typo tolerance for plot types
        
        3. Merge Results:
           - Use BERT intent (higher accuracy)
           - Use BERT slots for plot_type, model_type (trained vocabulary)
           - Use rule-based for features, target (dynamic from dataset)
        
        Args:
            text: User input text
            room_name: Room/session identifier for multi-turn tracking
            use_context: Whether to use dialogue context
        
        Returns:
            NLUResult with intent, slots, and dialogue state
        """
        state = self.get_dialogue_state(room_name)
        features = self.available_features.get(room_name, [])
        
        # Always run rule-based (for dynamic column extraction)
        rule_result = self.rule_based.understand(text, features)
        
        # If BERT is available, use hybrid approach
        if self.model is not None and self.tokenizer is not None:
            bert_result = self._bert_understand(text, state, use_context)
            
            # HYBRID MERGE:
            # - Intent: Use BERT (trained classifier)
            # - plot_type/model_type: Prefer BERT if found, else use rule-based
            # - features/target: Use rule-based (dynamic from dataset headers)
            merged_slots = {}
            
            # Use BERT-extracted slots for known types
            if bert_result.slots.get("plot_type"):
                merged_slots["plot_type"] = bert_result.slots["plot_type"]
            elif rule_result.slots.get("plot_type"):
                merged_slots["plot_type"] = rule_result.slots["plot_type"]
            
            if bert_result.slots.get("model_type"):
                merged_slots["model_type"] = bert_result.slots["model_type"]
            elif rule_result.slots.get("model_type"):
                merged_slots["model_type"] = rule_result.slots["model_type"]
            
            # Use rule-based for dynamic slots (features from dataset headers)
            if rule_result.slots.get("features"):
                merged_slots["features"] = rule_result.slots["features"]
                merged_slots["feature"] = rule_result.slots.get("feature")
            
            if rule_result.slots.get("target"):
                merged_slots["target"] = rule_result.slots["target"]
            
            result = NLUResult(
                intent=bert_result.intent,
                confidence=bert_result.confidence,
                slots=merged_slots,
                is_complete=self._check_complete(bert_result.intent, merged_slots),
                missing_slots=self._get_missing(bert_result.intent, merged_slots),
                clarification_question=self._get_clarification(bert_result.intent, merged_slots),
                raw_text=text
            )
        else:
            # Fallback to pure rule-based
            result = rule_result
        
        # Update dialogue state
        state.add_turn("user", text)
        
        # Handle multi-turn: if previous turn was incomplete, try to continue that intent
        if use_context and state.intent is not None and not state.is_complete:
            # Check if this looks like a follow-up (providing missing info)
            # E.g., user says "a1 and a2" after being asked for features
            is_followup = self._is_likely_followup(text, state, result)
            
            if is_followup or result.intent == "chat":
                # Try to extract slots for the pending intent
                pending_intent = state.intent
                merged_slots = {**state.slots}
                
                # Extract based on what's missing
                if pending_intent == "plot" and "features" in state.missing_slots:
                    # Extract features from this message
                    new_features = self.rule_based._extract_features(text, features)
                    if new_features:
                        existing_features = merged_slots.get("features", [])
                        merged_slots["features"] = existing_features + [f for f in new_features if f not in existing_features]
                        if merged_slots["features"]:
                            merged_slots["feature"] = merged_slots["features"][0]
                
                elif pending_intent == "ml_model":
                    # Extract target and/or model_type
                    if "target" in state.missing_slots:
                        new_target = self.rule_based._extract_target(text, features)
                        if new_target:
                            merged_slots["target"] = new_target
                    
                    if "model_type" in state.missing_slots:
                        # Try to extract model type from this message
                        text_lower = text.lower()
                        for mtype, keywords in self.rule_based.MODEL_TYPES.items():
                            for kw in keywords:
                                if kw in text_lower:
                                    merged_slots["model_type"] = mtype
                                    break
                            if "model_type" in merged_slots:
                                break
                
                # Also merge any other slots from current result
                for k, v in result.slots.items():
                    if k not in merged_slots:  # Don't overwrite existing slots
                        merged_slots[k] = v
                
                result = NLUResult(
                    intent=pending_intent,
                    confidence=result.confidence,
                    slots=merged_slots,
                    is_complete=self._check_complete(pending_intent, merged_slots),
                    missing_slots=self._get_missing(pending_intent, merged_slots),
                    clarification_question=self._get_clarification(pending_intent, merged_slots),
                    raw_text=text
                )
        
        # Handle multi-turn: merge slots from previous turns when intents match
        elif use_context and state.intent == result.intent:
            # Merge new slots with existing
            merged_slots = {**state.slots, **result.slots}
            
            # Special handling for features list
            if "features" in state.slots or "features" in result.slots:
                old_features = state.slots.get("features", [])
                new_features = result.slots.get("features", [])
                merged_slots["features"] = old_features + [f for f in new_features if f not in old_features]
            
            result = NLUResult(
                intent=result.intent,
                confidence=result.confidence,
                slots=merged_slots,
                is_complete=self._check_complete(result.intent, merged_slots),
                missing_slots=self._get_missing(result.intent, merged_slots),
                clarification_question=self._get_clarification(result.intent, merged_slots),
                raw_text=text
            )
        
        # Update state with new info
        state.intent = result.intent
        state.slots = result.slots
        state.missing_slots = result.missing_slots
        state.is_complete = result.is_complete
        
        # If complete, reset for next turn
        if result.is_complete and result.intent != "chat":
            state.add_turn("system", f"Executed {result.intent}")
        
        return result
    
    def _bert_understand(
        self, 
        text: str, 
        state: DialogueState,
        use_context: bool
    ) -> NLUResult:
        """Use BERT model for understanding."""
        
        # Prepare input
        context = state.get_context_string() if use_context else ""
        words = text.split()
        
        with torch.no_grad():
            if context:
                # Tokenize with context
                context_enc = self.tokenizer(
                    context,
                    add_special_tokens=True,
                    padding=False,
                    truncation=True,
                    max_length=self.config.max_length // 2,
                    return_tensors='pt'
                )
                context_ids = context_enc['input_ids'].squeeze().tolist()
                if isinstance(context_ids, int):
                    context_ids = [context_ids]
                context_len = len(context_ids)
                
                turn_enc = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.config.max_length - context_len - 1,
                    return_tensors='pt'
                )
                turn_ids = turn_enc['input_ids'].squeeze().tolist()
                if isinstance(turn_ids, int):
                    turn_ids = [turn_ids]
                word_ids = turn_enc.word_ids()
                
                sep_id = self.tokenizer.sep_token_id
                input_ids = context_ids + turn_ids + [sep_id]
            else:
                enc = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    add_special_tokens=True,
                    padding=False,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].squeeze().tolist()
                if isinstance(input_ids, int):
                    input_ids = [input_ids]
                word_ids = enc.word_ids()
                context_len = 1  # [CLS]
            
            # Pad
            pad_len = self.config.max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            else:
                input_ids = input_ids[:self.config.max_length]
            
            attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]
            
            # Convert to tensors
            input_ids_t = torch.tensor([input_ids]).to(self.config.device)
            attention_mask_t = torch.tensor([attention_mask]).to(self.config.device)
            
            # Forward pass
            outputs = self.model(input_ids_t, attention_mask_t)
            
            # Intent prediction
            intent_probs = torch.softmax(outputs['intent_logits'], dim=-1)
            intent_conf, intent_pred = intent_probs.max(dim=-1)
            intent = ID2INTENT[intent_pred.item()]
            confidence = intent_conf.item()
            
            # Slot prediction
            slot_preds = torch.argmax(outputs['slot_logits'], dim=-1).squeeze().tolist()
            
            # Extract slots from predictions
            slots = self._extract_slots(words, word_ids, slot_preds, context_len)
        
        # Check completeness
        is_complete = self._check_complete(intent, slots)
        missing = self._get_missing(intent, slots)
        clarification = self._get_clarification(intent, slots)
        
        return NLUResult(
            intent=intent,
            confidence=confidence,
            slots=slots,
            is_complete=is_complete,
            missing_slots=missing,
            clarification_question=clarification,
            raw_text=text
        )
    
    def _extract_slots(
        self, 
        words: List[str], 
        word_ids: List[int], 
        slot_preds: List[int],
        context_len: int
    ) -> Dict[str, str]:
        """Extract slot values from BIO predictions."""
        slots = {}
        current_slot = None
        current_value = []
        
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            
            pred_idx = context_len + i
            if pred_idx >= len(slot_preds):
                break
            
            tag = ID2SLOT.get(slot_preds[pred_idx], 'O')
            
            if tag.startswith('B-'):
                # Save previous slot if exists
                if current_slot and current_value:
                    slots[current_slot] = ' '.join(current_value)
                # Start new slot
                current_slot = tag[2:]  # Remove 'B-'
                current_value = [words[word_id]]
            elif tag.startswith('I-') and current_slot:
                slot_type = tag[2:]
                if slot_type == current_slot:
                    current_value.append(words[word_id])
            else:
                # O tag - save current slot if exists
                if current_slot and current_value:
                    slots[current_slot] = ' '.join(current_value)
                current_slot = None
                current_value = []
        
        # Save final slot
        if current_slot and current_value:
            slots[current_slot] = ' '.join(current_value)
        
        return slots
    
    def _is_likely_followup(self, text: str, state: DialogueState, result: NLUResult) -> bool:
        """
        Check if the current message is likely a follow-up to provide missing info.
        
        Args:
            text: Current user input
            state: Current dialogue state
            result: NLU result from current message
        
        Returns:
            True if this looks like a follow-up message
        """
        text_lower = text.lower().strip()
        
        # Very short messages are likely follow-ups
        if len(text.split()) <= 4:
            return True
        
        # Check if text contains potential column names (for scatter plot features)
        if state.intent == "plot" and "features" in state.missing_slots:
            # Look for column name patterns (a1, b2, column_name, etc.)
            if re.search(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text):
                return True
        
        # If current intent is chat but we're expecting specific info, it's likely a follow-up
        if result.intent == "chat" and state.missing_slots:
            return True
        
        return False
    
    def _check_complete(self, intent: str, slots: Dict[str, Any]) -> bool:
        """Check if all required slots are filled.
        
        Special handling for scatter plots which need 2 features.
        """
        required = REQUIRED_SLOTS.get(intent, [])
        basic_complete = all(s in slots for s in required)
        
        # Special check: scatter plots need exactly 2 features
        if intent == "plot" and slots.get("plot_type") == "scatter":
            features = slots.get("features", [])
            return basic_complete and len(features) >= 2
        
        return basic_complete
    
    def _get_missing(self, intent: str, slots: Dict[str, Any]) -> List[str]:
        """Get list of missing required slots.
        
        Special handling for scatter plots which need 2 features.
        """
        required = REQUIRED_SLOTS.get(intent, [])
        missing = [s for s in required if s not in slots]
        
        # Special check: scatter plots need 2 features
        if intent == "plot" and slots.get("plot_type") == "scatter":
            features = slots.get("features", [])
            if len(features) < 2:
                if "features" not in missing:
                    missing.append("features")
        
        return missing
    
    def _get_clarification(self, intent: str, slots: Dict[str, Any]) -> Optional[str]:
        """Get clarification question for first missing slot.
        
        Special handling for scatter plots which need 2 features.
        """
        missing = self._get_missing(intent, slots)
        if missing:
            # Special message for scatter plot features
            if "features" in missing and slots.get("plot_type") == "scatter":
                features = slots.get("features", [])
                if len(features) == 1:
                    return f"I have '{features[0]}' as one axis. Which column would you like for the other axis?"
                else:
                    return "I need 2 features for a scatter plot. Which columns would you like to compare? (e.g., 'a1 and a2')"
            return CLARIFICATION_QUESTIONS.get(missing[0])
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_nlu_module(model_path: str = None) -> NLUModule:
    """Create an NLU module with optional model path."""
    config = NLUConfig(model_path=model_path)
    return NLUModule(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the NLU module with trained model
    import sys
    sys.path.insert(0, '..')
    
    # Use the trained multi-turn model
    config = NLUConfig(
        model_path="models/multi_turn_model.pt"
    )
    nlu = NLUModule(config)
    
    # Set available features (simulating dataset columns)
    nlu.set_available_features("test", ["a1", "a2", "a3", "b1", "b2", "time", "value"])
    nlu.set_available_features("multi_test", ["a1", "a2", "a3", "b1", "b2", "time", "value"])
    
    print("=" * 60)
    print("NLU Module Test - Single Turn")
    print("=" * 60)
    
    test_inputs = [
        "show me a histogram of a1",
        "train a decision tree to predict value",
        "what is machine learning?",
        "scatter plot of a1 vs a2",
        "need scatter plot for a1 and a3",  # User's exact case
    ]
    
    for text in test_inputs:
        nlu.reset_dialogue("test")
        result = nlu.understand(text, room_name="test")
        print(f"\nInput: '{text}'")
        print(f"  Intent: {result.intent} (conf: {result.confidence:.2f})")
        print(f"  Slots: {result.slots}")
        print(f"  Complete: {result.is_complete}")
        if result.clarification_question:
            print(f"  Clarify: {result.clarification_question}")
    
    print("\n" + "=" * 60)
    print("NLU Module Test - Scatter Plot Multi-Turn")
    print("=" * 60)
    
    # Simulate user's exact conversation: "need scatter plot for a1 and a3" -> "a1 and a2"
    nlu.reset_dialogue("scatter_test")
    scatter_inputs = [
        "need scatter plot for a1",  # Only 1 feature
        "a1 and a2",  # Providing both features
    ]
    
    print("\n[Scatter plot multi-turn - needs 2 features]")
    for text in scatter_inputs:
        result = nlu.understand(text, room_name="scatter_test")
        print(f"\nUser: '{text}'")
        print(f"  → Intent: {result.intent} (conf: {result.confidence:.2f})")
        print(f"  → Slots: {result.slots}")
        if result.is_complete:
            print(f"  → [ACTION] Execute {result.intent}")
            nlu.reset_dialogue("scatter_test")
        elif result.clarification_question:
            print(f"  → Bot: {result.clarification_question}")
    
    print("\n" + "=" * 60)
    print("NLU Module Test - Multi-Turn Dialogue")
    print("=" * 60)
    
    # Simulate a multi-turn conversation
    nlu.reset_dialogue("multi_test")
    multi_turn_inputs = [
        "I want to see a chart",
        "histogram",
        "show me another plot",
        "scatter",
    ]
    
    print("\n[Multi-turn dialogue simulation]")
    for text in multi_turn_inputs:
        result = nlu.understand(text, room_name="multi_test")
        print(f"\nUser: '{text}'")
        print(f"  → Intent: {result.intent} (conf: {result.confidence:.2f})")
        print(f"  → Slots: {result.slots}")
        if result.is_complete:
            print(f"  → [ACTION] Execute {result.intent}")
            nlu.reset_dialogue("multi_test")  # Reset after action
        elif result.clarification_question:
            print(f"  → Bot: {result.clarification_question}")
