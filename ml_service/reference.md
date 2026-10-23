# Trace Analysis: Issue Description Through the ML Model

Comprehensive architectural and code-level trace of the issue description through the machine-learning pipeline in the **AI-Driven Customer Call Automation System**.

---

## Executive Summary

The codebase implements **two distinct machine-learning models**:

1. **Primary ML Triage Classifier (`DistilBERT`)**: Classifies ticket text into student follow-up triage actions (`YES`, `NO`, `REVIEW`), orchestrated by [`HuggingFaceModelLoader`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L52) and evaluated via [`FollowUpDecisionEngine`](file:///e:/ECS/ECS-2/ml_service/decision_engine.py#L107).
2. **Auxiliary ML Intent Classifier (`TF-IDF + Logistic Regression` baseline / `DistilBERT`)**: Classifies query text into 4 customer support topics (`payment_issue`, `get_refund`, `contact_customer_service`, `recover_password`), orchestrated by [`AuxiliaryIntentClassifier`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L66).

> [!IMPORTANT]
> **Priority is NOT predicted by Machine Learning.** Priority scores (`0.0 - 100.0`) and categories (`HIGH`, `MEDIUM`, `LOW`, `NOT_APPLICABLE`) are computed independently by a rule-based heuristic engine ([`PriorityAssessmentEngine`](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L216)) using transparent regex pattern matching and point accumulation.

---

## 1. Identify the ML Model Files

| Responsibility | Target Component | File Path | Key Functions / Classes / Artifacts |
|---|---|---|---|
| **Loading trained model** | Primary DistilBERT | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L94-L175) | `HuggingFaceModelLoader.load_model()`, `HuggingFaceModelLoader.inject_model_and_tokenizer()` |
| | Auxiliary Intent Baseline / Transformer | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L95-L191) | `AuxiliaryIntentClassifier.load_model()` |
| **Loading tokenizer / vectorizer / artifacts** | Primary DistilBERT Tokenizer | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L152-L155) | `transformers.AutoTokenizer.from_pretrained()` |
| | Auxiliary Vectorizer & Model Pipeline | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L117-L128) | `joblib.load("baseline_pipeline.joblib")` *(loads `TfidfVectorizer` + `LogisticRegression`)*, `json.load("label_mapping.json")` |
| **Input reception & extraction** | Input schema & pipeline entry point | [ml_service/schemas.py](file:///e:/ECS/ECS-2/ml_service/schemas.py#L73-L132)<br>[ml_service/predictor.py](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148) | `TicketInput.issue_description`<br>`StudentSupportPredictor.predict()` |
| **Preprocessing input text** | Text sanitization & validation | [ml_service/validator.py](file:///e:/ECS/ECS-2/ml_service/validator.py#L103-L191) | `TicketValidator.validate()`, `validate_ticket_input()` |
| | Transformer Tokenization | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L272-L278) | `self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")` |
| **Predicting issue category / class** | Follow-up triage prediction | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L229-L331) | `HuggingFaceModelLoader.classify_text()` |
| | Auxiliary intent prediction | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L192-L289) | `AuxiliaryIntentClassifier.predict_intent()` |
| **Assigning priority** | Separate rule-based heuristic engine *(No ML)* | [ml_service/priority_engine.py](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L374-L602) | `PriorityAssessmentEngine.assess()`, `assess_urgency()`, `assess_impact()` |
| **Converting predictions to readable output** | Output schema formatting | [ml_service/schemas.py](file:///e:/ECS/ECS-2/ml_service/schemas.py#L137-L192)<br>[ml_service/predictor.py](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148) | `TicketPrediction`<br>`StudentSupportPredictor._predict_core()`, `StudentSupportPredictor._attach_auxiliary_intent()` |

---

## 2. Trace the Issue Description

### 1. Which file and function receive the issue description as input?
The issue description enters the ML module boundary at:
- **File**: [`ml_service/predictor.py`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145)
- **Function**: `StudentSupportPredictor.predict(self, ticket: Union[TicketInput, Dict[str, Any], Any]) -> TicketPrediction`

From `predict()`, the ticket flows into:
- [`StudentSupportPredictor._predict_core()`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L150)
- [`StudentSupportPredictor._attach_auxiliary_intent()`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L108)

The underlying inference functions that directly receive the text are:
- Primary Model: `HuggingFaceModelLoader.classify_text(self, text: str)` in [`ml_service/model_loader.py:L229`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L229).
- Auxiliary Model: `AuxiliaryIntentClassifier.predict_intent(self, text: Optional[str])` in [`ml_service/intent_classifier.py:L192`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L192).

### 2. What is the exact input variable name?
- At the top-level schema: `issue_description` on `TicketInput` ([`ml_service/schemas.py:L106`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L106)).
- In `TicketValidator.validate()`: `raw_desc = ticket.issue_description` ([`ml_service/validator.py:L143`](file:///e:/ECS/ECS-2/ml_service/validator.py#L143)).
- In `StudentSupportPredictor._attach_auxiliary_intent()`: `raw_text` ([`ml_service/predictor.py:L128`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L128)).
- In `FollowUpDecisionEngine.evaluate()`: `text = cleaned.issue_description.strip()` ([`ml_service/decision_engine.py:L187`](file:///e:/ECS/ECS-2/ml_service/decision_engine.py#L187)).
- In `classify_text()` and `predict_intent()`: `text` ([`ml_service/model_loader.py:L229`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L229), [`ml_service/intent_classifier.py:L192`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L192)).

### 3. Is the input a raw transcript, cleaned text, extracted issue description, or another representation?
The input is an **extracted issue description** representing the student's problem narrative from the support ticket. It is not a raw speech or telephony transcript.
Before model inference, it is stripped of leading and trailing whitespace by [`TicketValidator.validate()`](file:///e:/ECS/ECS-2/ml_service/validator.py#L152) (`cleaned_desc = raw_desc.strip()`) to produce cleaned text.

### 4. Which function preprocesses the text before prediction?
- **Validation & Whitespace Sanitization**: `TicketValidator.validate()` / `validate_ticket_input()` in [`ml_service/validator.py:L103-L191`](file:///e:/ECS/ECS-2/ml_service/validator.py#L103-L191).
- **Primary Model Tokenization**: `self.tokenizer(...)` called inside `HuggingFaceModelLoader.classify_text()` in [`ml_service/model_loader.py:L272-L278`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L272-L278).
- **Auxiliary Intent Baseline Preprocessing**: `clean_text = text.strip()` in [`ml_service/intent_classifier.py:L217`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L217), followed by Scikit-Learn `TfidfVectorizer.transform()` inside `self.pipeline.predict_proba([clean_text])` ([`ml_service/intent_classifier.py:L222`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L222)).

### 5. Is the description passed directly to the model or transformed into numerical features first?
It is **transformed into numerical features first**:
- **Primary Model (`DistilBERT`)**: Tokenized into numerical input IDs and attention mask tensors using `AutoTokenizer` with truncation and padding up to 512 tokens:
  ```python
  inputs = self.tokenizer(
      text,
      max_length=max_len,
      padding=True,
      truncation=True,
      return_tensors="pt",
  )
  ```
- **Auxiliary Intent Baseline**: Transformed into a numerical TF-IDF feature vector (n-gram range 1–2, sublinear term frequency, max 10,000 features) by the Scikit-Learn `TfidfVectorizer` embedded in the loaded `joblib` pipeline.

### 6. ML Model Input Boundary
The boundary is the `TicketInput` model schema ([`ml_service/schemas.py:L73`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L73)) ingested via `StudentSupportPredictor.predict(ticket)` ([`ml_service/predictor.py:L145`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145)).

---

## 3. Identify the Prediction Function

### Primary Model: DistilBERT Triage Classifier
- **File path**: [`ml_service/model_loader.py`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L229-L331)
- **Class and function name**: `HuggingFaceModelLoader.classify_text(self, text: str)`
- **Input parameters**: `text: str`
- **Preprocessing steps**: Tokenization with truncation (`max_length=512`), padding (`padding=True`), tensor conversion (`return_tensors="pt"`), and device transfer (`v.to(self.device)`).
- **Model object called**: `self.model` (`transformers.AutoModelForSequenceClassification`)
- **Exact prediction method**: PyTorch model forward pass without gradients:
  ```python
  with torch.no_grad():
      outputs = self.model(**device_inputs)
  ```
  Followed by `torch.softmax(logits, dim=-1)` to calculate normalized class probabilities.
- **Return value and data type**: Python `Dict[str, Any]` with keys:
  - `"status"`: `ModelStatus`
  - `"status_reason"`: `str`
  - `"label"`: `Optional[str]` (`'YES'`, `'NO'`, or `'REVIEW'`)
  - `"score"`: `Optional[float]` (highest class probability)
  - `"probabilities"`: `Optional[Dict[str, float]]` (probabilities for each class)
- **Outputs returned**: Returns the triage class (`label`), confidence score (`score`), and per-class probabilities. It **does NOT predict priority**.

---

### Auxiliary Model: Support Intent Classifier
- **File path**: [`ml_service/intent_classifier.py`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L192-L289)
- **Class and function name**: `AuxiliaryIntentClassifier.predict_intent(self, text: Optional[str])`
- **Input parameters**: `text: Optional[str]`
- **Preprocessing steps**: Stripping whitespace via `clean_text = text.strip()`.
- **Model/Pipeline object called**: 
  - *Baseline*: `self.pipeline` (`sklearn.pipeline.Pipeline` containing `TfidfVectorizer` and `LogisticRegression`, loaded from [`models/auxiliary-intent-baseline/baseline_pipeline.joblib`](file:///e:/ECS/ECS-2/ml_service/models/auxiliary-intent-baseline/baseline_pipeline.joblib)).
  - *Transformer*: `self.model` (`transformers.AutoModelForSequenceClassification`).
- **Exact prediction method**:
  - *Baseline*: `self.pipeline.predict_proba([clean_text])`
  - *Transformer*: `self.model(**inputs)` forward pass + `torch.softmax(logits, dim=-1)`
- **Return value and data type**: `IntentPrediction` dataclass ([`ml_service/intent_classifier.py:L37-L54`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L37-L54)) containing:
  - `status`: `IntentModelStatus`
  - `predicted_intent`: `Optional[str]` (`'payment_issue'`, `'get_refund'`, `'contact_customer_service'`, `'recover_password'`)
  - `intent_confidence`: `Optional[float]` (rounded to 4 decimals)
  - `model_type`: `Optional[str]` (`'baseline_tfidf_logistic_regression'` or `'distilbert'`)
  - `probabilities`: `Optional[Dict[str, float]]`
  - `status_reason`: `str`
- **Outputs returned**: Returns the auxiliary intent category, confidence score, and per-class probabilities. It **does NOT predict priority**.

---

## 4. Trace the Prediction Output

### Raw Model Prediction vs. Final Formatted ML Output

```
┌─────────────────────────────────────────────────────────────┐
│ RAW MODEL PREDICTIONS                                       │
│ 1. DistilBERT: outputs.logits (tensor)                      │
│    -> torch.softmax -> probs_list                           │
│ 2. Intent Baseline: pipeline.predict_proba -> array of probs│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LABEL DECODING                                              │
│ 1. best_idx = argmax(probs_list)                            │
│    best_label = id2label[best_idx] ("YES" / "NO" / "REVIEW")│
│ 2. classes[best_idx] ("payment_issue", "get_refund", etc.)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ POST-PROCESSING ENGINES (HEURISTIC & BUSINESS RULES)        │
│ 1. FollowUpDecisionEngine (reconciles model with callback   │
│    regex, resolved regex, ticket status, and conflicts)     │
│ 2. PriorityAssessmentEngine (regex scoring for urgency,     │
│    impact, actionability -> 0-100 score + priority cat)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ FINAL FORMATTED ML OUTPUT                                   │
│ TicketPrediction Pydantic Model                             │
│ {                                                           │
│   "ticket_id": "...",                                       │
│   "follow_up_required": "YES",                              │
│   "interest_level": "UNKNOWN",                              │
│   "urgency": "HIGH",                                        │
│   "priority_score": 90.0,                                   │
│   "priority": "HIGH",                                       │
│   "recommended_action": "Urgent advisor callback...",       │
│   "model_status": "READY",                                  │
│   "predicted_intent": "payment_issue",                      │
│   "intent_confidence": 0.9966                               │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Trace of Output Variables:
1. **Raw model prediction storage**:
   - Primary: `logits = outputs.logits` ([`ml_service/model_loader.py:L299`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L299)) and `probs = torch.softmax(logits, dim=-1)` ([`ml_service/model_loader.py:L303`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L303)).
   - Auxiliary: `probs = self.pipeline.predict_proba([clean_text])[0]` ([`ml_service/intent_classifier.py:L222`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L222)).
2. **Label mapping (encoded to human-readable)**:
   - Primary: `best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))` and `best_label = self.id2label.get(best_idx, "REVIEW")` ([`ml_service/model_loader.py:L316-L317`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L316-L317)), using canonical mapping `{0: "NO", 1: "YES", 2: "REVIEW"}` defined in [`ml_service/training_contract.py:L45-L49`](file:///e:/ECS/ECS-2/ml_service/training_contract.py#L45-L49).
   - Auxiliary: `best_idx = int(probs.argmax())` and `best_label = str(classes[best_idx])` ([`ml_service/intent_classifier.py:L225-L226`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L225-L226)), corresponding to `{0: "payment_issue", 1: "get_refund", 2: "contact_customer_service", 3: "recover_password"}` in [`label_mapping.json`](file:///e:/ECS/ECS-2/ml_service/models/auxiliary-intent-baseline/label_mapping.json#L8-L13).
3. **Confidence or probability calculation**:
   - Primary: `best_score = float(probs_list[best_idx])` and per-class dictionary `prob_dict` ([`ml_service/model_loader.py:L318-L323`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L318-L323)).
   - Auxiliary: `best_conf = float(probs[best_idx])`, rounded to 4 decimals with `round(best_conf, 4)` ([`ml_service/intent_classifier.py:L227-L234`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L227-L234)).
4. **Post-processing and priority assignment**:
   - The ML predictions pass through [`FollowUpDecisionEngine.evaluate()`](file:///e:/ECS/ECS-2/ml_service/decision_engine.py#L129-L405), which cross-examines the classifier's output against explicit callback phrasing (`CALLBACK_PATTERNS`), resolved phrasing (`RESOLVED_PATTERNS`), and conflicting signals.
   - Priority is **not** assigned by ML. It is assigned by [`PriorityAssessmentEngine.assess()`](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L374-L602) via heuristic point aggregation (`urgency_pts` up to 50, `impact_pts` up to 30, `actionability_pts` up to 20), generating `priority_score` and categorical `priority` (`HIGH`, `MEDIUM`, `LOW`, `NOT_APPLICABLE`).
5. **Final function that returns the ML result to its caller**:
   - `StudentSupportPredictor.predict()` in [`ml_service/predictor.py:L145-L148`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148), returning a `TicketPrediction` instance.

---

## 5. Code-Level Evidence

| Stage | File path | Exact function | Input | Output | Line Numbers |
|---|---|---|---|---|---|
| **Model loading** | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L94-L175) | `HuggingFaceModelLoader.load_model()` | `model_name_or_path: Optional[str]`, `device: Optional[str]` | `bool` (`self.is_fine_tuned`) | Lines 94–175 |
| | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L95-L191) | `AuxiliaryIntentClassifier.load_model()` | `model_dir: Optional[Union[str, Path]]` | `bool` (`self.is_loaded`) | Lines 95–191 |
| **Input reception** | [ml_service/predictor.py](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148) | `StudentSupportPredictor.predict()` | `ticket: Union[TicketInput, Dict[str, Any], Any]` | `TicketPrediction` | Lines 145–148 |
| | [ml_service/schemas.py](file:///e:/ECS/ECS-2/ml_service/schemas.py#L106-L109) | `TicketInput` model definition | Payload JSON / dict | Pydantic model with `issue_description` field | Lines 106–109 |
| **Preprocessing** | [ml_service/validator.py](file:///e:/ECS/ECS-2/ml_service/validator.py#L103-L191) | `TicketValidator.validate()` | `raw_input: Union[TicketInput, Dict[str, Any], Any]` | `ValidationResult(cleaned_ticket=...)` | Lines 103–191 |
| **Feature transformation** | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L272-L278) | `HuggingFaceModelLoader.classify_text()` | `text: str` | PyTorch tensors: `inputs` (input_ids, attention_mask) | Lines 272–278 |
| | [ml_service/train_intent_baseline.py](file:///e:/ECS/ECS-2/ml_service/train_intent_baseline.py#L69-L72) / [intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L222) | `Pipeline.predict_proba()` | `[clean_text]` | TF-IDF sparse matrix (`TfidfVectorizer`) | Lines 220–222 |
| **Inference** | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L292-L303) | `HuggingFaceModelLoader.classify_text()` | `device_inputs: dict` of tensors | Raw logits: `outputs.logits`, Softmax probs: `probs` | Lines 292–303 |
| | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L222) | `AuxiliaryIntentClassifier.predict_intent()` | `[clean_text]` | `probs = self.pipeline.predict_proba([clean_text])[0]` | Lines 221–222 |
| **Label decoding** | [ml_service/model_loader.py](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L316-L318) | `HuggingFaceModelLoader.classify_text()` | `probs_list: List[float]` | `best_label: str` (`"YES"`, `"NO"`, or `"REVIEW"`), `best_score: float` | Lines 316–318 |
| | [ml_service/intent_classifier.py](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L225-L227) | `AuxiliaryIntentClassifier.predict_intent()` | `probs: np.ndarray` | `best_label: str` (e.g., `"payment_issue"`), `best_conf: float` | Lines 225–227 |
| **Priority processing** | [ml_service/priority_engine.py](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L374-L602) | `PriorityAssessmentEngine.assess()` | `ticket`, `follow_up_decision` | `PriorityAssessment` *(Heuristic scoring; NOT ML)* | Lines 374–602 |
| **Final ML output** | [ml_service/predictor.py](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148) | `StudentSupportPredictor.predict()` | Prediction objects, decision, intent | `TicketPrediction` Pydantic instance | Lines 145–148, 229–239 |

---

### Code Excerpts

#### 1. Input Reception & Validation
From [`ml_service/validator.py:L142-L153`](file:///e:/ECS/ECS-2/ml_service/validator.py#L142-L153):
```python
        # 2. Validate issue_description
        raw_desc = ticket.issue_description
        if raw_desc is None or not isinstance(raw_desc, str) or not raw_desc.strip():
            empty_description = True
            errors.append(
                "Missing or empty issue_description: Cannot perform NLP classification "
                "without textual description."
            )
            cleaned_desc = None
        else:
            cleaned_desc = raw_desc.strip()
```

#### 2. Feature Transformation & Primary Inference
From [`ml_service/model_loader.py:L272-L303`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L272-L303):
```python
        max_len = self.config.max_seq_length if self.config else 512
        inputs = self.tokenizer(
            text,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
...
        # Execute forward pass without gradients
        if hasattr(torch, "no_grad"):
            with torch.no_grad():
                outputs = self.model(**device_inputs)
        else:
            outputs = self.model(**device_inputs)

        # Extract logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Calculate softmax probabilities
        if hasattr(torch, "softmax"):
            probs = torch.softmax(logits, dim=-1)
```

#### 3. Primary Label Decoding & Return
From [`ml_service/model_loader.py:L316-L331`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L316-L331):
```python
        best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
        best_label = self.id2label.get(best_idx, "REVIEW")
        best_score = float(probs_list[best_idx])

        prob_dict = {
            self.id2label.get(idx, f"CLASS_{idx}"): float(p)
            for idx, p in enumerate(probs_list)
        }

        return {
            "status": ModelStatus.READY,
            "status_reason": self.status_reason,
            "label": best_label,
            "score": best_score,
            "probabilities": prob_dict,
        }
```

#### 4. Auxiliary Intent Baseline Inference
From [`ml_service/intent_classifier.py:L220-L238`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L220-L238):
```python
        # 1. Classical baseline inference
        if self.model_type == "baseline_tfidf_logistic_regression" and self.pipeline is not None:
            try:
                probs = self.pipeline.predict_proba([clean_text])[0]
                classes = list(self.pipeline.classes_)

                best_idx = int(probs.argmax())
                best_label = str(classes[best_idx])
                best_conf = float(probs[best_idx])

                prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}

                return IntentPrediction(
                    status=IntentModelStatus.READY,
                    predicted_intent=best_label,
                    intent_confidence=round(best_conf, 4),
                    model_type=self.model_type,
                    probabilities=prob_dict,
                    status_reason="Baseline TF-IDF + Logistic Regression inference (uncalibrated probability).",
                )
```

#### 5. Priority Engine Disclaimer & Heuristic Scoring (Non-ML Evidence)
From [`ml_service/priority_engine.py:L9-L14`](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L9-L14) and [`ml_service/priority_engine.py:L554-L556`](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L554-L556):
```python
# Lines 9-14:
# IMPORTANT DISCLAIMER & INSTITUTIONAL CONTEXT:
# The rules, criteria, weights, and scoring thresholds implemented in this module
# are preliminary heuristic baselines developed for transparent triage demonstration.
# They DO NOT represent officially approved or validated institutional policies of
# VIT-AP University. They are NOT machine-learned or statistically calibrated on
# historical datasets. They must not be claimed as accurate, optimal, or production-ready.

# Lines 554-556:
        raw_score = urgency_pts + impact_pts + action_pts
        # Bound score strictly to [0.0, 100.0]
        final_score = round(max(0.0, min(100.0, float(raw_score))), 1)
```

---

## 6. Execution Flow

The real execution call chain based on the actual codebase is:

```
StudentSupportPredictor.predict(ticket)
 │
 ├── 1. validate_ticket_input(ticket)
 │        └── TicketValidator.validate() -> cleans issue_description: raw_desc.strip()
 │
 ├── 2. _predict_core(ticket)
 │        │
 │        ├── FollowUpDecisionEngine.evaluate(cleaned, validation)
 │        │     │
 │        │     ├── HuggingFaceModelLoader.classify_text(text)
 │        │     │     ├── tokenizer(text, max_length=512, return_tensors="pt")
 │        │     │     ├── model(**device_inputs) -> outputs.logits
 │        │     │     ├── torch.softmax(logits, dim=-1) -> probs
 │        │     │     └── decode label: id2label[argmax(probs)] ("YES" / "NO" / "REVIEW")
 │        │     │
 │        │     └── Reconcile with regex patterns (CALLBACK_PATTERNS, RESOLVED_PATTERNS)
 │        │           └── returns FollowUpDecision(decision, reason, evidence)
 │        │
 │        └── PriorityAssessmentEngine.assess(cleaned, follow_up_decision, validation)
 │              ├── assess_urgency(text) -> (urgency_level, urgency_pts) [Heuristic Regex]
 │              ├── assess_impact(text) -> (impact_tier, impact_pts)     [Heuristic Regex]
 │              ├── assess_actionability(text) -> (action_tier, action_pts) [Heuristic Regex]
 │              ├── final_score = round(urgency_pts + impact_pts + action_pts, 1)
 │              └── returns PriorityAssessment(urgency, interest, priority_score, priority)
 │
 ├── 3. _attach_auxiliary_intent(prediction, ticket)
 │        │
 │        └── AuxiliaryIntentClassifier.predict_intent(clean_text)
 │              ├── baseline_pipeline.predict_proba([clean_text])
 │              │     ├── TfidfVectorizer.transform([clean_text])
 │              │     └── LogisticRegression.predict_proba()
 │              ├── decode label: classes[argmax(probs)]
 │              └── returns IntentPrediction(predicted_intent, intent_confidence)
 │
 └── 4. Assemble and return TicketPrediction
```

---

## 7. Final Answers

### 1. Where is the issue description received by the ML model?
It is received at the public ML service interface in `StudentSupportPredictor.predict(self, ticket)` located in [`ml_service/predictor.py:L145`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145), via the `ticket.issue_description` field defined in `TicketInput` ([`ml_service/schemas.py:L106`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L106)).

### 2. Which exact function processes the description?
- **Sanitization & validation**: `TicketValidator.validate()` in [`ml_service/validator.py:L103`](file:///e:/ECS/ECS-2/ml_service/validator.py#L103).
- **DistilBERT feature transformation**: `self.tokenizer()` inside `HuggingFaceModelLoader.classify_text()` in [`ml_service/model_loader.py:L272`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L272).
- **Auxiliary Intent feature transformation**: `TfidfVectorizer.transform()` within `self.pipeline.predict_proba()` in [`ml_service/intent_classifier.py:L222`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L222).

### 3. Which exact function performs the prediction?
- **Primary Follow-up Triage ML Model**: `HuggingFaceModelLoader.classify_text(self, text: str)` in [`ml_service/model_loader.py:L229`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L229).
- **Auxiliary Intent ML Model**: `AuxiliaryIntentClassifier.predict_intent(self, text: Optional[str])` in [`ml_service/intent_classifier.py:L192`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L192).

### 4. Which variable contains the predicted issue category?
- In `classify_text()`: `best_label` ([`ml_service/model_loader.py:L317`](file:///e:/ECS/ECS-2/ml_service/model_loader.py#L317)), containing `"YES"`, `"NO"`, or `"REVIEW"`.
- In `predict_intent()`: `best_label` ([`ml_service/intent_classifier.py:L226`](file:///e:/ECS/ECS-2/ml_service/intent_classifier.py#L226)), containing `"payment_issue"`, `"get_refund"`, `"contact_customer_service"`, or `"recover_password"`.
- In the final structured prediction object: `TicketPrediction.follow_up_required` ([`ml_service/schemas.py:L148`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L148)) and `TicketPrediction.predicted_intent` ([`ml_service/schemas.py:L182`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L182)).

### 5. Which function returns the final ML output?
`StudentSupportPredictor.predict(self, ticket)` in [`ml_service/predictor.py:L145`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145).

### 6. Does the ML model predict priority, or is priority assigned separately?
**Priority is assigned separately via a rule-based heuristic engine, NOT by any ML model.**
As documented in [`ml_service/priority_engine.py:L9-L14`](file:///e:/ECS/ECS-2/ml_service/priority_engine.py#L9-L14), priority scoring is performed by `PriorityAssessmentEngine.assess()` using transparent regex keyword matching and a weighted point formula (`urgency_pts` + `impact_pts` + `actionability_pts`). The codebase explicitly states that priority rules are preliminary heuristic baselines and are not statistically trained or calibrated on historical data.

### 7. What is the exact structure of the returned output?
The returned output is an instance of the `TicketPrediction` Pydantic model ([`ml_service/schemas.py:L137-L192`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L137-L192)):

```python
class TicketPrediction(BaseModel):
    ticket_id: Optional[str] = None
    follow_up_required: FollowUpRequired  # Enum: "YES" | "NO" | "REVIEW"
    interest_level: InterestLevel          # Enum: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
    urgency: UrgencyLevel                  # Enum: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
    follow_up_reason: Optional[str] = None # Human-readable explanation
    priority_score: Optional[float] = None # Float [0.0 - 100.0] or None
    priority: PriorityCategory             # Enum: "HIGH" | "MEDIUM" | "LOW" | "NOT_APPLICABLE"
    recommended_action: str                # Prescriptive operational instruction
    model_status: str                      # Diagnostic state (e.g., "READY", "PLACEHOLDER_PENDING_MODEL")
    predicted_intent: Optional[str] = None # e.g. "payment_issue", "get_refund"
    intent_confidence: Optional[float] = None # Float [0.0 - 1.0]
```

---

## 📌 TL;DR: ML Model Input & Output Summary

> [!IMPORTANT]
> ### 🎯 QUICK REFERENCE: INPUT & OUTPUT CONTRACT
>
> #### 📥 EXACT INPUT
> - **Input Field Name**: `issue_description` (type: `Optional[str]` / `str`)
> - **Received via Schema**: `TicketInput.issue_description` in [`ml_service/schemas.py`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L106-L109)
> - **Entry Function**: `StudentSupportPredictor.predict(ticket)` in [`ml_service/predictor.py`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145)
> - **Input State**: Extracted problem narrative from student ticket, stripped of leading/trailing whitespace (`raw_desc.strip()` via `TicketValidator.validate()`).
> - **Feature Transformation**:
>   - **Primary Model (`DistilBERT`)**: Transformed into PyTorch tensor IDs (`input_ids`, `attention_mask`) by `AutoTokenizer`.
>   - **Auxiliary Baseline**: Transformed into a numerical TF-IDF feature matrix by `TfidfVectorizer`.
>
> ---
>
> #### 📤 EXACT OUTPUT
> - **Returned Object**: `TicketPrediction` Pydantic instance in [`ml_service/schemas.py`](file:///e:/ECS/ECS-2/ml_service/schemas.py#L137-L192)
> - **Exit Function**: `StudentSupportPredictor.predict(ticket)` in [`ml_service/predictor.py`](file:///e:/ECS/ECS-2/ml_service/predictor.py#L145-L148)
> - **Core Output Fields**:
>   - **`follow_up_required`** (`str` enum): `"YES"` | `"NO"` | `"REVIEW"` *(ML prediction from DistilBERT sequence classification, reconciled with safety business rules)*
>   - **`predicted_intent`** (`Optional[str]`): `"payment_issue"` | `"get_refund"` | `"contact_customer_service"` | `"recover_password"` *(Auxiliary ML topic classification)*
>   - **`intent_confidence`** (`Optional[float]`): `0.0 - 1.0` *(Uncalibrated softmax / decision probability)*
>   - **`priority`** (`str` enum): `"HIGH"` | `"MEDIUM"` | `"LOW"` | `"NOT_APPLICABLE"` *(Assigned by heuristic regex engine, **NOT ML**)*
>   - **`priority_score`** (`Optional[float]`): `0.0 - 100.0` or `null` *(Assigned by heuristic point scoring, **NOT ML**)*
>   - **`urgency`** (`str` enum): `"HIGH"` | `"MEDIUM"` | `"LOW"` | `"UNKNOWN"` *(Keyword regex rule)*
>   - **`interest_level`** (`str` enum): `"HIGH"` | `"MEDIUM"` | `"LOW"` | `"UNKNOWN"` *(Keyword regex rule)*
>   - **`recommended_action`** (`str`): Prescriptive operational instruction for support staff.
>   - **`model_status`** (`str`): Operational status (e.g., `"READY"`, `"PLACEHOLDER_PENDING_MODEL"`).

