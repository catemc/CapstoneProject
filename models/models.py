from pydantic import BaseModel

class MutationObject(BaseModel):
    protein: str
    mutation: str
    subtype: str
    effect: str
    quote: str
    citation: str
    numbering: str | None = None  # optional
    type: str | None = None

class MutationList(BaseModel):
    mutations: list[MutationObject]

class ContainsTable(BaseModel):
    has_table: bool

class OptimizedSystemPrompt(BaseModel):
    system_prompt: str
    converged: bool
    rationale: str

class AnnotationMatch(BaseModel):
    mutation: str
    subtype: str
    effect: str

class EvaluationResult(BaseModel):
    matched_annotations: list[AnnotationMatch]
    missed_annotations: list[AnnotationMatch]
    hallucinated_annotations: list[AnnotationMatch]
    converged: bool
    system_prompt: str | None = None
    rationale: str
