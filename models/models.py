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

# The full return is an array of these
class MutationList(BaseModel):
    mutations: list[MutationObject]

class ContainsTable(BaseModel):
    has_table: bool

class OptimizedSystemPrompt(BaseModel):
    system_prompt: str
    converged: bool
    rationale: str