"""Request schemas shared by Web route modules."""
from pydantic import BaseModel, Field, StrictInt


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[str] = Field(min_length=1, max_length=50)
    use_model: bool = False


class QuizRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=50)
    count: int = Field(default=5, ge=1, le=10)
    target_concept: str | None = Field(default=None, min_length=2, max_length=40)
    section_id: str | None = Field(default=None, min_length=1, max_length=64)


class QuizGradeRequest(BaseModel):
    choices: dict[str, StrictInt] = Field(min_length=1, max_length=10)


class CompareRequest(BaseModel):
    concept: str = Field(min_length=2, max_length=80)
    document_ids: list[str] = Field(min_length=2, max_length=2)
    sense: str | None = Field(default=None, min_length=1, max_length=60)
