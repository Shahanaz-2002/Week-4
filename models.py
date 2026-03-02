from pydantic import BaseModel, Field
from typing import List


class CaseRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1)
    doctor_notes: str = Field(..., min_length=3)


class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SystemMetrics(BaseModel):
    response_time_ms: float = Field(..., ge=0)
    output_quality: str = Field(..., min_length=1)


class CaseResponse(BaseModel):
    similar_cases: List[SimilarCase]
    insight_summary: str = Field(..., min_length=1)
    confidence_reason: str = Field(..., min_length=1)
    system_metrics: SystemMetrics