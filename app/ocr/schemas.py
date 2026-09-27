"""
Data structures and schemas for Module 1 (OCR Extraction).
Defines PassportOCRResult dataclass and TypedDict, as well as Pydantic models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class PassportOCRDict(TypedDict, total=False):
    """TypedDict schema for Passport OCR Result."""
    document_type: str
    country_code: str
    surname: str
    given_names: str
    passport_number: str
    nationality: str
    date_of_birth: str
    sex: str
    date_of_expiry: str
    personal_number: str
    place_of_birth: str
    place_of_issue: str
    date_of_issue: str
    passport_type: str
    raw_ocr_text: List[str]
    mrz_validation: Dict[str, str]


@dataclass
class PassportOCRResult:
    """Dataclass holding complete structured OCR output from passport document."""
    document_type: str = "P"
    country_code: str = "IND"
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    nationality: str = "IND"
    date_of_birth: str = ""
    sex: str = ""
    date_of_expiry: str = ""
    personal_number: str = ""
    place_of_birth: str = ""
    place_of_issue: str = ""
    date_of_issue: str = ""
    passport_type: str = "P"
    raw_ocr_text: List[str] = field(default_factory=list)
    mrz_validation: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass instance to standard Python dictionary."""
        return {
            "document_type": self.document_type,
            "country_code": self.country_code,
            "surname": self.surname,
            "given_names": self.given_names,
            "passport_number": self.passport_number,
            "nationality": self.nationality,
            "date_of_birth": self.date_of_birth,
            "sex": self.sex,
            "date_of_expiry": self.date_of_expiry,
            "personal_number": self.personal_number,
            "place_of_birth": self.place_of_birth,
            "place_of_issue": self.place_of_issue,
            "date_of_issue": self.date_of_issue,
            "passport_type": self.passport_type,
            "raw_ocr_text": self.raw_ocr_text,
            "mrz_validation": self.mrz_validation,
        }


# Pydantic models for API serialization compatibility
class OCRFieldResult(BaseModel):
    value: str = Field(default="")
    confidence: float = Field(default=0.0)
    raw: str = Field(default="")


class MRZCheckDigitResult(BaseModel):
    value: str = Field(default="")
    expected: str = Field(default="")
    valid: bool = Field(default=False)


class MRZCheckDigits(BaseModel):
    passport_number: MRZCheckDigitResult = Field(default_factory=MRZCheckDigitResult)
    date_of_birth: MRZCheckDigitResult = Field(default_factory=MRZCheckDigitResult)
    date_of_expiry: MRZCheckDigitResult = Field(default_factory=MRZCheckDigitResult)
    optional_data: MRZCheckDigitResult = Field(default_factory=MRZCheckDigitResult)
    composite: MRZCheckDigitResult = Field(default_factory=MRZCheckDigitResult)


class MRZResult(BaseModel):
    line1: str = Field(default="")
    line2: str = Field(default="")
    document_type: str = Field(default="")
    issuing_country: str = Field(default="")
    surname: str = Field(default="")
    given_names: str = Field(default="")
    passport_number: str = Field(default="")
    nationality: str = Field(default="")
    date_of_birth: str = Field(default="")
    sex: str = Field(default="")
    date_of_expiry: str = Field(default="")
    optional_data: str = Field(default="")
    check_digits: MRZCheckDigits = Field(default_factory=MRZCheckDigits)
    all_check_digits_valid: bool = Field(default=False)
    parse_errors: List[str] = Field(default_factory=list)
    consistency: Dict[str, str] = Field(default_factory=dict)


class DocumentFields(BaseModel):
    name: OCRFieldResult = Field(default_factory=OCRFieldResult)
    passport_number: OCRFieldResult = Field(default_factory=OCRFieldResult)
    passport_type: OCRFieldResult = Field(default_factory=OCRFieldResult)
    country_code: OCRFieldResult = Field(default_factory=OCRFieldResult)
    nationality: OCRFieldResult = Field(default_factory=OCRFieldResult)
    date_of_birth: OCRFieldResult = Field(default_factory=OCRFieldResult)
    place_of_birth: OCRFieldResult = Field(default_factory=OCRFieldResult)
    date_of_issue: OCRFieldResult = Field(default_factory=OCRFieldResult)
    place_of_issue: OCRFieldResult = Field(default_factory=OCRFieldResult)
    date_of_expiry: OCRFieldResult = Field(default_factory=OCRFieldResult)
    sex: OCRFieldResult = Field(default_factory=OCRFieldResult)
    model_config = {"extra": "allow"}


class OCRResponse(BaseModel):
    success: bool = Field(description="Whether OCR succeeded.")
    document_type: str = Field(default="passport")
    fields: DocumentFields = Field(default_factory=DocumentFields)
    mrz: Optional[MRZResult] = Field(default=None)
    raw_text: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
