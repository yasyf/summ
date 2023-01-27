from typing import Optional, Self

from pydantic import BaseModel


class WithSafeParse(BaseModel):
    @classmethod
    def safe_parse(cls, obj: dict) -> Optional[Self]:
        try:
            return cls(**obj)
        except Exception:
            return None
