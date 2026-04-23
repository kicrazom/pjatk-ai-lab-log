from dataclasses import dataclass


@dataclass
class ScoreResult:
    score: int
    interpretation: str
    scoring_version: str = "1.0"


def calculate_gds(payload: dict, variant: str = "gds15") -> ScoreResult:
    answers: list[bool] = payload.get("answers", [])
    score = sum(1 for a in answers if bool(a))
    if variant == "gds30":
        interpretation = "Wynik podwyższony" if score >= 11 else "Wynik prawidłowy"
    else:
        interpretation = "Podejrzenie depresji" if score >= 6 else "Brak istotnych objawów"
    return ScoreResult(score=score, interpretation=interpretation, scoring_version=f"{variant}-1.0")


def calculate_sarc_f(payload: dict) -> ScoreResult:
    fields = ["strength", "assistance", "rise", "climb", "falls"]
    score = sum(int(payload.get(k, 0)) for k in fields)
    interpretation = "Screen positive" if score >= 4 else "Screen negative"
    return ScoreResult(score=score, interpretation=interpretation)


def calculate_g8(payload: dict) -> ScoreResult:
    values = payload.get("items", [])
    score = sum(int(v) for v in values)
    interpretation = "Nieprawidłowy przesiew" if score <= 14 else "Wynik prawidłowy"
    return ScoreResult(score=score, interpretation=interpretation)
