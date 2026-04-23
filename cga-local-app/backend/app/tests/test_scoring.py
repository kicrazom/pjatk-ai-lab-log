from app.services.scoring import calculate_g8, calculate_gds, calculate_sarc_f


def test_gds15_threshold() -> None:
    result = calculate_gds({"answers": [True, True, True, True, True, True]}, "gds15")
    assert result.score == 6
    assert "Podejrzenie" in result.interpretation


def test_sarc_f_positive() -> None:
    payload = {"strength": 2, "assistance": 1, "rise": 1, "climb": 0, "falls": 1}
    result = calculate_sarc_f(payload)
    assert result.score == 5
    assert result.interpretation == "Screen positive"


def test_g8_abnormal() -> None:
    result = calculate_g8({"items": [1, 1, 1, 1, 2, 2, 2, 2]})
    assert result.score == 12
    assert result.interpretation == "Nieprawidłowy przesiew"
