# CGA Local App (Polish, local-first)

## Założenia
- Brak połączeń cloud/telemetrii.
- Narzędzie wspierające klinicystę, nie autonomiczna diagnoza.
- Domyślne moduły scoringowe: GDS-15 (opcjonalnie GDS-30), SARC-F, G8, CDT (z oceną kliniczną).

## Architektura (skrót)
- **Backend:** FastAPI + SQLAlchemy + SQLite + Alembic
- **Frontend:** React + TypeScript + Vite
- **Auth:** lokalne konta, hashowanie haseł
- **Export:** JSON + PDF (ReportLab)

## Faza 1 — skrócona mapa
1. **Patient Registry:** pacjenci, opiekun, historia wizyt.
2. **CGA Core:** domeny CGA zapisane jako modułowe payloady.
3. **GDS/SARC-F/G8/CDT:** oddzielne moduły, niezależna logika scoringu.
4. **Summary:** agregacja wyników, export JSON/PDF.
5. **Senior UX:** duże elementy, wysoki kontrast, prosty flow.

## Schemat danych
- `users` (role: admin/clinician/reviewer)
- `patients`
- `assessment_visits`
- `assessment_module_data` (payload + score + wersja reguł)
- `audit_logs`

## API (główne endpointy)
- `POST /api/auth/login`
- `GET /api/auth/users`
- `POST /api/patients`, `GET /api/patients`
- `POST /api/assessments/visits`
- `POST /api/assessments/visits/{visit_id}/modules`
- `GET /api/assessments/visits/{visit_id}/summary`
- `GET /api/assessments/visits/{visit_id}/summary.pdf`

## Konfigurowalne części
- Wariant GDS (`gds15` / `gds30`)
- Progi interpretacyjne przez warstwę `services/scoring.py`
- Schemat punktacji CDT (manualny, konfiguracja docelowo w tabeli reguł)

## Wymaga walidacji klinicznej
- Interpretacja końcowa raportu
- Ocena CDT
- Rekomendacje planu opieki

## Uruchomienie (Docker)
```bash
cd cga-local-app
docker compose up
```
Frontend: `http://localhost:5173`  
Backend docs: `http://localhost:8000/docs`

## Uruchomienie bez Dockera
### Backend
```bash
cd cga-local-app/backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd cga-local-app/frontend
npm install
npm run dev
```

## Dane demo
- Login: `admin`
- Hasło: `admin123`

## Testy
```bash
cd cga-local-app/backend
PYTHONPATH=. pytest app/tests -q
```

## Roadmap
- Pełne formularze CGA (ADL/IADL/MNA/MMSE/CFS/Barthel)
- Zaawansowane audit trail + import/export backup
- Tryb audio prompts i pełne i18n
- Rozdzielenie UI na pełny tryb pacjent/klinicysta z wizardem
