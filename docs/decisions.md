# Architecture Decisions

## ADR-001 — Project Architecture

### Decision

Use a layered architecture consisting of:

Data Pipeline
→ ML Models
→ FastAPI Backend
→ Database
→ Frontend Dashboard

### Reason

This separates data processing, machine learning,
backend services, storage and user interface.

---

## ADR-002 — Database

### Decision

Start with SQLite.

### Reason

SQLite is appropriate for the course-project scale.
Migration to PostgreSQL can be considered later if required.

---

## ADR-003 — Version Control

### Decision

Use Git and GitHub.

### Reason

Git provides version history and allows the team
to track development progress through commits and branches.