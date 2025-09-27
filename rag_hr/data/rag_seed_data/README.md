# RAG Seed Dataset for HR/Employee Services

This bundle contains synthetic, self-consistent documents and tables for testing a Retrieval-Augmented Generation (RAG) chatbot.

## Structure
- `policies/`: Governing documents (leave, attendance, expense, travel, etc.).
- `hr/`: SLA, FAQ, glossary, and report definitions.
- `system/`: Roles, access rights, approval chains, and workflows.
- `data/`: Operational data (employees x100, managers, projects, assignments, attendance, requests, performance metrics, finance).

## Hints for Retrieval
- Leave rules and validations: `policies/leave_policy.md`
- Attendance/late rules: `policies/attendance_policy.md`
- Expense thresholds and director approvals: `policies/expense_policy.md`, `system/approval_chains.csv`
- SLA/escations: `hr/sla.md`
- Who can access what: `system/access_rights.csv`
- Approval levels per request type: `system/approval_chains.csv`
- Who reports to whom (for routing): `data/employees.csv` (`manager_id`) and `data/managers.csv`
- Manager/team context: `data/project_assignments.csv`
- Project health, budget: `data/projects.csv`, `data/finance_sample.csv`

## Suggested Indexing
- Chunk Markdown policies by headings.
- Treat CSVs as tables and enable semantic column mapping (e.g., `amount`, `status`, `manager_id`).
- Add metadata like `doc_type=policy/system/data`, `version`, and `effective_date` where available.
