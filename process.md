# Pipeline Architecture: Temporal Knowledge Graph & Portfolio Generation

## 1. System Overview
**Goal:** Automate the transformation of unstructured research documents into thematic investment portfolios via a Temporal Knowledge Graph (TKG).
**Core Constraint:** Traceability of data lineage (Raw $\to$ Clean) without excessive storage redundancy.

---

## 2. Data Schema & Storage Strategy
*To satisfy the "do not double volume" requirement, we utilize a Reference-Based Schema.*

* **Document Store:** `S3/MinIO` (Raw PDFs/HTML)
* **Graph Database:** `Neo4j` or `ArangoDB` (Stores nodes/edges)
    * *Strategy:* "Clean" entities are not new nodes. They are **Super-Nodes** or linked via `SAME_AS` edges to the Raw nodes. The "Clean View" is a query projection, not a full data copy.
* **Metadata Store:** `Postgres` (Run logs, LLM configs, Evaluation metrics)

---

## 3. Pipeline Stages (DAG)

### Stage I: Ingestion
**Function:** `ingest_documents()`
* **Input:** External API / Watch folder
* **Process:** * Detect new files.
    * Generate content hash (SHA-256) to prevent re-processing.
    * Extract text chunks.
* **Output:** `DocChunks` (Text + Source Metadata)

### Stage II: Extraction (LLM Processing)
**Function:** `extract_tkg_triples(doc_chunks)`
* **Process:**
    * Prompt LLM for triples `(Subject, Predicate, Object, Timestamp)`.
    * **Metadata Capture:** Log `model_version` (e.g., 'gpt-4-0613'), `prompt_hash`, and `generation_timestamp`.
* **Output:** `RawTriples`
* **Storage:** Write to GraphDB with property `status='raw'`.

### Stage III: Entity Resolution (Cleaning)
**Function:** `disambiguate_entities(raw_triples)`
* **Process:**
    * Fetch new unique entities.
    * Apply clustering/fuzzy matching algorithms against the existing `EntityDictionary`.
    * **Optimization:** Do not duplicate edges. Create a `CanonicalEntity` node and link `RawEntity` nodes to it via `[:MAPS_TO]` edges.
* **Output:** Updated `EntityDictionary` (Mapping Table).

### Stage IV: Portfolio Construction
**Function:** `generate_portfolios(clean_graph_view)`
* **Process:**
    * Project a "Clean Graph" (querying only Canonical Entities).
    * Run Graph Algorithms: `ShortestPath`, `CentralityRank`, `CommunityDetection`.
    * Apply financial logic to group entities into Themes.
* **Output:** `ThematicPortfolios` (JSON/Parquet)

### Stage V: Evaluation Suite
**Function:** `evaluate_portfolios(portfolios)`
* **Process:**
    * Calculate metrics: `SharpeRatio`, `Volatility`, `ThemeCoherenceScore` (Semantic density).
    * Compare against historical benchmarks.
* **Output:** `QualityReport` (Pass/Fail boolean + Metrics)

### Stage VI: Conditional Actions
**Function:** `handle_evaluation_result(report)`
* **Logic:**
    * `IF report.passed`: Push to Production DB / Dashboard.
    * `IF report.failed`: 
        * Trigger Alert (Slack/PagerDuty).
        * Trigger `adjust_parameters()` or fallback routine.

---

## 4. Metadata & Lineage Requirements
Every artifact must carry a `RunContext` object:
```json
{
  "run_id": "uuid-1234",
  "trigger": "scheduled",
  "llm_config": {
    "model": "gpt-4",
    "temperature": 0.2
  },
  "upstream_hash": "hash_of_input_docs"
}
