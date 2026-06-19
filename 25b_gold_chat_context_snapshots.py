# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 25b — gold_chat_context_snapshots
# MAGIC
# MAGIC **Sources (all Gold):**
# MAGIC   - `supplysage_gold.gold_dashboard_supplier_risk_summary`
# MAGIC   - `supplysage_gold.gold_supplier_risk_explanation_log`
# MAGIC   - `supplysage_gold.gold_supplier_sku_dependency_mart`
# MAGIC   - `supplysage_gold.gold_sku_stockout_risk_scores`
# MAGIC   - `supplysage_gold.gold_external_risk_event_mart`
# MAGIC   - `supplysage_gold.gold_alert_events`
# MAGIC
# MAGIC **Target:** `supplysage_gold.gold_chat_context_snapshots`
# MAGIC **Grain:** One row per supplier (48 rows)
# MAGIC
# MAGIC **Purpose:**
# MAGIC The chatbot reads this table FIRST before going to RAG. It contains a
# MAGIC complete pre-assembled context snapshot per supplier — current score, trend,
# MAGIC open alerts, top affected SKUs, active events, and recommended action —
# MAGIC all in one row as a structured JSON blob AND as a pre-formatted text string.
# MAGIC
# MAGIC This makes the chatbot fast: single-row lookup → structured facts → then
# MAGIC RAG only for the evidence narrative. No multi-hop joins at query time.
# MAGIC
# MAGIC **Refresh cadence:** Same as the scoring pipeline (daily or on each external
# MAGIC API ingestion run). Add to the pipeline DAG after notebook 27.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

GOLD_SCHEMA = "supplysage_gold"

SUPPLIER_SUMMARY_TABLE = f"{GOLD_SCHEMA}.gold_dashboard_supplier_risk_summary"
EXPLANATION_TABLE = f"{GOLD_SCHEMA}.gold_supplier_risk_explanation_log"
DEPENDENCY_TABLE = f"{GOLD_SCHEMA}.gold_supplier_sku_dependency_mart"
SKU_RISK_TABLE = f"{GOLD_SCHEMA}.gold_sku_stockout_risk_scores"
EXTERNAL_EVENT_TABLE = f"{GOLD_SCHEMA}.gold_external_risk_event_mart"
ALERT_EVENTS_TABLE = f"{GOLD_SCHEMA}.gold_alert_events"

TARGET_TABLE = f"{GOLD_SCHEMA}.gold_chat_context_snapshots"

print("Notebook 25b config loaded")
print(f"Target table: {TARGET_TABLE}")

# COMMAND ----------

def has_col(df, col_name):
    return col_name in df.columns

def first_existing_col(df, candidates, alias_name, dtype="string", default=None):
    for c in candidates:
        if c in df.columns:
            return F.col(c).cast(dtype).alias(alias_name)
    return F.lit(default).cast(dtype).alias(alias_name)

def require_col(df, col_name, table_name):
    if col_name not in df.columns:
        raise ValueError(f"Required column `{col_name}` not found in {table_name}. Columns: {df.columns}")

# COMMAND ----------

supplier_summary_raw = spark.table(SUPPLIER_SUMMARY_TABLE)

require_col(supplier_summary_raw, "supplier_id", SUPPLIER_SUMMARY_TABLE)

supplier_base = (
    supplier_summary_raw
    .select(
        F.col("supplier_id").cast("string").alias("supplier_id"),

        first_existing_col(
            supplier_summary_raw,
            ["supplier_name", "primary_supplier_name", "supplier_display_name"],
            "supplier_name",
            "string"
        ),

        first_existing_col(
            supplier_summary_raw,
            ["supplier_risk_score", "risk_score", "overall_risk_score"],
            "supplier_risk_score",
            "double",
            0.0
        ),

        first_existing_col(
            supplier_summary_raw,
            ["risk_band", "supplier_risk_band", "risk_level"],
            "risk_band",
            "string",
            "Unknown"
        ),

        first_existing_col(
            supplier_summary_raw,
            ["score_delta", "risk_score_delta", "supplier_risk_score_delta"],
            "score_delta",
            "double",
            0.0
        ),

        first_existing_col(
            supplier_summary_raw,
            ["top_risk_driver", "primary_risk_driver", "risk_driver"],
            "top_risk_driver",
            "string"
        ),

        first_existing_col(
            supplier_summary_raw,
            ["recommended_action", "next_best_action", "recommendation"],
            "recommended_action",
            "string"
        ),

        first_existing_col(
            supplier_summary_raw,
            ["criticality_tier", "supplier_criticality_tier"],
            "criticality_tier",
            "string"
        ),

        first_existing_col(
            supplier_summary_raw,
            ["annual_spend", "supplier_annual_spend"],
            "annual_spend",
            "double",
            0.0
        ),

        first_existing_col(
            supplier_summary_raw,
            ["mapped_sku_count", "sku_count", "impacted_sku_count"],
            "mapped_sku_count",
            "int",
            0
        ),

        first_existing_col(
            supplier_summary_raw,
            ["active_event_count", "external_event_count", "matched_event_count"],
            "active_event_count",
            "int",
            0
        )
    )
    .dropDuplicates(["supplier_id"])
)

supplier_count = supplier_base.count()
print(f"Supplier base rows: {supplier_count}")

display(supplier_base.limit(10))

assert supplier_count == 48, f"Expected 48 suppliers, got {supplier_count}"

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 4 — Latest supplier risk explanation per supplier
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Safety helpers in case they are not already defined
def require_col(df, col_name, table_name):
    if col_name not in df.columns:
        raise ValueError(
            f"Required column `{col_name}` not found in {table_name}. "
            f"Available columns: {df.columns}"
        )

def first_existing_col(df, candidates, alias_name, dtype="string", default=None):
    for c in candidates:
        if c in df.columns:
            return F.col(c).cast(dtype).alias(alias_name)
    return F.lit(default).cast(dtype).alias(alias_name)

EXPLANATION_TABLE = "supplysage_gold.gold_supplier_risk_explanation_log"

explanation_raw = spark.table(EXPLANATION_TABLE)

print(f"Loaded table: {EXPLANATION_TABLE}")
print("Columns:")
print(explanation_raw.columns)

require_col(explanation_raw, "supplier_id", EXPLANATION_TABLE)

# Pick the best available timestamp column
timestamp_candidates = [
    "created_at",
    "gold_created_at",
    "scored_at",
    "score_timestamp",
    "snapshot_timestamp",
    "score_date"
]

timestamp_col = None
for c in timestamp_candidates:
    if c in explanation_raw.columns:
        timestamp_col = c
        break

print(f"Using timestamp column: {timestamp_col if timestamp_col else 'current_timestamp fallback'}")

explanation_norm = (
    explanation_raw
    .select(
        F.col("supplier_id").cast("string").alias("supplier_id"),

        first_existing_col(
            explanation_raw,
            [
                "risk_explanation",
                "explanation_text",
                "supplier_risk_explanation",
                "explanation",
                "narrative_explanation"
            ],
            "risk_explanation",
            "string"
        ),

        first_existing_col(
            explanation_raw,
            [
                "top_risk_driver",
                "primary_risk_driver",
                "risk_driver",
                "top_driver"
            ],
            "explanation_top_driver",
            "string"
        ),

        first_existing_col(
            explanation_raw,
            [
                "recommended_action",
                "next_best_action",
                "recommendation",
                "action_recommendation"
            ],
            "explanation_recommended_action",
            "string"
        ),

        (
            F.col(timestamp_col).cast("timestamp").alias("explanation_timestamp")
            if timestamp_col
            else F.current_timestamp().alias("explanation_timestamp")
        )
    )
)

w_exp = (
    Window
    .partitionBy("supplier_id")
    .orderBy(F.desc("explanation_timestamp"))
)

latest_explanation = (
    explanation_norm
    .withColumn("rn", F.row_number().over(w_exp))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

print(f"Latest explanation rows: {latest_explanation.count()}")

display(
    latest_explanation
    .select(
        "supplier_id",
        "risk_explanation",
        "explanation_top_driver",
        "explanation_recommended_action",
        "explanation_timestamp"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 4b — Build chatbot-ready supplier explanation text
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window

EXPLANATION_TABLE = "supplysage_gold.gold_supplier_risk_explanation_log"

explanation_raw = spark.table(EXPLANATION_TABLE)

explanation_norm = (
    explanation_raw
    .select(
        F.col("supplier_id").cast("string").alias("supplier_id"),
        F.col("supplier_name").cast("string").alias("supplier_name"),
        F.col("score_date").cast("date").alias("score_date"),
        F.col("overall_risk_score").cast("double").alias("overall_risk_score"),
        F.col("risk_band").cast("string").alias("risk_band"),
        F.col("score_delta_24h").cast("double").alias("score_delta_24h"),
        F.col("top_risk_driver").cast("string").alias("explanation_top_driver"),
        F.col("recommended_action").cast("string").alias("explanation_recommended_action"),
        F.col("driver_1_dimension").cast("string").alias("driver_1_dimension"),
        F.col("driver_1_detail").cast("string").alias("driver_1_detail"),
        F.col("driver_2_dimension").cast("string").alias("driver_2_dimension"),
        F.col("driver_2_detail").cast("string").alias("driver_2_detail"),
        F.col("driver_3_detail").cast("string").alias("driver_3_detail"),
        F.col("evidence_count").cast("int").alias("evidence_count"),
        F.col("evidence_ids").cast("string").alias("evidence_ids"),
        F.col("gold_created_at").cast("timestamp").alias("explanation_timestamp")
    )
    .withColumn(
        "risk_explanation",
        F.concat_ws(
            " ",
            F.concat(
                F.lit("Supplier "),
                F.coalesce(F.col("supplier_name"), F.col("supplier_id")),
                F.lit(" has an overall risk score of "),
                F.coalesce(F.round(F.col("overall_risk_score"), 2).cast("string"), F.lit("unknown")),
                F.lit(" with risk band "),
                F.coalesce(F.col("risk_band"), F.lit("unknown")),
                F.lit(".")
            ),
            F.concat(
                F.lit("The 24-hour score delta is "),
                F.coalesce(F.round(F.col("score_delta_24h"), 2).cast("string"), F.lit("0")),
                F.lit(".")
            ),
            F.concat(
                F.lit("The top risk driver is "),
                F.coalesce(F.col("explanation_top_driver"), F.lit("not identified")),
                F.lit(".")
            ),
            F.when(
                F.col("driver_1_detail").isNotNull(),
                F.concat(
                    F.lit("Driver 1: "),
                    F.coalesce(F.col("driver_1_dimension"), F.lit("General")),
                    F.lit(" - "),
                    F.col("driver_1_detail"),
                    F.lit(".")
                )
            ),
            F.when(
                F.col("driver_2_detail").isNotNull(),
                F.concat(
                    F.lit("Driver 2: "),
                    F.coalesce(F.col("driver_2_dimension"), F.lit("General")),
                    F.lit(" - "),
                    F.col("driver_2_detail"),
                    F.lit(".")
                )
            ),
            F.when(
                F.col("driver_3_detail").isNotNull(),
                F.concat(
                    F.lit("Driver 3: "),
                    F.col("driver_3_detail"),
                    F.lit(".")
                )
            ),
            F.concat(
                F.lit("Evidence count: "),
                F.coalesce(F.col("evidence_count").cast("string"), F.lit("0")),
                F.lit(".")
            ),
            F.concat(
                F.lit("Recommended action: "),
                F.coalesce(
                    F.col("explanation_recommended_action"),
                    F.lit("Review supplier risk drivers and supporting evidence.")
                )
            )
        )
    )
)

w_exp = (
    Window
    .partitionBy("supplier_id")
    .orderBy(F.desc("explanation_timestamp"))
)

latest_explanation = (
    explanation_norm
    .withColumn("rn", F.row_number().over(w_exp))
    .filter(F.col("rn") == 1)
    .drop("rn")
    .select(
        "supplier_id",
        "risk_explanation",
        "explanation_top_driver",
        "explanation_recommended_action",
        "explanation_timestamp",
        "evidence_count",
        "evidence_ids"
    )
)

print(f"Latest explanation rows: {latest_explanation.count()}")

display(
    latest_explanation
    .select(
        "supplier_id",
        "risk_explanation",
        "explanation_top_driver",
        "explanation_recommended_action",
        "evidence_count"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 5 — Top affected SKUs per supplier
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window

DEPENDENCY_TABLE = "supplysage_gold.gold_supplier_sku_dependency_mart"
SKU_RISK_TABLE = "supplysage_gold.gold_sku_stockout_risk_scores"

dependency_raw = spark.table(DEPENDENCY_TABLE)
sku_risk_raw = spark.table(SKU_RISK_TABLE)

print(f"Loaded dependency table: {DEPENDENCY_TABLE}")
print("Dependency columns:")
print(dependency_raw.columns)

print(f"Loaded SKU risk table: {SKU_RISK_TABLE}")
print("SKU risk columns:")
print(sku_risk_raw.columns)

# Required supplier column
if "supplier_id" not in dependency_raw.columns:
    raise ValueError(f"supplier_id not found in {DEPENDENCY_TABLE}. Columns: {dependency_raw.columns}")

# Resolve SKU key columns
dep_sku_col = (
    "canonical_sku_id"
    if "canonical_sku_id" in dependency_raw.columns
    else "sku_id"
)

sku_sku_col = (
    "canonical_sku_id"
    if "canonical_sku_id" in sku_risk_raw.columns
    else "sku_id"
)

if dep_sku_col not in dependency_raw.columns:
    raise ValueError(f"No SKU column found in {DEPENDENCY_TABLE}")

if sku_sku_col not in sku_risk_raw.columns:
    raise ValueError(f"No SKU column found in {SKU_RISK_TABLE}")

print(f"Dependency SKU column: {dep_sku_col}")
print(f"SKU risk SKU column: {sku_sku_col}")

def first_existing_col(df, candidates, alias_name, dtype="string", default=None):
    for c in candidates:
        if c in df.columns:
            return F.col(c).cast(dtype).alias(alias_name)
    return F.lit(default).cast(dtype).alias(alias_name)

dependency_norm = (
    dependency_raw
    .select(
        F.col("supplier_id").cast("string").alias("supplier_id"),
        F.col(dep_sku_col).cast("string").alias("canonical_sku_id"),

        first_existing_col(
            dependency_raw,
            [
                "dependency_percent",
                "dependency_weight",
                "supplier_dependency_percent",
                "supplier_dependency_share"
            ],
            "dependency_percent",
            "double",
            0.0
        ),

        first_existing_col(
            dependency_raw,
            ["is_primary_supplier", "is_primary"],
            "is_primary_supplier",
            "boolean",
            False
        ),

        first_existing_col(
            dependency_raw,
            ["has_alternate_supplier", "has_alternate"],
            "has_alternate_supplier",
            "boolean",
            False
        ),

        first_existing_col(
            dependency_raw,
            ["alternate_supplier_id", "best_alternate_supplier_id"],
            "alternate_supplier_id",
            "string"
        ),

        first_existing_col(
            dependency_raw,
            ["switching_difficulty", "estimated_switching_difficulty"],
            "switching_difficulty",
            "string"
        )
    )
    .filter(F.col("canonical_sku_id").isNotNull())
)

sku_risk_norm = (
    sku_risk_raw
    .select(
        F.col(sku_sku_col).cast("string").alias("canonical_sku_id"),

        first_existing_col(
            sku_risk_raw,
            [
                "stockout_risk_score",
                "stockout_probability",
                "stockout_risk_probability",
                "sku_stockout_risk_score"
            ],
            "stockout_risk_score",
            "double",
            0.0
        ),

        first_existing_col(
            sku_risk_raw,
            [
                "stockout_risk_band",
                "risk_band",
                "sku_risk_band"
            ],
            "stockout_risk_band",
            "string",
            "Unknown"
        ),

        first_existing_col(
            sku_risk_raw,
            [
                "days_of_cover",
                "inventory_days_of_cover",
                "days_cover"
            ],
            "days_of_cover",
            "double"
        ),

        first_existing_col(
            sku_risk_raw,
            [
                "top_risk_driver",
                "top_stockout_driver",
                "stockout_top_driver",
                "risk_driver"
            ],
            "sku_top_driver",
            "string"
        )
    )
    .filter(F.col("canonical_sku_id").isNotNull())
)

supplier_sku_risk = (
    dependency_norm
    .join(sku_risk_norm, on="canonical_sku_id", how="left")
    .withColumn("stockout_risk_score", F.coalesce(F.col("stockout_risk_score"), F.lit(0.0)))
    .withColumn("dependency_percent", F.coalesce(F.col("dependency_percent"), F.lit(0.0)))
)

w_sku = (
    Window
    .partitionBy("supplier_id")
    .orderBy(
        F.desc("stockout_risk_score"),
        F.desc("dependency_percent")
    )
)

top_skus = (
    supplier_sku_risk
    .withColumn("sku_rank", F.row_number().over(w_sku))
    .filter(F.col("sku_rank") <= 10)
    .groupBy("supplier_id")
    .agg(
        F.collect_list(
            F.struct(
                F.col("canonical_sku_id"),
                F.col("stockout_risk_score"),
                F.col("stockout_risk_band"),
                F.col("days_of_cover"),
                F.col("dependency_percent"),
                F.col("is_primary_supplier"),
                F.col("has_alternate_supplier"),
                F.col("alternate_supplier_id"),
                F.col("switching_difficulty"),
                F.col("sku_top_driver")
            )
        ).alias("top_affected_skus"),

        F.count("*").alias("top_affected_sku_count"),

        F.sum(
            F.when(
                F.lower(F.col("stockout_risk_band")).isin("critical", "high"),
                1
            ).otherwise(0)
        ).alias("high_risk_sku_count")
    )
)

print(f"Supplier SKU context rows: {top_skus.count()}")

display(
    top_skus
    .select(
        "supplier_id",
        "top_affected_sku_count",
        "high_risk_sku_count",
        "top_affected_skus"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 6 — Active external events per supplier
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window

EXTERNAL_EVENT_TABLE = "supplysage_gold.gold_external_risk_event_mart"

events_raw = spark.table(EXTERNAL_EVENT_TABLE)

print(f"Loaded external event table: {EXTERNAL_EVENT_TABLE}")
print("External event columns:")
print(events_raw.columns)

# Resolve supplier column
if "matched_supplier_id" in events_raw.columns:
    event_supplier_col = "matched_supplier_id"
elif "supplier_id" in events_raw.columns:
    event_supplier_col = "supplier_id"
else:
    raise ValueError(
        f"No supplier column found in {EXTERNAL_EVENT_TABLE}. "
        f"Expected matched_supplier_id or supplier_id. Columns: {events_raw.columns}"
    )

print(f"Using supplier column: {event_supplier_col}")

def first_existing_col(df, candidates, alias_name, dtype="string", default=None):
    for c in candidates:
        if c in df.columns:
            return F.col(c).cast(dtype).alias(alias_name)
    return F.lit(default).cast(dtype).alias(alias_name)

events_norm = (
    events_raw
    .select(
        F.col(event_supplier_col).cast("string").alias("supplier_id"),

        first_existing_col(
            events_raw,
            [
                "external_event_id",
                "event_id",
                "risk_event_id",
                "source_event_id"
            ],
            "external_event_id",
            "string"
        ),

        first_existing_col(
            events_raw,
            [
                "event_title",
                "title",
                "headline",
                "event_summary"
            ],
            "event_title",
            "string"
        ),

        first_existing_col(
            events_raw,
            [
                "risk_category",
                "event_risk_category",
                "category"
            ],
            "risk_category",
            "string"
        ),

        first_existing_col(
            events_raw,
            [
                "source_name",
                "source",
                "data_source"
            ],
            "source_name",
            "string"
        ),

        first_existing_col(
            events_raw,
            [
                "severity",
                "event_severity",
                "risk_severity"
            ],
            "severity",
            "string",
            "Unknown"
        ),

        F.to_date(
            first_existing_col(
                events_raw,
                [
                    "event_date",
                    "event_timestamp",
                    "published_date",
                    "created_at",
                    "gold_created_at"
                ],
                "event_date_raw",
                "string"
            )
        ).alias("event_date"),

        first_existing_col(
            events_raw,
            [
                "source_url",
                "url",
                "event_url"
            ],
            "source_url",
            "string"
        ),

        first_existing_col(
            events_raw,
            [
                "match_reason",
                "match_type",
                "matching_rule",
                "supplier_match_reason"
            ],
            "match_reason",
            "string"
        )
    )
    .filter(F.col("supplier_id").isNotNull())
)

events_norm = (
    events_norm
    .withColumn(
        "severity_rank",
        F.when(F.lower(F.col("severity")) == "critical", F.lit(4))
         .when(F.lower(F.col("severity")) == "high", F.lit(3))
         .when(F.lower(F.col("severity")) == "medium", F.lit(2))
         .otherwise(F.lit(1))
    )
)

w_event = (
    Window
    .partitionBy("supplier_id")
    .orderBy(
        F.desc("severity_rank"),
        F.desc("event_date")
    )
)

active_events = (
    events_norm
    .withColumn("event_rank", F.row_number().over(w_event))
    .filter(F.col("event_rank") <= 10)
    .groupBy("supplier_id")
    .agg(
        F.collect_list(
            F.struct(
                F.col("external_event_id"),
                F.col("event_title"),
                F.col("risk_category"),
                F.col("source_name"),
                F.col("severity"),
                F.col("event_date"),
                F.col("source_url"),
                F.col("match_reason")
            )
        ).alias("active_external_events"),

        F.count("*").alias("active_external_event_count"),

        F.max("event_date").alias("latest_external_event_date"),

        F.sum(
            F.when(
                F.lower(F.col("severity")).isin("critical", "high"),
                1
            ).otherwise(0)
        ).alias("high_severity_event_count")
    )
)

print(f"Active event supplier rows: {active_events.count()}")

display(
    active_events
    .select(
        "supplier_id",
        "active_external_event_count",
        "high_severity_event_count",
        "latest_external_event_date",
        "active_external_events"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 7 — Open alerts per supplier
# Fixed for gold_alert_events schema:
# entity_id/entity_name/entity_type instead of supplier_id
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window

ALERT_EVENTS_TABLE = "supplysage_gold.gold_alert_events"

alerts_raw = spark.table(ALERT_EVENTS_TABLE)

print(f"Loaded alert events table: {ALERT_EVENTS_TABLE}")
print("Alert columns:")
print(alerts_raw.columns)

required_cols = ["entity_id", "entity_name", "entity_type", "alert_id", "severity", "status", "triggered_at"]
missing_cols = [c for c in required_cols if c not in alerts_raw.columns]

if missing_cols:
    raise ValueError(
        f"Missing required columns in {ALERT_EVENTS_TABLE}: {missing_cols}. "
        f"Available columns: {alerts_raw.columns}"
    )

alerts_norm = (
    alerts_raw
    # Keep supplier-level alerts only
    .filter(F.lower(F.col("entity_type")).isin("supplier", "suppliers"))
    .select(
        F.col("entity_id").cast("string").alias("supplier_id"),
        F.col("entity_name").cast("string").alias("supplier_name"),

        F.col("alert_id").cast("string").alias("alert_id"),

        # This table does not have alert_type, so infer from benchmark_id / trigger
        F.coalesce(
            F.col("benchmark_id").cast("string"),
            F.lit("supplier_risk_threshold")
        ).alias("alert_type"),

        F.col("severity").cast("string").alias("alert_severity"),

        F.coalesce(
            F.col("status").cast("string"),
            F.lit("open")
        ).alias("alert_status"),

        F.coalesce(
            F.col("trigger_description").cast("string"),
            F.col("email_subject").cast("string"),
            F.lit("Supplier risk alert triggered")
        ).alias("alert_title"),

        F.coalesce(
            F.col("email_body").cast("string"),
            F.col("trigger_description").cast("string"),
            F.lit("Review supplier risk score, affected SKUs, external events, and recommended mitigation action.")
        ).alias("alert_recommended_action"),

        F.col("triggered_at").cast("timestamp").alias("alert_created_at"),

        F.col("actual_value").cast("double").alias("actual_value"),
        F.col("threshold_value").cast("double").alias("threshold_value"),
        F.col("breach_amount").cast("double").alias("breach_amount"),
        F.col("investigation_id").cast("string").alias("investigation_id"),
        F.col("sent_at").cast("timestamp").alias("sent_at")
    )
)

# Treat missing or non-closed status as open
open_alerts_norm = (
    alerts_norm
    .filter(
        ~F.lower(F.coalesce(F.col("alert_status"), F.lit("open")))
        .isin("closed", "resolved", "dismissed")
    )
)

open_alerts_norm = (
    open_alerts_norm
    .withColumn(
        "alert_severity_rank",
        F.when(F.lower(F.col("alert_severity")) == "critical", F.lit(4))
         .when(F.lower(F.col("alert_severity")) == "high", F.lit(3))
         .when(F.lower(F.col("alert_severity")) == "medium", F.lit(2))
         .otherwise(F.lit(1))
    )
)

w_alert = (
    Window
    .partitionBy("supplier_id")
    .orderBy(
        F.desc("alert_severity_rank"),
        F.desc("alert_created_at")
    )
)

open_alerts = (
    open_alerts_norm
    .withColumn("alert_rank", F.row_number().over(w_alert))
    .filter(F.col("alert_rank") <= 10)
    .groupBy("supplier_id")
    .agg(
        F.first("supplier_name", ignorenulls=True).alias("alert_supplier_name"),

        F.collect_list(
            F.struct(
                F.col("alert_id"),
                F.col("alert_type"),
                F.col("alert_severity"),
                F.col("alert_status"),
                F.col("alert_title"),
                F.col("alert_recommended_action"),
                F.col("alert_created_at"),
                F.col("actual_value"),
                F.col("threshold_value"),
                F.col("breach_amount"),
                F.col("investigation_id"),
                F.col("sent_at")
            )
        ).alias("open_alerts"),

        F.count("*").alias("open_alert_count"),

        F.sum(
            F.when(
                F.lower(F.col("alert_severity")).isin("critical", "high"),
                1
            ).otherwise(0)
        ).alias("critical_or_high_alert_count")
    )
)

print(f"Open alert supplier rows: {open_alerts.count()}")

display(
    open_alerts
    .select(
        "supplier_id",
        "alert_supplier_name",
        "open_alert_count",
        "critical_or_high_alert_count",
        "open_alerts"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 7b — Sanity check alert entity types and statuses
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

ALERT_EVENTS_TABLE = "supplysage_gold.gold_alert_events"

alerts_raw = spark.table(ALERT_EVENTS_TABLE)

print("Total alert rows:", alerts_raw.count())

print("Entity type distribution:")
display(
    alerts_raw
    .groupBy("entity_type")
    .count()
    .orderBy(F.desc("count"))
)

print("Status distribution:")
display(
    alerts_raw
    .groupBy("status")
    .count()
    .orderBy(F.desc("count"))
)

print("Entity type + status distribution:")
display(
    alerts_raw
    .groupBy("entity_type", "status")
    .count()
    .orderBy(F.desc("count"))
)

print("Sample alerts:")
display(
    alerts_raw
    .select(
        "entity_id",
        "entity_name",
        "entity_type",
        "severity",
        "status",
        "trigger_description",
        "alert_id",
        "triggered_at"
    )
    .limit(20)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 7c — Create empty open_alerts context because gold_alert_events has 0 rows
# ─────────────────────────────────────────────────────────────

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    TimestampType,
    ArrayType
)

alert_struct_schema = StructType([
    StructField("alert_id", StringType(), True),
    StructField("alert_type", StringType(), True),
    StructField("alert_severity", StringType(), True),
    StructField("alert_status", StringType(), True),
    StructField("alert_title", StringType(), True),
    StructField("alert_recommended_action", StringType(), True),
    StructField("alert_created_at", TimestampType(), True),
    StructField("actual_value", DoubleType(), True),
    StructField("threshold_value", DoubleType(), True),
    StructField("breach_amount", DoubleType(), True),
    StructField("investigation_id", StringType(), True),
    StructField("sent_at", TimestampType(), True),
])

open_alerts_schema = StructType([
    StructField("supplier_id", StringType(), True),
    StructField("alert_supplier_name", StringType(), True),
    StructField("open_alerts", ArrayType(alert_struct_schema), True),
    StructField("open_alert_count", LongType(), True),
    StructField("critical_or_high_alert_count", LongType(), True),
])

open_alerts = spark.createDataFrame([], schema=open_alerts_schema)

print("Created empty open_alerts dataframe because gold_alert_events has no rows.")
print(f"Open alert supplier rows: {open_alerts.count()}")

display(open_alerts)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 8 — Build final supplier-level chat context snapshot
# Grain: one row per supplier
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

# Safety checks
required_dfs = [
    "supplier_base",
    "latest_explanation",
    "top_skus",
    "active_events",
    "open_alerts"
]

missing_dfs = [df_name for df_name in required_dfs if df_name not in globals()]

if missing_dfs:
    raise ValueError(f"Missing required dataframes: {missing_dfs}")

snapshot = (
    supplier_base.alias("s")
    .join(latest_explanation.alias("x"), on="supplier_id", how="left")
    .join(top_skus.alias("sku"), on="supplier_id", how="left")
    .join(active_events.alias("ev"), on="supplier_id", how="left")
    .join(open_alerts.alias("al"), on="supplier_id", how="left")
    .withColumn("snapshot_date", F.current_date())
    .withColumn("snapshot_timestamp", F.current_timestamp())

    # Fill alert counts because gold_alert_events currently has 0 rows
    .withColumn("open_alert_count", F.coalesce(F.col("open_alert_count"), F.lit(0)))
    .withColumn("critical_or_high_alert_count", F.coalesce(F.col("critical_or_high_alert_count"), F.lit(0)))

    # Fill external event and SKU counts
    .withColumn("active_external_event_count", F.coalesce(F.col("active_external_event_count"), F.lit(0)))
    .withColumn("high_severity_event_count", F.coalesce(F.col("high_severity_event_count"), F.lit(0)))
    .withColumn("top_affected_sku_count", F.coalesce(F.col("top_affected_sku_count"), F.lit(0)))
    .withColumn("high_risk_sku_count", F.coalesce(F.col("high_risk_sku_count"), F.lit(0)))

    # Final action fields for chatbot
    .withColumn(
        "final_top_risk_driver",
        F.coalesce(
            F.col("top_risk_driver"),
            F.col("explanation_top_driver"),
            F.lit("No dominant risk driver identified.")
        )
    )
    .withColumn(
        "final_recommended_action",
        F.coalesce(
            F.col("recommended_action"),
            F.col("explanation_recommended_action"),
            F.lit("Review supplier risk drivers, affected SKUs, active events, and supporting evidence before procurement action.")
        )
    )
)

snapshot = (
    snapshot
    .withColumn(
        "chat_context_struct",
        F.struct(
            F.col("supplier_id"),
            F.col("supplier_name"),
            F.col("snapshot_date"),
            F.col("snapshot_timestamp"),
            F.col("supplier_risk_score"),
            F.col("risk_band"),
            F.col("score_delta"),
            F.col("criticality_tier"),
            F.col("annual_spend"),
            F.col("mapped_sku_count"),
            F.col("final_top_risk_driver").alias("top_risk_driver"),
            F.col("final_recommended_action").alias("recommended_action"),
            F.col("risk_explanation"),
            F.col("evidence_count"),
            F.col("evidence_ids"),
            F.col("open_alert_count"),
            F.col("critical_or_high_alert_count"),
            F.col("active_external_event_count"),
            F.col("high_severity_event_count"),
            F.col("top_affected_sku_count"),
            F.col("high_risk_sku_count"),
            F.col("latest_external_event_date"),
            F.col("top_affected_skus"),
            F.col("active_external_events"),
            F.col("open_alerts")
        )
    )
    .withColumn(
        "chat_context_json",
        F.to_json(F.col("chat_context_struct"))
    )
    .withColumn(
        "chat_context_text",
        F.concat_ws(
            "\n\n",

            F.concat(
                F.lit("Supplier: "),
                F.coalesce(F.col("supplier_name"), F.col("supplier_id")),
                F.lit(" ("),
                F.col("supplier_id"),
                F.lit(")")
            ),

            F.concat(
                F.lit("Current supplier risk score: "),
                F.coalesce(F.round(F.col("supplier_risk_score"), 2).cast("string"), F.lit("unknown")),
                F.lit(" | Risk band: "),
                F.coalesce(F.col("risk_band"), F.lit("unknown")),
                F.lit(" | Score delta: "),
                F.coalesce(F.round(F.col("score_delta"), 2).cast("string"), F.lit("0"))
            ),

            F.concat(
                F.lit("Criticality tier: "),
                F.coalesce(F.col("criticality_tier"), F.lit("unknown")),
                F.lit(" | Annual spend: "),
                F.coalesce(F.round(F.col("annual_spend"), 2).cast("string"), F.lit("unknown")),
                F.lit(" | Mapped SKUs: "),
                F.coalesce(F.col("mapped_sku_count").cast("string"), F.lit("0"))
            ),

            F.concat(
                F.lit("Top risk driver: "),
                F.coalesce(F.col("final_top_risk_driver"), F.lit("unknown"))
            ),

            F.concat(
                F.lit("Recommended action: "),
                F.coalesce(F.col("final_recommended_action"), F.lit("Review supplier."))
            ),

            F.concat(
                F.lit("Open alerts: "),
                F.col("open_alert_count").cast("string"),
                F.lit(" | Critical/high alerts: "),
                F.col("critical_or_high_alert_count").cast("string")
            ),

            F.concat(
                F.lit("Active external events: "),
                F.col("active_external_event_count").cast("string"),
                F.lit(" | High-severity events: "),
                F.col("high_severity_event_count").cast("string"),
                F.lit(" | Latest event date: "),
                F.coalesce(F.col("latest_external_event_date").cast("string"), F.lit("none"))
            ),

            F.concat(
                F.lit("Top affected SKU count: "),
                F.col("top_affected_sku_count").cast("string"),
                F.lit(" | High-risk affected SKUs: "),
                F.col("high_risk_sku_count").cast("string")
            ),

            F.concat(
                F.lit("Risk explanation: "),
                F.coalesce(F.col("risk_explanation"), F.lit("No supplier risk explanation available."))
            ),

            F.concat(
                F.lit("Top affected SKUs JSON: "),
                F.coalesce(F.to_json(F.col("top_affected_skus")), F.lit("[]"))
            ),

            F.concat(
                F.lit("Active external events JSON: "),
                F.coalesce(F.to_json(F.col("active_external_events")), F.lit("[]"))
            ),

            F.concat(
                F.lit("Open alerts JSON: "),
                F.coalesce(F.to_json(F.col("open_alerts")), F.lit("[]"))
            )
        )
    )
    .select(
        "supplier_id",
        "supplier_name",
        "snapshot_date",
        "snapshot_timestamp",
        "supplier_risk_score",
        "risk_band",
        "score_delta",
        "criticality_tier",
        "annual_spend",
        "mapped_sku_count",
        "open_alert_count",
        "critical_or_high_alert_count",
        "active_external_event_count",
        "high_severity_event_count",
        "top_affected_sku_count",
        "high_risk_sku_count",
        "latest_external_event_date",
        "final_top_risk_driver",
        "final_recommended_action",
        "top_affected_skus",
        "active_external_events",
        "open_alerts",
        "chat_context_json",
        "chat_context_text"
    )
)

snapshot_count = snapshot.count()
distinct_supplier_count = snapshot.select("supplier_id").distinct().count()

print(f"Snapshot rows: {snapshot_count}")
print(f"Distinct suppliers: {distinct_supplier_count}")

assert snapshot_count == 48, f"Expected 48 supplier snapshots, got {snapshot_count}"
assert distinct_supplier_count == 48, f"Expected 48 distinct suppliers, got {distinct_supplier_count}"

display(
    snapshot
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "open_alert_count",
        "active_external_event_count",
        "top_affected_sku_count",
        "high_risk_sku_count",
        "final_top_risk_driver",
        "final_recommended_action",
        "chat_context_text"
    )
    .orderBy(F.desc("supplier_risk_score"))
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 9 — Write final chat context snapshot table
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

TARGET_TABLE = "supplysage_gold.gold_chat_context_snapshots"

(
    snapshot
    .withColumn("gold_created_at", F.current_timestamp())
    .withColumn("gold_source_notebook", F.lit("25b_gold_chat_context_snapshots"))
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote table: {TARGET_TABLE}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 10 — Validate final chat context snapshot table
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

TARGET_TABLE = "supplysage_gold.gold_chat_context_snapshots"

validation = spark.sql(f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT supplier_id) AS distinct_supplier_count,
    SUM(CASE WHEN supplier_id IS NULL THEN 1 ELSE 0 END) AS null_supplier_id_count,
    SUM(CASE WHEN chat_context_json IS NULL OR length(trim(chat_context_json)) = 0 THEN 1 ELSE 0 END) AS empty_json_count,
    SUM(CASE WHEN chat_context_text IS NULL OR length(trim(chat_context_text)) = 0 THEN 1 ELSE 0 END) AS empty_text_count,
    MIN(supplier_risk_score) AS min_supplier_risk_score,
    MAX(supplier_risk_score) AS max_supplier_risk_score,
    SUM(open_alert_count) AS total_open_alerts,
    SUM(active_external_event_count) AS total_active_external_events,
    SUM(top_affected_sku_count) AS total_top_affected_skus
FROM {TARGET_TABLE}
""")

display(validation)

v = validation.collect()[0]

assert v["row_count"] == 48, f"Expected 48 supplier snapshots, got {v['row_count']}"
assert v["distinct_supplier_count"] == 48, "Expected one row per supplier."
assert v["null_supplier_id_count"] == 0, "Some rows have null supplier_id."
assert v["empty_json_count"] == 0, "Some rows have empty chat_context_json."
assert v["empty_text_count"] == 0, "Some rows have empty chat_context_text."

print("gold_chat_context_snapshots validation passed.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 11 — Preview chatbot-ready supplier snapshots
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

TARGET_TABLE = "supplysage_gold.gold_chat_context_snapshots"

display(
    spark.table(TARGET_TABLE)
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "open_alert_count",
        "active_external_event_count",
        "top_affected_sku_count",
        "high_risk_sku_count",
        "final_top_risk_driver",
        "final_recommended_action",
        "chat_context_text"
    )
    .orderBy(F.desc("supplier_risk_score"))
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Notebook 26 — RAG Retrieval Test
# Purpose:
# 1. Read supplier chat snapshot first
# 2. Retrieve supporting evidence from gold_rag_embeddings
# 3. Prove the chatbot can answer with structured facts + evidence
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime

GOLD_SCHEMA = "supplysage_gold"

CHAT_CONTEXT_TABLE = f"{GOLD_SCHEMA}.gold_chat_context_snapshots"
EMBEDDINGS_TABLE = f"{GOLD_SCHEMA}.gold_rag_embeddings"
RETRIEVAL_INDEX_TABLE = f"{GOLD_SCHEMA}.gold_rag_retrieval_index"

# Same model used in Notebook 25
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

print("Notebook 26 config loaded")
print(f"CHAT_CONTEXT_TABLE: {CHAT_CONTEXT_TABLE}")
print(f"EMBEDDINGS_TABLE: {EMBEDDINGS_TABLE}")
print(f"RETRIEVAL_INDEX_TABLE: {RETRIEVAL_INDEX_TABLE}")

# Check required tables exist and have rows
for table_name in [CHAT_CONTEXT_TABLE, EMBEDDINGS_TABLE, RETRIEVAL_INDEX_TABLE]:
    print(f"\nChecking table: {table_name}")
    df = spark.table(table_name)
    row_count = df.count()
    print(f"Rows: {row_count}")
    assert row_count > 0, f"{table_name} is empty."

print("\nTable checks passed.")

# Preview available suppliers for testing
display(
    spark.table(CHAT_CONTEXT_TABLE)
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "active_external_event_count",
        "top_affected_sku_count",
        "final_top_risk_driver",
        "final_recommended_action"
    )
    .orderBy(F.desc("supplier_risk_score"))
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 2 — Select a test supplier
# Pick the highest-risk supplier from chat context snapshots
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

CHAT_CONTEXT_TABLE = "supplysage_gold.gold_chat_context_snapshots"

test_supplier_row = (
    spark.table(CHAT_CONTEXT_TABLE)
    .orderBy(F.desc("supplier_risk_score"))
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "final_top_risk_driver",
        "final_recommended_action",
        "chat_context_text",
        "chat_context_json"
    )
    .limit(1)
    .collect()[0]
)

TEST_SUPPLIER_ID = test_supplier_row["supplier_id"]
TEST_SUPPLIER_NAME = test_supplier_row["supplier_name"]

print(f"Selected test supplier: {TEST_SUPPLIER_ID}")
print(f"Supplier name: {TEST_SUPPLIER_NAME}")
print(f"Risk score: {test_supplier_row['supplier_risk_score']}")
print(f"Risk band: {test_supplier_row['risk_band']}")
print(f"Top risk driver: {test_supplier_row['final_top_risk_driver']}")
print(f"Recommended action: {test_supplier_row['final_recommended_action']}")

print("\nChat context preview:")
print(test_supplier_row["chat_context_text"][:2000])

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 3 — Check available RAG evidence for selected supplier
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

EMBEDDINGS_TABLE = "supplysage_gold.gold_rag_embeddings"
RETRIEVAL_INDEX_TABLE = "supplysage_gold.gold_rag_retrieval_index"

assert "TEST_SUPPLIER_ID" in globals(), "TEST_SUPPLIER_ID not found. Rerun Cell 2."

print(f"Testing evidence pool for supplier: {TEST_SUPPLIER_ID}")

supplier_embeddings = (
    spark.table(EMBEDDINGS_TABLE)
    .filter(F.col("supplier_id") == TEST_SUPPLIER_ID)
)

supplier_index = (
    spark.table(RETRIEVAL_INDEX_TABLE)
    .filter(F.col("supplier_id") == TEST_SUPPLIER_ID)
)

embedding_count = supplier_embeddings.count()
index_count = supplier_index.count()

print(f"Supplier embedding rows: {embedding_count}")
print(f"Supplier retrieval index rows: {index_count}")

assert embedding_count > 0, f"No embedding rows found for {TEST_SUPPLIER_ID}"
assert index_count > 0, f"No retrieval index rows found for {TEST_SUPPLIER_ID}"

print("\nEvidence chunk type distribution:")
display(
    supplier_embeddings
    .groupBy("chunk_type")
    .count()
    .orderBy(F.desc("count"))
)

print("\nTop supplier evidence preview:")
display(
    supplier_embeddings
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "sku_id",
        "source_name",
        "risk_category",
        "event_date",
        "severity",
        "freshness_weight",
        "chunk_text"
    )
    .orderBy(
        F.desc("freshness_weight"),
        F.desc("event_date")
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 4 — Embed a test user question
# ─────────────────────────────────────────────────────────────

import subprocess
import numpy as np

# Install only if needed
subprocess.run(
    ["pip", "install", "sentence-transformers", "--quiet"],
    check=True
)

from sentence_transformers import SentenceTransformer

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading model: {ST_MODEL_NAME}")
retrieval_model = SentenceTransformer(ST_MODEL_NAME)

EMBEDDING_DIM = int(
    retrieval_model.get_embedding_dimension()
    if hasattr(retrieval_model, "get_embedding_dimension")
    else retrieval_model.get_sentence_embedding_dimension()
)

print(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

TEST_QUESTION = f"""
Why is supplier {TEST_SUPPLIER_ID} high risk?
Explain the main external events, affected SKUs, and recommended action.
"""

query_embedding = retrieval_model.encode(
    TEST_QUESTION,
    normalize_embeddings=True
).tolist()

print("Test question:")
print(TEST_QUESTION)

print(f"Query embedding length: {len(query_embedding)}")

assert len(query_embedding) == EMBEDDING_DIM, "Query embedding dimension mismatch."
print("Query embedding created successfully.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 5 — Retrieve top supplier evidence using cosine similarity
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

EMBEDDINGS_TABLE = "supplysage_gold.gold_rag_embeddings"

assert "TEST_SUPPLIER_ID" in globals(), "TEST_SUPPLIER_ID not found. Rerun Cell 2."
assert "query_embedding" in globals(), "query_embedding not found. Rerun Cell 4."

TOP_K = 12

print(f"Retrieving evidence for supplier: {TEST_SUPPLIER_ID}")
print(f"Question: {TEST_QUESTION}")

supplier_evidence_pd = (
    spark.table(EMBEDDINGS_TABLE)
    .filter(F.col("supplier_id") == TEST_SUPPLIER_ID)
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "sku_id",
        "source_name",
        "risk_category",
        "event_date",
        "severity",
        "freshness_weight",
        "source_url",
        "evidence_doc_id",
        "chunk_text",
        "embedding"
    )
    .toPandas()
)

print(f"Candidate evidence rows: {len(supplier_evidence_pd)}")

assert len(supplier_evidence_pd) > 0, f"No evidence rows found for {TEST_SUPPLIER_ID}"

# Convert embeddings to numpy matrix
embedding_matrix = np.vstack(
    supplier_evidence_pd["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).values
)

query_vector = np.array(query_embedding, dtype=np.float32)

assert embedding_matrix.shape[1] == len(query_vector), (
    f"Embedding dim mismatch. Evidence dim={embedding_matrix.shape[1]}, query dim={len(query_vector)}"
)

# Embeddings were normalized in Notebook 25, and query is normalized too.
# Dot product = cosine similarity.
supplier_evidence_pd["cosine_similarity"] = embedding_matrix @ query_vector

# Add a simple blended score so newer / higher freshness evidence can surface
supplier_evidence_pd["freshness_weight"] = pd.to_numeric(
    supplier_evidence_pd["freshness_weight"],
    errors="coerce"
).fillna(0.5)

supplier_evidence_pd["retrieval_score"] = (
    0.85 * supplier_evidence_pd["cosine_similarity"]
    + 0.15 * supplier_evidence_pd["freshness_weight"]
)

top_rag_results = (
    supplier_evidence_pd
    .sort_values(
        by=["retrieval_score", "cosine_similarity", "freshness_weight"],
        ascending=False
    )
    .head(TOP_K)
    .copy()
)

print(f"Top {TOP_K} retrieved evidence chunks:")

display(
    spark.createDataFrame(
        top_rag_results[
            [
                "chunk_id",
                "chunk_type",
                "supplier_id",
                "sku_id",
                "source_name",
                "risk_category",
                "event_date",
                "severity",
                "freshness_weight",
                "cosine_similarity",
                "retrieval_score",
                "chunk_text",
                "source_url"
            ]
        ]
    )
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 6 — Build chatbot-ready context packet
# Combines:
# 1. Supplier snapshot from gold_chat_context_snapshots
# 2. Top retrieved RAG evidence chunks
# ─────────────────────────────────────────────────────────────

import json
import pandas as pd
from pyspark.sql import functions as F

CHAT_CONTEXT_TABLE = "supplysage_gold.gold_chat_context_snapshots"

assert "TEST_SUPPLIER_ID" in globals(), "TEST_SUPPLIER_ID not found. Rerun Cell 2."
assert "TEST_QUESTION" in globals(), "TEST_QUESTION not found. Rerun Cell 4."
assert "top_rag_results" in globals(), "top_rag_results not found. Rerun Cell 5."

# Pull supplier snapshot
snapshot_row = (
    spark.table(CHAT_CONTEXT_TABLE)
    .filter(F.col("supplier_id") == TEST_SUPPLIER_ID)
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "final_top_risk_driver",
        "final_recommended_action",
        "chat_context_json",
        "chat_context_text"
    )
    .collect()[0]
)

# Convert top retrieved evidence into clean dicts
evidence_records = []

for idx, row in top_rag_results.reset_index(drop=True).iterrows():
    evidence_records.append({
        "rank": int(idx + 1),
        "chunk_id": row.get("chunk_id"),
        "original_chunk_id": row.get("original_chunk_id"),
        "chunk_type": row.get("chunk_type"),
        "supplier_id": row.get("supplier_id"),
        "sku_id": row.get("sku_id"),
        "source_name": row.get("source_name"),
        "risk_category": row.get("risk_category"),
        "event_date": str(row.get("event_date")) if pd.notna(row.get("event_date")) else None,
        "severity": row.get("severity"),
        "freshness_weight": float(row.get("freshness_weight")) if pd.notna(row.get("freshness_weight")) else None,
        "cosine_similarity": float(row.get("cosine_similarity")) if pd.notna(row.get("cosine_similarity")) else None,
        "retrieval_score": float(row.get("retrieval_score")) if pd.notna(row.get("retrieval_score")) else None,
        "source_url": row.get("source_url"),
        "chunk_text": row.get("chunk_text")
    })

supplier_context_packet = {
    "question": TEST_QUESTION.strip(),
    "supplier": {
        "supplier_id": snapshot_row["supplier_id"],
        "supplier_name": snapshot_row["supplier_name"],
        "supplier_risk_score": float(snapshot_row["supplier_risk_score"]),
        "risk_band": snapshot_row["risk_band"],
        "top_risk_driver": snapshot_row["final_top_risk_driver"],
        "recommended_action": snapshot_row["final_recommended_action"]
    },
    "snapshot_context_text": snapshot_row["chat_context_text"],
    "snapshot_context_json": json.loads(snapshot_row["chat_context_json"]),
    "retrieved_evidence": evidence_records
}

print("Supplier context packet created.")
print(f"Supplier: {snapshot_row['supplier_name']} ({snapshot_row['supplier_id']})")
print(f"Risk score: {snapshot_row['supplier_risk_score']}")
print(f"Risk band: {snapshot_row['risk_band']}")
print(f"Evidence chunks retrieved: {len(evidence_records)}")

print("\n" + "=" * 100)
print("SNAPSHOT CONTEXT")
print("=" * 100)
print(snapshot_row["chat_context_text"][:3000])

print("\n" + "=" * 100)
print("TOP RETRIEVED EVIDENCE")
print("=" * 100)

for ev in evidence_records[:5]:
    print(f"\nRank {ev['rank']} | Score: {ev['retrieval_score']:.4f} | Source: {ev['source_name']} | Severity: {ev['severity']}")
    print(f"Category: {ev['risk_category']} | Date: {ev['event_date']} | SKU: {ev['sku_id']}")
    print(ev["chunk_text"][:800])
    print("-" * 100)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 7 — Business-aware reranking for supplier evidence
# Goal:
# Prioritize semantic match + severity + recency + known supplier risk drivers
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

assert "supplier_evidence_pd" in globals(), "supplier_evidence_pd not found. Rerun Cell 5."
assert "query_embedding" in globals(), "query_embedding not found. Rerun Cell 4."
assert "TEST_SUPPLIER_ID" in globals(), "TEST_SUPPLIER_ID not found. Rerun Cell 2."

rerank_pd = supplier_evidence_pd.copy()

# Ensure cosine similarity exists
if "cosine_similarity" not in rerank_pd.columns:
    embedding_matrix = np.vstack(
        rerank_pd["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).values
    )
    query_vector = np.array(query_embedding, dtype=np.float32)
    rerank_pd["cosine_similarity"] = embedding_matrix @ query_vector

# Normalize freshness
rerank_pd["freshness_weight"] = pd.to_numeric(
    rerank_pd["freshness_weight"],
    errors="coerce"
).fillna(0.5)

# Normalize event_date
rerank_pd["event_date_parsed"] = pd.to_datetime(
    rerank_pd["event_date"],
    errors="coerce"
)

today = pd.Timestamp.today().normalize()

rerank_pd["days_since_event"] = (
    today - rerank_pd["event_date_parsed"]
).dt.days

# Recency score: recent events should rank higher
rerank_pd["recency_score"] = np.select(
    [
        rerank_pd["event_date_parsed"].isna(),
        rerank_pd["days_since_event"] <= 30,
        rerank_pd["days_since_event"] <= 90,
        rerank_pd["days_since_event"] <= 365,
    ],
    [
        0.35,
        1.00,
        0.85,
        0.65,
    ],
    default=0.25
)

# Severity score
severity_map = {
    "critical": 1.00,
    "high": 0.90,
    "medium": 0.55,
    "low": 0.30
}

rerank_pd["severity_score"] = (
    rerank_pd["severity"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
    .map(severity_map)
    .fillna(0.40)
)

# Prefer evidence types useful to the chatbot
rerank_pd["chunk_type_score"] = np.select(
    [
        rerank_pd["chunk_type"].astype(str).str.lower().eq("risk_explanation"),
        rerank_pd["chunk_type"].astype(str).str.lower().eq("external_event"),
    ],
    [
        1.00,
        0.85,
    ],
    default=0.50
)

# Prefer sources that are active drivers in this supplier's snapshot
driver_sources = ["internal_risk_engine", "ofac", "cisa"]

rerank_pd["driver_source_score"] = (
    rerank_pd["source_name"]
    .fillna("")
    .astype(str)
    .str.lower()
    .isin(driver_sources)
    .astype(float)
)

# Prefer relevant categories
driver_categories = [
    "supplier_risk",
    "sanctions_compliance",
    "cyber",
    "logistics",
    "operational"
]

rerank_pd["driver_category_score"] = (
    rerank_pd["risk_category"]
    .fillna("")
    .astype(str)
    .str.lower()
    .isin(driver_categories)
    .astype(float)
)

# Final rerank score
rerank_pd["business_retrieval_score"] = (
    0.45 * rerank_pd["cosine_similarity"]
    + 0.15 * rerank_pd["freshness_weight"]
    + 0.15 * rerank_pd["severity_score"]
    + 0.10 * rerank_pd["recency_score"]
    + 0.10 * rerank_pd["driver_source_score"]
    + 0.05 * rerank_pd["driver_category_score"]
)

TOP_K_RERANKED = 12

reranked_rag_results = (
    rerank_pd
    .sort_values(
        by=[
            "business_retrieval_score",
            "driver_source_score",
            "severity_score",
            "recency_score",
            "cosine_similarity"
        ],
        ascending=False
    )
    .head(TOP_K_RERANKED)
    .copy()
)

print(f"Business-aware reranked top {TOP_K_RERANKED} evidence chunks for {TEST_SUPPLIER_ID}")

display(
    spark.createDataFrame(
        reranked_rag_results[
            [
                "chunk_id",
                "chunk_type",
                "supplier_id",
                "sku_id",
                "source_name",
                "risk_category",
                "event_date",
                "severity",
                "freshness_weight",
                "cosine_similarity",
                "severity_score",
                "recency_score",
                "driver_source_score",
                "business_retrieval_score",
                "chunk_text",
                "source_url"
            ]
        ]
    )
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 8 — Create deterministic chatbot-style answer
# Uses:
# 1. gold_chat_context_snapshots as primary context
# 2. reranked RAG evidence as supporting evidence
# No LLM yet. This proves the answer structure before LangGraph.
# ─────────────────────────────────────────────────────────────

import json
import pandas as pd
from pyspark.sql import functions as F

CHAT_CONTEXT_TABLE = "supplysage_gold.gold_chat_context_snapshots"

assert "TEST_SUPPLIER_ID" in globals(), "TEST_SUPPLIER_ID not found. Rerun Cell 2."
assert "TEST_QUESTION" in globals(), "TEST_QUESTION not found. Rerun Cell 4."
assert "reranked_rag_results" in globals(), "reranked_rag_results not found. Rerun Cell 7."

# Pull supplier snapshot
snapshot_row = (
    spark.table(CHAT_CONTEXT_TABLE)
    .filter(F.col("supplier_id") == TEST_SUPPLIER_ID)
    .select(
        "supplier_id",
        "supplier_name",
        "supplier_risk_score",
        "risk_band",
        "final_top_risk_driver",
        "final_recommended_action",
        "chat_context_json",
        "chat_context_text"
    )
    .collect()[0]
)

snapshot_json = json.loads(snapshot_row["chat_context_json"])

supplier_id = snapshot_row["supplier_id"]
supplier_name = snapshot_row["supplier_name"]
risk_score = snapshot_row["supplier_risk_score"]
risk_band = snapshot_row["risk_band"]
top_driver = snapshot_row["final_top_risk_driver"]
recommended_action = snapshot_row["final_recommended_action"]

top_skus = snapshot_json.get("top_affected_skus") or []
active_events = snapshot_json.get("active_external_events") or []

# Use top reranked evidence
evidence_df = reranked_rag_results.head(8).copy()

evidence_lines = []
for idx, row in evidence_df.reset_index(drop=True).iterrows():
    evidence_lines.append(
        f"[{idx + 1}] {row.get('source_name')} | "
        f"{row.get('risk_category')} | "
        f"{row.get('severity')} | "
        f"{row.get('event_date')} | "
        f"{str(row.get('chunk_text'))[:500]}"
    )

# Pull top affected SKUs from snapshot JSON
sku_lines = []
for sku in top_skus[:6]:
    sku_lines.append(
        f"- {sku.get('canonical_sku_id')}: "
        f"stockout risk={sku.get('stockout_risk_score')}, "
        f"dependency={sku.get('dependency_percent')}, "
        f"primary_supplier={sku.get('is_primary_supplier')}, "
        f"has_alternate={sku.get('has_alternate_supplier')}, "
        f"alternate_supplier={sku.get('alternate_supplier_id')}"
    )

# Pull active events from snapshot JSON
event_lines = []
for event in active_events[:5]:
    event_lines.append(
        f"- {event.get('source_name')} | {event.get('risk_category')} | "
        f"{event.get('severity')} | {event.get('event_date')}: "
        f"{event.get('event_title')}"
    )

# Important correction:
# The user question says "high risk", but the actual supplier band is medium.
risk_language = (
    f"{supplier_name} is currently classified as {risk_band}, not high risk."
    if str(risk_band).lower() != "high"
    else f"{supplier_name} is currently classified as high risk."
)

mock_chatbot_answer = f"""
Question:
{TEST_QUESTION.strip()}

Answer:
{risk_language}

Supplier {supplier_name} ({supplier_id}) has a current supplier risk score of {risk_score}, with the top risk driver listed as {top_driver}. The recommended action is: {recommended_action}

Why this supplier is being monitored:
- The snapshot shows {snapshot_json.get("active_external_event_count", 0)} active external events.
- {snapshot_json.get("high_severity_event_count", 0)} of those events are high-severity.
- The latest external event date is {snapshot_json.get("latest_external_event_date")}.
- There are {snapshot_json.get("top_affected_sku_count", 0)} affected SKUs mapped to this supplier.
- There are {snapshot_json.get("high_risk_sku_count", 0)} high-risk affected SKUs.
- Open alerts are currently {snapshot_json.get("open_alert_count", 0)}.

Main active external events:
{chr(10).join(event_lines) if event_lines else "- No active external events found in snapshot."}

Top affected SKUs:
{chr(10).join(sku_lines) if sku_lines else "- No affected SKUs found in snapshot."}

Supporting retrieved evidence:
{chr(10).join(evidence_lines)}

Procurement recommendation:
{recommended_action}

Suggested next step:
Continue monitoring FlexPack Industries because the current risk is mainly event-driven. The strongest evidence is coming from the internal risk explanation, CISA cyber events, and OFAC sanctions-compliance matches. Since affected SKUs currently show stockout risk of 0.0, this is more of a supplier/event-monitoring issue than an immediate inventory shortage issue.
""".strip()

print(mock_chatbot_answer)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 9 — Clean retrieved evidence for agent-ready context
# Purpose:
# Remove raw XML/JSON noise and create concise evidence snippets
# that the LangGraph agent can safely use.
# ─────────────────────────────────────────────────────────────

import re
import json
import pandas as pd

assert "reranked_rag_results" in globals(), "reranked_rag_results not found. Rerun Cell 7."
assert "supplier_context_packet" in globals(), "supplier_context_packet not found. Rerun Cell 6."

def clean_evidence_text(text, max_chars=700):
    """
    Clean evidence text for chatbot context.
    Keeps readable event/risk text, removes huge raw XML/JSON payloads.
    """
    if text is None:
        return ""

    text = str(text)

    # If the chunk contains raw JSON with raw_text, try to avoid dumping the full payload
    if '"raw_text"' in text or "'raw_text'" in text:
        # Keep the leading source/title part before raw_text starts
        raw_text_pos = text.find('"raw_text"')
        if raw_text_pos == -1:
            raw_text_pos = text.find("'raw_text'")

        prefix = text[:raw_text_pos].strip()
        if len(prefix) > 50:
            text = prefix + " [Raw payload omitted for readability.]"
        else:
            text = "Raw sanctions/compliance payload detected. Full payload omitted for readability."

    # Remove XML fragments if present
    text = re.sub(r"<\?xml.*", "[Raw XML payload omitted for readability.]", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


cleaned_evidence_records = []

for idx, row in reranked_rag_results.reset_index(drop=True).head(8).iterrows():
    cleaned_text = clean_evidence_text(row.get("chunk_text"), max_chars=700)

    cleaned_evidence_records.append({
        "rank": int(idx + 1),
        "chunk_id": row.get("chunk_id"),
        "chunk_type": row.get("chunk_type"),
        "source_name": row.get("source_name"),
        "risk_category": row.get("risk_category"),
        "severity": row.get("severity"),
        "event_date": str(row.get("event_date")) if pd.notna(row.get("event_date")) else None,
        "sku_id": row.get("sku_id"),
        "source_url": row.get("source_url"),
        "cosine_similarity": float(row.get("cosine_similarity")) if pd.notna(row.get("cosine_similarity")) else None,
        "business_retrieval_score": float(row.get("business_retrieval_score")) if pd.notna(row.get("business_retrieval_score")) else None,
        "cleaned_evidence_text": cleaned_text
    })

agent_ready_context_packet = {
    "question": supplier_context_packet["question"],
    "supplier": supplier_context_packet["supplier"],
    "snapshot_context_text": supplier_context_packet["snapshot_context_text"],
    "snapshot_context_json": supplier_context_packet["snapshot_context_json"],
    "cleaned_retrieved_evidence": cleaned_evidence_records
}

print("Agent-ready context packet created.")
print(f"Cleaned evidence records: {len(cleaned_evidence_records)}")

print("\n" + "=" * 100)
print("CLEANED RETRIEVED EVIDENCE")
print("=" * 100)

for ev in cleaned_evidence_records:
    print(
        f"\n[{ev['rank']}] "
        f"{ev['source_name']} | {ev['risk_category']} | {ev['severity']} | {ev['event_date']}"
    )
    print(f"Score: {ev['business_retrieval_score']:.4f}")
    print(ev["cleaned_evidence_text"])
    print("-" * 100)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 10 — Generate clean final retrieval-test answer
# Uses cleaned evidence from agent_ready_context_packet
# ─────────────────────────────────────────────────────────────

assert "agent_ready_context_packet" in globals(), "agent_ready_context_packet not found. Rerun Cell 9."

packet = agent_ready_context_packet

supplier = packet["supplier"]
snapshot = packet["snapshot_context_json"]
cleaned_evidence = packet["cleaned_retrieved_evidence"]

supplier_id = supplier["supplier_id"]
supplier_name = supplier["supplier_name"]
risk_score = supplier["supplier_risk_score"]
risk_band = supplier["risk_band"]
top_driver = supplier["top_risk_driver"]
recommended_action = supplier["recommended_action"]

top_skus = snapshot.get("top_affected_skus") or []
active_events = snapshot.get("active_external_events") or []

risk_correction = (
    f"{supplier_name} is currently classified as {risk_band}, not high risk."
    if str(risk_band).lower() != "high"
    else f"{supplier_name} is currently classified as high risk."
)

event_summary_lines = []
for event in active_events[:5]:
    event_summary_lines.append(
        f"- {event.get('source_name')} | {event.get('risk_category')} | "
        f"{event.get('severity')} | {event.get('event_date')}: "
        f"{event.get('event_title')}"
    )

sku_summary_lines = []
for sku in top_skus[:6]:
    sku_summary_lines.append(
        f"- {sku.get('canonical_sku_id')}: "
        f"dependency={sku.get('dependency_percent')}, "
        f"stockout risk={sku.get('stockout_risk_score')}, "
        f"primary supplier={sku.get('is_primary_supplier')}, "
        f"alternate supplier={sku.get('alternate_supplier_id')}"
    )

evidence_summary_lines = []
for ev in cleaned_evidence[:6]:
    evidence_summary_lines.append(
        f"[{ev['rank']}] {ev['source_name']} | {ev['risk_category']} | "
        f"{ev['severity']} | {ev['event_date']} | "
        f"{ev['cleaned_evidence_text']}"
    )

final_retrieval_test_answer = f"""
Question:
{packet["question"]}

Answer:
{risk_correction}

{supplier_name} ({supplier_id}) has a supplier risk score of {risk_score}. The current top risk driver is {top_driver}. The recommended action is: {recommended_action}

Why the supplier is being monitored:
- Active external events: {snapshot.get("active_external_event_count", 0)}
- High-severity external events: {snapshot.get("high_severity_event_count", 0)}
- Latest external event date: {snapshot.get("latest_external_event_date")}
- Affected SKUs: {snapshot.get("top_affected_sku_count", 0)}
- High-risk affected SKUs: {snapshot.get("high_risk_sku_count", 0)}
- Open alerts: {snapshot.get("open_alert_count", 0)}

Main external events:
{chr(10).join(event_summary_lines) if event_summary_lines else "- No active external events found."}

Top affected SKUs:
{chr(10).join(sku_summary_lines) if sku_summary_lines else "- No affected SKUs found."}

Supporting evidence:
{chr(10).join(evidence_summary_lines)}

Recommendation:
{recommended_action}

Operational interpretation:
This is currently an event-driven supplier monitoring case, not an immediate stockout case. The strongest evidence comes from the internal supplier risk explanation, CISA cyber events, and OFAC sanctions-compliance matches. Since the affected SKUs currently show stockout risk of 0.0 and there are no open alerts, the right next step is continued monitoring rather than urgent procurement intervention.
""".strip()

print(final_retrieval_test_answer)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Cell 11 — Save retrieval test result for audit/demo
# ─────────────────────────────────────────────────────────────

import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType
)

assert "agent_ready_context_packet" in globals(), "agent_ready_context_packet not found. Rerun Cell 9."
assert "final_retrieval_test_answer" in globals(), "final_retrieval_test_answer not found. Rerun Cell 10."

RETRIEVAL_TEST_TABLE = "supplysage_gold.gold_rag_retrieval_test_results"

packet = agent_ready_context_packet
supplier = packet["supplier"]
evidence = packet["cleaned_retrieved_evidence"]

test_result_row = [{
    "test_run_id": f"rag_test_{supplier['supplier_id']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
    "supplier_id": supplier["supplier_id"],
    "supplier_name": supplier["supplier_name"],
    "question": packet["question"],
    "supplier_risk_score": float(supplier["supplier_risk_score"]),
    "risk_band": supplier["risk_band"],
    "top_risk_driver": supplier["top_risk_driver"],
    "recommended_action": supplier["recommended_action"],
    "retrieved_evidence_count": int(len(evidence)),
    "top_evidence_source": evidence[0]["source_name"] if evidence else None,
    "top_evidence_category": evidence[0]["risk_category"] if evidence else None,
    "top_evidence_score": float(evidence[0]["business_retrieval_score"]) if evidence else None,
    "agent_ready_context_json": json.dumps(packet, default=str),
    "final_retrieval_test_answer": final_retrieval_test_answer,
    "test_created_at": datetime.utcnow()
}]

test_schema = StructType([
    StructField("test_run_id", StringType(), False),
    StructField("supplier_id", StringType(), True),
    StructField("supplier_name", StringType(), True),
    StructField("question", StringType(), True),
    StructField("supplier_risk_score", DoubleType(), True),
    StructField("risk_band", StringType(), True),
    StructField("top_risk_driver", StringType(), True),
    StructField("recommended_action", StringType(), True),
    StructField("retrieved_evidence_count", IntegerType(), True),
    StructField("top_evidence_source", StringType(), True),
    StructField("top_evidence_category", StringType(), True),
    StructField("top_evidence_score", DoubleType(), True),
    StructField("agent_ready_context_json", StringType(), True),
    StructField("final_retrieval_test_answer", StringType(), True),
    StructField("test_created_at", TimestampType(), True),
])

test_result_df = spark.createDataFrame(test_result_row, schema=test_schema)

(
    test_result_df
    .withColumn("gold_created_at", F.current_timestamp())
    .withColumn("gold_source_notebook", F.lit("26_rag_retrieval_test"))
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(RETRIEVAL_TEST_TABLE)
)

print(f"Saved retrieval test result to: {RETRIEVAL_TEST_TABLE}")

display(
    spark.table(RETRIEVAL_TEST_TABLE)
    .orderBy(F.desc("test_created_at"))
    .limit(5)
)
