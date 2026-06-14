# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 22 — gold_supplier_risk_scores + gold_supplier_risk_explanation_log
# MAGIC **Sources (all Gold, so run after notebooks 16–21):**
# MAGIC   - `supplysage_gold.gold_dim_suppliers`
# MAGIC   - `supplysage_gold.gold_supplier_performance_mart`
# MAGIC   - `supplysage_gold.gold_supplier_sku_dependency_mart`
# MAGIC   - `supplysage_gold.gold_external_risk_event_mart`
# MAGIC **Targets:**
# MAGIC   - `supplysage_gold.gold_supplier_risk_scores`
# MAGIC   - `supplysage_gold.gold_supplier_risk_explanation_log`
# MAGIC **Grain:** One row per supplier (48 rows, latest scoring run)
# MAGIC
# MAGIC ## Risk score weight configuration
# MAGIC Weights must sum to 1.0. Stored here as constants — move to a config table in v2.
# MAGIC   - Operational performance: 0.25
# MAGIC   - Dependency / concentration: 0.20
# MAGIC   - External events: 0.20
# MAGIC   - Logistics / route risk: 0.15
# MAGIC   - Sanctions / compliance (OFAC): 0.10
# MAGIC   - Cyber risk (CISA KEV): 0.10

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()

SCORE_DATE = date.today().isoformat()

# ── Score weights ────────────────────────────────────────────────
W_OPERATIONAL  = 0.25
W_DEPENDENCY   = 0.20
W_EXTERNAL     = 0.20
W_LOGISTICS    = 0.15
W_SANCTIONS    = 0.10
W_CYBER        = 0.10

assert abs(W_OPERATIONAL + W_DEPENDENCY + W_EXTERNAL + W_LOGISTICS + W_SANCTIONS + W_CYBER - 1.0) < 0.001, "Weights must sum to 1.0"

# ── Band thresholds ──────────────────────────────────────────────
BAND_CRITICAL = 75
BAND_HIGH     = 55
BAND_MEDIUM   = 35

print(f"Scoring date: {SCORE_DATE}")
print(f"Weights: operational={W_OPERATIONAL}, dependency={W_DEPENDENCY}, external={W_EXTERNAL}, logistics={W_LOGISTICS}, sanctions={W_SANCTIONS}, cyber={W_CYBER}")

# COMMAND ----------

dim_suppliers = spark.table("supplysage_gold.gold_dim_suppliers")
performance   = spark.table("supplysage_gold.gold_supplier_performance_mart")
dependency    = spark.table("supplysage_gold.gold_supplier_sku_dependency_mart")
events        = spark.table("supplysage_gold.gold_external_risk_event_mart")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Operational performance sub-score (0-100)
# MAGIC Based on latest scorecard month. Higher defect rate / lower fill rate / lower OTD = higher risk.

# COMMAND ----------

# Get latest scorecard per supplier
perf_window = Window.partitionBy("supplier_id").orderBy(F.col("scorecard_month").desc())

latest_perf = (
    performance
    .withColumn("rn", F.row_number().over(perf_window))
    .filter(F.col("rn") == 1)
    .select(
        "supplier_id",
        "fill_rate",
        "on_time_delivery_rate",
        "defect_rate",
        "avg_lead_time_days",
        "lead_time_variance",
        "deterioration_flag",
        "fill_rate_3mo_slope",
        "otd_3mo_slope",
        "late_po_line_count"
    )
)

# Operational score: penalize poor fill rate, OTD, high defect rate
# All components normalized 0–100; lower performance = higher risk score
operational_score = latest_perf.withColumn(
    "operational_score",
    F.least(F.lit(100.0), F.greatest(F.lit(0.0),
        # fill_rate penalty: 1.0 fill_rate = 0 risk; 0.7 fill_rate = 30 points
        ((1.0 - F.col("fill_rate")) * 100.0) * 0.35 +
        # OTD penalty
        ((1.0 - F.col("on_time_delivery_rate")) * 100.0) * 0.30 +
        # Defect rate penalty (defect_rate is already 0-1 proportion)
        (F.col("defect_rate") * 500.0) * 0.20 +  # scale: 0.05 defect = 25 pts
        # Deterioration bonus
        F.when(F.col("deterioration_flag") == True, F.lit(15.0)).otherwise(F.lit(0.0)) * 0.15
    ))
).select("supplier_id", "operational_score", "fill_rate", "on_time_delivery_rate",
         "defect_rate", "deterioration_flag", "fill_rate_3mo_slope", "otd_3mo_slope")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dependency / concentration sub-score (0-100)
# MAGIC Based on: single_source_flag, max dependency_percent per SKU, no-alternate SKU count.

# COMMAND ----------

dep_summary = (
    dependency
    .filter(F.col("is_primary_supplier") == True)
    .groupBy("supplier_id")
    .agg(
        F.count("canonical_sku_id").alias("total_sku_count"),
        F.avg("dependency_percent").alias("avg_dependency_pct"),
        F.max("dependency_percent").alias("max_dependency_pct"),
        F.sum(F.when(F.col("alternate_status") == "none", 1).otherwise(0)).alias("no_alternate_sku_count")
    )
)

dependency_score = (
    dim_suppliers.select("supplier_id", "single_source_flag", "criticality_tier")
    .join(dep_summary, on="supplier_id", how="left")
    .withColumn("total_sku_count", F.coalesce(F.col("total_sku_count"), F.lit(0)))
    .withColumn("no_alternate_sku_count", F.coalesce(F.col("no_alternate_sku_count"), F.lit(0)))
    .withColumn("avg_dependency_pct", F.coalesce(F.col("avg_dependency_pct"), F.lit(0.5)))
    .withColumn(
        "dependency_score",
        F.least(F.lit(100.0), F.greatest(F.lit(0.0),
            # Single source = 40 base points
            F.when(F.col("single_source_flag") == True, F.lit(40.0)).otherwise(F.lit(0.0)) +
            # High average dependency
            F.col("avg_dependency_pct") * 30.0 +
            # No-alternate SKU proportion
            F.when(F.col("total_sku_count") > 0,
                (F.col("no_alternate_sku_count") / F.col("total_sku_count")) * 20.0
            ).otherwise(F.lit(0.0)) +
            # Tier 1 amplifier (more critical = higher base risk from dependency)
            F.when(F.col("criticality_tier") == "Tier 1", F.lit(10.0))
             .when(F.col("criticality_tier") == "Tier 2", F.lit(5.0))
             .otherwise(F.lit(0.0))
        ))
    )
    .select("supplier_id", "dependency_score", "total_sku_count", "no_alternate_sku_count", "avg_dependency_pct")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. External event sub-score (0-100)
# MAGIC Sum of score_contributions from matched events in last 30 days, capped at 100.

# COMMAND ----------

external_score = (
    events
    .filter(
        F.col("matched_supplier_id").isNotNull() &
        (F.col("event_date") >= F.date_sub(F.current_date(), 30))
    )
    .groupBy(F.col("matched_supplier_id").alias("supplier_id"))
    .agg(
        F.sum("score_contribution").alias("raw_external_score"),
        F.count("external_event_id").alias("active_event_count"),
        F.max("event_date").alias("latest_event_date"),
        F.collect_list(
            F.struct(
                F.col("source_name"),
                F.col("event_title"),
                F.col("severity"),
                F.col("score_contribution"),
                F.col("match_type")
            )
        ).alias("event_details")
    )
    .withColumn(
        "external_event_score",
        F.least(F.lit(100.0), F.coalesce(F.col("raw_external_score"), F.lit(0.0)))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Logistics / route risk sub-score (0-100)
# MAGIC Based on number of high-risk routes, transport mode concentration.

# COMMAND ----------

route_risk = spark.table("supplysage_silver.silver_shipment_routes")

logistics_score = (
    route_risk
    .groupBy("supplier_id")
    .agg(
        F.count("route_id").alias("route_count"),
        # Ocean and air have higher risk
        F.sum(F.when(F.col("transport_mode") == "ocean", F.lit(1)).otherwise(F.lit(0))).alias("ocean_route_count"),
        F.countDistinct("risk_region").alias("distinct_risk_regions")
    )
    .withColumn(
        "logistics_score",
        F.least(F.lit(100.0), F.greatest(F.lit(0.0),
            # Ocean-heavy routes = higher logistics risk
            F.when(F.col("route_count") > 0,
                (F.col("ocean_route_count") / F.col("route_count")) * 40.0
            ).otherwise(F.lit(0.0)) +
            # Multiple distinct risk regions = more exposure
            F.least(F.col("distinct_risk_regions") * 15.0, F.lit(60.0))
        ))
    )
    .select("supplier_id", "logistics_score", "route_count", "ocean_route_count", "distinct_risk_regions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sanctions sub-score — OFAC matches (0-100)

# COMMAND ----------

sanctions_score = (
    events
    .filter(
        (F.col("source_name") == "OFAC") &
        F.col("matched_supplier_id").isNotNull()
    )
    .groupBy(F.col("matched_supplier_id").alias("supplier_id"))
    .agg(
        F.count("external_event_id").alias("ofac_match_count"),
        F.max("severity").alias("ofac_max_severity")
    )
    .withColumn(
        "sanctions_score",
        F.when(F.col("ofac_match_count") > 0, F.lit(80.0))  # Any OFAC match is critical
         .otherwise(F.lit(0.0))
    )
    .select("supplier_id", "sanctions_score", "ofac_match_count")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cyber sub-score — CISA KEV matches (0-100)

# COMMAND ----------

cyber_score = (
    events
    .filter(
        (F.col("source_name") == "CISA") &
        F.col("matched_supplier_id").isNotNull() &
        (F.col("event_date") >= F.date_sub(F.current_date(), 90))
    )
    .groupBy(F.col("matched_supplier_id").alias("supplier_id"))
    .agg(
        F.count("external_event_id").alias("cisa_match_count")
    )
    .withColumn(
        "cyber_score",
        F.least(F.lit(100.0), F.col("cisa_match_count") * 10.0)
    )
    .select("supplier_id", "cyber_score", "cisa_match_count")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble composite risk score

# COMMAND ----------

composite = (
    dim_suppliers.select("supplier_id", "supplier_name", "country", "criticality_tier", "single_source_flag")
    .join(operational_score, on="supplier_id", how="left")
    .join(dependency_score, on="supplier_id", how="left")
    .join(external_score, on="supplier_id", how="left")
    .join(logistics_score, on="supplier_id", how="left")
    .join(sanctions_score, on="supplier_id", how="left")
    .join(cyber_score, on="supplier_id", how="left")
    .withColumn("operational_score",   F.coalesce(F.col("operational_score"),   F.lit(0.0)))
    .withColumn("dependency_score",    F.coalesce(F.col("dependency_score"),    F.lit(0.0)))
    .withColumn("external_event_score",F.coalesce(F.col("external_event_score"),F.lit(0.0)))
    .withColumn("logistics_score",     F.coalesce(F.col("logistics_score"),     F.lit(0.0)))
    .withColumn("sanctions_score",     F.coalesce(F.col("sanctions_score"),     F.lit(0.0)))
    .withColumn("cyber_score",         F.coalesce(F.col("cyber_score"),         F.lit(0.0)))
    .withColumn(
        "overall_risk_score",
        F.round(
            F.col("operational_score")   * W_OPERATIONAL +
            F.col("dependency_score")    * W_DEPENDENCY  +
            F.col("external_event_score")* W_EXTERNAL    +
            F.col("logistics_score")     * W_LOGISTICS   +
            F.col("sanctions_score")     * W_SANCTIONS   +
            F.col("cyber_score")         * W_CYBER,
            1
        )
    )
    .withColumn(
        "risk_band",
        F.when(F.col("overall_risk_score") >= BAND_CRITICAL, F.lit("critical"))
         .when(F.col("overall_risk_score") >= BAND_HIGH,     F.lit("high"))
         .when(F.col("overall_risk_score") >= BAND_MEDIUM,   F.lit("medium"))
         .otherwise(F.lit("low"))
    )
    .withColumn("score_date", F.lit(SCORE_DATE))
    .withColumn("active_event_count", F.coalesce(F.col("active_event_count"), F.lit(0)))
    .withColumn("total_sku_count",    F.coalesce(F.col("total_sku_count"),    F.lit(0)))
    .withColumn("no_alternate_sku_count", F.coalesce(F.col("no_alternate_sku_count"), F.lit(0)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute 24h and 7d deltas (if prior run exists)

# COMMAND ----------

# Check for prior scores
prior_scores_exist = spark.catalog.tableExists("supplysage_gold.gold_supplier_risk_scores")

if prior_scores_exist:
    prior = spark.table("supplysage_gold.gold_supplier_risk_scores").select(
        "supplier_id",
        F.col("overall_risk_score").alias("prior_score"),
        F.col("score_date").alias("prior_date")
    )
    composite = composite.join(prior, on="supplier_id", how="left")
    composite = composite.withColumn(
        "score_delta_24h",
        F.round(F.col("overall_risk_score") - F.coalesce(F.col("prior_score"), F.col("overall_risk_score")), 1)
    ).drop("prior_score", "prior_date")
else:
    composite = composite.withColumn("score_delta_24h", F.lit(0.0))

# score_delta_7d would need a 7-day historical table — seeded as 0 for first run
composite = composite.withColumn("score_delta_7d", F.lit(0.0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Derive top risk driver and recommended action

# COMMAND ----------

# Top driver = dimension with highest absolute sub-score × weight
composite = composite.withColumn(
    "top_risk_driver",
    F.when(
        (F.col("external_event_score") * W_EXTERNAL) == F.greatest(
            F.col("operational_score") * W_OPERATIONAL,
            F.col("dependency_score") * W_DEPENDENCY,
            F.col("external_event_score") * W_EXTERNAL,
            F.col("logistics_score") * W_LOGISTICS,
            F.col("sanctions_score") * W_SANCTIONS,
            F.col("cyber_score") * W_CYBER
        ), F.lit("External events")
    ).when(
        (F.col("dependency_score") * W_DEPENDENCY) == F.greatest(
            F.col("operational_score") * W_OPERATIONAL,
            F.col("dependency_score") * W_DEPENDENCY,
            F.col("external_event_score") * W_EXTERNAL,
            F.col("logistics_score") * W_LOGISTICS,
            F.col("sanctions_score") * W_SANCTIONS,
            F.col("cyber_score") * W_CYBER
        ), F.lit("Supplier dependency")
    ).when(
        (F.col("sanctions_score") * W_SANCTIONS) > 0, F.lit("OFAC sanctions match")
    ).when(
        (F.col("operational_score") * W_OPERATIONAL) >= (F.col("logistics_score") * W_LOGISTICS),
        F.lit("Operational performance decline")
    ).otherwise(F.lit("Logistics / route risk"))
)

composite = composite.withColumn(
    "recommended_action",
    F.when(
        (F.col("risk_band") == "critical") & (F.col("single_source_flag") == True) & (F.col("no_alternate_sku_count") > 0),
        F.lit("Escalate immediately — no fallback exists")
    ).when(
        (F.col("risk_band") == "critical") & (F.col("no_alternate_sku_count") == 0),
        F.lit("Initiate supplier switch — alternate pre-qualified")
    ).when(
        (F.col("risk_band") == "critical"),
        F.lit("Expedite open POs and escalate to procurement lead")
    ).when(
        (F.col("risk_band") == "high") & (F.col("deterioration_flag") == True),
        F.lit("Schedule supplier review — scorecard trend declining")
    ).when(
        (F.col("risk_band") == "high"),
        F.lit("Monitor closely — external event in progress")
    ).when(
        F.col("risk_band") == "medium", F.lit("Monitor — no immediate action required")
    ).otherwise(F.lit("No action required"))
)

final_scores = composite.select(
    "supplier_id", "supplier_name", "country", "criticality_tier", "single_source_flag",
    "score_date", "overall_risk_score", "risk_band",
    "operational_score", "dependency_score", "external_event_score",
    "logistics_score", "sanctions_score", "cyber_score",
    "score_delta_24h", "score_delta_7d",
    "top_risk_driver", "recommended_action",
    "active_event_count", "total_sku_count", "no_alternate_sku_count",
    "deterioration_flag",
    F.lit(datetime.utcnow().isoformat()).alias("gold_created_at"),
    F.lit("22_gold_supplier_risk_scores").alias("gold_source_notebook")
)

print(f"gold_supplier_risk_scores: {final_scores.count()} rows")
display(final_scores.select("supplier_name", "overall_risk_score", "risk_band", "top_risk_driver", "recommended_action").orderBy(F.col("overall_risk_score").desc()))

# COMMAND ----------

(
    final_scores
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_supplier_risk_scores")
)
print(f"✅ gold_supplier_risk_scores written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build gold_supplier_risk_explanation_log
# MAGIC Pre-computes the "why" narrative for the chatbot.

# COMMAND ----------

# Get top 3 events per supplier for the explanation
top_events = (
    events
    .filter(
        F.col("matched_supplier_id").isNotNull() &
        (F.col("event_date") >= F.date_sub(F.current_date(), 30))
    )
    .withColumn("rn", F.row_number().over(
        Window.partitionBy("matched_supplier_id")
              .orderBy(F.col("score_contribution").desc())
    ))
    .filter(F.col("rn") <= 3)
    .groupBy(F.col("matched_supplier_id").alias("supplier_id"))
    .agg(
        F.collect_list(
            F.concat_ws(" · ",
                F.col("source_name"),
                F.col("event_title"),
                F.concat(F.lit("+"), F.col("score_contribution").cast("string"), F.lit(" pts"))
            )
        ).alias("top_events_list"),
        F.count("external_event_id").alias("evidence_count"),
        F.collect_list("external_event_id").alias("evidence_ids")
    )
)

explanation_log = (
    final_scores.select(
        "supplier_id", "supplier_name", "overall_risk_score", "risk_band",
        "score_delta_24h", "score_date",
        "operational_score", "dependency_score", "external_event_score",
        "logistics_score", "sanctions_score", "cyber_score",
        "deterioration_flag", "top_risk_driver", "recommended_action"
    )
    .join(top_events, on="supplier_id", how="left")
    .withColumn(
        "driver_1_dimension",
        F.col("top_risk_driver")
    )
    .withColumn(
        "driver_1_detail",
        F.when(
            F.size(F.col("top_events_list")) > 0,
            F.col("top_events_list").getItem(0)
        ).otherwise(F.lit("No recent external events"))
    )
    .withColumn(
        "driver_2_dimension",
        F.when(F.col("deterioration_flag") == True, F.lit("Operational performance decline"))
         .when(F.col("dependency_score") > 50, F.lit("High supplier dependency"))
         .otherwise(F.lit("Logistics risk"))
    )
    .withColumn(
        "driver_2_detail",
        F.when(
            F.size(F.col("top_events_list")) > 1,
            F.col("top_events_list").getItem(1)
        ).when(
            F.col("deterioration_flag") == True,
            F.lit("Scorecard deterioration detected in last 3 months")
        ).otherwise(F.lit("Route / logistics risk"))
    )
    .withColumn(
        "driver_3_detail",
        F.when(
            F.size(F.col("top_events_list")) > 2,
            F.col("top_events_list").getItem(2)
        ).otherwise(F.lit(None).cast("string"))
    )
    .withColumn("evidence_count", F.coalesce(F.col("evidence_count"), F.lit(0)))
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("22_gold_supplier_risk_explanation_log"))
    .select(
        "supplier_id", "supplier_name", "score_date",
        "overall_risk_score", "risk_band", "score_delta_24h",
        "operational_score", "dependency_score", "external_event_score",
        "logistics_score", "sanctions_score", "cyber_score",
        "deterioration_flag", "top_risk_driver", "recommended_action",
        "driver_1_dimension", "driver_1_detail",
        "driver_2_dimension", "driver_2_detail", "driver_3_detail",
        "evidence_count", "evidence_ids",
        "gold_created_at", "gold_source_notebook"
    )
)

(
    explanation_log
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_supplier_risk_explanation_log")
)
print(f"✅ gold_supplier_risk_explanation_log written: {spark.table('supplysage_gold.gold_supplier_risk_explanation_log').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

results = []
scores = spark.table("supplysage_gold.gold_supplier_risk_scores")

rc = scores.count()
results.append({"check": "row_count_48", "status": "PASS" if rc == 48 else "FAIL", "detail": str(rc)})

bad_score = scores.filter((F.col("overall_risk_score") < 0) | (F.col("overall_risk_score") > 100)).count()
results.append({"check": "score_in_range_0_100", "status": "PASS" if bad_score == 0 else "FAIL", "detail": str(bad_score)})

bad_band = scores.filter(~F.col("risk_band").isin("critical", "high", "medium", "low")).count()
results.append({"check": "risk_band_valid_vocab", "status": "PASS" if bad_band == 0 else "FAIL", "detail": str(bad_band)})

critical_count = scores.filter(F.col("risk_band") == "critical").count()
results.append({"check": "has_critical_suppliers", "status": "PASS" if critical_count > 0 else "WARN", "detail": f"{critical_count} critical suppliers"})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("22_gold_supplier_risk_scores")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
