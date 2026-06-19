# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 21 — gold_external_risk_event_mart
# MAGIC **Sources:**
# MAGIC   - `supplysage_silver.silver_domain_external_risk_events` (11,508 rows)
# MAGIC   - `supplysage_silver.silver_external_evidence_documents` (15 rows)
# MAGIC   - `supplysage_silver.silver_suppliers` (48 rows)
# MAGIC   - `supplysage_silver.silver_supplier_aliases` (133 rows)
# MAGIC   - `supplysage_silver.silver_shipment_routes` (97 rows)
# MAGIC   - `supplysage_silver.silver_supplier_sku_map` (355 rows)
# MAGIC **Target:** `supplysage_gold.gold_external_risk_event_mart`
# MAGIC **Grain:** One row per external_event_id × matched_supplier_id
# MAGIC **Purpose:** Rule-based v1 matching of external events to suppliers, routes, SKUs.
# MAGIC Unmatched events are also stored (matched_supplier_id = NULL).

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

events    = spark.table("supplysage_silver.silver_domain_external_risk_events")
evidence  = spark.table("supplysage_silver.silver_external_evidence_documents")
suppliers = spark.table("supplysage_silver.silver_suppliers")
aliases   = spark.table("supplysage_silver.silver_supplier_aliases")
routes    = spark.table("supplysage_silver.silver_shipment_routes")
sku_map   = spark.table("supplysage_silver.silver_supplier_sku_map")

print(f"silver_domain_external_risk_events:   {events.count()} rows")
print(f"silver_external_evidence_documents:   {evidence.count()} rows")
print(f"silver_suppliers:                     {suppliers.count()} rows")
print(f"silver_supplier_aliases:              {aliases.count()} rows")
print(f"silver_shipment_routes:               {routes.count()} rows")
print(f"silver_supplier_sku_map:              {sku_map.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 1 — Alias match
# MAGIC Check if any supplier alias appears in event_title or event_summary.
# MAGIC This is the highest-confidence match (e.g., "Pacific Rim Holdings" in a GDELT headline).

# COMMAND ----------

alias_matches = (
    events.crossJoin(
        aliases.select(
            F.col("supplier_id"),
            F.col("alias_name"),
            F.col("match_confidence").alias("alias_confidence")
        )
    )
    .filter(
        F.lower(F.col("event_title")).contains(F.lower(F.col("alias_name"))) |
        F.lower(F.col("event_summary")).contains(F.lower(F.col("alias_name")))
    )
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("alias").alias("match_type"),
        F.col("alias_name").alias("match_value"),
        F.col("alias_confidence").alias("match_confidence")
    )
)

print(f"Alias matches: {alias_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 2 — Supplier name match
# MAGIC Check if supplier_name (or its first two words) appears in the event text.

# COMMAND ----------

name_matches = (
    events.crossJoin(
        suppliers.select(
            F.col("supplier_id"),
            F.col("supplier_name"),
            # Also match on shortened name (first 2 words)
            F.concat_ws(" ",
                F.split(F.col("supplier_name"), " ").getItem(0),
                F.split(F.col("supplier_name"), " ").getItem(1)
            ).alias("supplier_name_short")
        )
    )
    .filter(
        F.lower(F.col("event_title")).contains(F.lower(F.col("supplier_name"))) |
        F.lower(F.col("event_summary")).contains(F.lower(F.col("supplier_name"))) |
        F.lower(F.col("event_title")).contains(F.lower(F.col("supplier_name_short"))) |
        F.lower(F.col("event_summary")).contains(F.lower(F.col("supplier_name_short")))
    )
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("name").alias("match_type"),
        F.col("supplier_name").alias("match_value"),
        F.lit(0.85).alias("match_confidence")
    )
)

print(f"Name matches: {name_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 3 — Country match
# MAGIC For logistics/weather/sanctions events, match on supplier country = event country.

# COMMAND ----------

country_match_sources = ["NWS", "GDELT", "OFAC", "SEC"]

country_matches = (
    events.filter(F.col("source_name").isin(country_match_sources))
    .crossJoin(
        suppliers.select("supplier_id", "country")
    )
    .filter(
        F.lower(F.col("event_country")) == F.lower(F.col("country"))
    )
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("country").alias("match_type"),
        F.col("country").alias("match_value"),
        F.lit(0.65).alias("match_confidence")
    )
)

print(f"Country matches: {country_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 4 — Route / risk region match
# MAGIC For weather and logistics events, match event_region to route risk_region.

# COMMAND ----------

region_matches = (
    events.filter(F.col("source_name").isin("NWS", "GDELT"))
    .join(
        routes.select("supplier_id", "risk_region", "origin_country"),
        F.lower(F.col("event_region")).contains(F.lower(F.col("risk_region"))) |
        F.lower(F.col("risk_region")).contains(F.lower(F.col("event_region"))),
        how="inner"
    )
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("region").alias("match_type"),
        F.col("risk_region").alias("match_value"),
        F.lit(0.60).alias("match_confidence")
    )
)

print(f"Region/route matches: {region_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 5 — Category match (CPSC recalls)
# MAGIC CPSC recall categories matched to supplier_category.

# COMMAND ----------

category_matches = (
    events.filter(F.col("source_name") == "CPSC")
    .crossJoin(
        suppliers.select("supplier_id", "supplier_category")
    )
    .filter(
        F.lower(F.col("event_title")).contains(F.lower(F.col("supplier_category"))) |
        F.lower(F.col("event_summary")).contains(F.lower(F.col("supplier_category"))) |
        # Check reverse
        F.lower(F.col("supplier_category")).contains(F.lower(F.col("risk_category")))
    )
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("category").alias("match_type"),
        F.col("supplier_category").alias("match_value"),
        F.lit(0.55).alias("match_confidence")
    )
)

print(f"Category matches (CPSC): {category_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 6 — Fuel / logistics (EIA → all ocean/truck suppliers)

# COMMAND ----------

ocean_truck_suppliers = routes.filter(
    F.col("transport_mode").isin("ocean", "truck")
).select("supplier_id").distinct()

fuel_matches = (
    events.filter(F.col("source_name") == "EIA")
    .crossJoin(ocean_truck_suppliers)
    .select(
        F.col("external_event_id"),
        F.col("supplier_id"),
        F.lit("fuel_logistics").alias("match_type"),
        F.lit("ocean_or_truck_route").alias("match_value"),
        F.lit(0.50).alias("match_confidence")
    )
)

print(f"Fuel/logistics matches: {fuel_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union all matches, deduplicate keeping highest confidence per event × supplier

# COMMAND ----------

all_matches = (
    alias_matches
    .union(name_matches)
    .union(country_matches)
    .union(region_matches)
    .union(category_matches)
    .union(fuel_matches)
)

# Keep highest-confidence match per event × supplier
from pyspark.sql.window import Window
dedup_window = Window.partitionBy("external_event_id", "supplier_id").orderBy(F.col("match_confidence").desc())

best_matches = (
    all_matches
    .withColumn("rn", F.row_number().over(dedup_window))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

print(f"Unique event × supplier matches: {best_matches.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Count impacted SKUs per event × supplier match

# COMMAND ----------

sku_counts = (
    sku_map
    .groupBy("supplier_id")
    .agg(F.count("sku_id").alias("matched_sku_count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join events + matches + evidence documents
# MAGIC Keep unmatched events too (left join from events to matches)

# COMMAND ----------

# Severity → score contribution mapping
severity_score_map = {
    "critical": 20,
    "high": 12,
    "medium": 6,
    "low": 2
}

score_contribution_expr = (
    F.when(F.col("severity") == "critical", F.lit(20))
     .when(F.col("severity") == "high", F.lit(12))
     .when(F.col("severity") == "medium", F.lit(6))
     .otherwise(F.lit(2))
)

evidence_renamed = evidence.select(
    F.col("api_run_id").alias("_ev_api_run_id"),
    F.col("payload_hash").alias("_ev_payload_hash"),
    F.col("evidence_document_id").alias("_ev_evidence_document_id"),
)

events_joined = (
    events
    .join(best_matches, on="external_event_id", how="left")
    .join(sku_counts, on="supplier_id", how="left")
)

event_mart = (
    events_joined
    .join(
        evidence_renamed,
        on=[
            events_joined["api_run_id"] == evidence_renamed["_ev_api_run_id"],
            events_joined["source_payload_hash"] == evidence_renamed["_ev_payload_hash"]
        ],
        how="left"
    )
    .withColumn(
        "score_contribution",
        F.when(events_joined["supplier_id"].isNotNull(), score_contribution_expr)
         .otherwise(F.lit(None).cast("int"))
    )
    .withColumn(
        "match_type",
        F.coalesce(events_joined["match_type"], F.lit("none"))
    )
    .withColumn(
        "matched_sku_count",
        F.coalesce(events_joined["matched_sku_count"], F.lit(0))
    )
    .select(
        events_joined["external_event_id"],
        events_joined["source_name"],
        events_joined["risk_category"],
        events_joined["event_type"],
        events_joined["event_title"],
        events_joined["event_summary"],
        events_joined["severity"],
        events_joined["event_date"],
        events_joined["event_timestamp"],
        events_joined["source_url"],
        events_joined["event_country"],
        events_joined["event_region"],
        events_joined["language"],
        events_joined["supplier_id"].alias("matched_supplier_id"),
        events_joined["match_type"],
        events_joined["match_value"],
        events_joined["match_confidence"],
        events_joined["matched_sku_count"],
        F.col("score_contribution"),
        evidence_renamed["_ev_evidence_document_id"].alias("evidence_doc_id"),
        events_joined["api_run_id"],
        F.lit(datetime.utcnow().isoformat()).alias("gold_created_at"),
        F.lit("21_gold_external_risk_event_mart").alias("gold_source_notebook")
    )
)

row_count = event_mart.count()
print(f"gold_external_risk_event_mart row count: {row_count}")

# COMMAND ----------

(
    event_mart
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_external_risk_event_mart")
)

print(f"✅ written: {spark.table('supplysage_gold.gold_external_risk_event_mart').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation + match summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     source_name,
# MAGIC     match_type,
# MAGIC     COUNT(*) AS event_count,
# MAGIC     COUNT(matched_supplier_id) AS matched_count,
# MAGIC     AVG(score_contribution) AS avg_score_contribution
# MAGIC FROM supplysage_gold.gold_external_risk_event_mart
# MAGIC GROUP BY source_name, match_type
# MAGIC ORDER BY source_name, matched_count DESC

# COMMAND ----------

results = []
mart = spark.table("supplysage_gold.gold_external_risk_event_mart")

rc = mart.count()
results.append({"check": "row_count_gte_11508", "status": "PASS" if rc >= 11508 else "FAIL", "detail": str(rc)})

matched = mart.filter(F.col("matched_supplier_id").isNotNull()).count()
results.append({"check": "has_matched_events", "status": "PASS" if matched > 0 else "FAIL", "detail": f"{matched} matched rows"})

bad_match = mart.filter(
    ~F.col("match_type").isin("alias", "name", "country", "region", "category", "fuel_logistics", "none")
).count()
results.append({"check": "match_type_valid_vocab", "status": "PASS" if bad_match == 0 else "FAIL", "detail": str(bad_match)})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("21_gold_external_risk_event_mart")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
