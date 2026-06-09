# Databricks notebook source
# MAGIC %md
# MAGIC # SupplySage AI — Notebook 03: Generate Synthetic Supplier Data
# MAGIC
# MAGIC **Purpose:**
# MAGIC Because public Kaggle datasets (M5, Retail Inventory, DataCo) do not include a real enterprise
# MAGIC supplier system, this notebook generates realistic synthetic internal tables that bridge raw
# MAGIC supply chain data to supplier-level risk intelligence.
# MAGIC
# MAGIC **Tables generated (all written to `supplysage_bronze`):**
# MAGIC
# MAGIC | Table | Grain | Rows (approx) |
# MAGIC |---|---|---|
# MAGIC | `bronze_suppliers` | 1 row per supplier | 50 |
# MAGIC | `bronze_supplier_aliases` | supplier × alias | ~120 |
# MAGIC | `bronze_supplier_sku_map` | supplier × SKU | ~400 |
# MAGIC | `bronze_alternate_suppliers` | SKU × alt supplier | ~200 |
# MAGIC | `bronze_supplier_scorecards` | supplier × month | ~600 |
# MAGIC | `bronze_purchase_orders` | PO line | ~2,000 |
# MAGIC | `bronze_shipment_routes` | route | ~80 |
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - Supplier IDs anchor to real M5 SKU taxonomy (FOODS, HOBBIES, HOUSEHOLD categories)
# MAGIC - Geography is grounded in real supply chain trade lanes (China, Mexico, Vietnam, India, US)
# MAGIC - Scorecards have realistic variance: Tier 1 suppliers perform better than Tier 3
# MAGIC - Purchase orders align with M5 calendar date range (2011–2016)
# MAGIC - All tables are Delta-format, Unity-Catalog-compatible, lowercase+underscore named

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: imports and random seed

# COMMAND ----------

import random
import uuid
from datetime import date, timedelta, datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType, BooleanType, DateType
)

# Reproducible synthetic data
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

spark = SparkSession.builder.getOrCreate()

print("Spark session ready.")
print(f"Random seed: {RANDOM_SEED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference: M5 categories and store IDs
# MAGIC These are pulled from the real M5 dataset already loaded in bronze.

# COMMAND ----------

# Pull distinct category / department combos from real M5 Bronze table
m5_categories_df = spark.sql("""
    SELECT DISTINCT cat_id, dept_id
    FROM supplysage_bronze.bronze_m5_sales_train_validation
    ORDER BY cat_id, dept_id
""").toPandas()

# Pull distinct item_ids (SKUs) — limit to a representative 200 for supplier mapping
m5_skus_df = spark.sql("""
    SELECT DISTINCT item_id
    FROM supplysage_bronze.bronze_m5_sales_train_validation
    ORDER BY item_id
    LIMIT 200
""").toPandas()

# Pull distinct store IDs
m5_stores_df = spark.sql("""
    SELECT DISTINCT store_id, state_id
    FROM supplysage_bronze.bronze_m5_sales_train_validation
    ORDER BY store_id
""").toPandas()

print(f"M5 categories loaded: {len(m5_categories_df)} rows")
print(f"M5 sample SKUs loaded: {len(m5_skus_df)} rows")
print(f"M5 stores loaded: {len(m5_stores_df)} rows")
display(m5_categories_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: `bronze_suppliers`
# MAGIC
# MAGIC 50 synthetic suppliers, modeled after real retail supply chains.
# MAGIC - Tiers 1–3 map to criticality (Tier 1 = highest spend, most critical)
# MAGIC - Geography: China, Mexico, Vietnam, India, US, Canada, Germany, Brazil, South Korea
# MAGIC - Categories align to M5 taxonomy: FOODS, HOBBIES, HOUSEHOLD

# COMMAND ----------

# ── Lookup tables ────────────────────────────────────────────────────────────

SUPPLIER_NAMES = [
    # FOODS suppliers
    "Pacific Rim Foods Co.", "Golden Harvest Distributors", "Sun Valley Provisions",
    "FreshLink Supply LLC", "NatureSource Growers", "Prairie Grain Partners",
    "Tropical Origins Ltd.", "Blue Ridge Dairy Group", "Coastal Seafood Logistics",
    "Heartland Meats Inc.", "Sunrise Organics Co.", "Delta Beverage Supply",
    "Mountain Spring Waters", "Gulf Coast Produce", "Great Lakes Grains LLC",
    "Sierra Frozen Foods", "AgriCore International", "Rio Farms Export",

    # HOBBIES suppliers
    "PlayCraft Manufacturing", "CreativeEdge Toys Ltd.", "SportLine Gear Co.",
    "ArtSupply Global", "EduToys International", "GameZone Distributors",
    "OutdoorPro Equipment", "CraftMaster Supply", "TechToy Innovations",
    "VitalPlay Products",

    # HOUSEHOLD suppliers
    "CleanHome Products Inc.", "HomeEssentials Corp.", "PurePath Chemical Co.",
    "BrightLine Packaging", "DuraHome Manufacturing", "EcoClean Supplies Ltd.",
    "FlexPack Industries", "HomeGuard Safety LLC", "KitchenPro Suppliers",
    "LightBright Fixtures",

    # Mixed / multi-category
    "Apex Supply Chain Group", "Meridian Trade Partners", "Summit Distribution Co.",
    "Horizon Global Logistics", "CrossBorder Fulfillment", "Atlas Supply Network",
    "Pacific Gateway Traders", "Unified Commerce Supply", "NexGen Retail Partners",
    "Alliance Procurement LLC"
]

PARENT_COMPANIES = {
    "Pacific Rim Foods Co.": "Pacific Rim Holdings",
    "Golden Harvest Distributors": "AgriCorp International",
    "Sun Valley Provisions": "Sun Valley Holdings",
    "FreshLink Supply LLC": None,
    "NatureSource Growers": "NatureSource Inc.",
    "Prairie Grain Partners": None,
    "Tropical Origins Ltd.": "TropAg Group",
    "Blue Ridge Dairy Group": "Blue Ridge Foods",
    "Coastal Seafood Logistics": None,
    "Heartland Meats Inc.": "Heartland Foods Corp.",
    "Sunrise Organics Co.": None,
    "Delta Beverage Supply": "Delta Brands Inc.",
    "Mountain Spring Waters": "Hydra Beverages",
    "Gulf Coast Produce": None,
    "Great Lakes Grains LLC": "Midwest Agri Group",
    "Sierra Frozen Foods": "Sierra Foods Holdings",
    "AgriCore International": "AgriCore Global",
    "Rio Farms Export": "Rio Agri S.A.",
    "PlayCraft Manufacturing": "PlayCraft Holdings",
    "CreativeEdge Toys Ltd.": "CreativeEdge Group",
    "SportLine Gear Co.": "SportLine International",
    "ArtSupply Global": None,
    "EduToys International": "EduGroup Ltd.",
    "GameZone Distributors": "GameZone Corp.",
    "OutdoorPro Equipment": None,
    "CraftMaster Supply": None,
    "TechToy Innovations": "TechToy Holdings",
    "VitalPlay Products": None,
    "CleanHome Products Inc.": "CleanHome Corp.",
    "HomeEssentials Corp.": "HomeEssentials Holdings",
    "PurePath Chemical Co.": "PurePath Industries",
    "BrightLine Packaging": "BrightLine Group",
    "DuraHome Manufacturing": None,
    "EcoClean Supplies Ltd.": "EcoGroup International",
    "FlexPack Industries": "FlexPack Corp.",
    "HomeGuard Safety LLC": None,
    "KitchenPro Suppliers": "KitchenPro Holdings",
    "LightBright Fixtures": None,
    "Apex Supply Chain Group": "Apex Global",
    "Meridian Trade Partners": "Meridian Holdings",
    "Summit Distribution Co.": None,
    "Horizon Global Logistics": "Horizon Logistics Group",
    "CrossBorder Fulfillment": None,
    "Atlas Supply Network": "Atlas Commerce",
    "Pacific Gateway Traders": "Pacific Gateway Holdings",
    "Unified Commerce Supply": None,
    "NexGen Retail Partners": "NexGen Global",
    "Alliance Procurement LLC": None
}

COUNTRIES = [
    "China", "China", "China", "China",  # weighted — China is dominant
    "Mexico", "Mexico", "Mexico",
    "United States", "United States", "United States",
    "Vietnam", "Vietnam",
    "India", "India",
    "Canada",
    "Germany",
    "Brazil",
    "South Korea",
    "Thailand",
    "Indonesia"
]

COUNTRY_REGION_MAP = {
    "China": "Asia Pacific",
    "Vietnam": "Asia Pacific",
    "South Korea": "Asia Pacific",
    "Thailand": "Asia Pacific",
    "Indonesia": "Asia Pacific",
    "India": "South Asia",
    "Germany": "Europe",
    "Mexico": "North America",
    "United States": "North America",
    "Canada": "North America",
    "Brazil": "Latin America"
}

SUPPLIER_CATEGORIES = [
    "FOODS", "FOODS", "FOODS",  # weighted heavier — M5 is food-heavy
    "HOBBIES",
    "HOUSEHOLD",
    "FOODS_HOBBIES",
    "HOUSEHOLD_FOODS",
    "MULTI_CATEGORY"
]

CRITICALITY_TIERS = ["Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 2", "Tier 3"]

ANNUAL_SPEND_BY_TIER = {
    "Tier 1": (15_000_000, 80_000_000),
    "Tier 2": (3_000_000, 15_000_000),
    "Tier 3": (500_000, 3_000_000)
}

LEAD_TIME_BY_COUNTRY = {
    "China": (21, 42),
    "Vietnam": (25, 45),
    "South Korea": (18, 35),
    "Thailand": (20, 40),
    "Indonesia": (25, 45),
    "India": (20, 40),
    "Germany": (10, 21),
    "Mexico": (5, 14),
    "United States": (2, 7),
    "Canada": (3, 8),
    "Brazil": (14, 28)
}

# ── Generate suppliers ────────────────────────────────────────────────────────

suppliers = []
for i, name in enumerate(SUPPLIER_NAMES):
    supplier_id = f"SUP_{str(i + 100).zfill(3)}"
    country = random.choice(COUNTRIES)
    region = COUNTRY_REGION_MAP[country]
    tier = random.choice(CRITICALITY_TIERS)
    spend_range = ANNUAL_SPEND_BY_TIER[tier]
    annual_spend = round(random.uniform(*spend_range), 2)
    lead_range = LEAD_TIME_BY_COUNTRY[country]
    lead_time = random.randint(*lead_range)
    parent = PARENT_COMPANIES.get(name, None)
    supplier_cat = random.choice(SUPPLIER_CATEGORIES)
    single_source = random.random() < 0.20  # 20% are single-source (high risk)

    suppliers.append({
        "supplier_id": supplier_id,
        "supplier_name": name,
        "parent_company": parent,
        "country": country,
        "region": region,
        "supplier_category": supplier_cat,
        "criticality_tier": tier,
        "annual_spend": annual_spend,
        "single_source_flag": single_source,
        "default_lead_time_days": lead_time
    })

suppliers_pdf = pd.DataFrame(suppliers)
print(f"Suppliers generated: {len(suppliers_pdf)}")
print(suppliers_pdf[["supplier_id", "supplier_name", "country", "criticality_tier", "annual_spend", "single_source_flag"]].head(10))

# COMMAND ----------

# ── Write to Delta ────────────────────────────────────────────────────────────

suppliers_sdf = spark.createDataFrame(suppliers_pdf)

(
    suppliers_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_suppliers")
)

row_count = spark.table("supplysage_bronze.bronze_suppliers").count()
print(f"✅ bronze_suppliers written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: `bronze_supplier_aliases`
# MAGIC
# MAGIC Suppliers are known by different names in news, government databases, and internal ERPs.
# MAGIC This table enables fuzzy entity resolution in the RAG pipeline.

# COMMAND ----------

ALIAS_TYPES = ["news_abbreviation", "ticker_symbol", "dba_name", "parent_name", "erp_alias"]
ALIAS_SOURCES = ["internal_erp", "gdelt_news_match", "sec_edgar", "ofac_screening", "manual_entry"]

aliases = []

for sup in suppliers:
    sid = sup["supplier_id"]
    name = sup["supplier_name"]
    parent = sup["parent_company"]

    # Always add a short abbreviation alias
    words = name.replace(".", "").replace(",", "").split()
    abbr = "".join([w[0] for w in words if w[0].isupper()])
    if len(abbr) >= 2:
        aliases.append({
            "supplier_id": sid,
            "alias_name": abbr,
            "alias_type": "news_abbreviation",
            "match_confidence": round(random.uniform(0.70, 0.95), 2),
            "source": "gdelt_news_match"
        })

    # Add a shortened version of the name
    short_name = " ".join(words[:2]) if len(words) >= 2 else name
    if short_name != name:
        aliases.append({
            "supplier_id": sid,
            "alias_name": short_name,
            "alias_type": "dba_name",
            "match_confidence": round(random.uniform(0.80, 0.99), 2),
            "source": "internal_erp"
        })

    # If parent company exists, add parent name as alias
    if parent:
        aliases.append({
            "supplier_id": sid,
            "alias_name": parent,
            "alias_type": "parent_name",
            "match_confidence": round(random.uniform(0.60, 0.90), 2),
            "source": "sec_edgar"
        })

    # 40% chance of an OFAC/sanctions screening alias
    if random.random() < 0.40:
        aliases.append({
            "supplier_id": sid,
            "alias_name": name.upper().replace(" CO.", "").replace(" LLC", "").replace(" INC.", "").strip(),
            "alias_type": "erp_alias",
            "match_confidence": round(random.uniform(0.75, 0.95), 2),
            "source": "ofac_screening"
        })

aliases_pdf = pd.DataFrame(aliases)
print(f"Supplier aliases generated: {len(aliases_pdf)}")
print(aliases_pdf.head(10))

# COMMAND ----------

aliases_sdf = spark.createDataFrame(aliases_pdf)

(
    aliases_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_supplier_aliases")
)

row_count = spark.table("supplysage_bronze.bronze_supplier_aliases").count()
print(f"✅ bronze_supplier_aliases written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: `bronze_supplier_sku_map`
# MAGIC
# MAGIC Maps each supplier to the SKUs they supply.
# MAGIC - Each SKU has exactly 1 primary supplier and 0–2 alternate suppliers
# MAGIC - `dependency_percent` reflects how much of the retailer's supply of that SKU comes from this supplier
# MAGIC - `origin_port` is consistent with the supplier's country

# COMMAND ----------

SKU_LIST = m5_skus_df["item_id"].tolist()
SUPPLIER_IDS = [s["supplier_id"] for s in suppliers]

TRANSPORT_MODES = ["ocean", "ocean", "ocean", "air", "truck", "rail"]

PORT_BY_COUNTRY = {
    "China": ["Shanghai", "Shenzhen", "Ningbo", "Tianjin"],
    "Vietnam": ["Ho Chi Minh City", "Hai Phong"],
    "South Korea": ["Busan", "Incheon"],
    "Thailand": ["Laem Chabang", "Bangkok"],
    "Indonesia": ["Tanjung Priok", "Surabaya"],
    "India": ["Nhava Sheva", "Chennai", "Mundra"],
    "Germany": ["Hamburg", "Bremen"],
    "Mexico": ["Manzanillo", "Veracruz", "Lazaro Cardenas"],
    "United States": ["Long Beach", "Los Angeles", "Houston", "Savannah", "Newark"],
    "Canada": ["Vancouver", "Montreal"],
    "Brazil": ["Santos", "Paranagua"]
}

DESTINATION_DCS = ["DC_CA_001", "DC_TX_002", "DC_WI_003", "DC_CA_004", "DC_TX_005"]

sku_supplier_map = []
assigned_primary = {}  # sku_id → primary supplier_id

# Assign 1 primary supplier per SKU
for sku in SKU_LIST:
    primary_sid = random.choice(SUPPLIER_IDS)
    assigned_primary[sku] = primary_sid
    sup = next(s for s in suppliers if s["supplier_id"] == primary_sid)

    port_options = PORT_BY_COUNTRY.get(sup["country"], ["Unknown Port"])
    transport = "truck" if sup["country"] in ("United States", "Mexico", "Canada") else random.choice(TRANSPORT_MODES)

    sku_supplier_map.append({
        "supplier_id": primary_sid,
        "sku_id": sku,
        "dependency_percent": round(random.uniform(0.60, 1.00), 2),
        "is_primary_supplier": True,
        "alternate_supplier_available": random.random() < 0.60,  # 60% have an alternate
        "standard_lead_time_days": sup["default_lead_time_days"] + random.randint(-3, 5),
        "origin_country": sup["country"],
        "origin_port": random.choice(port_options),
        "destination_dc": random.choice(DESTINATION_DCS),
        "transport_mode": transport,
        "minimum_order_quantity": random.choice([50, 100, 200, 500, 1000])
    })

# Add 1–2 secondary/alternate suppliers per SKU (subset only)
for sku in random.sample(SKU_LIST, k=int(len(SKU_LIST) * 0.55)):
    n_alts = random.choice([1, 2])
    primary_sid = assigned_primary[sku]
    alternate_pool = [sid for sid in SUPPLIER_IDS if sid != primary_sid]

    for alt_sid in random.sample(alternate_pool, k=min(n_alts, len(alternate_pool))):
        sup = next(s for s in suppliers if s["supplier_id"] == alt_sid)
        port_options = PORT_BY_COUNTRY.get(sup["country"], ["Unknown Port"])
        transport = "truck" if sup["country"] in ("United States", "Mexico", "Canada") else random.choice(TRANSPORT_MODES)
        dep_pct = round(random.uniform(0.10, 0.40), 2)

        sku_supplier_map.append({
            "supplier_id": alt_sid,
            "sku_id": sku,
            "dependency_percent": dep_pct,
            "is_primary_supplier": False,
            "alternate_supplier_available": True,
            "standard_lead_time_days": sup["default_lead_time_days"] + random.randint(-2, 7),
            "origin_country": sup["country"],
            "origin_port": random.choice(port_options),
            "destination_dc": random.choice(DESTINATION_DCS),
            "transport_mode": transport,
            "minimum_order_quantity": random.choice([50, 100, 200, 500])
        })

sku_map_pdf = pd.DataFrame(sku_supplier_map)
print(f"Supplier-SKU map entries generated: {len(sku_map_pdf)}")
print(f"  Primary mappings: {sku_map_pdf[sku_map_pdf.is_primary_supplier].shape[0]}")
print(f"  Alternate mappings: {sku_map_pdf[~sku_map_pdf.is_primary_supplier].shape[0]}")
print(sku_map_pdf.head(10))

# COMMAND ----------

sku_map_sdf = spark.createDataFrame(sku_map_pdf)

(
    sku_map_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_supplier_sku_map")
)

row_count = spark.table("supplysage_bronze.bronze_supplier_sku_map").count()
print(f"✅ bronze_supplier_sku_map written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: `bronze_alternate_suppliers`
# MAGIC
# MAGIC Explicit alternate-supplier qualification table.
# MAGIC This is what the agent checks when it asks: "Can we switch suppliers?"

# COMMAND ----------

SWITCHING_COST_LEVELS = ["low", "medium", "high"]

alt_suppliers = []

# Build from sku_supplier_map where is_primary_supplier = False
alt_map = sku_map_pdf[~sku_map_pdf.is_primary_supplier][["sku_id", "supplier_id"]].copy()
alt_map.rename(columns={"supplier_id": "alternate_supplier_id"}, inplace=True)
alt_map["primary_supplier_id"] = alt_map["sku_id"].map(assigned_primary)

for _, row in alt_map.iterrows():
    alt_suppliers.append({
        "sku_id": row["sku_id"],
        "primary_supplier_id": row["primary_supplier_id"],
        "alternate_supplier_id": row["alternate_supplier_id"],
        "approved_flag": random.random() < 0.75,           # 75% are pre-approved
        "switching_cost_level": random.choice(SWITCHING_COST_LEVELS),
        "estimated_switch_days": random.randint(3, 45),
        "capacity_available_pct": round(random.uniform(0.30, 1.00), 2)
    })

alt_suppliers_pdf = pd.DataFrame(alt_suppliers)
print(f"Alternate supplier records generated: {len(alt_suppliers_pdf)}")
print(f"  Approved: {alt_suppliers_pdf.approved_flag.sum()}")
print(f"  Switching cost distribution:\n{alt_suppliers_pdf.switching_cost_level.value_counts()}")

# COMMAND ----------

alt_sdf = spark.createDataFrame(alt_suppliers_pdf)

(
    alt_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_alternate_suppliers")
)

row_count = spark.table("supplysage_bronze.bronze_alternate_suppliers").count()
print(f"✅ bronze_alternate_suppliers written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5: `bronze_supplier_scorecards`
# MAGIC
# MAGIC Monthly supplier performance metrics (12 months history).
# MAGIC Tier 1 suppliers perform better. Tier 3 have higher variance.
# MAGIC Some suppliers have deteriorating performance over time (storyline for risk events).

# COMMAND ----------

# 12 months back from a reference date (aligned to M5 data range)
REFERENCE_DATE = date(2016, 6, 1)
SCORECARD_MONTHS = [
    (REFERENCE_DATE - timedelta(days=30 * i)).strftime("%Y-%m")
    for i in range(12)
]

# Tier-based performance parameters: (mean, std) for each metric
PERF_PARAMS = {
    "Tier 1": {
        "fill_rate":            (0.96, 0.02),
        "on_time_delivery_rate":(0.94, 0.03),
        "quality_issue_rate":   (0.01, 0.01),
        "avg_lead_time_mult":   (1.00, 0.05),
        "lead_time_variance":   (1.0,  0.5),
        "defect_rate":          (0.005, 0.003)
    },
    "Tier 2": {
        "fill_rate":            (0.90, 0.05),
        "on_time_delivery_rate":(0.85, 0.07),
        "quality_issue_rate":   (0.03, 0.02),
        "avg_lead_time_mult":   (1.05, 0.10),
        "lead_time_variance":   (2.5,  1.0),
        "defect_rate":          (0.015, 0.008)
    },
    "Tier 3": {
        "fill_rate":            (0.82, 0.08),
        "on_time_delivery_rate":(0.74, 0.12),
        "quality_issue_rate":   (0.07, 0.04),
        "avg_lead_time_mult":   (1.15, 0.15),
        "lead_time_variance":   (5.0,  2.0),
        "defect_rate":          (0.03, 0.015)
    }
}

# Suppliers with intentionally deteriorating performance (for risk storylines)
DETERIORATING_SUPPLIERS = random.sample(SUPPLIER_IDS, k=8)

scorecards = []

for sup in suppliers:
    sid = sup["supplier_id"]
    tier = sup["criticality_tier"]
    params = PERF_PARAMS[tier]
    base_lead = sup["default_lead_time_days"]
    is_deteriorating = sid in DETERIORATING_SUPPLIERS

    for month_idx, month_str in enumerate(SCORECARD_MONTHS):
        # Deteriorating suppliers get worse in more recent months (month_idx=0 is most recent)
        degrade_factor = 1.0
        if is_deteriorating and month_idx < 4:
            degrade_factor = 1.0 - (0.08 * (4 - month_idx))  # progressively worse

        fill_rate = float(np.clip(np.random.normal(
            params["fill_rate"][0] * degrade_factor,
            params["fill_rate"][1]), 0.5, 1.0))

        on_time = float(np.clip(np.random.normal(
            params["on_time_delivery_rate"][0] * degrade_factor,
            params["on_time_delivery_rate"][1]), 0.3, 1.0))

        quality_issue = float(np.clip(np.random.normal(
            params["quality_issue_rate"][0] / degrade_factor,
            params["quality_issue_rate"][1]), 0.0, 0.30))

        lead_time_mult = float(np.clip(np.random.normal(
            params["avg_lead_time_mult"][0] / degrade_factor,
            params["avg_lead_time_mult"][1]), 0.8, 2.5))

        avg_lead_time = round(base_lead * lead_time_mult, 1)

        lead_variance = float(np.clip(np.random.normal(
            params["lead_time_variance"][0],
            params["lead_time_variance"][1]), 0.5, 15.0))

        defect_rate = float(np.clip(np.random.normal(
            params["defect_rate"][0] / degrade_factor,
            params["defect_rate"][1]), 0.0, 0.15))

        scorecards.append({
            "supplier_id": sid,
            "scorecard_month": month_str,
            "fill_rate": round(fill_rate, 4),
            "on_time_delivery_rate": round(on_time, 4),
            "quality_issue_rate": round(quality_issue, 4),
            "avg_lead_time_days": round(avg_lead_time, 1),
            "lead_time_variance": round(lead_variance, 2),
            "defect_rate": round(defect_rate, 4),
            "is_deteriorating": is_deteriorating
        })

scorecards_pdf = pd.DataFrame(scorecards)
print(f"Scorecard records generated: {len(scorecards_pdf)}")
print(f"Deteriorating suppliers: {DETERIORATING_SUPPLIERS}")
print(scorecards_pdf.groupby("is_deteriorating")[["fill_rate", "on_time_delivery_rate"]].mean())

# COMMAND ----------

scorecards_sdf = spark.createDataFrame(scorecards_pdf)

(
    scorecards_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_supplier_scorecards")
)

row_count = spark.table("supplysage_bronze.bronze_supplier_scorecards").count()
print(f"✅ bronze_supplier_scorecards written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6: `bronze_purchase_orders`
# MAGIC
# MAGIC Open and closed POs from the last 18 months of M5 data range.
# MAGIC Statuses: open, in_transit, delivered, cancelled, delayed.
# MAGIC Used by the agent to check exposure when a supplier is at risk.

# COMMAND ----------

PO_STATUSES = ["open", "in_transit", "delivered", "delivered", "delivered", "cancelled", "delayed"]

# Date range: 18 months back from reference
PO_START_DATE = date(2015, 1, 1)
PO_END_DATE   = date(2016, 6, 15)

po_records = []
po_counter = 1

# Generate ~40 POs per active supplier (not all have open POs)
ACTIVE_SUPPLIERS_FOR_PO = random.sample(SUPPLIER_IDS, k=45)

for sid in ACTIVE_SUPPLIERS_FOR_PO:
    n_pos = random.randint(25, 55)
    # Get SKUs supplied by this supplier
    sup_skus = sku_map_pdf[sku_map_pdf.supplier_id == sid]["sku_id"].tolist()
    if not sup_skus:
        sup_skus = random.sample(SKU_LIST, k=5)

    for _ in range(n_pos):
        po_id = f"PO_{str(po_counter).zfill(6)}"
        po_counter += 1

        sku = random.choice(sup_skus)
        sup = next(s for s in suppliers if s["supplier_id"] == sid)
        lead = sup["default_lead_time_days"]

        order_date = PO_START_DATE + timedelta(
            days=random.randint(0, (PO_END_DATE - PO_START_DATE).days - lead))

        expected_delivery = order_date + timedelta(days=lead + random.randint(-2, 5))

        status = random.choice(PO_STATUSES)

        # Actual delivery date logic
        if status == "delivered":
            delay = random.randint(-2, 10)
            actual_delivery = expected_delivery + timedelta(days=delay)
        elif status == "delayed":
            actual_delivery = None  # still outstanding
        elif status in ("open", "in_transit"):
            actual_delivery = None
        else:  # cancelled
            actual_delivery = None

        qty_ordered = random.choice([100, 200, 500, 750, 1000, 1500, 2000])
        if status == "delivered":
            fill_pct = random.uniform(0.90, 1.00) if sid not in DETERIORATING_SUPPLIERS else random.uniform(0.70, 0.95)
            qty_received = int(qty_ordered * fill_pct)
        else:
            qty_received = 0

        po_records.append({
            "po_id": po_id,
            "po_line_id": f"{po_id}_L001",
            "supplier_id": sid,
            "sku_id": sku,
            "order_date": order_date.isoformat(),
            "expected_delivery_date": expected_delivery.isoformat(),
            "actual_delivery_date": actual_delivery.isoformat() if actual_delivery else None,
            "quantity_ordered": qty_ordered,
            "quantity_received": qty_received,
            "status": status
        })

po_pdf = pd.DataFrame(po_records)
print(f"PO records generated: {len(po_pdf)}")
print(f"Status distribution:\n{po_pdf.status.value_counts()}")
print(f"Open/In-transit POs (active exposure): {po_pdf[po_pdf.status.isin(['open','in_transit','delayed'])].shape[0]}")

# COMMAND ----------

po_sdf = spark.createDataFrame(po_pdf)

(
    po_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_purchase_orders")
)

row_count = spark.table("supplysage_bronze.bronze_purchase_orders").count()
print(f"✅ bronze_purchase_orders written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7: `bronze_shipment_routes`
# MAGIC
# MAGIC Trade lane definitions used by the agent to match geographic disruption events
# MAGIC (port strikes, weather, tariffs) to specific supplier routes.

# COMMAND ----------

CARRIERS = {
    "ocean": ["Maersk", "MSC", "CMA CGM", "COSCO", "Hapag-Lloyd", "ONE", "Evergreen"],
    "air":   ["FedEx Freight", "UPS Air Cargo", "DHL Express", "Amazon Air"],
    "truck": ["J.B. Hunt", "Werner Enterprises", "XPO Logistics", "Schneider National"],
    "rail":  ["Union Pacific", "BNSF Railway", "CSX Transportation"]
}

RISK_REGIONS = {
    "Asia Pacific":   ["South China Sea", "Taiwan Strait", "Port of Shanghai congestion"],
    "South Asia":     ["Indian Ocean routes", "Bangladesh port delays"],
    "Europe":         ["North Sea weather", "Rotterdam congestion"],
    "North America":  ["US-Mexico border crossing", "Gulf of Mexico weather"],
    "Latin America":  ["Panama Canal congestion", "Brazil port delays"]
}

routes = []
route_counter = 1

for sid in SUPPLIER_IDS:
    sup = next(s for s in suppliers if s["supplier_id"] == sid)
    country = sup["country"]
    region = sup["region"]
    port_options = PORT_BY_COUNTRY.get(country, ["Unknown Port"])

    # Each supplier has 1–3 routes
    n_routes = random.randint(1, 3)
    for _ in range(n_routes):
        route_id = f"ROUTE_{str(route_counter).zfill(4)}"
        route_counter += 1

        is_domestic = country in ("United States",)
        transport = "truck" if is_domestic else (
            "air" if random.random() < 0.10 else
            "rail" if country in ("Mexico", "Canada") and random.random() < 0.25 else
            "ocean"
        )

        carriers = CARRIERS.get(transport, CARRIERS["ocean"])
        carrier = random.choice(carriers)

        base_transit = {
            "ocean": random.randint(18, 40),
            "air":   random.randint(2, 5),
            "truck": random.randint(1, 7),
            "rail":  random.randint(5, 14)
        }[transport]

        risk_options = RISK_REGIONS.get(region, ["General route risk"])
        risk_region = random.choice(risk_options)

        routes.append({
            "route_id": route_id,
            "supplier_id": sid,
            "origin_country": country,
            "origin_port": random.choice(port_options),
            "destination_dc": random.choice(DESTINATION_DCS),
            "transport_mode": transport,
            "carrier": carrier,
            "standard_transit_days": base_transit,
            "risk_region": risk_region
        })

routes_pdf = pd.DataFrame(routes)
print(f"Shipment routes generated: {len(routes_pdf)}")
print(f"Transport mode distribution:\n{routes_pdf.transport_mode.value_counts()}")
print(routes_pdf.head(10))

# COMMAND ----------

routes_sdf = spark.createDataFrame(routes_pdf)

(
    routes_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_bronze.bronze_shipment_routes")
)

row_count = spark.table("supplysage_bronze.bronze_shipment_routes").count()
print(f"✅ bronze_shipment_routes written: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final verification: Row counts for all 7 synthetic Bronze tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'bronze_suppliers'          AS table_name, COUNT(*) AS row_count FROM supplysage_bronze.bronze_suppliers
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_supplier_aliases',                 COUNT(*) FROM supplysage_bronze.bronze_supplier_aliases
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_supplier_sku_map',                 COUNT(*) FROM supplysage_bronze.bronze_supplier_sku_map
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_alternate_suppliers',              COUNT(*) FROM supplysage_bronze.bronze_alternate_suppliers
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_supplier_scorecards',              COUNT(*) FROM supplysage_bronze.bronze_supplier_scorecards
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_purchase_orders',                  COUNT(*) FROM supplysage_bronze.bronze_purchase_orders
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_shipment_routes',                  COUNT(*) FROM supplysage_bronze.bronze_shipment_routes
# MAGIC ORDER BY table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot check: Supplier SKU map — join back to M5 item hierarchy

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     m.supplier_id,
# MAGIC     s.supplier_name,
# MAGIC     s.country,
# MAGIC     s.criticality_tier,
# MAGIC     m.sku_id,
# MAGIC     m.dependency_percent,
# MAGIC     m.is_primary_supplier,
# MAGIC     m.transport_mode,
# MAGIC     m.origin_port,
# MAGIC     m.destination_dc
# MAGIC FROM supplysage_bronze.bronze_supplier_sku_map m
# MAGIC JOIN supplysage_bronze.bronze_suppliers s ON m.supplier_id = s.supplier_id
# MAGIC WHERE m.is_primary_supplier = TRUE
# MAGIC ORDER BY s.criticality_tier, m.dependency_percent DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot check: Scorecards — deteriorating vs stable suppliers

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     sc.supplier_id,
# MAGIC     s.supplier_name,
# MAGIC     s.criticality_tier,
# MAGIC     sc.scorecard_month,
# MAGIC     sc.fill_rate,
# MAGIC     sc.on_time_delivery_rate,
# MAGIC     sc.defect_rate,
# MAGIC     sc.is_deteriorating
# MAGIC FROM supplysage_bronze.bronze_supplier_scorecards sc
# MAGIC JOIN supplysage_bronze.bronze_suppliers s ON sc.supplier_id = s.supplier_id
# MAGIC WHERE sc.is_deteriorating = TRUE
# MAGIC ORDER BY sc.supplier_id, sc.scorecard_month DESC
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot check: Open POs — active supplier exposure

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     po.supplier_id,
# MAGIC     s.supplier_name,
# MAGIC     s.criticality_tier,
# MAGIC     po.status,
# MAGIC     COUNT(*) AS po_count,
# MAGIC     SUM(po.quantity_ordered) AS total_units_at_risk
# MAGIC FROM supplysage_bronze.bronze_purchase_orders po
# MAGIC JOIN supplysage_bronze.bronze_suppliers s ON po.supplier_id = s.supplier_id
# MAGIC WHERE po.status IN ('open', 'in_transit', 'delayed')
# MAGIC GROUP BY po.supplier_id, s.supplier_name, s.criticality_tier, po.status
# MAGIC ORDER BY total_units_at_risk DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Notebook complete
# MAGIC
# MAGIC All 7 synthetic Bronze tables have been written to `supplysage_bronze`.
# MAGIC
# MAGIC **Full Bronze table inventory is now:**
# MAGIC
# MAGIC | Layer | Table | Source |
# MAGIC |---|---|---|
# MAGIC | Raw (Kaggle) | `bronze_m5_calendar` | M5 Kaggle |
# MAGIC | Raw (Kaggle) | `bronze_m5_sell_prices` | M5 Kaggle |
# MAGIC | Raw (Kaggle) | `bronze_m5_sales_train_validation` | M5 Kaggle |
# MAGIC | Raw (Kaggle) | `bronze_retail_inventory` | Retail Inventory Kaggle |
# MAGIC | Raw (Kaggle) | `bronze_dataco_supply_chain` | DataCo Kaggle |
# MAGIC | Raw (Kaggle) | `bronze_dataco_description` | DataCo Kaggle |
# MAGIC | Synthetic | `bronze_suppliers` | This notebook |
# MAGIC | Synthetic | `bronze_supplier_aliases` | This notebook |
# MAGIC | Synthetic | `bronze_supplier_sku_map` | This notebook |
# MAGIC | Synthetic | `bronze_alternate_suppliers` | This notebook |
# MAGIC | Synthetic | `bronze_supplier_scorecards` | This notebook |
# MAGIC | Synthetic | `bronze_purchase_orders` | This notebook |
# MAGIC | Synthetic | `bronze_shipment_routes` | This notebook |
# MAGIC
# MAGIC **Next step:** Run `04_silver_transform_m5` to begin Silver layer transformations.
