# Building `01_Road_Segments`

**Project:** Saskatchewan Road Deterioration Research  
**Document type:** Data provenance, preparation, integration, and quality-assurance notes  
**Snapshot documented:** 21 October 2026  
**Geographic scope:** Saskatchewan's **provincial highway network**; this is a Saskatchewan-wide project, **not** a Saskatoon-only project. It does not yet cover every municipal or rural road in the province.

> **What this document is:** A reconstruction of the documented workflow, checked against the saved workbook, exported CSV, and the Ministry's original shapefile. It is **not** a claim that I possess the complete original ETL script or its execution logs. Exact source-download settings, any unpublished intermediate code, and the source layer's effective date require further documentation.

> **Dataset included here:** The analysis uses `data/road_segments.csv`, a project-prepared snapshot assembled from multiple sources for learning and exploration. The original Ministry-supplied archive and shapefile are **not included** in this repository. The preparation described below includes surface attributes derived from that source.

> **Reuse:** The source terms and permissions applicable to sharing this prepared snapshot, including derived attributes, have not been confirmed; an educational purpose alone does not establish them.

## 1. Why I built this dataset

The research started with a practical question: *Why do some Saskatchewan roads deteriorate faster than others, and can I predict when they will need maintenance?*

The original research motivation included historical road condition, traffic, maintenance and environmental factors, and the possibility of estimating maintenance needs **1, 3, and 5 years** ahead. This showcase does not make those predictions.

The present `01_Road_Segments` dataset is the **road-network and surface-type foundation** for that research. It is **not yet** a longitudinal pavement-condition dataset, and it does **not** contain verified maintenance outcomes sufficient to train the intended deterioration model.

The original project reflection is not included as a linked external resource; the research purpose is summarized above.

## 2. The sources

| Source | Origin | What it provides | Status |
|---|---|---|---|
| **`HIGHWAY_OFFICIAL`** | Government of Saskatchewan / Saskatchewan GeoHub | Provincial/national highway segments, kilometer references, road identifiers, attributes and geometry | Imported as the foundation |
| **`highways_surfType.zip`** | Saskatchewan Ministry of Highways; sent directly in response to a data request, **14 Oct 2026** | GIS line features with official highway surface classes and kilometre intervals | Used in preparation **21 Oct 2026**; original archive not included |
| Official surface-type map and public metadata | Government of Saskatchewan | Surface-class definitions and background documentation | Reference material |
| Original working workbook | My working data preparation | Formula-linked cleaned records, mapping decisions, integration QA | Preserved separately; not needed to use the project CSV |

**Original network references:**

- [Saskatchewan GeoHub — HIGHWAY_OFFICIAL](https://geohub.saskatchewan.ca/datasets/saskatchewan::highway-official)
- [Government of Canada open-data catalogue — HIGHWAY_OFFICIAL](https://open.canada.ca/data/en/dataset/988a8b95-fa49-adc8-bbe7-6e69bb26099c)
- [ArcGIS — Highway Official Road Network](https://www.arcgis.com/home/item.html?id=2e32c6c48fb0468fb6c1b32cd306a3c5)
- Ministry-supplied archive: `highways_surfType.zip` (received directly from the Ministry)

`HIGHWAY_OFFICIAL` is based on a **linear referencing system (LRS)**: `ROADNAME` identifies a highway/control-section alignment, while `KM_FROM` and `KM_TO` describe a section along it. The network also contains `NID` identifiers and other roadway attributes. These kilometre references were especially useful when joining the two sources.

**Source-date distinction:** The Ministry archive was *received* on 14 Oct 2026. Its ZIP members have modification timestamps around 7–8 Oct 2026. **Neither date establishes when the road surface conditions were surveyed or became effective.** I deliberately left `SURFACE_SOURCE_YEAR` blank rather than labeling the classification as “2026” without confirmation.

## 3. Timeline

| When | What happened | Result |
|---|---|---|
| **29 Sep 2026** | Acquired/imported Saskatchewan's `HIGHWAY_OFFICIAL` road-network data. | Foundation dataset with approximately 25,415 working road-segment records and a 47-field source schema. |
| **29 Sep 2026** | Created a cleaner, formula-linked `Clean_Master` view. | 22 retained/derived working columns; calculated segment length; preserved geometry and original source. |
| **29–30 Sep 2026** | Created `Surface_Analysis` as a separate working layer. | Six surface classification/quality fields; 2,562 initially identified gravel records from original pavement codes. Other surface classes remained unassigned pending an authoritative layer. |
| **29 Sep 2026** | Requested the underlying GIS/vector surface-type layer from Saskatchewan's eGIS/Ministry contacts. | Sought classifications by road segment and kilometre interval, rather than estimating them from a map image. |
| **14 Oct 2026** | Received `highways_surfType.zip` from the Ministry of Highways. | Official surface-type shapefile available for segment matching. |
| **21 Oct 2026** | Joined the Ministry intervals to the highway records; validated and applied the Ministry-first classification policy. | 25,035 classified records; 380 remain flagged. |
| **21 Oct 2026** | Copied the workbook for the integrated version; renamed/reordered the working tabs and hid the original views. | `01_Road_Segments` became the main working table; raw/pre-integration data remains preserved. |
| **21 Oct 2026** | Exported `01_Road_Segments` as a standalone CSV. | **25,415 data rows × 33 columns**, excluding the header row. |

## 4. How the road-network base table was prepared

### 4.1 Import and preservation

The initial working source was the provincial highway layer, **`HIGHWAY_OFFICIAL`**. In my original workbook, the full network was retained in `Highway_Official_Road_Network` (now renamed **`90_Raw_Road_Network`**). Its underlying columns include official identifiers, road and route attributes, pavement codes, linear-reference kilometre ranges, and spatial geometry.

The original source tab was **not overwritten** by my surface classification work. This matters because it allows me to revisit my cleaning and matching decisions later.

### 4.2 Cleaned, formula-linked view

A derived view called **`Clean_Master`** (now **`91_Clean_Pre_Integration`**) selected the useful fields through an array formula, filtered out rows with blank `ROADNAME`, and kept the records linked back to the source network. A second working view, `Surface_Analysis`, used the same 22-field foundation. That view was later renamed `01_Road_Segments`.

The retained base schema is:

| Group | Fields |
|---|---|
| Road identity / location | `ROADNAME`, `PLACENAME`, `POSTEDNAME`, `NID`, `PLACENAME_` |
| Chainage and measurements | `KM_FROM`, `KM_TO`, `SEGMENT_LENGTH_KM`, `Shape__Length` |
| Roadway properties | `SPEEDLIMIT`, `TRAVELDIR`, `ROADCLASS`, `NBRLANES` |
| Route identification | `RTENAME1EN`, `RTENAME2EN`, `RTNUMBER1`, `RTNUMBER2` |
| Existing pavement fields | `PAVSTATUS`, `UNPAVSURF` |
| Structure information | `STRUCTID`, `STRUCTTYPE` |
| Spatial geometry | `geometry_json` |

**Derived measurement:**

```text
SEGMENT_LENGTH_KM = KM_TO - KM_FROM
```

In the exported CSV, this arithmetic matches the saved values to floating-point precision. `Shape__Length` is a **separate source field**; it should not be assumed to be the same type of distance or in the same units until the originating GIS metadata is checked. The `geometry_json` field preserves serialized geographic paths for later spatial work; four rows in the current export have this field missing.

**Identifier caution:** `NID` alone is **not unique** in the exported table. In this snapshot, the combination `ROADNAME + KM_FROM + KM_TO + NID` distinguishes all 25,415 rows. Continue to use a stable composite/source identifier rather than assuming `NID` alone is a primary key.

### 4.3 Preliminary gravel classification

Before the Ministry supplied the GIS layer, I could identify some gravel roads directly from the original network fields:

```text
PAVSTATUS = 2  AND  UNPAVSURF = 1  →  Gravel
```

This produced **2,562 provisional gravel classifications** as of 30 Sep 2026. I did **not** infer Asphalt, Granular Paved, TMS, or Winter Ice Road for all other segments from those two fields. They were left blank until the underlying official surface layer was received.

This provisional step explains why the later Ministry data could disagree with the original gravel-coded fields. It was a preliminary direct attribute classification, not an independently surveyed surface inventory.

## 5. What was inside the Ministry's shapefile

The archive `highways_surfType.zip` contains the standard shapefile components, including `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg`, in a `highways_surfType/` directory.

**Verified geometry and attributes:**

| Property | Actual value |
|---|---|
| Features | **3,903** |
| Geometry | Polyline |
| `ROADSEGMEN` | Road-segment reference code; corresponds to the network's `ROADNAME` field |
| `FROMKM` | Start kilometre of the government's classified interval |
| `TOKM` | End kilometre of the government's classified interval |
| `SURFACETYP` | Official surface classification code |
| Projection in the shapefile `.prj` | **NAD83 / UTM Zone 13 North**, projected in **metres** |

**Official source codes used in my integration:**

| `SURFACETYP` | `SURFACE_TYPE` in my table | Number of *source shapefile features* |
|---|---|---:|
| `AC` | Asphalt (AC Paved) | 2,044 |
| `TMS` | Thin Membrane Surface (TMS) | 764 |
| `GRAN` | Granular Paved | 335 |
| `Gravel` | Gravel | 755 |
| `IceRoad` | Winter Ice Road | 5 |
| **Total** | | **3,903** |

These counts describe **source GIS features**, not final road-network rows. One source interval can overlap many shorter road-network segments, so the final per-class segment counts are different.

## 6. How I integrated the datasets

### 6.1 Why I used a road-and-kilometre join

The two datasets divide the highway system into **different lengths of road sections**. A row-by-row positional join would be incorrect. Instead, the integration matched:

1. `01_Road_Segments.ROADNAME` to `highways_surfType.ROADSEGMEN`;
2. the base segment's interval `[KM_FROM, KM_TO]` to the government's `[FROMKM, TOKM]`;
3. the surface class that overlapped that interval, taking account of coverage and class dominance.

The fundamental interval-overlap calculation is:

```text
overlap_km = max(0, min(KM_TO, TOKM) - max(KM_FROM, FROMKM))
```

For each base segment, I evaluated matching source intervals, how much of the base segment they covered, and whether one government surface class clearly dominated. The saved output records **matched coverage percentage** (`SURFACE_MATCH_PCT`) and **dominant-class percentage** (`GOV_DOMINANT_PCT`).

> **Reproducibility note:** The original per-row matching script is not retained in the workbook. The calculations above describe the traceable join approach; they are not a verbatim copy of that script. Inspection of the final output shows the accepted Ministry rows have both coverage and dominant-class percentages **at least 99.5%** (allowing for the displayed two-decimal rounding), while the mixed/partial group falls below that standard. Treat **99.5% as the reconstructable acceptance rule**, and confirm it against saved ETL code if that code becomes available. Overlapping GIS intervals, rounding and boundary conditions need deliberate handling in any independent reimplementation.

### 6.2 Source priority and disagreements

I chose the **directly supplied Ministry surface-type layer as the primary surface classification** when its road/kilometre match was reliable.

- If the new Ministry class agreed with the old pavement attributes, keep that class.
- If it **disagreed**, use the Ministry class in `SURFACE_TYPE`, while **retaining** the original `PAVSTATUS` and `UNPAVSURF` values.
- Record the disagreement in `SOURCE_DISAGREEMENT` and describe it in `SURFACE_QA_NOTE`.
- If the new Ministry coverage was partial/mixed, do **not** automatically assign one class to the whole base segment.
- If there was no reliable Ministry overlap but the original source had an unambiguous gravel code, keep that direct-source gravel class.

This is a **source-precedence policy**, not proof that every value in either source is survey-current. A newer *delivery date* is not sufficient evidence of a newer *observation date*.

### 6.3 Joined fields added to the original 22 columns

| Export column | Role |
|---|---|
| `SURFACE_TYPE` | Chosen final human-readable class, where defensible |
| `SURFACE_SOURCE_YEAR` | Actual observation year, **left blank** because not verified |
| `SURFACE_MATCH_METHOD` | How the class was obtained, or why it could not be assigned |
| `SURFACE_MATCH_PCT` | Percent of the base interval covered by matched government intervals |
| `SURFACE_CONFIDENCE` | `HIGH`, `REVIEW`, or `UNMATCHED` |
| `SURFACE_QA_NOTE` | Human-readable explanation of the result |
| `GOV_SURFACETYP` | Raw/dominant government surface code, including on review records |
| `GOV_DOMINANT_PCT` | Percent attributed to the leading government class |
| `GOV_SOURCE_INDEX` | Source feature reference for tracing back to the shapefile; treat its indexing convention as an implementation detail pending script review |
| `SURFACE_SOURCE_FILE` | Archive name and received-date provenance |
| `SOURCE_DISAGREEMENT` | Nonblank where the preserved network attributes disagree with the accepted Ministry class |

**Total schema:** 22 base columns + 11 integration/QA columns = **33 columns**.

## 7. What my validation found

All counts below were **recomputed from `data/road_segments.csv`** rather than taken only from the Overview sheet.

### 7.1 Matching status

| Match result | Records | Treatment |
|---|---:|---|
| `GOV_ROADSEG_KM_RANGE` | **25,033** | Accept Ministry class; confidence `HIGH` |
| `DIRECT_UNPAVSURF_CODE` | **2** | Retain original verified gravel class where no Ministry overlap exists; confidence `HIGH` |
| `INCOMPLETE_OR_MIXED` | **350** | Leave `SURFACE_TYPE` blank; flag `REVIEW` |
| `NO_RELIABLE_MATCH` | **30** | Leave `SURFACE_TYPE` blank; flag `UNMATCHED` |
| **Total** | **25,415** | |

Thus **25,035 of 25,415** rows have a final surface class (**98.50%** to two decimal places); **380** remain unresolved.

### 7.2 Final surface classifications

| Final `SURFACE_TYPE` | Road-network rows |
|---|---:|
| Asphalt (AC Paved) | 14,595 |
| Thin Membrane Surface (TMS) | 4,475 |
| Granular Paved | 3,126 |
| Gravel | 2,823 |
| Winter Ice Road | 16 |
| **Classified total** | **25,035** |
| **Unclassified** | **380** |

### 7.3 Disagreement checks

**388** road segments had a reliable government match but were coded as unpaved gravel in the earlier `HIGHWAY_OFFICIAL` pavement fields. I **accepted the Ministry's class** for the main `SURFACE_TYPE` while preserving the discrepancy:

| Accepted Ministry class | Previously disputed rows |
|---|---:|
| Thin Membrane Surface (TMS) | 202 |
| Asphalt (AC Paved) | 97 |
| Granular Paved | 81 |
| Winter Ice Road | 8 |
| **Total** | **388** |

Those **388 are included in the 25,033 accepted Ministry matches**; they are not additional rows, and they should not be counted again. There are exactly 388 nonblank `SOURCE_DISAGREEMENT` cells in the exported snapshot.

### 7.4 Additional checks and caveats

- The exported `SEGMENT_LENGTH_KM` values equal `KM_TO - KM_FROM` to floating-point precision, and no records have zero or negative derived lengths.
- Four records have missing `geometry_json`, despite having usable road/km-based surface classifications. **Do not equate a successful linear-reference match with a successful geometry validation for every row.**
- The network includes repeated `NID` values. Use a suitable composite ID for analysis and joins.
- The first geographic spot checks of matched roads were encouraging; they **do not** replace exhaustive GIS overlay QA for the whole province.
- Some of the 350 `INCOMPLETE_OR_MIXED` records have essentially full overall coverage but contain **more than one surface type**. These should be **split into homogeneous sections or retained as mixed**, rather than assigned their dominant class without review.
- Exact survey/effective date, original survey methodology, and source terms applicable to sharing the prepared dataset and its derived attributes remain unconfirmed.

## 8. Project data file

Use the local **`data/road_segments.csv`** in this repository as the working dataset. It contains all 25,415 data rows and 33 columns. It is a snapshot of the prepared road-segment data, not a live workbook or a copy of formulas.

`geometry_json` contains commas and quotation marks; any CSV reader must use proper CSV quoting rather than splitting lines on commas.

## 9. Minimal audit checklist

For a quick independently repeatable check of the exported file:

```python
import pandas as pd

df = pd.read_csv("data/road_segments.csv", low_memory=False)

assert df.shape == (25_415, 33)
assert (df["SEGMENT_LENGTH_KM"] > 0).all()
assert (df["KM_TO"] - df["KM_FROM"] - df["SEGMENT_LENGTH_KM"]).abs().max() < 1e-8
assert df["SURFACE_TYPE"].notna().sum() == 25_035
assert df["SOURCE_DISAGREEMENT"].notna().sum() == 388
assert df["SURFACE_MATCH_METHOD"].value_counts().to_dict() == {
    "GOV_ROADSEG_KM_RANGE": 25_033,
    "INCOMPLETE_OR_MIXED": 350,
    "NO_RELIABLE_MATCH": 30,
    "DIRECT_UNPAVSURF_CODE": 2,
}
assert not df.duplicated(["ROADNAME", "KM_FROM", "KM_TO", "NID"]).any()

print("Snapshot integrity checks passed.")
```

For a **full re-run** from raw sources, an independent processing script must additionally read the source shapefile, implement interval matching and boundary handling, record which GIS feature(s) contribute to each match, and compare its output against the 33-column snapshot. This audit snippet **checks** the export; it does not recreate the integration.

## 10. Scope and data limitations

- The snapshot retains 350 mixed/incomplete surface overlaps and 30 records
  without a reliable Ministry match. The Ministry layer's effective date is
  unconfirmed.
- Dated condition, maintenance, traffic and environmental histories are not
  part of this snapshot. Surface class alone cannot establish when a road
  will fail, and the regression showcase does not predict deterioration.
- Source versions, effective dates, field definitions, join scripts and
  exhaustive GIS quality assurance are not all available for this export.

## 11. Files and provenance references

- **Project dataset:** `data/road_segments.csv` (created 21 Oct 2026; included in this project)
- **Methodology document:** `docs/road_segments_methodology.md`
- **Ministry source archive:** `highways_surfType.zip` (received directly; not included in this repository)
- **Integration audit and review files:** retained as separate provenance materials; not required to access or analyze the project CSV.
- **Government base-network description:** [HIGHWAY_OFFICIAL public catalogue](https://open.canada.ca/data/en/dataset/988a8b95-fa49-adc8-bbe7-6e69bb26099c)

---

**Bottom line:** `01_Road_Segments` is a documented **Saskatchewan provincial highway inventory with Ministry-priority surface-type enrichment**. The exported snapshot can be audited, but the original integration cannot be fully reproduced from the retained materials; it is not a historical deterioration-prediction training set.
