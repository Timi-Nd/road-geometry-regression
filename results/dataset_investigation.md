# Dataset investigation and feature selection

## Scope and reproducibility

This is a descriptive audit of the full [road segment snapshot](../data/road_segments.csv),
not a model comparison. It contains 25,415 rows and 33 columns. The audited
file's SHA-256 is
`97897e1c149900e28e0fe6b7e9d976f10200fc81096a2042a87b6aa63dca9313`.
The original CSV is not changed. From the repository root, reproduce the
machine-readable profile with:

```sh
python src/investigate_dataset.py
```

The script writes JSON to standard output. It counts empty or whitespace-only
fields as blanks and separately counts the literal string `None`; it does not
silently assume that token's meaning. Pandas' default CSV NA handling treats
`None` as missing in some versions, so its counts may differ from the raw blank
counts below. Numeric quality checks additionally distinguish nonnumeric and
non-finite values. Quantiles use pandas' linear interpolation. Correlations
are Pearson correlations over available finite pairs.

The descriptive target correlations below use **all rows**, including rows
that were later used for testing. They characterize this snapshot only and
must not be used to select features or tune models for a held-out evaluation.
The provisional feature choices below are based on field semantics and
potential availability, not on those correlations; their actual availability
at prediction time is unconfirmed.

## Target and available numerical fields

The target for continuity with the existing experiment is `Shape__Length`
(GIS geometry length, **source units unconfirmed**). `SEGMENT_LENGTH_KM` is
the recorded kilometre-reference interval length, not an independent road
condition measurement. See [dataset provenance](../docs/road_segments_methodology.md)
and [previous findings](findings.md).

### Current public-service metadata

The [Government of Canada catalogue entry](https://open.canada.ca/data/en/dataset/988a8b95-fa49-adc8-bbe7-6e69bb26099c)
links to the Saskatchewan `HIGHWAY_OFFICIAL` feature service. Its
[current layer metadata](https://services3.arcgis.com/zcv98lgAl8xQ04cW/arcgis/rest/services/HIGHWAY_OFFICIAL/FeatureServer/0?f=pjson),
accessed 28 October 2026, declares spatial reference WKID 102100 / latest
WKID 3857 (Web Mercator), and identifies `Shape__Length` as the shape-length
field with `esriMeters` units. This supports the earlier projection
interpretation for the **current service**, not proof of the units or CRS of
the historical CSV values: the exact source download and original ETL are
not retained, and the snapshot's `geometry_json` coordinates have their own
unconfirmed export CRS. Web Mercator metres are projected distances, not
necessarily ground metres.

The same metadata describes `SPEEDLIMIT` as a double and `NBRLANES` as an
integer, with no coded domain listed for either. It does not establish the
meaning of a zero speed value or these fields' availability at prediction
time. The catalogue links a public-data licence for the base layer, but this
does not establish the terms for sharing the prepared snapshot or its surface
attributes described as derived from a separately supplied Ministry layer.
The original Ministry file is not included in the repository.

| Field | Role and assessment |
|---|---|
| `SEGMENT_LENGTH_KM` | Baseline predictor, kilometres; derived as `KM_TO - KM_FROM`. A closely related measurement of the target, so this is not a general road-condition prediction task. |
| `SPEEDLIMIT` | Possible additional numeric road attribute, provided its unit, coding and availability before prediction are verified. 157 values are zero; do not assume these mean a real zero or missing without a data dictionary. |
| `NBRLANES` | Possible additional numeric road attribute, if its definition and availability are confirmed. Count is heavily concentrated at two; a linear effect is an assumption, not established fact. |
| `KM_FROM`, `KM_TO` | Chainage in kilometres. Do not combine both with `SEGMENT_LENGTH_KM` and an intercept: the exact linear relation makes the design rank deficient. Their value also depends on the road's reference system. |
| `SURFACE_MATCH_PCT`, `GOV_DOMINANT_PCT` | Percentages describing the surface integration's coverage and class dominance, not intrinsic physical road predictors. Exclude from initial models; 32 blanks each. |
| `ROADCLASS`, `TRAVELDIR`, `PAVSTATUS`, `UNPAVSURF`, `STRUCTTYPE` | Numeric **codes**, not measured continuous quantities. Do not treat the code values as ordered numeric predictors without definitions and appropriate encoding. |
| `RTNUMBER1`, `RTNUMBER2`, `GOV_SOURCE_INDEX` | Route or source identifiers, not continuous measurements; exclude. |
| `SURFACE_SOURCE_YEAR` | Entirely blank; the source observation date is unconfirmed. |
| `geometry_json` | Source geometry, not an independent numeric input. Computing geometry length from it would directly reconstruct the target. Coordinates derived from the same geometry may also be unavailable or leak information in a missing-geometry use case. |

`SURFACE_TYPE` is a categorical road attribute, but its source effective date
is unconfirmed and 380 values are blank. It would require explicit encoding,
availability checks and training-only preprocessing; it is not included in
the initial numerical comparison. No traffic, pavement age, historical
condition, climate or maintenance outcome columns are present in this
snapshot.

## Missing values and quality

Only fields with nonzero raw blanks or literal `None` are listed; all other
fields have zero of both. Counts are out of 25,415 rows.

| Field | Blank | Blank % | Literal `None` |
|---|---:|---:|---:|
| `POSTEDNAME` | 20,870 | 82.12 | 0 |
| `RTENAME1EN` | 22,869 | 89.98 | 116 |
| `RTENAME2EN` | 24,123 | 94.92 | 116 |
| `RTNUMBER1` | 5,406 | 21.27 | 0 |
| `RTNUMBER2` | 24,007 | 94.46 | 105 |
| `STRUCTID` | 24,460 | 96.24 | 115 |
| `PLACENAME_` | 1,377 | 5.42 | 0 |
| `geometry_json` | 4 | 0.02 | 0 |
| `SURFACE_TYPE` | 380 | 1.50 | 0 |
| `SURFACE_SOURCE_YEAR` | 25,415 | 100.00 | 0 |
| `SURFACE_MATCH_PCT` | 32 | 0.13 | 0 |
| `GOV_SURFACETYP` | 32 | 0.13 | 0 |
| `GOV_DOMINANT_PCT` | 32 | 0.13 | 0 |
| `GOV_SOURCE_INDEX` | 32 | 0.13 | 0 |
| `SOURCE_DISAGREEMENT` | 25,027 | 98.47 | 0 |

These are field-presence counts, not necessarily errors: for example, a blank
disagreement field can mean no recorded disagreement. The audit found zero
blank, nonnumeric or non-finite values in the target, chainage, segment length,
speed-limit and lane-count fields. Zero-coded speed limits remain an unresolved
quality question rather than verified missingness.

The geometry check found 25,411 nonblank JSON objects with nonempty paths and
finite coordinate pairs, four blanks, and no structural failures under that
check. Coordinate bounds are approximately x = -110.056 to -101.362 and
y = 48.999 to 59.611. These ranges are consistent with longitude/latitude,
but do **not** establish the CRS or prove geometries are accurate, correctly
matched, or topologically valid. The units of `Shape__Length` are also not
verified by this check.

## Distributions

All shown values are finite; the target and proposed numerical predictors have
25,415 values each. Speed-limit units and any special codes need source
confirmation; lane count is a count.

| Field | Min | 25% | Median | 75% | 99% | Max |
|---|---:|---:|---:|---:|---:|---:|
| `SEGMENT_LENGTH_KM` (km) | 0.003 | 0.120 | 0.550 | 1.627 | 6.544 | 100.806 |
| `Shape__Length` (source units) | 4.683 | 194.627 | 885.732 | 2,564.074 | 11,216.354 | 187,320.368 |
| `SPEEDLIMIT` (source units) | 0 | 60 | 90 | 100 | 110 | 110 |
| `NBRLANES` (count) | 1 | 2 | 2 | 2 | 4 | 5 |

Speed limits are concentrated at 100 (9,064 rows), 80 (3,999), 90 (3,583),
40 (2,522) and 50 (2,495); 157 are zero. Lane counts are 1: 604, 2: 24,178,
3: 357, 4: 251, and 5: 25. Thus 95.13% are two-lane records, which limits
the support for estimating differences among lane counts.

## Associations and leakage

Pearson correlations on all 25,415 rows:

| | `SEGMENT_LENGTH_KM` | `Shape__Length` | `KM_FROM` | `KM_TO` | `SPEEDLIMIT` | `NBRLANES` |
|---|---:|---:|---:|---:|---:|---:|
| `SEGMENT_LENGTH_KM` | 1.0000 | 0.9977 | 0.1246 | 0.2515 | 0.2011 | -0.0277 |
| `Shape__Length` | 0.9977 | 1.0000 | 0.1209 | 0.2475 | 0.1771 | -0.0254 |
| `KM_FROM` | 0.1246 | 0.1209 | 1.0000 | 0.9917 | 0.2474 | 0.0231 |
| `KM_TO` | 0.2515 | 0.2475 | 0.9917 | 1.0000 | 0.2674 | 0.0189 |
| `SPEEDLIMIT` | 0.2011 | 0.1771 | 0.2474 | 0.2674 | 1.0000 | -0.0400 |
| `NBRLANES` | -0.0277 | -0.0254 | 0.0231 | 0.0189 | -0.0400 | 1.0000 |

The maximum absolute discrepancy between `KM_TO - KM_FROM` and
`SEGMENT_LENGTH_KM` is approximately 1.93e-14 km. The target-to-recorded
length ratio, `Shape__Length / (1000 * SEGMENT_LENGTH_KM)`, has median 1.6067;
228 ratios lie outside 1.5 to 2.0. Their cause has not been established.
The earlier report's Web Mercator explanation is an inference for this
snapshot; the current public-service metadata supports it but does not confirm
the historical export. The very high length correlation can primarily reflect two
measurements of the same geometry; it does not establish predictive
capability for road deterioration.

## Experimental design and limitations

The [completed comparison](model_comparison.md) uses the following
**exploratory** feature sets; field definitions and prediction-time
availability remain unconfirmed:

1. A: `SEGMENT_LENGTH_KM` (the existing baseline).
2. B: `SEGMENT_LENGTH_KM` + `SPEEDLIMIT`.
3. C: `SEGMENT_LENGTH_KM` + `SPEEDLIMIT` + `NBRLANES`.

The comparison excludes the 157 zero-speed rows consistently across all
models, without assuming that zero means missing. Fixed feature sets are not
a basis for causal interpretation; the report records the separate validation
and test results and their limitations.

There are 2,111 distinct `ROADNAME` values; the provenance document describes
them as highway/control-section alignments, not necessarily whole highways.
The comparison uses connected groups of `ROADNAME` and `NID`, alongside a
random-row diagnostic, with common partitions across models. Disjoint
identifiers do not guarantee independent highways or geographic separation.
This snapshot lacks the dated condition, maintenance, traffic and
environmental records needed for deterioration prediction. See the
[methodology](../docs/road_segments_methodology.md) for source provenance.
