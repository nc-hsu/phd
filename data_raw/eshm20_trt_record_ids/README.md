# ESM record ID lists of the ESHM20 regional datasets

The ESHM20 ground-motion models for subduction in-slab, subduction interface and Vrancea
(non-subduction deep) earthquakes were derived from three datasets provided by
G. Weatherill (GFZ German Research Centre for Geosciences, Potsdam). Every record in those
datasets is a record of the ESM flatfile (Lanzano et al. 2019,
DOI `10.13127/esm/flatfile.1.0`).

The files here list, for each dataset, the ESM flatfile records it contains:

| file | dataset | region tag in nb 002 |
|---|---|---|
| `sslab_esm_record_ids.csv` | subduction in-slab (`hellenic_inslab_db`) | `sslab` |
| `sinter_esm_record_ids.csv` | subduction interface (`hellenic_interface_db`) | `sinter` |
| `vran_esm_record_ids.csv` | Vrancea (`vrancea_db`) | `vran` |

Each row identifies one ESM record by four ESM flatfile columns: `event_id`,
`network_code`, `station_code` and `sensor_depth_m`. The three code columns are text and
must be read as text (station `0703` is not station `703`); `sensor_depth_m` is a number in
metres, and an empty cell means the depth is not given (it matches ESM records whose depth is
also empty). One row per record, no duplicates, sorted on the four columns.

nb 002 rebuilds each regional flatfile by selecting these records from the ESM flatfile
(`manage_flatfiles.load_esm_record_ids` and `extract_flatfile_for_record_ids`). The lists
are written by nb `002c-extract_gfz_esm_record_ids`.
