import zipfile
from pathlib import Path

# ChatGPT helper code to unzip and rename ESHM20 hazard map shapefiles

src = Path(r"C:\Users\clemettn\Desktop\Neuer Ordner")
tgt = Path(r"C:\Users\clemettn\Documents\phd\data\eshm20_hazard_maps")


tgt.mkdir(parents=True, exist_ok=True)

for zip_path in src.glob("hmap_*_*.zip"):
    stem = zip_path.stem              # e.g., "hmap_12_34"
    suffix = stem.replace("hmap", "") # e.g., "_12_34"

    with zipfile.ZipFile(zip_path, "r") as zf:
        # find the shapefile inside output/ folder
        shp_base = None
        for name in zf.namelist():
            if name.startswith("output/hmap") and \
               name.endswith(".shp"):
                shp_base = Path(name).stem  # e.g., "hmap9876"
                break

        if shp_base is None:
            print(f"No shapefile in {zip_path}")
            continue

        # construct new base name
        new_base = f"hmap{suffix}"

        # extract all related shapefile components
        for name in zf.namelist():
            if name.startswith("output/") and \
               Path(name).stem == shp_base:
                ext = Path(name).suffix
                out_name = f"{new_base}{ext}"
                out_path = tgt / out_name
                with zf.open(name) as src_f, \
                     open(out_path, "wb") as dst_f:
                    dst_f.write(src_f.read())

print("Done.")
