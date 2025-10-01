import re
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from pathlib import Path
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point, Polygon





"""
# --- TIFF ---
tiff = TifHandler("example.tif").read(bands=[1, 3])  # only load bands 1 and 3
band1 = tiff.get_band(1)

# --- Bridge ---
pq_handler = ParquetHandler.from_tiff(tiff, "output.parquet")

# --- Parquet ---
pq_loaded = ParquetHandler.read("output.parquet")
subset = pq_loaded.get_bands([1])  # returns only band_1 with x,y
print(subset.head())

"""



import rasterio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path




class SafeHandler:
    def __init__(self,path:str,level:str):
        self.path = Path(path)
        self.level = level
        self.data :np.ndarray | None = None
        self.meta :dict | None = None

    # read and stack bands(all bands, or a list of specific) based on level(L1C,L2A)

    def read(self,stack:bool = True,resample:bool= False,bands:list[int] | None = None) -> "SafeHandler":
        if self.level == "L1C":
            subdir_filter = "IMG_DATA"
            if bands is None:
                patterns = ['B01', 'B02', 'B03', 'B04','B05', 'B06','B07','B08','B8A','B09', 'B10','B11', 'B12']
            else:
                patterns = [f'B{str(b).zfill(2)}' for b in bands]
            regexes = [re.compile(p) for p in patterns]
            img_data_dirs = [d for d in self.path.rglob("*") if d.is_dir() and subdir_filter in d.name]
            if len(img_data_dirs) == 0:
                raise ValueError(f"No {subdir_filter} directory found in {self.path}")
            img_dir = img_data_dirs[0]# ---> Img dirs
            
            files = []

            for p in img_dir.glob("*"): 
                if not p.is_file()  : 
                    continue
                files.append(p)
            
            # sort files
            sorted_files = list(sorted(
                files,
                key=lambda s: next((i for i, r in enumerate(regexes) if r.search(str(s))), len(regexes))
            ))
            
            # refernce metadata from the first band
            with rasterio.open(sorted_files[0]) as src0:
                self.meta = src0.meta.copy()

            arrays = {}
            ref_profile = None
            for f in sorted_files:
                band_id = int(re.search(r"B(\d+)", f.name).group(1))
                with rasterio.open(f) as src:
                    if resample and ref_profile is not None:
                        arr = src.read(
                            1,
                            out_shape=(ref_profile['height'], ref_profile['width']),
                            resampling=Resampling.cubic # verify if TCI also needs cubic or nearest

                        )
                    else:
                        arr = src.read(1)
                        ref_profile = src.profile
                    arrays[band_id] = arr

            if stack:
                self.data = np.stack([arrays[b] for b in sorted(arrays.keys())], axis=0)
            else:
                self.data = arrays
            self.meta.update(ref_profile)
            return self



        elif self.level == "L2A":
            raise NotImplementedError("L2A level not implemented yet")

       


class TifHandler:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: np.ndarray | None = None
        self.meta: dict | None = None

    def read(self, bands: list[int] | None = None) -> "TifHandler":
        with rasterio.open(self.path) as src:
            if bands is None:
                self.data = src.read()
            else:
                self.data = src.read(bands)
            self.meta = src.meta
        return self
    
    def get_band(self, index: int) -> np.ndarray:
        return self.data[index - 1]
    

    def spatial_query(self, bbox: tuple[float, float, float, float]) -> np.ndarray:
        with rasterio.open(self.path) as src:
            window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
            window = window.round_offsets().round_shape()
            data_window = src.read(window=window)
        return data_window




class ParquetHandler:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: pa.Table | None = None
        self.gdf: gpd.GeoDataFrame | None = None

    @classmethod
    def from_tiff(cls, tiff: TifHandler, output_path: str | Path) -> "ParquetHandler":
        bands, height, width = tiff.data.shape
        transform = tiff.meta['transform']
        crs = tiff.meta['crs']

        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs = np.array(xs)
        ys = np.array(ys)

        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitudes, latitudes = transformer.transform(xs.flatten(), ys.flatten())

        # Prepare data dict
        data_dict = {
            "x": xs.flatten(),
            "y": ys.flatten(),
            "lat": latitudes,
            "long": longitudes
        }
        for b in range(bands):
            data_dict[f"band_{b+1}"] = tiff.data[b].flatten()

        df = pd.DataFrame(data_dict)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['long'], df['lat']),
            crs="EPSG:4326"
        )

        # Create handler and write
        handler = cls(output_path)
        handler.gdf = gdf
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        handler.write()
        return handler

    def write(self):
        self.gdf.to_parquet(self.path, index=False)

    @classmethod
    def read(cls, path: str, ref_tiff: str | None = None) -> "ParquetHandler":
        handler = cls(path)
        df = pd.read_parquet(path)
        
        if ref_tiff is not None:
            from pyproj import Transformer
            from shapely.geometry import Point

            with rasterio.open(ref_tiff) as src:
                transform = src.transform
                crs = src.crs

            # xs, ys = rasterio.transform.xy(transform, df["y"].to_numpy(), df["x"].to_numpy())
            # transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            # longitudes, latitudes = transformer.transform(xs, ys)
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            longitudes, latitudes = transformer.transform(df["x"].to_numpy(), df["y"].to_numpy())
            
            df["lat"] = latitudes
            df["long"] = longitudes
            df["geometry"] = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
            handler.gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            raise ValueError("No ref_tiff provided for coordinate transformation.")

        return handler

    
    def get_bands(self, bands: list[int] | None = None) -> gpd.GeoDataFrame:
        if bands is None:
            return self.gdf
        cols = ["x", "y", "lat", "long", "geometry"] + [f"band_{b}" for b in bands]
        return self.gdf[cols]
    
    def spatial_query(self, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:

        if self.gdf is None:
            raise ValueError("Data not loaded. Please read the Parquet file first.")
        
        min_x, min_y, max_x, max_y = bbox
        from shapely.geometry import box
        bbox_poly = box(min_x, min_y, max_x, max_y)
        
        # spatial index automatically with sjoin or direct intersection
        return self.gdf[self.gdf.intersects(bbox_poly)]
    
    def spatial_query_polygon(self, polygon: Polygon) -> gpd.GeoDataFrame:

        if self.gdf is None:
            raise ValueError("Data not loaded. Please read the Parquet file first.")
        
        #  spatial index automatically 
        return self.gdf[self.gdf.intersects(polygon)]
    
    def spatial_query_within(self, polygon: Polygon) -> gpd.GeoDataFrame:
        """
        Query points completely within polygon (stricter than intersects)
        """
        if self.gdf is None:
            raise ValueError("Data not loaded. Please read the Parquet file first.")
        
        return self.gdf[self.gdf.within(polygon)]



        
       

