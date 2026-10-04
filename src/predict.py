import joblib
import pandas as pd

from src.utils import Root, load_config, load_data

# Loads the saved final model, predicts predicted_rate for every row of the preprocessed validation file,
# and fills those predictions into the validarion template file by matching on load_id
def predict_validation(cfg):
    cols = cfg["columns"]
    model = joblib.load(Root / cfg["paths"]["final_model"])

    df = load_data(cfg["paths"]["validation_preprocessed"], cfg)
    predicted = pd.DataFrame({
                                "load_id": df[cols["Load_Id"]],
                                "predicted_rate": model.predict(df),
                            })

    template = pd.read_csv(Root / cfg["paths"]["validation_predictions"])
    result = template[["load_id"]].merge(predicted, on="load_id", how="left")
    result["predicted_rate"] = result["predicted_rate"].round(2)

    missing = result["predicted_rate"].isna().sum()
    if missing:
        print(f"WARNING: {missing} load_id values in the template got no prediction")

    out_path = Root / cfg["paths"]["validation_predictions"]
    result.to_csv(out_path, index=False)
    print(f"saved {result.shape} -> {out_path}")
    print(f"predicted_rate: min {result['predicted_rate'].min():.2f}, "
          f"max {result['predicted_rate'].max():.2f}, "
          f"any non-positive: {(result['predicted_rate'] <= 0).any()}")
    return result

# Builds a {city name: (lat, lon)} table from every pickup and delivery city seen in train. Each city has one fixed coordinate pair.
def city_coordinates(cfg):
    cols = cfg["columns"]
    train = load_data(cfg["paths"]["train"], cfg)
    pickups = train[[cols["Pickup"], cols["Pickup_Lat"], cols["Pickup_Lon"]]].set_axis(
        ["city", "lat", "lon"], axis=1)
    deliveries = train[[cols["Delivery"], cols["Delivery_Lat"], cols["Delivery_Lon"]]].set_axis(
        ["city", "lat", "lon"], axis=1)
    return pd.concat([pickups, deliveries]).drop_duplicates(subset="city").set_index("city")

# Attaches pickup/delivery coordinates to a dataframe by looking up each row's city names in the given coordinate table then returns a new dataframe.
def attach_coordinates(df, cfg, coords):
    cols = cfg["columns"]
    df = df.copy()
    df[cols["Pickup_Lat"]] = df[cols["Pickup"]].map(coords["lat"])
    df[cols["Pickup_Lon"]] = df[cols["Pickup"]].map(coords["lon"])
    df[cols["Delivery_Lat"]] = df[cols["Delivery"]].map(coords["lat"])
    df[cols["Delivery_Lon"]] = df[cols["Delivery"]].map(coords["lon"])
    return df

# Fills predicted_rate in the December chart file.
def predict_december(cfg):
    cols = cfg["columns"]
    path = Root / cfg["paths"]["december"]
    df = pd.read_csv(path)
    original_columns = list(df.columns)

    coords = city_coordinates(cfg)
    enriched = attach_coordinates(df, cfg, coords)

    missing_coords = enriched[[cols["Pickup_Lat"], cols["Pickup_Lon"],
                                cols["Delivery_Lat"], cols["Delivery_Lon"]]].isna().any(axis=1)
    if missing_coords.any():
        print(f"WARNING: {missing_coords.sum()} December rows have a pickup or delivery city "
              f"not found in train; no coordinates could be looked up for them")

    model = joblib.load(Root / cfg["paths"]["final_model"])
    result = df[original_columns].copy()
    result["predicted_rate"] = model.predict(enriched).round(2)

    result.to_csv(path, index=False)
    print(f"saved {result.shape} -> {path}")
    print(f"predicted_rate: min {result['predicted_rate'].min():.2f}, "
          f"max {result['predicted_rate'].max():.2f}, "
          f"any non-positive: {(result['predicted_rate'] <= 0).any()}")
    return result

def main():
    cfg = load_config()
    predict_validation(cfg)
    predict_december(cfg)

if __name__ == "__main__":
    main()