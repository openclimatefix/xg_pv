"""Script to load results from the database"""

import json
from datetime import timedelta, timezone

import boto3
import pandas as pd
import plotly.graph_objects as go
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import ForecastSQL, ForecastValueSQL, GSPYieldSQL, MLModelSQL
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.models.gsp import LocationSQL

client = boto3.client("secretsmanager")
response = client.get_secret_value(
    SecretId="development/rds/forecast/",
)
secret = json.loads(response["SecretString"])
""" We have used a ssh tunnel to 'localhost' """
db_url = f'postgresql://{secret["username"]}:{secret["password"]}@localhost:5433/{secret["dbname"]}'


connection = DatabaseConnection(url=db_url, base=Base_Forecast, echo=True)

start_date = "2023-03-28"


with connection.get_session() as session:
    # 24 hours results
    query = session.query(ForecastValueSQL)
    query = query.distinct(ForecastValueSQL.target_time)
    query = query.join(ForecastSQL)
    query = query.join(LocationSQL)
    query = query.join(MLModelSQL)
    query = query.where(MLModelSQL.name == "National_xg")
    query = query.where(LocationSQL.gsp_id == 0)
    query = query.where(ForecastValueSQL.target_time >= start_date)
    query = query.where(
        ForecastValueSQL.created_utc <= ForecastValueSQL.target_time - timedelta(hours=24)
    )
    query = query.order_by(ForecastValueSQL.target_time, ForecastValueSQL.created_utc.desc())

    national_xg = query.all()

    # truth - pvlive
    query = session.query(GSPYieldSQL)
    query = query.distinct(GSPYieldSQL.datetime_utc)
    query = query.join(LocationSQL)
    query = query.where(LocationSQL.gsp_id == 0)
    query = query.where(GSPYieldSQL.created_utc > start_date)
    query = query.order_by(GSPYieldSQL.datetime_utc, GSPYieldSQL.created_utc.desc())

    pvlive = query.all()

    # get values
    national_xg_datetimes = [forecast_value.target_time for forecast_value in national_xg]
    national_xg_datetimes = [
        datetime.replace(tzinfo=timezone.utc) for datetime in national_xg_datetimes
    ]
    national_xg_values = [
        forecast_value.expected_power_generation_megawatts for forecast_value in national_xg
    ]

    national_pvlive_datetimes = [gsp_yield.datetime_utc for gsp_yield in pvlive]
    national_pvlive_datetimes = [
        datetime.replace(tzinfo=timezone.utc) for datetime in national_pvlive_datetimes
    ]
    national_pvlive_values = [gsp_yield.solar_generation_kw / 1000 for gsp_yield in pvlive]

    # make figure
    fig = go.Figure(
        data=[
            go.Scatter(x=national_xg_datetimes, y=national_xg_values, name="xg"),
            go.Scatter(x=national_pvlive_datetimes, y=national_pvlive_values, name="pvlive"),
        ]
    )

    # format
    nation_xg = pd.DataFrame(data=[national_xg_datetimes, national_xg_values]).T
    nation_xg.columns = ["datetime_utc", "power_xg"]
    nation_xg.set_index("datetime_utc", inplace=True)

    nation_pvlive = pd.DataFrame(data=[national_pvlive_datetimes, national_pvlive_values]).T
    nation_pvlive.columns = ["datetime_utc", "power_pvlive"]
    nation_pvlive.set_index("datetime_utc", inplace=True)

    nationa_df = nation_xg.merge(nation_pvlive, on="datetime_utc")

    # make mae
    mae = (nationa_df["power_xg"] - nationa_df["power_pvlive"]).abs().mean()
    print(mae)

    fig.update_layout(
        title=f"Compare PV live and 24 hour forecast: MAE {mae:.2f} MW",
        xaxis_title="MW",
        yaxis_title="Time",
    )
    fig.show(renderer="browser")
