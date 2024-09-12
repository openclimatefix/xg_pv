"""
Script to adjust the predictions of the model by using the last week average error.

"""
import pandas as pd

folder = "./results"

forecast_horizon_max = 37

df_all = []
for i in range(0, forecast_horizon_max):
    print(i)
    # open file
    df = pd.read_csv(f"{folder}/errors_{i}_test.csv")

    df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
    df = df.set_index("Unnamed: 0", drop=True)
    # df.index = pd.DatetimeIndex(df.index).to_period("H")

    df = df.rename(columns={"prediction": f"prediction_{i}"})
    df = df.rename(columns={"target": f"target_{i}"})

    # shift by hours
    print("shift")
    df = df.shift(periods=2 * i)

    print("append")
    df_all.append(df)

df_all = pd.concat(df_all, axis=1)

# Minus the rolling mean of the last 7 days for each half and hour
df_adjust = []
for hour in range(0, 24):
    for minute in [0, 30]:
        print(hour, minute)

        df_one_sp = df_all.loc[df_all.index.hour == hour]
        df_one_sp = df_one_sp.loc[df_one_sp.index.minute == minute]

        for i in range(0, forecast_horizon_max):
            df_one_sp[f"mae_{i}"] = (
                (df_one_sp[f"prediction_{i}"] - df_one_sp[f"target_{i}"]).rolling(7).mean()
            )
            df_one_sp[f"mae_{i}"] = df_one_sp[f"mae_{i}"].shift(1)

            df_one_sp[f"prediction_{i}"] = df_one_sp[f"prediction_{i}"] - df_one_sp[f"mae_{i}"]

            df_adjust.append(df_one_sp)

df_adjust = pd.concat(df_adjust)
df_adjust.sort_index(inplace=True)

# Mae with adjust
mae_adjust = {}
for i in range(0, forecast_horizon_max):
    mae_adjust[i] = (df_adjust[f"target_{i}"] - df_adjust[f"prediction_{i}"]).abs().mean()

# mae
mae = {}
for i in range(0, forecast_horizon_max):
    mae[i] = (df_all[f"target_{i}"] - df_all[f"prediction_{i}"]).abs().mean()


print(mae)
print(mae_adjust)

# plot resutls
mae_df = pd.DataFrame(mae, index=["mae"]).T
mae_adjust_df = pd.DataFrame(mae_adjust, index=["mae_adjust"]).T
mae_df["mae_adjust"] = mae_adjust_df["mae_adjust"]
fig = mae_df.plot()
fig.show(renderer="browser")
