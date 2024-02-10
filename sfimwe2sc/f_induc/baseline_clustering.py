import pandas as pd


class OnecpvClustering:
    def step(self, df):
        df_list = []
        for i, verb in enumerate(sorted(set(df["verb"]))):
            df_verb = df[df["verb"] == verb].copy()
            df_verb["frame_cluster"] = i + 1
            df_list.append(df_verb)
        return pd.concat(df_list, axis=0)
