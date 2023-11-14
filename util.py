import pandas as pd
import numpy as np
import glob
from functools import lru_cache, cache, partial
from sklearn.metrics import make_scorer, f1_score, precision_score
import os
import json

PROJECT_NAME = "Dream11"
ENTITY = None


replace_venue_dict = {
    "M Chinnaswamy Stadium, Bangalore": "M Chinnaswamy Stadium",
    "Vidarbha Cricket Association Stadium, Jamtha, Nagpur": "Vidarbha Cricket Association Stadium, Jamtha",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Eden Park, Auckland": "Eden Park",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Hyderabad": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Cricket Stadium, Dehradun": "Rajiv Gandhi International Cricket Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "McLean Park, Napier": "McLean Park",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "New Wanderers Stadium": "Wanderers",
    "The Wanderers Stadium, Johannesburg": "Wanderers",
    "The Wanderers Stadium": "Wanderers",
    "Wanderers Cricket Ground, Windhoek": "Wanderers Cricket Ground",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Ministry Turf 2": "Ministry Turf 1",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
    "Gahanga International Cricket Stadium. Rwanda": "Gahanga International Cricket Stadium, Rwanda",
    "Shere Bangla National Stadium, Mirpur": "Shere Bangla National Stadium",
    "Desert Springs Cricket Ground, Almeria": "Desert Springs Cricket Ground",
    "R.Premadasa Stadium, Khettarama": "R Premadasa Stadium",
    "R Premadasa Stadium, Colombo": "R Premadasa Stadium",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "SuperSport Park, Centurion": "SuperSport Park",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "ICC Academy, Dubai": "ICC Academy",
    "ICC Academy Ground No 2": "ICC Academy",
    "ICC Global Cricket Academy": "ICC Academy",
    "Zahur Ahmed Chowdhury Stadium, Chattogram": "Zahur Ahmed Chowdhury Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Kensington Oval, Bridgetown, Barbados": "Kensington Oval, Bridgetown",
    "Queens Sports Club, Bulawayo": "Queens Sports Club",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "Gaddafi Stadium, Lahore": "Gaddafi Stadium",
    "Tribhuvan University International Cricket Ground, Kirtipur": "Tribhuvan University International Cricket Ground",
    "Indian Association Ground, Singapore": "Indian Association Ground",
    "Saurashtra Cricket Association Stadium, Rajkot": "Saurashtra Cricket Association Stadium",
    "The Village, Malahide, Dublin": "The Village, Malahide",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Sportpark Westvliet, The Hague": "Sportpark Westvliet",
    "Bready Cricket Club, Magheramason, Bready": "Bready",
    "Bready Cricket Club, Magheramason": "Bready",
    "Greenfield International Stadium, Thiruvananthapuram": "Greenfield International Stadium",
    "Edgbaston, Birmingham": "Edgbaston",
    "Barabati Stadium, Cuttack": "Barabati Stadium",
    "Bay Oval, Mount Maunganui": "Bay Oval",
    "College Field, St Peter Port": "College Field",
    "Holkar Cricket Stadium, Indore": "Holkar Cricket Stadium",
    "National Cricket Stadium, St George's, Grenada": "National Cricket Stadium, Grenada",
    "Daren Sammy National Cricket Stadium, Gros Islet, St Lucia": "Darren Sammy National Cricket Stadium, St Lucia",
    "Windsor Park, Roseau, Dominica": "Windsor Park, Roseau",
    "Manuka Oval, Canberra": "Manuka Oval",
    "Brisbane Cricket Ground, Woolloongabba, Brisbane": "Brisbane Cricket Ground, Woolloongabba",
    "Gymkhana Club Ground, Nairobi": "Gymkhana Club Ground",
    "Trent Bridge, Nottingham": "Trent Bridge",
    "County Ground, Bristol": "County Ground",
    "Sophia Gardens, Cardiff": "Sophia Gardens",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Moara Vlasiei Cricket Ground, Ilfov County": "Moara Vlasiei Cricket Ground",
    "National Stadium, Karachi": "National Stadium",
}

replace_team_dict = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Kings XI Punjab": "Punjab Kings",
}


def get_json_info(jsondata):
    keys = [
        "event",
        "match_type",
        "officials",
        "outcome",
        "season",
        "teams",
        "players",
        "toss",
        "venue",
    ]
    match_info, player_info = {}, []
    for key in keys:
        # print(f"{key}: {jsondata['info'][key]}")
        if key == "event":
            try:
                # print(f"{key}: {jsondata['info'][key]}")
                match_info[key] = jsondata["info"][key]["name"]
            except:
                match_info[key] = "NA"
        elif key == "match_type":
            try:
                # print(f"{key}: {jsondata['info'][key]}")
                match_info[key] = jsondata["info"][key]
            except:
                match_info[key] = "NA"
        elif key == "officials":
            try:
                # print(f"{key}: {jsondata['info'][key]}")
                match_info["umpire_1"] = jsondata["info"][key]["umpires"][0]
                match_info["umpire_2"] = jsondata["info"][key]["umpires"][1]
            except:
                match_info["umpire_1"] = "NA"
                match_info["umpire_2"] = "NA"
        elif key == "outcome":
            try:
                match_info["winner"] = jsondata["info"][key]["winner"]
                try:
                    match_info["win_by_runs"] = jsondata["info"][key]["by"]["runs"]
                except:
                    match_info["win_by_wickets"] = jsondata["info"][key]["by"][
                        "wickets"
                    ]
            except:
                match_info["winner"] = "Draw"  # change this later
        elif key == "teams":
            match_info["team_1"] = jsondata["info"][key][0]
            match_info["team_2"] = jsondata["info"][key][1]
        elif key == "players":
            try:
                player_info = pd.DataFrame(jsondata["info"][key])
            except:
                player_info = pd.DataFrame()
        elif key == "toss":
            try:
                match_info["toss_winner"] = jsondata["info"][key]["winner"]
                match_info["toss_decision"] = jsondata["info"][key]["decision"]
            except:
                match_info["toss_winner"] = "NA"
                match_info["toss_decision"] = "NA"
        elif key == "venue":
            try:
                match_info[key] = jsondata["info"][key]
            except:
                match_info[key] = "NA"
    return (match_info, player_info)


@cache
def read_data(path_csv, path_json):
    # read all csv files in the folder
    # seperate files with '_info' in the name
    all_files = glob.glob(path_csv + "/*.csv")
    data_files = [file for file in all_files if "_info" not in file]
    if path_csv + "/all_matches.csv" in data_files:
        data_files.remove(path_csv + "/all_matches.csv")
    json_files = [
        pos_json for pos_json in os.listdir(path_json) if pos_json.endswith(".json")
    ]
    match_dict = {
        file.split("/")[-1].split("_")[0][:-4]: [
            pd.read_csv(file, index_col=None, header=0, low_memory=False)
        ]
        for file in data_files
    }

    for file in json_files:
        if file[:-5] in match_dict:
            with open(os.path.join(path_json, file)) as json_file:
                json_data = json.load(json_file)
                match_info, player_info = get_json_info(json_data)
                match_dict[file[:-5]].append(match_info)
                match_dict[file[:-5]].append(player_info)
            # break;
    return match_dict


def collate_datasets(kinds):
    dfs = []
    for kind in kinds:
        path_csv = f"../Inputs/{kind}_csv"
        path_json = f"../Inputs/{kind}_json"
        match_dict = read_data(path_csv, path_json)
        dfs.append(pd.concat([match_dict[match][0] for match in match_dict]))
    return pd.concat(dfs, axis=0, ignore_index=True)


def get_player_dict():
    player_dict = {}
    kinds = ["ipl"]
    for kind in kinds:
        path_json = f"../Inputs/{kind}_json"
        json_files = [
            pos_json for pos_json in os.listdir(path_json) if pos_json.endswith(".json")
        ]

        for file in json_files:
            with open(os.path.join(path_json, file)) as json_file:
                json_data = json.load(json_file)
                for key, val in json_data["info"]["registry"]["people"].items():
                    player_dict[val] = key
    return player_dict


player_dict = get_player_dict()


def player2id(df):
    df[["striker", "non_striker", "bowler"]] = df[
        ["striker", "non_striker", "bowler"]
    ].applymap(lambda x: next((k for k, v in player_dict.items() if v == x), x))
    return df


# check this function
def id2player(df):
    df[["striker", "non_striker", "bowler"]] = df[
        ["striker", "non_striker", "bowler"]
    ].applymap(lambda x: player_dict.get(x, x))
    return df


# slow takes 40 seconds
def clean_data(df):
    # clean venue column
    print("-------Cleaning Venue Column-------")

    def clean_venues(df):  # slow
        df["venue"] = df["venue"].map(lambda x: replace_venue_dict.get(x, x))
        return df

    df = clean_venues(df)

    # clean teams column
    print("-------Cleaning Teams Column-------")

    def clean_teams(df):
        df["batting_team"] = df["batting_team"].map(
            lambda x: replace_team_dict.get(x, x)
        )
        df["bowling_team"] = df["bowling_team"].map(
            lambda x: replace_team_dict.get(x, x)
        )
        return df

    df = clean_teams(df)

    # clean player column
    print("-------Cleaning Player Column-------")
    df = player2id(df)
    # remove the unwanted columns
    print("-------Removing Unwanted Columns-------")
    df["overs"] = (
        df["ball"]
        .apply(lambda s: np.array(str(s).split(".")).reshape(-1, 1))
        .apply(lambda s: int(s[0][0]) + 1)
        .astype("category")
    )
    df["balls"] = (
        df["ball"]
        .apply(lambda s: np.array(str(s).split(".")).reshape(-1, 1))
        .apply(lambda s: int(s[1][0]))
        .astype("category")
    )
    df = df.drop(
        ["penalty", "other_wicket_type", "other_player_dismissed", "season"], axis=1
    )
    df["delivery_id"] = (
        df["match_id"].astype(str)
        + "_"
        + df["innings"].astype(str)
        + "_"
        + df["ball"].astype(str)
    )
    df = df.set_index("delivery_id")
    df = df.drop(["match_id"], axis=1)
    return df


def custom_scorer(y_true, y_pred, important_classes):
    # Compute F1 score for all classes
    f1 = f1_score(y_true, y_pred, average=None)

    # Create a mask for important classes
    mask = [1 if c in important_classes else 0 for c in range(len(f1))]

    # Compute weighted F1 score
    weighted_f1 = (f1 * mask).sum() / sum(mask)

    return weighted_f1


def get_custom_scorer(important_classes):
    # Create partial function with important_classes argument
    custom_scorer_partial = partial(custom_scorer, important_classes=important_classes)
    custom_scorer_partial.__name__ = "custom_f1"
    # Create scorer object
    custom_scorer_obj = make_scorer(custom_scorer_partial, greater_is_better=True)
    return custom_scorer_obj
