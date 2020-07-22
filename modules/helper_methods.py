import pandas as pd
import numpy as np


def determine_outcome(row_type, row_subtype, rowahead_type, rowahead_subtype):
    lostball = rowahead_type == "BALL LOST" and rowahead_subtype != np.nan and \
               not ("FORCED" or "THEFT" or "CLEARANCE" or "END HALF") in str(rowahead_subtype)
    outcome = {
        "PASS": 1 if not lostball else 0,
        "SHOT": 1 if "GOAL" in str(row_subtype) else 0,
    }

    return outcome.get(row_type, None)


def get_events_dataframe(data_location):
    eventsdf = pd.read_csv(data_location, error_bad_lines=False)
    eventsdf.columns = ['team', 'type', 'subtype', 'period', 'start_frame', 'start_time', 'end_frame',
                        'end_time', 'from_player', 'to_player', 'start_x', 'start_y', 'end_x', 'end_y']

    all_events = []
    field_dimen = (106., 68.)
    row_iterator = eventsdf.iterrows()
    _, row = next(row_iterator)  # Get first row
    for index, rowahead in row_iterator:
        attributes = {
            "team": row.team,
            "period": row.period,
            "type": row.type,
            "subtype": row.subtype,
            "outcome": determine_outcome(row.type, row.subtype, rowahead.type, rowahead.subtype),
            "from_player": row.from_player,
            "to_player": row.to_player,

            "start_frame": row.start_frame,
            "end_frame": row.end_frame,

            "start_time": row.start_time,
            "end_time": row.end_time,

            "start_x": (row.start_x - .5) * 106.,  # Change field dimensions to 106x68 meters
            "start_y": (row.start_y - .5) * 68.,
            "end_x": (row.end_x - .5) * 106.,
            "end_y": (row.end_y - .5) * 68.,
        }
        all_events.append(attributes)
        row = rowahead
    all_eventsdf = pd.DataFrame(all_events)

    return all_eventsdf


def get_tracking_dataframe(data_location):
    trackingdf = pd.read_csv(data_location, error_bad_lines=False, dtype=str)
    trackingdf = trackingdf.drop([0, 1]).reset_index(drop=True)
    trackingdf.columns = ["period", "frame", "time", "player11x", "player11y", "player1x", "player1y", "player2x",
                          "player2y",
                          "player3x", "player3y", "player4x", "player4y", "player5x", "player5y", "player6x",
                          "player6y",
                          "player7x", "player7y", "player8x", "player8y", "player9x", "player9y", "player10x",
                          "player10y",
                          "player12x", "player12y", "player13x", "player13y", "player14x", "player14y", "ballx",
                          "bally"]
    return trackingdf


def get_all_action(event_dataframe, action):
    all_actions = []

    for index, row in event_dataframe.iterrows():
        if row.type == action:
            attributes = {
                "team": row.team,
                "type": row.type,
                "subtype": row.subtype,
                "outcome": row.outcome,
                "from_player": row.from_player,
                "to_player": row.to_player,

                "start_frame": row.start_frame,
                "end_frame": row.end_frame,

                "start_time": row.start_time,
                "end_time": row.end_time,

                "start_x": row.start_x,
                "start_y": row.start_y,
                "end_x": row.end_x,
                "end_y": row.end_y,
            }
            all_actions.append(attributes)

    actiondf = pd.DataFrame(all_actions)
    return actiondf


def action_exception():
    raise Exception("Invalid Action")


def get_seperate_action(event_dataframe, action):
    action_switch = {
        "PASS": {
            "home_passes_1": [],
            "away_passes_1": [],
            "home_passes_2": [],
            "away_passes_2": []
        },
        "SHOT": {
            "home_shots_1": [],
            "away_shots_1": [],
            "home_shots_2": [],
            "away_shots_2": []
        }
    }
    seperate_actions = action_switch.get(action, lambda: action_exception())

    for index, row in event_dataframe.iterrows():
        if row.type == action:
            attributes = {
                "team": row.team,
                "period": row.period,
                "type": row.type,
                "subtype": row.subtype,
                "outcome": row.outcome,
                "from_player": row.from_player,
                "to_player": row.to_player,

                "start_frame": row.start_frame,
                "end_frame": row.end_frame,

                "start_time": row.start_time,
                "end_time": row.end_time,

                "start_x": row.start_x,
                "start_y": row.start_y,
                "end_x": row.end_x,
                "end_y": row.end_y,
            }
            assign_passes(seperate_actions, attributes)

    for key, value in seperate_actions.items():
        # noinspection PyTypeChecker
        seperate_actions[key] = pd.DataFrame(value)

    return seperate_actions


def distance_to_goal(shot_loc):
    if shot_loc[0] > 0:
        goal_loc = np.array([53., 0.])
    else:
        goal_loc = np.array([-53., 0.])

    return np.sqrt(np.sum((shot_loc - goal_loc) ** 2))


def goal_angle(shot_loc):
    if shot_loc[0] > 0:
        p0 = np.array((53., 4.))  # Left Post
        p1 = np.array(shot_loc, dtype=np.float)
        p2 = np.array((53., -4.))  # Right Post

        v0 = p0 - p1
        v1 = p2 - p1
    else:
        p0 = np.array((-53., -4.))  # Left Post
        p1 = np.array(shot_loc, dtype=np.float)
        p2 = np.array((-53., 4.))  # Right Post

        v0 = p0 - p1
        v1 = p2 - p1

    angle = np.abs(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))

    return angle


def determine_starters(dataframe):
    dataframe = dataframe.iloc[:1, 3:31]
    players = []
    i = 0
    for col in dataframe:
        if (dataframe[col][0] is not np.nan) and i % 2 != 0:
            player = col[:len(col) - 1]
            players.append(player)
        i += 1
    return players


def assign_passes(match_dict, pass_attributes):
    if pass_attributes["team"] == "Home":
        if pass_attributes["period"] == 1:
            match_dict["home_passes_1"].append(pass_attributes)
        else:
            match_dict["home_passes_2"].append(pass_attributes)
    else:
        if pass_attributes["period"] == 1:
            match_dict["away_passes_1"].append(pass_attributes)
        else:
            match_dict["away_passes_2"].append(pass_attributes)
