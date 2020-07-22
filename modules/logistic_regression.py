import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import modules.helper_methods as helper
from sklearn.linear_model import LogisticRegression
import numpy as np

url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/' \
      'Sample_Game_1_RawEventsData.csv'

eventsdf = helper.get_events_dataframe(url)


all_shotsdf = helper.get_all_action(eventsdf, action="SHOT")

all_shotsdf['distance_to_goal'] = all_shotsdf.apply(lambda q: helper.distance_to_goal(q[['start_x', 'start_y']]), axis=1)
all_shotsdf['goal_angle'] = all_shotsdf.apply(lambda q: helper.goal_angle(q[['start_x', 'start_y']]), axis=1)
all_shotsdf['head'] = all_shotsdf.apply(lambda q: 1 if ("HEAD" in q["subtype"]) else 0, axis=1)

model = LogisticRegression()

features = all_shotsdf[['distance_to_goal', 'goal_angle', 'head']]
labels = all_shotsdf['outcome']

fit = model.fit(features, labels)

predictions = model.predict_proba(features)[:, 1]

xnew = np.linspace(0, len(predictions), 300)
spl = make_interp_spline(range(len(predictions)), sorted(predictions), k=3)  # type: BSpline
power_smooth = spl(xnew)
plt.plot(xnew, power_smooth)

plt.show()

print("----------------")
