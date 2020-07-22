import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from soccerutils.pitch import Pitch
import modules.helper_methods as helper

url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/' \
      'Sample_Game_1_RawEventsData.csv'

eventdf = helper.get_events_dataframe(url)

passdf_dict = helper.get_seperate_action(eventdf, action="PASS")


def make_model(homedf, awaydf):
    homemodel = KMeans(n_clusters=30)
    awaymodel = KMeans(n_clusters=30)

    homefeatures = homedf[['start_x', 'start_y', 'end_x', 'end_y']]
    homefit = homemodel.fit(homefeatures)

    awayfeatures = awaydf[['start_x', 'start_y', 'end_x', 'end_y']]
    awayfit = awaymodel.fit(awayfeatures)

    homedf["cluster"] = homemodel.predict(homefeatures)
    awaydf["cluster"] = awaymodel.predict(awayfeatures)

    return homefit, awayfit


def plot_arrows(model_fits, axis1, axis2):
    for period in range(2):  # Two periods
        for team in range(2):  # Two teams
            for i, (start_x, start_y, end_x, end_y) in enumerate(model_fits[period][team].cluster_centers_):
                axis = axis1 if period == 0 else axis2
                axis.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                           head_width=1,
                           head_length=1,
                           color='blue' if team == 0 else 'red',
                           alpha=0.5,
                           length_includes_head=True)

                # ax1.text((start_x + end_x) / 2, (start_y + end_y) / 2, str(i + 1))


match_fits = [make_model(passdf_dict["home_passes_1"], passdf_dict["away_passes_1"]),
              make_model(passdf_dict["home_passes_2"], passdf_dict["away_passes_2"])]
# match_fits = [[period1], [period2]]; period1 = [homefit1, awayfit1]; period2 = [homefit2, awayfit2]

fig, (ax1, ax2) = plt.subplots(2, sharex="all", sharey="all", figsize=(10, 8))
plot_arrows(match_fits, ax1, ax2)

# Plot properties
red_patch = mpatches.Patch(color='red', label='Away Team')
blue_patch = mpatches.Patch(color='blue', label='Home Team')
fig.legend(handles=[red_patch, blue_patch])

ax1.set_title("First Half")
ax2.set_title("Second Half")
plt.xlim(-53, 53)
plt.ylim(-34, 34)
# fig.savefig('passing.png', dpi=100)

plt.show()

# Maybe work on pass difficulty making use of pass_distance and pass_angle?
