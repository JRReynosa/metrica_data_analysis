import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import modules.helper_methods as helper

url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/' \
      'Sample_Game_1_RawEventsData.csv'

eventsdf = helper.get_events_dataframe(url)

passesdf = helper.get_all_action(eventsdf, action="PASS")
total_passesdf = passesdf.groupby('from_player')['outcome'].count()
pass_accuracydf = passesdf.groupby('from_player')['outcome'].mean() * 100

fig, ax = plt.subplots()

scatter = ax.scatter(total_passesdf, pass_accuracydf)
ax.set_xlabel("Total Passes")
ax.set_ylabel("Pass Completion")
plt.yticks(np.arange(0, 110, 10))

for player, total in total_passesdf.items():
    x = total
    y = pass_accuracydf[player]
    plt.annotate(player,
                 (x, y),
                 textcoords="offset points",  # how to position the text
                 xytext=(-5, 10),  # distance from text to points (x,y)
                 ha='center',
                 arrowprops=dict(facecolor='black', arrowstyle="-")
                 )
    # t = ax.text(x, y, player, fontsize=8)

model = LinearRegression()
fit = model.fit([[x] for x in total_passesdf], pass_accuracydf)
print("Coefficients: {}".format(fit.coef_))
print("Intercept: {}".format(fit.intercept_))

xfit = [0, 90]  # This is the x-axis range of the chart
yfit = model.predict([[x] for x in xfit])

plt.plot(xfit, yfit, 'r')
plt.show()
