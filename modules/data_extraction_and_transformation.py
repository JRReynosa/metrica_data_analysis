import modules.helper_methods as helper

url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/' \
      'Sample_Game_1_RawEventsData.csv'

eventsdf = helper.get_events_dataframe(url)
shotsdf = helper.get_all_shots(eventsdf)

print(shotsdf)


