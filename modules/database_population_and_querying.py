import modules.helper_methods as helper
from sqlalchemy import create_engine
import pandas as pd

url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/' \
      'Sample_Game_1_RawEventsData.csv'
eventsdf = helper.get_events_dataframe(url)

engine = create_engine('sqlite://')
eventsdf.to_sql('events', engine)

top_passers = """
select from_player as player , count(*) as passes
from events
where outcome=1
and type = "PASS"
group by from_player
order by passes desc
"""

print(pd.read_sql(top_passers, engine).head(10))

# This was supposed to be xG but I did not have enough data to make
# a solid calculation
top_shots = """
select from_player as player, count(*) as shots
from events
where outcome=1
and type = "SHOT"
group by from_player
order by shots desc
"""

print(pd.read_sql(top_shots, engine).head(10))
