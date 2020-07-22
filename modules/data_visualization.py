from soccerutils.pitch import Pitch
import numpy as np
import pandas as pd
import modules.helper_methods as helper
import matplotlib as plt

tracking_path = 'C:\\Users\\reynosaj\\PycharmProjects\\metrica_data_analysis\\data\\TrackingData.csv'
trackingdf = helper.get_tracking_data(tracking_path)

events_path = 'C:\\Users\\reynosaj\\PycharmProjects\\metrica_data_analysis\\data\\EventsData.csv'
eventdf = helper.get_events_data(events_path)

starters = helper.determine_starters(trackingdf)
eventsdf = helper.get_all_events(eventdf)
