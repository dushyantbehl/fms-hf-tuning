
from .tracker import Tracker
from .aimstack_tracker import AimStackTracker

def get_tracker(tracker_name, tracker_config):
    if tracker_name == 'aim':
        if tracker_config is not None:
            tracker = AimStackTracker(tracker_config)
    else:
        tracker = Tracker()
    return tracker