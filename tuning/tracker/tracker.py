# Generic Tracker API

class Tracker:
    def __init__(self, tracker_config=None) -> None:
        if tracker_config is not None:
            self.config = tracker_config

    def get_hf_callback():
        return None

    def track(self, metric, name, stage):
        pass

    # Object passed here is supposed to be a KV object
    # for the parameters to be associated with a run
    def set_params(self, params, name):
        pass