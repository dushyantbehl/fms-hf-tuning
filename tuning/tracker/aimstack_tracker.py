# Standard
import os

from .tracker import Tracker

# Third Party
from aim.hugging_face import AimCallback

class CustomAimCallback(AimCallback):

    aim_run_hash_export_path = None
    is_world_process_zero = None

    def on_init_end(self, args, state, control, **kwargs):
        if state:
            self.is_world_process_zero = state.is_world_process_zero

        if not self.is_world_process_zero:
            return

        self.setup()

        # store the run hash
        if self.aim_run_hash_export_path:
            with open(self.aim_run_hash_export_path, 'w') as f:
                f.write('{\"run_hash\":\"'+str(self._run.hash)+'\"}\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # call directly to make sure hyper parameters and model info is recorded.
        self.setup(args=args, state=state, model=model)

    def track_metrics(self, metric, name, context):
        if self.is_world_process_zero:
            self._run.track(metric, name=name, context=context)

    def set_params(self, params, name):
        if self.is_world_process_zero:
            for key, value in params.items():
                self._run.set((name, key), value, strict=False)

class AimStackTracker(Tracker):

    def __init__(self, tracker_config):
        super().__init__(tracker_config)

    def get_hf_callback(self):
        c = self.config
        exp = c.experiment
        ip = c.aim_remote_server_ip
        port = c.aim_remote_server_port
        repo = c.aim_repo
        hash_export_path = c.aim_run_hash_export_path

        if (ip is not None and port is not None):
            aim_callback = CustomAimCallback(
                                repo="aim://" + ip +":"+ port + "/",
                                experiment=exp
                            )
        if repo:
            aim_callback = CustomAimCallback(repo=repo, experiment=exp)
        else:
            aim_callback = CustomAimCallback(experiment=exp)

        aim_callback.aim_run_hash_export_path = hash_export_path

        # Save Internal State
        self.hf_callback = aim_callback

        return self.hf_callback

    def track(self, metric, name, stage='additional_metrics'):
        context={'subset' : stage}
        self.hf_callback.track_metrics(metric, name=name, context=context)

    def set_params(self, params, name='extra_params'):
        try:
            self.hf_callback.set_params(params, name)
        except:
            pass