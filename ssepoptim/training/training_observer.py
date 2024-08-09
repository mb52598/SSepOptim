from abc import ABCMeta


class TrainingObserver(metaclass=ABCMeta):
    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_fine_tuning_start(self):
        pass

    def on_fine_tuning_end(self):
        pass

    def on_testing_start(self):
        pass

    def on_testing_end(self):
        pass
