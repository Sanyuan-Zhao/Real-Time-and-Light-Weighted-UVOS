from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/home/zhao/datasets/DAVIS'

    @staticmethod
    def save_root_dir():
        return './pretrained_models'

    @staticmethod
    def models_dir():
        return "./models"

