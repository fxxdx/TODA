def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 0},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'TODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TODAwoDeep': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                              'im': 0.2983, 'TOV_wt': 0.169},
            'TODAwoShallow': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TODAwoTC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                     'im': 0.2983, 'TOV_wt': 0.169},
            'SHOTTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                             'im':  0.2983, 'TOV_wt': 0.169, 'target_cls_wt': 0.1},
            'SHOTMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                             'im': 0.2983, 'TOV_wt': 0.169, 'target_cls_wt': 0.2},
            'AaDTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'beta': 5, 'alpha': 1, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'AaDMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1, 'ent_loss_wt': 0.8467,
                         'im': 0.2983, 'TOV_wt': 0.169},
            'NRCMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5,
                        'ent_loss_wt': 0.8467,
                        'im': 0.2983, 'TOV_wt': 0.169},
            'NRCTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'epsilon': 1e-5,
                            'ent_loss_wt': 0.8467, 'im': 0.2983, 'TOV_wt': 0.10, 'Con_wt':1.0},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.0081},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'TODA': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'TODAwoShallow': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'TODAwoDeep': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.4216,
                              'im': 0.5514, 'TOV_wt': 0.6385},

            'TODAwoTC': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'SHOTTODA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.4216,
                             'im': 0.5514, 'TOV_wt': 0.6385, 'target_cls_wt': 0.0081},
            'AaDTODA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'beta': 9, 'alpha': 1,
                            'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'SHOTMAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.0081, 'TOV_wt': 0.6385},
            'AaDMAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'NRCMAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'NRCTODA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5, 'lr': 3e-4, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'RAINCOAT': {'learning_rate': 5e-4, 'src_cls_loss_wt': 0.5, 'domain_loss_wt': 0.5},
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'TODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'lr': 3e-4, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TODAwoTC': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'lr': 3e-4, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TODAwoShallow': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'lr': 3e-4, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TODAwoDeep': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'lr': 3e-4, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'SHOTTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.6709,
                             'im': 0.8969, 'TOV_wt': 0.5, 'target_cls_wt': 0.3312},
            'AaDTODA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'lr': 3e-4, 'beta': 10, 'alpha': 1,
                            'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'SHOTMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312, 'TOV_wt': 0.5},
            'AaDMAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'NRCTODA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5, 'lr': 3e-4, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'NRCMAPU': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'CoDATS': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
       }

class Sensor1():
    def __init__(self):
        super(Sensor1, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 0},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'TODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TODAwoShallow': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TODAwoTC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
           'SHOTTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                             'im':  0.2983, 'TOV_wt': 0.169, 'target_cls_wt': 0.1},
            'SHOTMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'ent_loss_wt': 0.8467,
                             'im': 0.2983, 'TOV_wt': 1, 'target_cls_wt': 0.2},
            'AaDTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'beta': 5, 'alpha': 1, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'AaDMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1, 'ent_loss_wt': 0.8467,
                         'im': 0.2983, 'TOV_wt': 1},
            'NRCMAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5,
                        'ent_loss_wt': 0.8467,
                        'im': 0.2983, 'TOV_wt': 1},
            'NRCTODA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'lr': 3e-4, 'epsilon': 1e-5,
                            'ent_loss_wt': 0.8467, 'im': 0.2983, 'TOV_wt': 0.169},
        }
