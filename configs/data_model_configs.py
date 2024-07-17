def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14"),("2", "11"),("9", "5"),("12", "11"),("17", "11"),("16", "11"),("6", "9")]
        # self.scenarios = [("2", "11"),("9", "5"),("12", "11"),("17", "11"),("16", "11"),("6", "9")]
        # self.scenarios = [("12", "5"),  ("7", "18"),("6", "9")]
        # self.scenarios = [("9", "14")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2
        self.batch_size = 32

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 65 # for my model
        self.AR_hid_dim = 8

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1

        #Content parameters
        self.temperature = 0.2
        self.use_cosine_similarity = True

        #增强
        self.jitter_ratio = 0.001 #默认
        # self.jitter_ratio = 0.1
        # self.max_seg = 5
        self.max_seg = 15  #默认
        self.timesteps = 10
        self.num_splits = 4

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        # self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        # self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0")]
        # self.scenarios = [("2", "0")]
        self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3"), ("1", "3"),("0", "2"),("0", "3"),("2", "0"),("2", "1"), ("3", "0")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5
        self.batch_size = 32

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 109

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.AR_hid_dim = 128
        #增强的相关参数
        self.jitter_ratio = 0.001 #默认
        # self.jitter_ratio = 0.1
        self.max_seg = 15   #默认

        # self.max_seg = 15

        #Content parameters
        self.temperature = 0.2
        self.use_cosine_similarity = True

        self.timesteps = 10
        self.num_splits = 4

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),("3", "10"),("11", "9"),("5", "10"),("7", "10"),("10", "16"),("5", "9")]
        # self.scenarios = [("3", "10"),("11", "9"),("5", "10"),("7", "10"),("10", "16"),("5", "9")]
        # self.scenarios = [("5", "1")]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.batch_size = 32
        self.fourier_modes = 64

        self.dim1 = 128
        self.dim2 = 128
        # self.dim3 = 128  # Deep_Coral
        self.dim3 = 256  #raincoat

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        # self.features_len = 1
        self.features_len = 18 #for sequential methods
        self.AR_hid_dim = 128


        # Content parameters
        self.temperature = 0.2
        self.use_cosine_similarity = True

        #增强
        self.jitter_ratio = 0.8  #默认
        # self.jitter_ratio = 0.1
        self.max_seg = 8   #默认
        # self.max_seg = 5
        self.timesteps = 6
        self.num_splits = 4

class Sensor1():
    def __init__(self):
        super(Sensor1, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['0', '1', '2', '3', '4']
        self.sequence_len = 300
        # self.scenarios = [("1", "9"),("2", "9"),("9", "1"), ("5", "9"),("3", "6"),("6", "1"), ("4", "5"),("1", "2"), ("1", "5"),("2", "1"), ("2", "3"), ("2", "4"), ("2", "5"), ("3", "1"), ("3", "2"), ("3", "4"), ("3", "5"), ("4", "1"), ("4", "2")]
        # self.scenarios = [("2", "3"), ("3", "4"), ("4", "5")]
        # self.scenarios = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"),("5", "1")]
        self.scenarios = [("5", "1")]
        # self.scenarios = [("10", "9"),("2", "10"),("10", "1")]
        # self.scenarios = [ ("1", "5"), ("2", "1"), ("3", "2"), ("3", "5"), ("4", "1"), ("4", "2"), ("2", "10")]

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2
        self.batch_size = 32
        self.fourier_modes = 150

        self.dim1 = 64
        self.dim2 = 300
        # self.dim3 = 364   #Raincoat
        self.dim3 = 64  # DIRT

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 8 # for my model
        self.AR_hid_dim = 8

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1

        #Content parameters
        self.temperature = 0.2
        self.use_cosine_similarity = True

        #增强
        self.jitter_ratio = 0.001
        # self.jitter_ratio = 0.01
        self.max_seg = 5

        self.timesteps = 4
        self.num_splits = 10

