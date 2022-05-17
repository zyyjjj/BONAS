import os
# modify the directory paths to get started
taskname = "supermodel_random_100"
local_root_dir = "/home/yz685/BONAS/" # root working directory
local_data_dir = "/home/yz685/BONAS/" # data root
results_dir = "trained_results"
trained_pickle_file = "trained_models.pkl"
trained_csv_file = "trained_models.csv"
logfile = 'BOGCN_open_domain.log'

# regression_type = 'quantile' # alternative: "linear"
regression_type = 'linear'
taskname += '_' + regression_type

io_config = dict(
    trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
    trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
)


# configs for BO search
search_config = dict(
    gcn_epochs=10, #TODO: reduced; epochs to train the GCN using evaluated networks
    gcn_lr=0.001,
    loss_num=3,
    generate_num=50, # TODO: reduced
    iterations=50, # total number of search iterations, #evaluated networks = #iterations x bo_sample_num
    bo_sample_num=50, # number of subnets to be selected in each BO iteration
    sample_method="random", # using random sampler or EA sampler
    if_init_samples=True, # whether use randomly selected models to initialize GCN predictor
    init_num=10, # TODO: change to larger numbers later
    regression_type = regression_type # or "linear" as in original model
)

# configs for network training (evaluation)
# TODO: these can probably be tuned
training_config = dict(
    train_supernet_epochs=10, # epochs to train the supermodel (merged by subnets) as a whole
    data_path=os.path.join(local_data_dir, 'data'),
    super_batch_size=64,
    sub_batch_size=128,
    learning_rate=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    report_freq=50,
    epochs=50, 
    init_channels=36,
    layers=20,
    drop_path_prob=0.2,
    seed=0,
    grad_clip=5,
    parallel=False,
    mode='random' # use uniform sampling or random sampling for subnet training
)

distributed = False

#OPS to allow in the search space
# TODO: can these be expanded?
OPS = ['input', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'dil_conv_3x3', 'output']
