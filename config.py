class config:
    #### DataBase ####
    MP_key = "" ## fill your own Key

    #### Graph Loader ####
    cif_folder = "./data/cif/"
    atom_init_file = "./data/atom_init.json"
    
    radius = 8
    dmin = 0
    step = .2
    max_neighbor = 12
    
    random_seed = 1234
    
    #### GCNN ####
    atom_in_dim = 92
    atom_dim = 90
    bond_dim = 41
    conv_depth = 3
    hid_dim = 180
    n_hid = 1
    
    dropout = .5
    is_cls = True
    
    #### RUN ####
    PU_iter = 50
    epoch = 30
    batch_size = 256
    # Ensemble_mode
    n_bag = 5
    bag_PU_iter = 10 
    
    #### Environment ####
    device = "cuda:0"
    result_file = None #"./result/sample.pt"
    use_board = True
    use_tqdm = True

    