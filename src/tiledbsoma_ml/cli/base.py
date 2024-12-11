from click import group


DEFAULT_CENSUS_VERSION = '2024-07-01'
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_IO_CHUNK_SIZE = 2**16
DEFAULT_SHUFFLE_CHUNK_SIZE = 64


@group
def tdbsml():
    pass
