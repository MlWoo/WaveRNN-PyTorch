import pickle
import glob
from utils import *
from dsp import *


def convert_file(path, bits):
    wav = load_wav(path, encode=False)
    mel = melspectrogram(wav)
    quant = (wav + 1.) * (2**bits - 1) / 2
    return mel.astype(np.float32), quant.astype(np.int)


def preprocess_data(wav_files, output_path, mel_path, quant_path, bits):
    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(wav_files):
        id_ = path.split('/')[-1][:-4]
        dataset_ids += [id_]
        m, x = convert_file(path, bits)
        np.save(f'{mel_path}{id_}.npy', m)
        np.save(f'{quant_path}{id_}.npy', x)
        display('%i/%i', (i + 1, len(wav_files)))

    with open(output_path + 'dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)


def get_files(path, extension='.wav'):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames

