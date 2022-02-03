"""
    @date 03.03.2021
    It extracts the acoustic features from audio files and it creates h5py files for training the PC models. 
"""
import argparse
import gc
import io
import json
import pathlib
import tarfile
import zipfile
from typing import Optional, List, Union, Tuple

import h5py
import librosa
import numpy as np
import numpy.matlib as mb

__docformat__ = ['reStructuredText']
__all__ = ['extract_acoustic_features']


def extract_acoustic_features(file_paths: Union[List[pathlib.Path], List[str]],
                              output_file: Union[pathlib.Path, str], sample_length: int,
                              zip_path: Optional[Union[pathlib.Path, str]] = None,
                              window_length: Optional[float] = 0.025, window_shift: Optional[float] = 0.01,
                              num_features: Optional[int] = 13, deltas: Optional[bool] = True,
                              deltas_deltas: Optional[bool] = True, cmvn: Optional[bool] = True,
                              name: Optional[str] = 'mfcc',
                              target_sampling_freq: Optional[int] = 16000, reset_dur: Optional[int] = 0,
                              include_indices: Optional[bool] = False,
                              limit_h5py_size: Optional[int] = -1) -> List[np.ndarray]:
    features = []
    files_names = []
    duration_h5py_file = 0

    window_length_sample = int(target_sampling_freq * window_length)
    window_shift_sample = int(target_sampling_freq * window_shift)

    if zip_path != None:
        if zipfile.is_zipfile(zip_path):
            zf = zipfile.ZipFile(zip_path)
        elif tarfile.is_tarfile(zip_path):
            zf = tarfile.TarFile(zip_path)

    for idx, file in enumerate(file_paths):
        files_names.append(file)
        if zip_path:
            with zf.open(file) as audio_file:
                file = io.BytesIO(audio_file.read())

        signal, sampling_freq = librosa.load(file, sr=target_sampling_freq)

        duration = (len(signal)/target_sampling_freq)/3600  # in hours
        duration_h5py_file += duration

        if name == 'mfcc':
            tmp_feats = librosa.feature.mfcc(signal, target_sampling_freq, n_mfcc=num_features,
                                             n_fft=window_length_sample, hop_length=window_shift_sample)
        else:
            tmp_feats = librosa.feature.melspectrogram(signal, target_sampling_freq, n_fft=window_length_sample,
                                                       hop_length=window_shift_sample, n_mels=num_features)

        if name == 'mfcc' and deltas:
            mfcc_tmp = tmp_feats
            mfcc_deltas = librosa.feature.delta(mfcc_tmp)
            tmp_feats = np.concatenate([tmp_feats, mfcc_deltas])
            if deltas_deltas:
                mfcc_deltas_deltas = librosa.feature.delta(mfcc_tmp, order=2)
                tmp_feats = np.concatenate([tmp_feats, mfcc_deltas_deltas])

        tmp_feats = np.transpose(tmp_feats)
        # Replace zeros
        min_feats = np.min(np.abs(tmp_feats[np.nonzero(tmp_feats)]))
        tmp_feats = np.where(tmp_feats == 0, min_feats, tmp_feats)

        if name == 'logmel':
            tmp_feats = 10 * np.log10(tmp_feats)

        # Normalisation
        if cmvn:
            # mean = np.expand_dims(np.mean(mfcc, axis=0), 0)
            # std = np.expand_dims(np.std(mfcc, axis=0), 0)
            mean = mb.repmat(np.mean(tmp_feats, axis=0), tmp_feats.shape[0], 1)
            std = mb.repmat(np.std(tmp_feats, axis=0), tmp_feats.shape[0], 1)
            tmp_feats = np.divide((tmp_feats - mean), std)

            # mfcc = (mfcc - mean) / std

        features.append(tmp_feats)
        print('{}/{} file processed'.format(idx, len(file_paths)))
        print(tmp_feats.shape)

        # Check h5py size limit
        if limit_h5py_size>0:
            if duration_h5py_file >= limit_h5py_size:
                _create_h5py_file(output_file, files_names, features, sample_length, reset_dur, include_indices)
                # reset values
                del features
                del files_names
                gc.collect()
                features= []
                files_names = []
                duration_h5py_file = 0

    if (limit_h5py_size>0 and duration_h5py_file>0) or limit_h5py_size<0:
        _create_h5py_file(output_file, files_names, features, sample_length, reset_dur, include_indices)

    if zip_path:
        zf.close()

    return features


def _create_h5py_file(output_path: str, file_paths: Union[List[pathlib.Path], List[str]], features: List[np.ndarray],
                     sample_length: int, reset_dur: Optional[int] = 0, include_indices: Optional[bool] = False) -> None:
    file_id = 0
    file_mapping = {}
    frame_indices = []

    for file, feature in zip(file_paths, features):
        file_mapping[file_id] = file
        frames = feature.shape[0]
        idx = np.zeros((frames, 2))
        idx[:, 0] = file_id
        idx[:, 1] = np.arange(0, frames, 1)
        frame_indices.append(idx)

        if reset_dur > 0:
            reset = np.zeros((reset_dur, 2))
            reset[:, 0] = -1
            frame_indices.append(reset)
        file_id += 1

    indices = np.concatenate(frame_indices)

    if reset_dur == 0:
        data = np.concatenate(features)
    else:
        reset = np.zeros((reset_dur, features[0].shape[-1]))
        total_files = len(features)
        [features.insert(i * 2 + 1, reset) for i in range(0, total_files)]
        data = np.concatenate(features)

    total_frames = data.shape[0]
    n_feats = data.shape[1]
    extra_frames = sample_length - (total_frames % sample_length)

    if extra_frames:
        data = np.concatenate((data, np.zeros((extra_frames, n_feats))))
        idx = np.zeros((extra_frames, 2))
        idx[:, 0] = -1
        indices = np.concatenate((indices, idx))

    total_samples = int(data.shape[0] / sample_length)
    data = data.reshape((total_samples, sample_length, -1))
    indices = indices.reshape((total_samples, sample_length, -1))

    with h5py.File(pathlib.Path(output_path), 'w') as out_file:
        out_file.create_dataset('data', data=data)
        if include_indices:
            file_mapping = np.array(list([file_mapping[i] for i in sorted(file_mapping.keys())]),
                                    dtype=h5py.special_dtype(vlen=str))
            out_file.create_dataset('file_list', data=file_mapping)
            out_file.create_dataset('indices', data=indices)


def _read_config_file(config_path: Union[pathlib.Path, str]) -> Tuple[List, Union[pathlib.Path, str],
                                                                      Union[pathlib.Path, str, None], int, dict]:
    with open(args.config) as config_file:
        config = json.load(config_file)

        audios_path = pathlib.Path(config["audios_path"])
        if audios_path.exists():
            zip_file_bool = zipfile.is_zipfile(audios_path) or tarfile.is_tarfile(audios_path)
            file_paths = []
            zip_file = None
            if not zip_file_bool:
                file_paths = list(pathlib.Path(audios_path).glob('**/*.flac|**/*.wav'))
                file_paths = sorted(file_paths)
            else:
                zip_file = audios_path
            return file_paths, config["output_file"], zip_file, config["sample_length"], config["optional_args"]
        else:
            Exception(f"audios_path: {audios_path} does not exist.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to preprocess audio files, extract acostic features and '
                                                 'create an output h5py file. '
                                                 '\nUsage: python preprocess_training_data.py '
                                                 '--config path_to_json_file')

    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    file_paths, output_file, zip_file, sample_length, optional_args = _read_config_file(args.config)
    extract_acoustic_features(file_paths, output_file, sample_length, zip_file, **optional_args)
