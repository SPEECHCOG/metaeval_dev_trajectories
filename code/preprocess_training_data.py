"""
    @date 03.03.2021
    It extracts the acoustic features from audio files and it creates h5py files for training the PC models. 
"""
import argparse
import gc
import io
import json
import pathlib
import random
import tarfile
import warnings
import zipfile
from typing import Optional, List, Union, Tuple

import h5py
import librosa
import numpy as np
import numpy.matlib as mb

__docformat__ = ['reStructuredText']
__all__ = ['extract_acoustic_features']


def extract_acoustic_features(audio_paths: List[Tuple[str, str]],
                              compressed_files_paths: List[str],
                              output_file: Union[pathlib.Path, str], sample_length: int,
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

    single_file = True
    file_counter = 0
    output_file = pathlib.Path(output_file)

    all_compressed_files = dict()
    for file_path in compressed_files_paths:
        if zipfile.is_zipfile(file_path):
            all_compressed_files[file_path] = (zipfile.ZipFile(file_path), 'zip')
        elif tarfile.is_tarfile(file_path):
            all_compressed_files[file_path] = (tarfile.open(file_path), 'tar')

    for idx, file_path in enumerate(audio_paths):
        compressed_file_path = file_path[0]
        audio_file_path = file_path[1]
        files_names.append(compressed_file_path + '/'+ audio_file_path)  # path including compressed file
        compressed_file = all_compressed_files[compressed_file_path][0]  # tarfile or zipfile object
        compressed_file_type = all_compressed_files[compressed_file_path][1]
        if 'zip' == compressed_file_type:
            with compressed_file.open(audio_file_path) as audio_file:
                file_IO = io.BytesIO(audio_file.read())
        elif 'tar' == compressed_file_type:
            with compressed_file.extractfile(audio_file_path) as audio_file:
                file_IO = io.BytesIO(audio_file.read())
        else:
            continue

        signal, sampling_freq = librosa.load(file_IO, sr=target_sampling_freq)

        duration = (len(signal) / target_sampling_freq) / 3600  # in hours
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
        if limit_h5py_size > 0:
            if duration_h5py_file >= limit_h5py_size:
                output_file_tmp = output_file.parent.joinpath(
                    output_file.stem + f"_{file_counter}" + output_file.suffix)
                _create_h5py_file(output_file_tmp, files_names, features, sample_length, reset_dur, include_indices)
                file_counter += 1
                single_file = False
                # reset values
                del features
                del files_names
                gc.collect()
                features = []
                files_names = []
                duration_h5py_file = 0

    if (limit_h5py_size > 0 and duration_h5py_file > 0) or limit_h5py_size < 0:
        if not single_file:
            output_file_tmp = output_file.parent.joinpath(output_file.stem + f"_{file_counter}" + output_file.suffix)
        else:
            output_file_tmp = output_file
        _create_h5py_file(output_file_tmp, files_names, features, sample_length, reset_dur, include_indices)

    for file_path in all_compressed_files:
        all_compressed_files[file_path][0].close()

    return features


def _create_h5py_file(output_path: Union[pathlib.Path, str], file_paths: Union[List[pathlib.Path], List[str]],
                      features: List[np.ndarray], sample_length: int, reset_dur: Optional[int] = 0,
                      include_indices: Optional[bool] = False) -> None:
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


def _read_config_file(config_path: Union[pathlib.Path, str]) -> Tuple[
    List[Tuple[str, str]], List[str], Union[pathlib.Path, str], int, dict]:
    """
    Reads zipfile/tarfile to extract audio files names and shuffles them if indicated in the configuration file.
    """
    with open(config_path) as config_file:
        config = json.load(config_file)

        compressed_files_paths = config["audios_paths"]
        file_paths = []
        final_compressed_files_paths = []
        for compressed_file_path in compressed_files_paths:
            if pathlib.Path(compressed_file_path).exists():
                file_paths_tmp = []
                if zipfile.is_zipfile(compressed_file_path):
                    file_paths_tmp = zipfile.ZipFile(compressed_file_path).namelist()
                elif tarfile.is_tarfile(compressed_file_path):
                    file_paths_tmp = tarfile.open(compressed_file_path).getnames()
                else:
                    warnings.warn(f"compressed file format not recognised: {compressed_file_path}. File ignored")
                    continue
                final_compressed_files_paths.append(compressed_file_path)
                file_paths_tmp = [(compressed_file_path, audio_file) for audio_file in file_paths_tmp if
                                  pathlib.Path(audio_file).suffix in ['.flac', '.wav']]
                file_paths += file_paths_tmp
            else:
                warnings.warn(f"audios_path: {compressed_file_path} does not exist. File ignored")
                continue
        if config["shuffle_files"]:
            random.shuffle(file_paths)
        return file_paths, final_compressed_files_paths, config["output_file"], config["sample_length"], \
               config["optional_args"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to preprocess audio files, extract acostic features and '
                                                 'create an output h5py file. '
                                                 '\nUsage: python preprocess_training_data.py '
                                                 '--config path_to_json_file')

    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    file_paths, compressed_files_paths, output_file, sample_length, optional_args = _read_config_file(args.config)
    extract_acoustic_features(file_paths, compressed_files_paths, output_file, sample_length, **optional_args)
