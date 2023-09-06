# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Basic run (on GPU for 10 epochs for 2 class training):
EXP_NAME=sample_run
python ./speaker_reco.py --config-path='conf' --config-name='SpeakerNet_recognition_3x2x512.yaml' \
    trainer.max_epochs=10  \
    model.train_ds.batch_size=64 model.validation_ds.batch_size=64 \
    model.train_ds.manifest_filepath="<train_manifest>" model.validation_ds.manifest_filepath="<dev_manifest>" \
    model.test_ds.manifest_filepath="<test_manifest>" \
    trainer.devices=1 \
    model.decoder.params.num_classes=2 \
    exp_manager.name=$EXP_NAME +exp_manager.use_datetime_version=False \
    exp_manager.exp_dir='./speaker_exps'

See https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb for notebook tutorial

Optional: Use tarred dataset to speech up data loading.
   Prepare ONE manifest that contains all training data you would like to include. Validation should use non-tarred dataset.
   Note that it's possible that tarred datasets impacts validation scores because it drop values in order to have same amount of files per tarfile; 
   Scores might be off since some data is missing. 
   
   Use the `convert_to_tarred_audio_dataset.py` script under <NEMO_ROOT>/speech_recognition/scripts in order to prepare tarred audio dataset.
   For details, please see TarredAudioToClassificationLabelDataset in <NEMO_ROOT>/nemo/collections/asr/data/audio_to_label.py
"""

seed_everything(42)

try:
    torch.set_float32_matmul_precision("highest")
except AttributeError:
    pass


class LgaiSpeakerModel(EncDecSpeakerLabelModel):
    @typecheck()
    def forward(self, input_signal, input_signal_length):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                processed_signal, processed_signal_len = self.preprocessor(
                    input_signal=input_signal, length=input_signal_length,
                )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)

        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_output=encoded, length=length)
        return logits, embs

    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss = self.loss(logits=logits, labels=labels)
        self._accuracy(logits=logits, labels=labels)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/lr', self._optimizer.param_groups[0]['lr'], prog_bar=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        acc = self._accuracy.compute()[0]
        self._accuracy.reset()
        self.log('train/acc', acc, sync_dist=True)

    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.eval_loss(logits=logits, labels=labels)
        acc_top_k = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        self._macro_accuracy.update(preds=logits, target=labels)
        stats = self._macro_accuracy._final_state()
        self.log(f'{tag}_loss', loss_value, on_step=False, on_epoch=True, sync_dist=True)
        return {
            f'{tag}_loss': loss_value,
            f'{tag}_correct_counts': correct_counts,
            f'{tag}_total_counts': total_counts,
            f'{tag}_acc_micro_top_k': acc_top_k,
            f'{tag}_acc_macro_stats': stats,
        }


@hydra_runner(config_path="conf", config_name="titanet-large.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = LgaiSpeakerModel(cfg=cfg.model, trainer=trainer)

    # save labels to file
    if log_dir is not None:
        with open(os.path.join(log_dir, 'labels.txt'), 'w') as f:
            if speaker_model.labels is not None:
                for label in speaker_model.labels:
                    f.write(f'{label}\n')

    trainer.fit(speaker_model)

    if not trainer.fast_dev_run:
        model_path = os.path.join(log_dir, '..', 'spkr.nemo')
        speaker_model.save_to(model_path)

    torch.distributed.destroy_process_group()
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if trainer.is_global_zero:
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator, strategy=cfg.trainer.strategy)
            if speaker_model.prepare_test(trainer):
                trainer.test(speaker_model)


if __name__ == '__main__':
    main()
