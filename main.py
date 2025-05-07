import os
import argparse
import torch
import pytorch_lightning as pl
from deim_nn_model import AeroCoeffModel
from deim_nn_data import AeroCoeffDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser('AeroCoeff')
    parser.add_argument('--n_in', type=int, default=5)
    parser.add_argument('--n_out', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--n_early_stopping', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--sch_step', type=int, default=50)
    parser.add_argument('--sch_gamma', type=float, default=0.95)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--nn_dir', type=str, default='training_logs')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--restart', type=bool, default=False)
    # parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.n_fc = [args.n_in]
    args.nn_dir = os.path.join('basis'+str(args.n_in), args.nn_dir)
    for i in range(args.n_layers):
        args.n_fc.append(args.n_hidden)
    args.n_fc.append(args.n_out)
    print(args.n_fc)

    acm  = AeroCoeffModel(args)
    acdm = AeroCoeffDataModule(args)
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min',
            patience=args.n_early_stopping)
    check_point = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1,
            filename="{epoch:d}-{step:d}-{val_loss:.6e}")
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.nn_dir)
    trainer = pl.Trainer(
           accelerator='gpu',
           devices=[1],
           auto_select_gpus=True,
           max_epochs=args.max_epochs,
           log_every_n_steps=1,
           check_val_every_n_epoch=args.val_freq,
           callbacks=[early_stopping, check_point],
           logger=tb_logger,
           # callbacks=[check_point]
           enable_progress_bar=True
           )
    if args.normalize:
        acdm.normalize()
    print(acm)
    trainer.fit(model=acm, datamodule=acdm)
    val_result = trainer.validate(model=acm, ckpt_path='best', datamodule=acdm)
    test_result = trainer.test(model=acm, ckpt_path='best', datamodule=acdm)
    tb_logger.log_metrics({'Best model val_loss': val_result[0]['val_loss']}, 0)
    tb_logger.log_metrics({'Best model test_loss': test_result[0]['test_loss']}, 0)

    f = open('basis'+str(args.n_in)+'/grid_results.csv', 'a')
    f.write('{:d},{:d},{:d},{:.3e},{:.8e},{:.8e}\n'.format(trainer.logger.version, args.n_layers,\
                args.n_hidden, args.weight_decay,\
                val_result[0]['val_loss'], test_result[0]['test_loss']))
    f.close()

