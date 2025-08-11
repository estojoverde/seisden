from .templates import PML_BasicCallback
import torch
import numpy as np
import math



class ModelCheckpoint(PML_BasicCallback):
    """
    Callback para salvar o modelo após cada época se a performance melhorar.
    """
    def __init__(self, filepath='model.pth', monitor='val_loss', save_best_only=True, mode='min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Modo '{mode}' não suportado. Use 'min' ou 'max'.")

    def on_epoch_end(self):
        current_score = self.state.get(self.monitor)
        if current_score is None:
            return # A métrica monitorada não está disponível

        if self.save_best_only:
            if self.monitor_op(current_score, self.best):
                print(f"\nModelCheckpoint: {self.monitor} melhorou de {self.best:.4f} para {current_score:.4f}. Salvando modelo em '{self.filepath}'")
                self.best = current_score
                torch.save(self.state['model'].state_dict(), self.filepath)
        else:
            epoch_filepath = self.filepath.replace('.pth', f'_epoch{self.state["epoch"]}.pth')
            print(f"\nModelCheckpoint: Salvando modelo da época {self.state['epoch']} em '{epoch_filepath}'")
            torch.save(self.state['model'].state_dict(), epoch_filepath)


class EarlyStopping(PML_BasicCallback):
    """
    Callback para interromper o treinamento quando uma métrica monitorada para de melhorar.
    """
    def __init__(self, monitor='val_loss', patience=5, min_delta=0, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Modo '{mode}' não suportado. Use 'min' ou 'max'.")

    def on_epoch_end(self):
        current_score = self.state.get(self.monitor)
        if current_score is None:
            return

        # Verifica se o score atual é melhor que o melhor score já visto,
        # considerando o min_delta.
        if self.monitor_op(current_score - self.min_delta, self.best):
            self.best = current_score
            self.wait = 0
            print(f"\nEarlyStopping: {self.monitor} melhorou. Resetando paciência.")
        else:
            self.wait += 1
            print(f"\nEarlyStopping: Sem melhoria em {self.monitor}. Contador: {self.wait}/{self.patience}.")
            if self.wait >= self.patience:
                print("EarlyStopping: Paciência esgotada. Interrompendo o treinamento.")
                self.state['stop_training'] = True


class LRSchedulerWarmupCosine(PML_BasicCallback):
    """
    Callback para warmup e decaimento cossenoidal do learning rate.
    """
    def __init__(self, 
                 base_lr, 
                 warmup_epochs=10, 
                 max_epochs=100, 
                 min_lr=0.0):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self):
        epoch = self.state.get('epoch', 0)
        optimizer = self.state['optimizer']

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\n[LRSchedulerWarmupCosine] Epoch {epoch}: lr set to {lr:.6f}")