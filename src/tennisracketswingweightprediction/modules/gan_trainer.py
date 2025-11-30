import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.tennisracketswingweightprediction.modules.models import (
    Discriminator,
    Generator,
    ResidualGenerator,
    ResidualDiscriminator,
)


class GANTrainer:
    """Manages the training of a GAN with auxiliary losses for distribution matching.

    Attributes:
        generator: Generator network
        discriminator: Discriminator network
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        criterion: Loss function (BCEWithLogitsLoss or BCELoss)
        device: Compute device
        scaler: sklearn scaler for inverse transformation
    """

    def __init__(
        self,
        input_length: int,
        use_residual: bool = False,
        g_lr: float = 0.0002,
        d_lr: float = 0.0001,
        betas: Tuple[float, float] = (0.5, 0.999),
        device: str = "cpu",
        scaler: Optional[Any] = None,
        use_logits: bool = True,
        g_hidden_dim: int = 100,
        d_hidden_dim: int = 100,
        g_num_blocks: int = 3,
        d_num_blocks: int = 3,
        gradient_clip_val: float = 0.0,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        diversity_weight: float = 0.0,
        sinkhorn_weight: float = 0.0,
        correlation_weight: float = 0.0,
    ):
        """Initialize GAN trainer with models and optimizers.

        Args:
            input_length: Dimension of input/output vectors
            use_residual: Whether to use residual blocks in networks
            g_lr: Generator learning rate (TTUR: typically higher than discriminator)
            d_lr: Discriminator learning rate
            betas: Adam optimizer beta parameters
            device: Compute device ('cpu' or 'cuda')
            scaler: sklearn scaler for data transformation
            use_logits: Output logits (True) or probabilities (False)
            g_hidden_dim: Generator hidden layer size
            d_hidden_dim: Discriminator hidden layer size
            g_num_blocks: Number of blocks in generator
            d_num_blocks: Number of blocks in discriminator
            gradient_clip_val: Max gradient norm (0 to disable)
            scheduler_patience: Epochs before reducing learning rate
            scheduler_factor: Learning rate reduction factor
            diversity_weight: Weight for mean/std matching loss
            sinkhorn_weight: Weight for Sinkhorn divergence loss
            correlation_weight: Weight for correlation matrix matching loss
        """
        self.device = torch.device(device)
        self.input_length = input_length
        self.scaler = scaler
        self.gradient_clip_val = gradient_clip_val
        self.diversity_weight = diversity_weight
        self.sinkhorn_weight = sinkhorn_weight
        self.correlation_weight = correlation_weight

        if use_residual:
            self.generator = ResidualGenerator(
                input_length, hidden_dim=g_hidden_dim, num_blocks=g_num_blocks
            ).to(self.device)
            self.discriminator = ResidualDiscriminator(
                input_length,
                hidden_dim=d_hidden_dim,
                num_blocks=d_num_blocks,
                use_logits=use_logits,
            ).to(self.device)
        else:
            self.generator = Generator(
                input_length, hidden_dim=g_hidden_dim, num_blocks=g_num_blocks
            ).to(self.device)
            self.discriminator = Discriminator(
                input_length,
                hidden_dim=d_hidden_dim,
                num_blocks=d_num_blocks,
                use_logits=use_logits,
            ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=betas
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=betas
        )

        self.g_scheduler = ReduceLROnPlateau(
            self.g_optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )
        self.d_scheduler = ReduceLROnPlateau(
            self.d_optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )

        self.criterion = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

        self.history: Dict[str, List[float]] = {
            "g_loss": [],
            "d_loss": [],
            "d_real_loss": [],
            "d_fake_loss": [],
        }

    def compute_sinkhorn_loss(
        self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1, n_iters: int = 5
    ) -> torch.Tensor:
        """Compute Sinkhorn divergence between two distributions.

        Approximates the Wasserstein distance using entropy regularization.

        Args:
            x: Real data batch (B, D)
            y: Fake data batch (B, D)
            epsilon: Regularization strength
            n_iters: Number of Sinkhorn iterations

        Returns:
            Transport cost scalar
        """
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        C = torch.sum((x_col - y_lin) ** 2, dim=2)

        K = torch.exp(-C / epsilon)

        batch_size = x.size(0)
        u = torch.ones(batch_size, device=self.device) / batch_size
        v = torch.ones(batch_size, device=self.device) / batch_size

        for _ in range(n_iters):
            u = 1.0 / (torch.matmul(K, v) + 1e-8)
            v = 1.0 / (torch.matmul(K.t(), u) + 1e-8)

        transport_cost = torch.sum(u * torch.matmul(K * C, v))
        return transport_cost

    def compute_correlation_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Frobenius norm of difference between correlation matrices.

        Args:
            x: Real data batch
            y: Fake data batch

        Returns:
            Correlation structure difference
        """

        def get_corr(batch):
            batch_size = batch.size(0)
            mean = torch.mean(batch, dim=0, keepdim=True)
            batch_centered = batch - mean
            cov = torch.matmul(batch_centered.t(), batch_centered) / (batch_size - 1)
            std = torch.std(batch, dim=0, keepdim=True) + 1e-8
            corr = cov / torch.matmul(std.t(), std)
            return corr

        real_corr = get_corr(x)
        fake_corr = get_corr(y)

        return torch.norm(real_corr - fake_corr)

    def train_step(
        self,
        real_data: torch.Tensor,
        update_generator: bool = True,
        label_smoothing: bool = True,
        instance_noise_std: float = 0.0,
    ) -> Dict[str, float]:
        """Execute one training step.

        Args:
            real_data: Batch of real samples
            update_generator: Whether to update generator this step
            label_smoothing: Use 0.9 instead of 1.0 for real labels
            instance_noise_std: Std dev of noise added to inputs

        Returns:
            Dictionary of loss metrics
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        real_target = 0.9 if label_smoothing else 1.0
        real_labels = torch.full((batch_size, 1), real_target, device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)

        self.d_optimizer.zero_grad()

        if instance_noise_std > 0:
            real_data_noisy = (
                real_data + torch.randn_like(real_data) * instance_noise_std
            )
        else:
            real_data_noisy = real_data

        d_out_real = self.discriminator(real_data_noisy)
        d_loss_real = self.criterion(d_out_real, real_labels)

        noise = torch.randn(batch_size, self.input_length).to(self.device)
        fake_data = self.generator(noise)

        fake_data_detached = fake_data.detach()
        if instance_noise_std > 0:
            fake_data_noisy = (
                fake_data_detached
                + torch.randn_like(fake_data_detached) * instance_noise_std
            )
        else:
            fake_data_noisy = fake_data_detached

        d_out_fake = self.discriminator(fake_data_noisy)
        d_loss_fake = self.criterion(d_out_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()

        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.gradient_clip_val
            )

        self.d_optimizer.step()

        metrics = {
            "d_loss": d_loss.item(),
            "d_real_loss": d_loss_real.item(),
            "d_fake_loss": d_loss_fake.item(),
        }

        if update_generator:
            self.g_optimizer.zero_grad()

            gen_noise = torch.randn(batch_size, self.input_length).to(self.device)
            gen_data = self.generator(gen_noise)

            g_out = self.discriminator(gen_data)
            g_loss_fake = self.criterion(g_out, real_labels)

            aux_loss = 0.0

            if self.diversity_weight > 0:
                real_mean = torch.mean(real_data, dim=0)
                real_std = torch.std(real_data, dim=0)
                fake_mean = torch.mean(gen_data, dim=0)
                fake_std = torch.std(gen_data, dim=0)
                loss_mean = torch.norm(real_mean - fake_mean)
                loss_std = torch.norm(real_std - fake_std)
                aux_loss += self.diversity_weight * (loss_mean + loss_std)

                metrics["real_std_avg"] = real_std.mean().item()
                metrics["fake_std_avg"] = fake_std.mean().item()

            if self.sinkhorn_weight > 0:
                sinkhorn_dist = self.compute_sinkhorn_loss(real_data, gen_data)
                aux_loss += self.sinkhorn_weight * sinkhorn_dist
                metrics["sinkhorn_dist"] = sinkhorn_dist.item()

            if self.correlation_weight > 0:
                corr_loss = self.compute_correlation_loss(real_data, gen_data)
                aux_loss += self.correlation_weight * corr_loss
                metrics["corr_loss"] = corr_loss.item()

            g_loss = g_loss_fake + aux_loss

            g_loss.backward()

            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.gradient_clip_val
                )

            self.g_optimizer.step()

            metrics["g_loss"] = g_loss.item()

        return metrics

    def train_loop(
        self,
        dataloader: DataLoader,
        epochs: int,
        g_update_freq: int = 1,
        label_smoothing: bool = True,
        initial_noise_std: float = 0.1,
        noise_decay: float = 0.95,
    ):
        """Run full training loop.

        Args:
            dataloader: Training data loader
            epochs: Number of epochs to train
            g_update_freq: Update generator every N discriminator steps
            label_smoothing: Apply label smoothing
            initial_noise_std: Initial instance noise level
            noise_decay: Multiplicative decay for instance noise per epoch
        """
        self.generator.train()
        self.discriminator.train()

        total_steps = 0
        current_noise = initial_noise_std

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_d_losses = []
            epoch_g_losses = []

            real_stds = []
            fake_stds = []
            sinkhorn_dists = []

            for i, (batch_data,) in enumerate(dataloader):
                update_g = total_steps % g_update_freq == 0

                step_metrics = self.train_step(
                    batch_data,
                    update_generator=update_g,
                    label_smoothing=label_smoothing,
                    instance_noise_std=current_noise,
                )

                self.history["d_loss"].append(step_metrics["d_loss"])
                self.history["d_real_loss"].append(step_metrics["d_real_loss"])
                self.history["d_fake_loss"].append(step_metrics["d_fake_loss"])
                epoch_d_losses.append(step_metrics["d_loss"])

                if "g_loss" in step_metrics:
                    self.history["g_loss"].append(step_metrics["g_loss"])
                    epoch_g_losses.append(step_metrics["g_loss"])

                    if "real_std_avg" in step_metrics:
                        real_stds.append(step_metrics["real_std_avg"])
                        fake_stds.append(step_metrics["fake_std_avg"])

                    if "sinkhorn_dist" in step_metrics:
                        sinkhorn_dists.append(step_metrics["sinkhorn_dist"])

                total_steps += 1

            current_noise *= noise_decay
            if current_noise < 0.001:
                current_noise = 0

            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0.0

            avg_real_std = np.mean(real_stds) if real_stds else 0.0
            avg_fake_std = np.mean(fake_stds) if fake_stds else 0.0
            avg_sinkhorn = np.mean(sinkhorn_dists) if sinkhorn_dists else 0.0

            self.g_scheduler.step(avg_g_loss)
            self.d_scheduler.step(avg_d_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
            )
            if self.diversity_weight > 0:
                print(
                    f"    >> Std Dev (Real vs Fake): {avg_real_std:.4f} vs {avg_fake_std:.4f}"
                )
            if self.sinkhorn_weight > 0:
                print(f"    >> Sinkhorn Dist: {avg_sinkhorn:.4f}")

    def get_generated_samples(
        self, n_samples: int, inverse_transform: bool = True
    ) -> np.ndarray:
        """Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate
            inverse_transform: Apply scaler inverse transform if available

        Returns:
            Generated samples as numpy array
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.input_length).to(self.device)
            generated_tensor = self.generator(noise)

            generated_np = generated_tensor.cpu().numpy()

            if inverse_transform and self.scaler is not None:
                return self.scaler.inverse_transform(generated_np)

        self.generator.train()
        return generated_np

    def plot_losses(self, smoothing_window: int = 50):
        """Plot training loss history with smoothing."""
        import matplotlib.pyplot as plt
        import pandas as pd

        d_loss = pd.Series(self.history["d_loss"])
        g_loss = pd.Series(self.history["g_loss"])

        d_x = np.arange(len(d_loss))
        g_x = np.linspace(0, len(d_loss), len(g_loss))

        plt.figure(figsize=(12, 6))

        plt.plot(d_x, d_loss, label="D Loss (raw)", alpha=0.2, color="tab:blue")
        plt.plot(g_x, g_loss, label="G Loss (raw)", alpha=0.2, color="tab:orange")

        if len(d_loss) > smoothing_window:
            d_smooth = d_loss.rolling(window=smoothing_window, center=True).mean()
            plt.plot(
                d_x,
                d_smooth,
                label=f"D Loss ({smoothing_window}-avg)",
                color="tab:blue",
                linewidth=2,
            )

        if len(g_loss) > smoothing_window:
            g_smooth = g_loss.rolling(window=smoothing_window, center=True).mean()
            plt.plot(
                g_x,
                g_smooth,
                label=f"G Loss ({smoothing_window}-avg)",
                color="tab:orange",
                linewidth=2,
            )

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("GAN Training Dynamics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_distribution_comparison(
        self,
        real_data: np.ndarray,
        fake_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Visualize real vs generated data distributions.

        Args:
            real_data: Real samples (N, D)
            fake_data: Generated samples (N, D)
            feature_names: Names for each feature
        """
        import matplotlib.pyplot as plt

        num_features = real_data.shape[1]
        cols = 3
        rows = (num_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_features > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(num_features):
            ax = axes[i]
            name = feature_names[i] if feature_names else f"Feature {i}"

            ax.hist(
                real_data[:, i],
                bins=30,
                alpha=0.5,
                label="Real",
                density=True,
                color="tab:blue",
            )
            ax.hist(
                fake_data[:, i],
                bins=30,
                alpha=0.5,
                label="Generated",
                density=True,
                color="tab:orange",
            )
            ax.set_title(name)
            ax.legend()

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def evaluate_quality(
        self, real_data: np.ndarray, fake_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate quality metrics comparing real and generated data.

        Args:
            real_data: Real samples
            fake_data: Generated samples

        Returns:
            Dictionary of quality metrics
        """
        from scipy.stats import wasserstein_distance

        metrics = {}

        w_dists = [
            wasserstein_distance(real_data[:, i], fake_data[:, i])
            for i in range(real_data.shape[1])
        ]
        metrics["avg_wasserstein_dist"] = np.mean(w_dists)

        real_corr = np.corrcoef(real_data, rowvar=False)
        fake_corr = np.corrcoef(fake_data, rowvar=False)

        if np.ndim(real_corr) == 0:
            metrics["correlation_diff"] = 0.0
        else:
            real_corr = np.nan_to_num(real_corr)
            fake_corr = np.nan_to_num(fake_corr)
            metrics["correlation_diff"] = np.linalg.norm(real_corr - fake_corr)

        metrics["mean_abs_diff"] = np.mean(
            np.abs(real_data.mean(0) - fake_data.mean(0))
        )
        metrics["std_abs_diff"] = np.mean(np.abs(real_data.std(0) - fake_data.std(0)))

        return metrics
