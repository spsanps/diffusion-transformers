# Based on:
# https://github.com/albertfgu/diffwave-sashimi

import inspect

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def calc_diffusion_hyperparams(T, beta_0, beta_T, beta=None, fast=False):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    if fast and beta is not None:
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
            1 - Alpha_bar[t]
        )  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
        T,
        Beta.cuda(),
        Alpha.cuda(),
        Alpha_bar.cuda(),
        Sigma,
    )
    return _dh


def convert_to_diffusion(model_class):
    """
    Converts a huggingface model to a diffusion model class
    """

    # check if it is a class and not an instance
    assert inspect.isclass(model_class), "model_class must be a class"

    class DiffusionModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Set default configs if not provided
            # T: 200
            # beta_0: 0.0001
            # beta_T: 0.02
            # beta: null

            # set diffusion parameters
            self.config.T = kwargs.get("T", 200)
            self.config.beta_0 = kwargs.get("beta_0", 0.0001)
            self.config.beta_T = kwargs.get("beta_T", 0.02)
            self.config.beta = kwargs.get("beta", None)

            # set training parameters
            self.config.lr = kwargs.get("lr", 2e-4)
            self.config.epochs = kwargs.get("epochs", 10)
            self.config.batch_size = kwargs.get("batch_size", 8)

            # set token length
            self.config.max_length = kwargs.get("max_length", 128)

            # device
            self.config.device = kwargs.get("device", "cuda")

            # send to device
            self.to(self.config.device)

        def training_loss(
            self, loss_fn, input_tensor, diffusion_hyperparams, file_log=False
        ):
            """
            Compute the training loss of epsilon and epsilon_theta

            Parameters:
            net (torch network):              the network model
            loss_fn (torch loss function):    the loss function, default is nn.MSELoss()
            input_tensor (torch.tensor):      training data, shape can be arbitrary but should match with what the network expects
            diffusion_hyperparams (dict):     dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                            note, the tensors need to be cuda tensors

            Returns:
            training loss
            """

            _dh = diffusion_hyperparams
            T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

            B = input_tensor.shape[
                0
            ]  # B is batchsize, assumes that batchsize is the first dimension
            diffusion_steps = torch.randint(
                T, size=(B, 1, 1)
            ).cuda()  # randomly sample diffusion steps from 1~T

            # convert to FloatTensor
            diffusion_embed = diffusion_steps.type(torch.cuda.FloatTensor)

            ## Can we start with FloatTensor instead of converting?
            # diffusion_steps = torch.rand(B, self.config.max_length, self.config.d_model).cuda() * T

            # analysis shows that the input tensor has variance 0.3
            # rescale the input tensor to have variance 1
            input_tensor = input_tensor  # / torch.sqrt(torch.tensor(0.3)).cuda()
            # normalize the input tensor to have mean 0 and variance 1
            input_tensor = (input_tensor - torch.mean(input_tensor)) / torch.std(
                input_tensor
            )

            z = torch.normal(0, 1, size=input_tensor.shape).cuda()  # [:2] + (1,)
            transformed_input = (
                torch.sqrt(Alpha_bar[diffusion_steps]) * input_tensor
                + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
            ).cuda()
            # compute x_t from q(x_t|x_0)

            # compute mean and variance of transformed input per sample,
            # ie output should be of shape B, 1, 1

            if file_log:
                mean_transformed_input = torch.mean(transformed_input, dim=[1, 2])
                var_transformed_input = torch.var(transformed_input, dim=[1, 2])
                with open("diffusion_input.txt", "a") as f:
                    # f.write("input")
                    f.write(str(diffusion_steps))
                    f.write("\n")
                    f.write(str(mean_transformed_input))
                    f.write("\n")
                    f.write(str(var_transformed_input))
                    f.write("\n\n")

            epsilon_theta = self.forward(
                transformed_input, diffusion_embed
            )  # predict \epsilon according to \epsilon_\theta

            if file_log:
                # compute mean and variance of epsilon_theta
                mean_epsilon_theta = torch.mean(epsilon_theta, dim=[1, 2])
                var_epsilon_theta = torch.var(epsilon_theta, dim=[1, 2])
                with open("diffusion_output.txt", "a") as f:
                    # f.write("output")
                    f.write(str(diffusion_steps))
                    f.write("\n")
                    f.write(str(mean_epsilon_theta))
                    f.write("\n")
                    f.write(str(var_epsilon_theta))
                    f.write("\n\n")

            return loss_fn(epsilon_theta, input_tensor)

        def train_diffusion(
            self, dataset, save_path, wandb_cfg=None, log_interval=100, *args, **kwargs
        ):
            """
            Train the model with diffusion
            """
            self.diffusion_hyperparams = calc_diffusion_hyperparams(
                T=self.config.T,
                beta_0=self.config.beta_0,
                beta_T=self.config.beta_T,
                beta=self.config.beta,
                fast=False,
            )

            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

            loss_fn = torch.nn.MSELoss()

            # Initialize WandB if config provided
            if wandb_cfg:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=wandb_cfg,
                )
                wandb.watch(self)

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            for epoch in range(self.config.epochs):
                epoch_loss = 0
                step = 0

                for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                    self.optimizer.zero_grad()

                    input_tensor = batch[
                        "input"
                    ].cuda()  # Replace with the actual input tensor from your dataset
                    loss = self.training_loss(
                        loss_fn, input_tensor, self.diffusion_hyperparams
                    )

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    step += 1

                    # Log every log_interval steps
                    if step % log_interval == 0:
                        avg_loss = epoch_loss / step
                        print(f"Epoch {epoch}, Step {step} - Avg Loss: {avg_loss}")

                        # Log to WandB
                        if wandb_cfg:
                            wandb.log(
                                {"epoch": epoch, "step": step, "avg_loss": avg_loss}
                            )

                # Log at the end of the epoch
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch} - Loss: {avg_loss}")

                # Log to WandB
                if wandb_cfg:
                    wandb.log({"epoch": epoch, "loss": avg_loss})

            # save model
            torch.save(self.state_dict(), save_path)

            # Finish WandB run
            if wandb_cfg:
                wandb.finish()

        def generate_diffusion(self, n, save_path):
            # use the diffusion process to generate samples
            # n: number of samples to generate

            # initialize the diffusion process
            # diffusion hyperparameters are already stored in self.diffusion_hyperparams

            # initialize the input tensor
            z = torch.normal(0, 128, 768).cuda()

            # input_tensor is a copy of z
            input_tensor = z + 0
            

            # do noising and denoising diffusion_steps
            # ie reverse the diffusion process

            T = self.diffusion_hyperparams["T"]

            for t in range(T - 1, 0, -1):
                diffusion_embed = torch.tensor(
                    [t]*n,
                    dtype=torch.float32,
                ).cuda()
                # convert to shape (n, 1, 1)
                diffusion_embed = diffusion_embed.view(-1, 1, 1)
                input_tensor = self.forward(input_tensor, diffusion_embed)

                # add noise
                input_tensor = (
                    torch.sqrt(self.diffusion_hyperparams["Alpha_bar"][t])
                    * input_tensor
                    + torch.sqrt(1 - self.diffusion_hyperparams["Alpha_bar"][t]) * z
                )

            # inference for t=0
            diffusion_embed = torch.tensor([0] * n, dtype=torch.float32).cuda()
            input_tensor = self.forward(input_tensor, diffusion_embed)

            # save as binary file with each sample as a row
            with open(save_path, "wb") as f:
                for i in range(n):
                    f.write(input_tensor[i].cpu().numpy().tobytes())

    return DiffusionModel
