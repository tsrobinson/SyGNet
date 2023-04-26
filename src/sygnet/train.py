### TRAINING FUNCTIONS
from .requirements import *

logger = logging.getLogger(__name__)

# def train_wgan(
#     training_data, 
#     generator, 
#     critic,
#     epochs, 
#     num_cols,
#     batch_size,
#     learning_rates,
#     lmbda,
#     use_tensorboard,
#     device,
#     noise_std = 0.01,
#     ):
#     """Training function for Wasserstein-GP GAN method

#     Args:
#         training_data (Dataset): Real data used to train GAN
#         generator (nn.Module): Generator model object
#         critic (nn.Module): Critic model object
#         epochs (int): Number of training epochs
#         num_cols (list): List indicating which *numeric* columns to apply random noise to
#         batch_size (int): Number of training observations per batch
#         learning_rates (list): The learning rates for the generator and critic AdamW optimizers
#         lmbda (float): Scalar penalty term for applying gradient penalty as part of Wasserstein loss
#         use_tensorboard (boolean): If True, creates tensorboard output capturing key training metrics (default = True)
#         device (str): Either 'cuda' for GPU training, or 'cpu'

#     Note: 
#         The generator and critic models are modified in-place so this is not returned by the function

#     Returns:
#         generator_losses (list), critic_losses (list): lists of losses at the batch-level

#     """

#     data_loader = DataLoader(dataset = training_data, batch_size=batch_size, shuffle=True)

#     if use_tensorboard:
#         from torch.utils.tensorboard import SummaryWriter
#         writer = SummaryWriter(f'outputs/run_wgan_{datetime.now()}')
#     # Models
#     critic_model = critic.to(device)
#     generator_model = generator.to(device)

#     # Optimizers
#     learning_rate_g = learning_rates[0]
#     learning_rate_c = learning_rates[1]

#     generator_optimizer = torch.optim.AdamW(generator_model.parameters(),
#                                             lr=learning_rate_g, weight_decay=0.01)
#     critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=learning_rate_c,
#                                          weight_decay=0.01)
    
#     base_lambda_g = lambda epoch: 0.9995 ** epoch
#     base_lambda_c = lambda epoch: 0.9995 ** epoch

#     scheduler_g = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=base_lambda_g)
#     scheduler_c = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=base_lambda_c)

#     # Loss recording
#     generator_losses = []
#     critic_losses = []

#     num_cols = num_cols.to(device)

#     # Training loop
#     tbar = trange(epochs, desc="Epoch")
#     for epoch in tbar:

#         for i, (features, _) in enumerate(data_loader):
            
#             # Get info on batches
#             gen_obs = features.size(dim=0)
#             gen_cols = features.size(dim=1)

#             # Add random noise for stability
#             noise_g = torch.randn(size=(gen_obs, num_cols.shape[0])) * noise_std
#             noise_c = torch.randn(size=(gen_obs, num_cols.shape[0])) * noise_std
#             noise_g = noise_g.to(device)
#             noise_c = noise_c.to(device)
            
#             real_data = features.to(device)
            
#             # add gausian noise
#             real_data[:, num_cols] = real_data[:, num_cols] + noise_c

#             ## Sort out the critic
#             # Zero gradients and get critic scores
#             generator_model.zero_grad()
#             critic_model.zero_grad()

#             critic_score_real = critic_model(real_data)
            
#             fake_input = torch.rand(size=(gen_obs, gen_cols))
#             fake_input = fake_input.to(device)
#             fake_data = generator_model(fake_input).to(device)

#             critic_score_fake = critic_model(fake_data)

#             # Calculate gradient penalty
#             eps_shape = [gen_obs] + [1]*(len(features.shape)-1)
#             eps = torch.rand(eps_shape).to(device)
#             mixed_data = eps*real_data + (1-eps)*fake_data
#             mixed_output = critic_model(mixed_data)

#             grad = torch.autograd.grad(
#                 outputs = mixed_output,
#                 inputs = mixed_data,
#                 grad_outputs = torch.ones_like(mixed_output),
#                 create_graph = True,
#                 retain_graph = True,
#                 only_inputs=True,
#                 allow_unused=True
#             )[0]

#             critic_grad_penalty = ((grad.norm(2, dim=1) -1)**2)

#             error_critic = (critic_score_fake - critic_score_real).mean() + critic_grad_penalty.mean()*lmbda
#             error_critic.backward()
#             critic_optimizer.step()
#             scheduler_c.step()

#             critic_losses.append(error_critic.item())

#             ## Sort out the generator
#             # Re-zero gradients
#             generator_model.zero_grad()
#             critic_model.zero_grad()
#             # Refresh random fake data
#             fake_input2 = torch.rand(size=(gen_obs, gen_cols))
#             fake_input2 = fake_input2.to(device)
#             fake_data2 = generator_model(fake_input2).to(device)

#             neg_critic_score_fake = -critic_model(fake_data2)
#             error_gen = neg_critic_score_fake.mean()
#             error_gen.backward()
#             generator_optimizer.step()
#             scheduler_g.step()

#             generator_losses.append(error_gen.item())
            
#         if use_tensorboard:
#             writer.add_scalar('Critic loss', critic_losses[-1], global_step=epoch)
#             writer.add_scalar('Generator loss', generator_losses[-1], global_step=epoch)

#         logger.info("Epoch %s summary: Generator loss: %s; Critic loss = %s" % (epoch, round(generator_losses[-1],5), round(critic_losses[-1],5)))
#         tbar.set_postfix(loss = critic_losses[-1])
#     return generator_losses, critic_losses


def train(
    training_data, 
    generator, 
    critic,
    conditional,
    epochs, 
    num_cols,
    batch_size,
    learning_rates,
    lmbda,
    use_tensorboard,
    device,
    noise_std = 0.01,
    ):
    """Training function for Wasserstein-GP GAN method

    Args:
        training_data (Dataset): Real data used to train GAN
        generator (nn.Module): Generator model object
        critic (nn.Module): Critic model object
        epochs (int): Number of training epochs
        num_cols (list): List indicating which *numeric* columns to apply random noise to
        batch_size (int): Number of training observations per batch
        learning_rates (list): The learning rates for the generator and critic AdamW optimizers
        lmbda (float): Scalar penalty term for applying gradient penalty as part of Wasserstein loss
        use_tensorboard (boolean): If True, creates tensorboard output capturing key training metrics (default = True)
        device (str): Either 'cuda' for GPU training, or 'cpu'

    Note: 
        The generator and critic models are modified in-place so this is not returned by the function

    Returns:
        generator_losses (list), critic_losses (list): lists of losses at the batch-level

    """

    data_loader = DataLoader(dataset = training_data, batch_size=batch_size, shuffle=True)

    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f'outputs/run_wgan_{datetime.now()}')
    # Models
    critic_model = critic.to(device)
    generator_model = generator.to(device)

    # Optimizers
    learning_rate_g = learning_rates[0]
    learning_rate_c = learning_rates[1]

    generator_optimizer = torch.optim.AdamW(generator_model.parameters(),
                                            lr=learning_rate_g, weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=learning_rate_c,
                                         weight_decay=0.01)
    
    base_lambda_g = lambda epoch: 0.9995 ** epoch
    base_lambda_c = lambda epoch: 0.9995 ** epoch

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=base_lambda_g)
    scheduler_c = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=base_lambda_c)

    # Loss recording
    generator_losses = []
    critic_losses = []

    num_cols = num_cols.to(device)

    # Training loop
    tbar = trange(epochs, desc="Epoch")
    for epoch in tbar:

        for i, (features, labels) in enumerate(data_loader):
            
            # Get info on batches
            gen_obs = features.size(dim=0)
            gen_cols = features.size(dim=1)

            # Add random noise for stability
            noise_g = torch.randn(size=(gen_obs, num_cols.shape[0])) * noise_std
            noise_c = torch.randn(size=(gen_obs, num_cols.shape[0])) * noise_std
            noise_g = noise_g.to(device)
            noise_c = noise_c.to(device)

            if conditional:
                train_args = [
                    features.to(device),
                    labels.to(device)
                ]
            else:
                train_args = [
                    features.to(device)
                ]
            
            
            # add gausian noise
            train_args[0][:, num_cols] = train_args[0][:, num_cols] + noise_c

            ## Sort out the critic
            # Zero gradients and get critic scores
            generator_model.zero_grad()
            critic_model.zero_grad()

            critic_score_real = critic_model(*train_args)
            
            fake_input = torch.rand(size=(gen_obs, gen_cols))
            fake_input = fake_input.to(device)
            
            

            if conditional:
                fake_data = generator_model(fake_input, train_args[1]).to(device)
                critic_score_fake = critic_model(fake_data, train_args[1])
            else:
                fake_data = generator_model(fake_input).to(device)
                critic_score_fake = critic_model(fake_data)

            # Calculate gradient penalty
            eps_shape = [gen_obs] + [1]*(len(features.shape)-1)
            eps = torch.rand(eps_shape).to(device)
            mixed_data = eps*train_args[0] + (1-eps)*fake_data

            if conditional:
                mixed_output = critic_model(mixed_data, train_args[1])
            else:
                mixed_output = critic_model(mixed_data)

            grad = torch.autograd.grad(
                outputs = mixed_output,
                inputs = mixed_data,
                grad_outputs = torch.ones_like(mixed_output),
                create_graph = True,
                retain_graph = True,
                only_inputs=True,
                allow_unused=True
            )[0]

            critic_grad_penalty = ((grad.norm(2, dim=1) -1)**2)

            error_critic = (critic_score_fake - critic_score_real).mean() + critic_grad_penalty.mean()*lmbda
            error_critic.backward()
            critic_optimizer.step()
            scheduler_c.step()

            critic_losses.append(error_critic.item())

            ## Sort out the generator
            # Re-zero gradients
            generator_model.zero_grad()
            critic_model.zero_grad()
            
            # Refresh random fake data
            fake_input2 = torch.rand(size=(gen_obs, gen_cols))
            fake_input2 = fake_input2.to(device)
            if conditional:
                fake_data2 = generator_model(fake_input2, train_args[1]).to(device)
                neg_critic_score_fake = -critic_model(fake_data2, train_args[1])
            else:
                fake_data2 = generator_model(fake_input2).to(device)
                neg_critic_score_fake = -critic_model(fake_data2)

            error_gen = neg_critic_score_fake.mean()
            error_gen.backward()
            generator_optimizer.step()
            scheduler_g.step()

            generator_losses.append(error_gen.item())
            
        if use_tensorboard:
            writer.add_scalar('Critic loss', critic_losses[-1], global_step=epoch)
            writer.add_scalar('Generator loss', generator_losses[-1], global_step=epoch)

        logger.info("Epoch %s summary: Generator loss: %s; Critic loss = %s" % (epoch, round(generator_losses[-1],5), round(critic_losses[-1],5)))
        tbar.set_postfix(loss = critic_losses[-1])
    return generator_losses, critic_losses