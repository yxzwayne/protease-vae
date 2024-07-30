import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial


def init_layer(key, in_dim, out_dim):
    k1, k2 = random.split(key)
    w = random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = random.normal(k2, (out_dim,)) * 0.1
    return w, b


def init_vae(key, input_dim, hidden_dims, latent_dim):
    keys = random.split(key, len(hidden_dims) * 2 + 2)
    encoder_params = []
    decoder_params = []

    # Encoder
    in_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        encoder_params.append(init_layer(keys[i], in_dim, h_dim))
        in_dim = h_dim
    encoder_params.append(init_layer(keys[len(hidden_dims)], in_dim, latent_dim * 2))

    # Decoder
    in_dim = latent_dim
    for i, h_dim in enumerate(reversed(hidden_dims)):
        decoder_params.append(init_layer(keys[len(hidden_dims) + 1 + i], in_dim, h_dim))
        in_dim = h_dim
    decoder_params.append(init_layer(keys[-1], in_dim, input_dim))

    return encoder_params, decoder_params


def encoder(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    x = jnp.dot(x, w) + b
    mean, log_var = jnp.split(x, 2, axis=-1)
    return mean, log_var


def decoder(params, z):
    for w, b in params[:-1]:
        z = jnp.tanh(jnp.dot(z, w) + b)
    w, b = params[-1]
    return jnp.dot(z, w) + b


def reparameterize(key, mean, log_var):
    std = jnp.exp(0.5 * log_var)
    eps = random.normal(key, mean.shape)
    return mean + eps * std


def vae_forward(params, key, x):
    encoder_params, decoder_params = params
    mean, log_var = encoder(encoder_params, x)
    z = reparameterize(key, mean, log_var)
    recon_x = decoder(decoder_params, z)
    return recon_x, mean, log_var


def kl_divergence(mean, log_var):
    return -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var), axis=-1)


def reconstruction_loss(recon_x, x):
    return jnp.sum(jnp.square(recon_x - x), axis=-1)


def vae_loss(params, key, x):
    recon_x, mean, log_var = vae_forward(params, key, x)
    recon_loss = reconstruction_loss(recon_x, x)
    kl_loss = kl_divergence(mean, log_var)
    return jnp.mean(recon_loss + kl_loss)


@jit
def train_step(params, key, x, optimizer_state, optimizer_update, get_params):
    loss, grads = jax.value_and_grad(vae_loss)(params, key, x)
    updates, optimizer_state = optimizer_update(grads, optimizer_state, params)
    params = jax.tree_map(lambda p, u: p + u, params, updates)
    return params, optimizer_state, loss


def train_vae(
    key,
    x_train,
    input_dim,
    hidden_dims,
    latent_dim,
    batch_size,
    num_epochs,
    learning_rate,
):
    init_key, train_key = random.split(key)

    encoder_params, decoder_params = init_vae(
        init_key, input_dim, hidden_dims, latent_dim
    )
    params = (encoder_params, decoder_params)

    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    num_batches = x_train.shape[0] // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in range(num_batches):
            batch_key, train_key = random.split(train_key)
            batch_idx = random.choice(
                batch_key, x_train.shape[0], shape=(batch_size,), replace=False
            )
            x_batch = x_train[batch_idx]

            params, optimizer_state, loss = train_step(
                params,
                batch_key,
                x_batch,
                optimizer_state,
                optimizer.update,
                optimizer.params_fn,
            )
            epoch_loss += loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}")

    return params


# Example usage
key = random.PRNGKey(0)
input_dim = 99
hidden_dims = [512, 256, 128]
latent_dim = 8
batch_size = 64
num_epochs = 50
learning_rate = 1e-3

# Generate random data for demonstration
x_train = random.normal(key, (10000, input_dim))

trained_params = train_vae(
    key,
    x_train,
    input_dim,
    hidden_dims,
    latent_dim,
    batch_size,
    num_epochs,
    learning_rate,
)


# Function to generate new samples
def generate_samples(params, key, num_samples):
    _, decoder_params = params
    z = random.normal(key, (num_samples, latent_dim))
    return decoder(decoder_params, z)


# Generate 10 new samples
new_samples = generate_samples(trained_params, random.PRNGKey(1), 10)
print("Generated samples shape:", new_samples.shape)
