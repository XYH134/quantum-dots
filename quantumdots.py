import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import matplotlib.pyplot as plt 
from IPython import display


def energy_fn(x, n, dim):
    i, j = np.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    tem=-10.0*jnp.exp(-0.1*((jnp.reshape(x,(n,dim))[:,0]-2.0)**2+(jnp.reshape(x,(n,dim))[:,1]-2.0)**2))-10.0*jnp.exp(-0.1*((jnp.reshape(x,(n,dim))[:,0]+2.0)**2+(jnp.reshape(x,(n,dim))[:,1]+2.0)**2))
    return jnp.sum(tem) + v_ee
def make_network(key, n, dim, hidden_sizes):

    '''
    z <-> x mapping
    '''
    @hk.without_apply_rng
    @hk.transform
    def network(z):
        return hk.nets.MLP(hidden_sizes + [n*dim], 
                           activation=jax.nn.tanh, 
                           w_init=hk.initializers.TruncatedNormal(0.1), 
                           b_init=hk.initializers.TruncatedNormal(2.5),
                           )(z)
    tem=jax.random.normal(key,(int(n*dim/2),))+jnp.concatenate((2.0*jnp.ones(int(n*dim/4)),-2.0*jnp.ones(int(n*dim/4))),axis=0)
    z=jnp.concatenate((tem,-tem),axis=0)
    params = network.init(key, z)
    return params, network.apply
def make_flow(network):
    def flow(params, z):
        x = network(params, z)
        jac = jax.jacfwd(network,argnums=1)(params, z)
        _, logabsdet = jnp.linalg.slogdet(jac)
        try:            
            1/jnp.linalg.det(jac)        
        except ZeroDivisionError:            
            return z,jnp.sum(jax.scipy.stats.norm.logpdf(z))        
        else:            
            return x, jnp.sum(jax.scipy.stats.norm.logpdf(z)) - logabsdet
    return flow
def make_loss(batch_flow, n, dim, beta):
    batch_energy = jax.vmap(energy_fn, (0, None, None), 0)
    def loss(params, z):
        x, logp = batch_flow(params, z)
        energy = batch_energy(x, n, dim)
        f = logp/beta + energy
        return jnp.mean(f), (jnp.std(f)/jnp.sqrt(x.shape[0]), x)
    return loss
batchsize = 8192
n = 20
dim = 2 
beta = 10.0
hidden_sizes = [64, 64]
key = jax.random.PRNGKey(42)
params, network = make_network(key, n, dim, hidden_sizes)
flow = make_flow(network)
batch_flow = jax.vmap(flow, (None, 0), (0, 0))
loss = make_loss(batch_flow, n, dim, beta)
value_and_grad = jax.value_and_grad(loss, has_aux=True)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(key, params, opt_state):
    tem=jax.random.normal(key,(batchsize,int(n*dim/2)))+jnp.concatenate((2.0*jnp.ones((batchsize,int(n*dim/4))),-2.0*jnp.ones((batchsize,int(n*dim/4)))),axis=1)
    z=jnp.concatenate((tem,-tem),axis=1)
    value, grad = value_and_grad(params, z)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return value, params, opt_state
loss_history = []
for i in range(200):
    key, subkey = jax.random.split(key)
    value,  params, opt_state = step(subkey, params, opt_state)
    f_mean, (f_err, x) = value
    loss_history.append([f_mean, f_err])
    print(i, f_mean)
    x = jnp.reshape(x, (batchsize*n, dim)) 
    display.clear_output(wait=True)
    if i % 10 == 0:
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1],bins=100,range=((-4, 4), (-4, 4)),density=True)
        plt.imshow(H, interpolation="nearest", 
               extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
               cmap="inferno")
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.subplot(1, 2, 2)
        y = np.reshape(np.array(loss_history), (-1, 2))
        plt.errorbar(np.arange(i+1), y[:, 0], yerr=y[:, 1], marker='o', capsize=8)
        plt.xlabel('epochs')
        plt.ylabel('variational free energy')
        plt.savefig("quantumdot6.png")      
print(loss_history[-1])
x = x.reshape(batchsize, n, dim)
jax.vmap(energy_fn, (0, None, None), 0)(x, n, dim).mean()
