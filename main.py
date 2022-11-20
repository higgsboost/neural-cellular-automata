import cv2
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training import checkpoints, train_state
import optax

from dataclasses import dataclass

@dataclass
class Config:
    state_grid_w: int = 50 
    state_grid_h: int = 50 
    kernel_size: int = 3
    alpha_channel: int = 3
    pv_len: int = 16 # Length of perception vector
    
    @property
    def cx(self):
        return self.state_grid_w//2 

    @property
    def cy(self):
        return self.state_grid_h//2

config = Config()

def draw(x):
    img = (np.array(x[:,:,:3])  * 255).astype(np.uint8)
    print(img.shape)
    img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('im', img)
    cv2.waitKey(1)

def conv2d(x, kernel):
    # TODO: remove transpose if everything is same
    x = x[..., jnp.newaxis]
    x = jnp.transpose(x, [0,3,1,2]) 
    kernel = jnp.transpose(kernel, [3,2,0,1])
    x = jax.lax.conv(x, kernel, (1, 1), "SAME") 
    x = jnp.transpose(x, [0, 2, 3, 1])
    return x

@jax.jit
def perceive(state_grid):
    state_grid = state_grid[jnp.newaxis, ...]
    sobel_x = jnp.array([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ])[..., jnp.newaxis, jnp.newaxis]
    #sobel_x = jnp.broadcast_to(sobel_x[..., jnp.newaxis, jnp.newaxis]) # , [3, 3, state_grid.shape[-1], 1])
    sobel_y = jnp.transpose(sobel_x, [1, 0, 2, 3])


    print('soble_x :', sobel_x.shape)
    print('soble_y :', sobel_y.shape)
   
    grad_x = jax.vmap(lambda state_grid: conv2d(state_grid, sobel_x), in_axes=(-1,))(state_grid)
    grad_x = jnp.transpose(grad_x, [1, 2, 3, 4, 0])

    grad_y = jax.vmap(lambda state_grid: conv2d(state_grid, sobel_y), in_axes=(-1,))(state_grid)
    grad_y = jnp.transpose(grad_y, [1, 2, 3, 4, 0])

    grad_x = jnp.squeeze(grad_x, 3)
    grad_y = jnp.squeeze(grad_y, 3)

    perception_grid = jnp.concatenate(
        [state_grid, grad_x, grad_y],
        axis=-1
    )
    return perception_grid[0]


def stochastic_update(key, state_grid, ds):
    mask = jax.random.uniform(key, state_grid.shape[:-1])
    print('mask shape', mask.shape)
    ds = mask[..., jnp.newaxis] * ds 
    state_grid = state_grid + ds
    return state_grid

def alive_masking(state_grid):
    alive = jax.lax.reduce_window(state_grid[..., 3], 0., jax.lax.max, (3,3,), (1,1,), 'SAME')    
    alive = alive > 0.1 
    state_grid = alive[..., jnp.newaxis] * state_grid
    return state_grid

def get_mlp(len_pv):
    class Model(nn.Module):
    
      @nn.compact
      def __call__(self, z):
        z = nn.Dense(128)(z)
        z = nn.relu(z)
        z = nn.Dense(len_pv)(z)
        return z 
    
    model = Model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, len_pv * 2 + len_pv)))
    return model, params

model, params = get_mlp(config.pv_len)

def update(model, params, pv):
    ds = model.apply(params, pv)
    return ds
#kernel = jnp.array([[1, 1, 0],

#                     [1, 0,-1],
#                     [0,-1,-1]]* 3)[:, :, jnp.newaxis]
##kernel = jax.random.uniform(key, (1, 3, 3, 3))
#kernel = jnp.transpose(kernel, [3,2,0,1])
#print(kernel.shape)

key = jax.random.PRNGKey(1)


state_grid = jnp.zeros((config.state_grid_h, config.state_grid_w, config.pv_len))

#state_grid = state_grid.at[:,:, 3].set(jax.random.uniform(key, (300, 300)))
state_grid = state_grid.at[config.cy,config.cx, config.alpha_channel].set(1.)

tx = optax.adam(0.001)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)
 
def run_one(key, state, state_grid, num_steps):

    def loss_fn(params):
        x = state_grid
        _key = key
        for _ in range(num_steps):
            pg = perceive(x)
            ds = update(model, params, pg)
            x= stochastic_update(_key, x, ds)
            x = alive_masking(x)

            _key, _ = jax.random.split(_key)
        return state_grid
    
    loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
run_one(key, state, state_grid, 96)

#params = init_network_params([16*2 + 16, 128, 16], key)
'''
for i in range(10000):

    print('-------')
    print(state_grid.shape)
    #image = image[..., :3] 
    pg = perceive(state_grid)
     
    ds = update(model, params, pg)
    state_grid = stochastic_update(key, state_grid, ds)
    state_grid = alive_masking(state_grid)
    #image = conv2d(image, kernel)
    #image = jnp.transpose(image, [0,3,1,2]) 
    #image = jax.lax.conv(image, kernel, (1, 1), "SAME") 
    #image = jnp.transpose(image, [0, 2, 3, 1])
    #import pdb; pdb.set_trace()
    draw(state_grid[:,:,:]) 
    key, _ = jax.random.split(key)
'''
